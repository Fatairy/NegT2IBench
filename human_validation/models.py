"""Image/text metric implementations used by the human-metric comparison.

Models are deliberately loaded lazily: importing this module does not allocate a
GPU or require transformers.  Every scorer accepts a Hugging Face model id *or*
a local checkpoint directory, which also makes weight-only variants cheap to add.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from PIL import Image


@dataclass(frozen=True)
class MetricSpec:
    key: str
    display_name: str
    default_checkpoint: str | None
    scorer_class: type["ImageTextScorer"]
    default_model_name: str | None = None


class ImageTextScorer(ABC):
    """Common interface for metrics producing one score per image/caption pair."""

    def __init__(
        self,
        checkpoint: str,
        *,
        model_name: str | None = None,
        device: str = "cuda",
        dtype: str = "auto",
        cache_dir: str | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.model_name = model_name
        self.device = torch.device(device)
        self.dtype = _resolve_dtype(dtype, self.device)
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self) -> "ImageTextScorer":
        """Load processor and weights and return self."""

    @abstractmethod
    def score(self, images: Sequence["Image.Image"], captions: Sequence[str], **kwargs) -> list[float]:
        """Score aligned image/caption pairs."""

    def _model_kwargs(self) -> dict:
        kwargs = {"cache_dir": self.cache_dir}
        if self.dtype != torch.float32:
            kwargs["torch_dtype"] = self.dtype
        return kwargs


class CLIPScoreScorer(ImageTextScorer):
    """Reference CLIPScore: 2.5 * max(cosine(image, text), 0)."""

    def load(self) -> "CLIPScoreScorer":
        from evaluation.clip_utils import load_open_clip

        self.model, self.processor, self.tokenizer = load_open_clip(
            self.model_name or "ViT-L-14", pretrained=self.checkpoint,
            device=str(self.device), cache_dir=self.cache_dir,
            precision={torch.float32: "fp32", torch.float16: "fp16",
                       torch.bfloat16: "bf16"}[self.dtype])
        return self

    @torch.inference_mode()
    def score(self, images: Sequence["Image.Image"], captions: Sequence[str], **kwargs) -> list[float]:
        _validate_batch(images, captions)
        image_tensor = torch.stack([self.processor(image) for image in images]).to(
            device=self.device, dtype=self.dtype)
        tokens = self.tokenizer(list(captions)).to(self.device)
        image_features = F.normalize(self.model.encode_image(image_tensor), dim=-1)
        text_features = F.normalize(self.model.encode_text(tokens), dim=-1)
        return (2.5 * (image_features * text_features).sum(-1).clamp(min=0)).float().cpu().tolist()


class BLIPScoreScorer(ImageTextScorer):
    """BLIP image-text contrastive (ITC) cosine similarity."""

    def load(self) -> "BLIPScoreScorer":
        from transformers import AutoProcessor, BlipForImageTextRetrieval

        self.processor = AutoProcessor.from_pretrained(self.checkpoint, cache_dir=self.cache_dir)
        self.model = BlipForImageTextRetrieval.from_pretrained(
            self.checkpoint, **self._model_kwargs())
        self.model.eval().to(self.device)
        return self

    @torch.inference_mode()
    def score(self, images: Sequence["Image.Image"], captions: Sequence[str], **kwargs) -> list[float]:
        _validate_batch(images, captions)
        inputs = self.processor(
            text=list(captions), images=list(images), return_tensors="pt", padding=True,
            truncation=True,
        ).to(self.device)
        inputs["pixel_values"] = inputs.pixel_values.to(dtype=self.dtype)
        vision_outputs = self.model.vision_model(pixel_values=inputs.pixel_values)
        image_features = F.normalize(self.model.vision_proj(vision_outputs.last_hidden_state[:, 0, :]), dim=-1)
        text_outputs = self.model.text_encoder(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            return_dict=True,
        )
        text_features = F.normalize(self.model.text_proj(text_outputs.last_hidden_state[:, 0, :]), dim=-1)
        return (image_features * text_features).sum(-1).float().cpu().tolist()


class OpenCLIPCosineScorer(CLIPScoreScorer):
    """Raw cosine score for compatible CLIP-family fine-tuned checkpoints."""

    def load(self) -> "OpenCLIPCosineScorer":
        self.checkpoint = _resolve_checkpoint(self.checkpoint, self.cache_dir)
        super().load()
        return self

    @torch.inference_mode()
    def score(self, images: Sequence["Image.Image"], captions: Sequence[str], **kwargs) -> list[float]:
        _validate_batch(images, captions)
        image_tensor = torch.stack([self.processor(image) for image in images]).to(
            device=self.device, dtype=self.dtype)
        tokens = self.tokenizer(list(captions)).to(self.device)
        image_features = F.normalize(self.model.encode_image(image_tensor), dim=-1)
        text_features = F.normalize(self.model.encode_text(tokens), dim=-1)
        return (image_features * text_features).sum(-1).float().cpu().tolist()


class ImageRewardScorer(ImageTextScorer):
    """Official ImageReward BLIP cross-attention encoder plus MLP reward head."""

    def load(self) -> "ImageRewardScorer":
        try:
            import ImageReward as RM
        except ImportError as exc:
            raise RuntimeError("ImageReward requires the `image-reward` package") from exc
        self.model = RM.load(
            self.checkpoint, device=self.device,
            download_root=self.cache_dir,
        )
        return self

    @torch.inference_mode()
    def score(self, images: Sequence["Image.Image"], captions: Sequence[str], **kwargs) -> list[float]:
        _validate_batch(images, captions)
        # The official implementation only batches images sharing one prompt;
        # our aligned pairs generally have different prompts, so score per pair.
        return [float(self.model.score(caption, image))
                for image, caption in zip(images, captions)]


class PickScoreScorer(ImageTextScorer):
    """PickScore's fine-tuned CLIP-H embeddings and learned logit scale."""

    PROCESSOR_CHECKPOINT = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    def load(self) -> "PickScoreScorer":
        from transformers import AutoModel, AutoProcessor

        processor_checkpoint = self.model_name or self.PROCESSOR_CHECKPOINT
        self.processor = AutoProcessor.from_pretrained(
            processor_checkpoint, cache_dir=self.cache_dir)
        self.model = AutoModel.from_pretrained(
            self.checkpoint, **self._model_kwargs()).eval().to(self.device)
        return self

    @torch.inference_mode()
    def score(self, images: Sequence["Image.Image"], captions: Sequence[str], **kwargs) -> list[float]:
        _validate_batch(images, captions)
        image_inputs = self.processor(
            images=list(images), padding=True, truncation=True, max_length=77,
            return_tensors="pt",
        ).to(self.device)
        image_inputs["pixel_values"] = image_inputs.pixel_values.to(dtype=self.dtype)
        text_inputs = self.processor(
            text=list(captions), padding=True, truncation=True, max_length=77,
            return_tensors="pt",
        ).to(self.device)
        image_features = F.normalize(self.model.get_image_features(**image_inputs), dim=-1)
        text_features = F.normalize(self.model.get_text_features(**text_inputs), dim=-1)
        # Return the unnormalised per-pair logit, not softmax probability: with
        # one image per prompt the latter would always equal one.
        logits = self.model.logit_scale.exp() * (image_features * text_features).sum(-1)
        return logits.float().cpu().tolist()


class HPSv2Scorer(ImageTextScorer):
    """Official HPS v2.1 ViT-H/14 preference model, loaded once per run."""

    def load(self) -> "HPSv2Scorer":
        # hpsv2 1.2.0 wheels occasionally omit this tokenizer asset. Its
        # bundled open_clip fork expects the standard OpenCLIP vocabulary,
        # which is already installed by open-clip-torch.
        import site
        for base in site.getsitepackages():
            target = Path(base) / "hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz"
            source = Path(base) / "open_clip/bpe_simple_vocab_16e6.txt.gz"
            if not target.exists() and source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                break
        try:
            from hpsv2 import img_score as hps
        except ImportError as exc:
            raise RuntimeError("HPSv2 requires the `hpsv2` package") from exc

        # Reuse the official package's exact architecture and preprocessing, but
        # load the checkpoint once. hpsv2.score() reloads it on every call.
        hps.initialize_model()
        self.model = hps.model_dict["model"]
        self.processor = hps.model_dict["preprocess_val"]
        self.tokenizer = hps.get_tokenizer("ViT-H-14")
        checkpoint_path = _resolve_checkpoint(self.checkpoint, self.cache_dir)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval().to(self.device)
        self.checkpoint = checkpoint_path
        return self

    @torch.inference_mode()
    def score(self, images: Sequence["Image.Image"], captions: Sequence[str], **kwargs) -> list[float]:
        _validate_batch(images, captions)
        image_tensor = torch.stack([self.processor(image) for image in images]).to(self.device)
        text_tokens = self.tokenizer(list(captions)).to(self.device)
        amp = (torch.autocast("cuda", dtype=self.dtype)
               if self.device.type == "cuda" and self.dtype != torch.float32
               else nullcontext())
        with amp:
            outputs = self.model(image_tensor, text_tokens)
        image_features = outputs["image_features"]
        text_features = outputs["text_features"]
        return (image_features * text_features).sum(-1).float().cpu().tolist()


METRICS: dict[str, MetricSpec] = {
    "clipscore": MetricSpec(
        "clipscore", "CLIPScore", "openai", CLIPScoreScorer),
    "blipscore": MetricSpec(
        "blipscore", "BLIPScore", "Salesforce/blip-itm-base-coco", BLIPScoreScorer),
    "negclip": MetricSpec(
        "negclip", "NegCLIP",
        "hf://jerryray/negationclip/negationclip_ViT-B32.pth",
        OpenCLIPCosineScorer, "ViT-B-32-quickgelu"),
    # No official public TNG-CLIP checkpoint could be verified. The backend is
    # ready, but callers must provide the released/local weights explicitly.
    "tngclip": MetricSpec(
        "tngclip", "TNG-CLIP", None, OpenCLIPCosineScorer,
        "ViT-B-32-quickgelu"),
    "imagereward": MetricSpec(
        "imagereward", "ImageReward", "ImageReward-v1.0", ImageRewardScorer),
    "pickscore": MetricSpec(
        "pickscore", "PickScore", "yuvalkirstain/PickScore_v1", PickScoreScorer),
    "hpsv2": MetricSpec(
        "hpsv2", "HPSv2",
        "hf://xswu/HPSv2/HPS_v2.1_compressed.pt", HPSv2Scorer,
        "ViT-H-14"),
}


def load_metric(
    key: str,
    *,
    checkpoint: str | None = None,
    model_name: str | None = None,
    device: str = "cuda",
    dtype: str = "auto",
    cache_dir: str | None = None,
) -> ImageTextScorer:
    """Instantiate and load a registered metric, optionally replacing its weights."""
    try:
        spec = METRICS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown metric {key!r}; choose from {', '.join(METRICS)}") from exc
    selected_checkpoint = checkpoint or spec.default_checkpoint
    if selected_checkpoint is None:
        raise ValueError(
            f"Metric {key!r} has no verified public default checkpoint; pass --checkpoint")
    return spec.scorer_class(
        selected_checkpoint,
        model_name=model_name or spec.default_model_name,
        device=device,
        dtype=dtype,
        cache_dir=cache_dir,
    ).load()


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    choices = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    try:
        dtype = choices[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; choose from auto, {', '.join(choices)}") from exc
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 inference on CPU is unsupported; use float32 or bfloat16")
    return dtype


def _validate_batch(images: Sequence["Image.Image"], captions: Sequence[str]) -> None:
    if not images or len(images) != len(captions):
        raise ValueError("images and captions must be non-empty and have equal lengths")


def _resolve_checkpoint(checkpoint: str, cache_dir: str | None) -> str:
    """Resolve ``hf://repo/name/file`` while leaving tags/local paths intact."""
    if not checkpoint.startswith("hf://"):
        return checkpoint
    from huggingface_hub import hf_hub_download

    repo_and_file = checkpoint.removeprefix("hf://").split("/")
    if len(repo_and_file) < 3:
        raise ValueError(f"Invalid Hugging Face checkpoint URI: {checkpoint}")
    repo_id = "/".join(repo_and_file[:2])
    filename = "/".join(repo_and_file[2:])
    return str(Path(hf_hub_download(
        repo_id=repo_id, filename=filename, cache_dir=cache_dir)))
