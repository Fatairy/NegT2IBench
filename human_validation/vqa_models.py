"""Yes/No probability adapters for rows 8--11 of the human comparison.

Every backend returns ``P(Yes) / (P(Yes) + P(No))`` from model likelihoods.
Generated strings are never parsed. The implementations follow the official
CLIP-FlanT5/t2v-metrics, mPLUG, and InternVL inference paths.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class VQAMetric:
    name: str
    model_name: str
    device: str = "cuda"
    checkpoint: str | None = None
    cache_dir: str | None = None
    backend_root: str | None = None

    def load(self) -> "VQAMetric":
        raise NotImplementedError

    def yes_probability(self, image_paths: Sequence[str], questions: Sequence[str]) -> list[float]:
        raise NotImplementedError


def _normalise_likelihoods(yes: Sequence[float], no: Sequence[float]) -> list[float]:
    if len(yes) != len(no):
        raise ValueError("Yes and No likelihood lists have different lengths")
    values = []
    for y, n in zip(yes, no):
        y, n = float(y), float(n)
        if not math.isfinite(y) or not math.isfinite(n) or y < 0 or n < 0:
            raise ValueError(f"Invalid Yes/No likelihoods: {y}, {n}")
        values.append(y / (y + n) if y + n > 0 else 0.5)
    return values


class T2VMetricsVQA(VQAMetric):
    """Official t2v-metrics scorer (Qwen2.5-VL or legacy CLIP-FlanT5)."""

    def load(self) -> "T2VMetricsVQA":
        try:
            from t2v_metrics import VQAScore
        except ImportError as exc:
            raise RuntimeError("Install the appropriate metric_evaluation VQA requirements") from exc
        kwargs = {"model": self.model_name, "device": self.device}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        self._model = VQAScore(**kwargs)
        return self

    def yes_probability(self, image_paths, questions):
        # Upstream returns an M-images x N-texts matrix. Select its diagonal to
        # retain the aligned-pair contract used by this evaluator.
        yes_matrix = self._model(
            images=list(image_paths), texts=list(questions),
            question_template="{}", answer_template="Yes")
        no_matrix = self._model(
            images=list(image_paths), texts=list(questions),
            question_template="{}", answer_template="No")
        yes = yes_matrix.diagonal().detach().cpu().tolist()
        no = no_matrix.diagonal().detach().cpu().tolist()
        return _normalise_likelihoods(yes, no)


class InternVLHFVQA(VQAMetric):
    """InternVL Hugging Face checkpoint with direct next-token logit readout."""

    def load(self) -> "InternVLHFVQA":
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        checkpoint = self.checkpoint or self.model_name
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self._processor = AutoProcessor.from_pretrained(checkpoint, cache_dir=self.cache_dir)
        self._model = AutoModelForImageTextToText.from_pretrained(
            checkpoint, cache_dir=self.cache_dir, torch_dtype=dtype,
            device_map=self.device if self.device != "cuda" else "auto",
        ).eval()
        self._device = next(self._model.parameters()).device
        self.checkpoint = checkpoint
        return self

    def _answer_token_id(self, answer: str) -> int:
        # Require a single token so both classes use the exact same readout.
        tokenizer = self._processor.tokenizer
        candidates = [tokenizer.encode(answer, add_special_tokens=False),
                      tokenizer.encode(" " + answer, add_special_tokens=False)]
        ids = next((value for value in candidates if len(value) == 1), None)
        if ids is None:
            raise RuntimeError(
                f"InternVL tokenizer does not encode {answer!r} as one token; "
                "a sequence-likelihood adapter is required for this checkpoint")
        return ids[0]

    def yes_probability(self, image_paths, questions):
        import torch
        if len(image_paths) != len(questions):
            raise ValueError("image_paths and questions must have equal lengths")
        yes_id, no_id = self._answer_token_id("Yes"), self._answer_token_id("No")
        result = []
        for path, question in zip(image_paths, questions):
            messages = [{"role": "user", "content": [
                {"type": "image", "url": str(Path(path).resolve())},
                {"type": "text", "text": question},
            ]}]
            inputs = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = self._model(**inputs).logits[:, -1, [yes_id, no_id]].float()
            result.append(float(logits.softmax(dim=-1)[0, 0].cpu()))
        return result


class MPLUGVQA(VQAMetric):
    """Official mPLUG-large-v2 encoder/decoder with Yes/No sequence scoring."""

    # The X-PLUG/mPLUG GitHub README's own checkpoint host
    # (alice-open.oss-cn-zhangjiakou.aliyuncs.com) now returns 403 AccessDenied
    # ("no right to access this object because of bucket acl") for every real
    # checkpoint. TIFA (github.com/Yushi-Hu/tifa), which this comparison cites
    # for using "the VQA backbones used by TIFA and DSG (mPLUG)", does not use
    # that host either -- it loads this exact architecture from ModelScope
    # (damo/mplug_visual-question-answering_coco_large_en), which is still
    # live. Verified: config_bert.json/config.yaml there match this adapter's
    # config (module_setting expects the same encoder/fusion/decoder layer
    # counts) except image_res=504, and pytorch_model.bin resolves through a
    # real redirect to a torch.save zip archive, not an error page.
    OFFICIAL_CHECKPOINT = (
        "https://modelscope.cn/models/damo/"
        "mplug_visual-question-answering_coco_large_en/resolve/master/pytorch_model.bin"
    )
    OFFICIAL_VIT_L14_BACKBONE = (
        "https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/ViT-L-14.tar"
    )

    def load(self) -> "MPLUGVQA":
        import os
        import torch
        from torchvision import transforms
        from PIL import Image

        root = Path(self.backend_root or "external/mPLUG").resolve()
        if not (root / "models/model_vqa_mplug.py").is_file():
            raise RuntimeError(
                f"mPLUG source not found at {root}; clone https://github.com/X-PLUG/mPLUG "
                "and pass --backend-root")
        sys.path.insert(0, str(root))
        try:
            from models.model_vqa_mplug import MPLUG
            from models.tokenization_bert import BertTokenizer
        except ImportError as exc:
            raise RuntimeError("Could not import the official mPLUG repository") from exc

        config = {
            "bert_config": str(root / "configs/config_bert.json"),
            "image_res": 504,  # must match the checkpoint's config.yaml, or the ViT position embeddings mis-size
            "vision_width": 1024,
            "distill": True,
            "clip_name": "ViT-L-14",
            "use_checkpoint": False,
            "text_encoder": "bert-base-uncased",
            "text_decoder": "bert-base-uncased",
            "min_length": 1,
            "max_length": 10,
            "beam_size": 1,
        }
        self._tokenizer = BertTokenizer.from_pretrained(
            config["text_encoder"], cache_dir=self.cache_dir)

        # models.visual_transformers.initialize_clip() calls the bundled
        # models.clip.clip.load("ViT-L-14.tar", ...), which resolves that
        # filename relative to the process cwd. Fetch the official backbone
        # archive once into the cache dir and cd there only while the MPLUG
        # constructor (encoder + its distillation twin) builds it.
        clip_dir = Path(self.cache_dir or ".").resolve() / "mplug"
        clip_dir.mkdir(parents=True, exist_ok=True)
        clip_archive = clip_dir / "ViT-L-14.tar"
        if not clip_archive.is_file():
            torch.hub.download_url_to_file(
                self.OFFICIAL_VIT_L14_BACKBONE, str(clip_archive), progress=True)
        cwd = os.getcwd()
        os.chdir(clip_dir)
        try:
            self._model = MPLUG(config=config, tokenizer=self._tokenizer)
        finally:
            os.chdir(cwd)

        checkpoint = self.checkpoint or self.OFFICIAL_CHECKPOINT
        if checkpoint.startswith(("http://", "https://")):
            model_dir = str(Path(self.cache_dir or ".") / "mplug")
            state = torch.hub.load_state_dict_from_url(
                checkpoint, model_dir=model_dir, map_location="cpu", progress=True)
        else:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = state.get("model", state.get("module", state))
        incompatible = self._model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                "[mPLUG] non-fatal checkpoint mismatch: "
                f"missing={incompatible.missing_keys[:5]}, "
                f"unexpected={incompatible.unexpected_keys[:5]}",
                file=sys.stderr,
            )
        self._model.eval().to(self.device)
        self.checkpoint = checkpoint
        self._transform = transforms.Compose([
            transforms.Resize((config["image_res"], config["image_res"]),
                               interpolation=Image.Resampling.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)),
        ])
        return self

    def _encode(self, images, questions):
        import torch

        image_embeds = self._model.visual_encoder.visual(
            images, skip_last_layer=True, use_checkpoint=False)
        image_embeds = self._model.dropout(
            self._model.visn_layer_norm(self._model.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=images.device)
        question_input = self._tokenizer(
            list(questions), padding="longest", truncation=True,
            max_length=80, return_tensors="pt").to(images.device)
        text_output = self._model.text_encoder(
            question_input.input_ids, attention_mask=question_input.attention_mask,
            return_dict=True)
        image_output, question_output = self._model.fusion_encoder(
            encoder_embeds=text_output.last_hidden_state,
            attention_mask=question_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False)
        states = torch.cat([image_output, question_output], dim=1)
        atts = torch.cat([image_atts, question_input.attention_mask], dim=1)
        return states, atts

    def _sequence_likelihood(self, states, atts, answer: str):
        import torch.nn.functional as F

        answers = self._tokenizer(
            [answer + "[SEP]"] * states.shape[0], padding="longest",
            return_tensors="pt").to(states.device)
        logits = self._model.text_decoder(
            answers.input_ids, attention_mask=answers.attention_mask,
            encoder_hidden_states=states, encoder_attention_mask=atts,
            return_logits=True)
        targets = answers.input_ids[:, 1:]
        mask = answers.attention_mask[:, 1:].bool()
        token_logp = F.log_softmax(logits.float(), dim=-1).gather(
            -1, targets.unsqueeze(-1)).squeeze(-1)
        # Geometric mean matches VQAScore's sequence-likelihood convention.
        return ((token_logp * mask).sum(-1) / mask.sum(-1)).exp().cpu().tolist()

    def yes_probability(self, image_paths, questions):
        import torch
        from PIL import Image

        if len(image_paths) != len(questions):
            raise ValueError("image_paths and questions must have equal lengths")
        images = []
        for path in image_paths:
            with Image.open(path) as handle:
                images.append(self._transform(handle.convert("RGB")))
        image_batch = torch.stack(images).to(self.device)
        with torch.inference_mode():
            states, atts = self._encode(image_batch, questions)
            yes = self._sequence_likelihood(states, atts, "yes")
            no = self._sequence_likelihood(states, atts, "no")
        return _normalise_likelihoods(yes, no)


def load_vqa_metric(
    metric: str,
    *,
    device: str = "cuda",
    checkpoint: str | None = None,
    cache_dir: str | None = None,
    backend_root: str | None = None,
) -> VQAMetric:
    common = dict(device=device, checkpoint=checkpoint, cache_dir=cache_dir,
                  backend_root=backend_root)
    if metric == "qwen25vl":
        return T2VMetricsVQA(
            "Qwen2.5-VL (ours)", "qwen2.5-vl-7b", **common).load()
    if metric == "vqascore":
        return T2VMetricsVQA(
            "VQAScore", checkpoint or "clip-flant5-xxl", **common).load()
    if metric == "mplug":
        return MPLUGVQA(
            "mPLUG (ours)", "mplug.en.large.v2", **common).load()
    if metric == "internvl":
        return InternVLHFVQA(
            "InternVL (ours)", "OpenGVLab/InternVL3-8B-hf", **common).load()
    raise ValueError(f"Unknown VQA metric: {metric}")
