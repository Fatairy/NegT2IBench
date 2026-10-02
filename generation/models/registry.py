"""
registry.py — text-to-image model registry.

Maps a short `--model` key to (a) a loader that builds a diffusers pipeline and
(b) a generate function matching the protocol

    generate(pipe, prompt, steps, guidance, seed, device) -> PIL.Image

Standard diffusers models share one generic generate function; NegationCLIP uses a
custom CLIP-guided denoising loop (negationclip_generate.py, imported lazily). Every
model writes GenEval's <idx>/samples layout.

These are the ten models evaluated in the benchmark. To add one, append a ModelSpec
to MODELS.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

# Checkout of the BLIP3o repo's `BLIP3o-NEXT` branch (blip3o-next only).
BLIP3O_NEXT_ROOT = Path(
    os.environ.get("BLIP3O_NEXT_ROOT", "external/BLIP3o-NEXT")).expanduser()


# ── Generic generate (standard diffusers pipelines) ──────────────────────────

def std_generate(pipe, prompt: str, steps: int, guidance: float, seed: int, device: str):
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
    return result.images[0]


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_flux2(model_id: str, device: str, cache_dir: str | None):
    from diffusers import DiffusionPipeline
    return DiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda", cache_dir=cache_dir)


def _load_flux2_4bit(model_id: str, device: str, cache_dir: str | None):
    from diffusers import Flux2Pipeline
    return Flux2Pipeline.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda", cache_dir=cache_dir)


def _load_sd35(model_id: str, device: str, cache_dir: str | None):
    from diffusers import StableDiffusion3Pipeline
    return StableDiffusion3Pipeline.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda", cache_dir=cache_dir)


def _load_qwen(model_id: str, device: str, cache_dir: str | None):
    from diffusers import DiffusionPipeline
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, cache_dir=cache_dir)
    return pipe.to(device)


def _load_zimage(model_id: str, device: str, cache_dir: str | None):
    """Z-Image Turbo: official diffusers pipeline with 1024px generation.

    The official Hugging Face card shows both `DiffusionPipeline` and the newer
    `ZImagePipeline` entrypoint. We prefer the ZImage-specific class when the
    installed diffusers build provides it, and fall back to the generic pipeline
    so the registry still works in older environments that only expose the model
    files via `DiffusionPipeline.from_pretrained`.
    """
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    try:
        from diffusers import ZImagePipeline
    except Exception:
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, cache_dir=cache_dir)
    else:
        pipe = ZImagePipeline.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=False, cache_dir=cache_dir)

    return pipe.to(device)


def _load_hidream_4bit(model_id: str, device: str, cache_dir: str | None):
    """HiDream-I1 with the transformer AND the Llama text encoder quantized to NF4.

    Quantizes the OFFICIAL weights at load time rather than pulling a pre-quantized
    re-upload, so the provenance is the official checkpoint.

    text_encoder_4 is `meta-llama/Meta-Llama-3.1-8B-Instruct`, which the repo does
    NOT ship and which is gated: accept its licence and `huggingface-cli login`, or
    point HIDREAM_LLAMA_ID at a local copy. NF4 cuts VRAM roughly 4x on the ~17B
    transformer and the 8B Llama.
    """
    import os
    from diffusers import (HiDreamImagePipeline, HiDreamImageTransformer2DModel,
                           BitsAndBytesConfig as DiffusersBnB)
    from transformers import (LlamaForCausalLM, PreTrainedTokenizerFast,
                              BitsAndBytesConfig as TransformersBnB)

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    llama_id = os.environ.get("HIDREAM_LLAMA_ID",
                              "meta-llama/Meta-Llama-3.1-8B-Instruct")
    nf4 = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
               bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    try:
        tok4 = PreTrainedTokenizerFast.from_pretrained(llama_id, cache_dir=cache_dir)
        te4 = LlamaForCausalLM.from_pretrained(
            llama_id, quantization_config=TransformersBnB(**nf4),
            torch_dtype=dtype, cache_dir=cache_dir,
            output_hidden_states=True, output_attentions=True)
    except Exception as e:
        raise SystemExit(
            f"[GEN] hidream-4bit needs the 4th text encoder '{llama_id}', which is "
            f"gated.\n       Accept the licence at https://huggingface.co/{llama_id} "
            f"and run `huggingface-cli login`.\n"
            f"       underlying error: {type(e).__name__}: {e}") from e

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        model_id, subfolder="transformer", quantization_config=DiffusersBnB(**nf4),
        torch_dtype=dtype, cache_dir=cache_dir)
    pipe = HiDreamImagePipeline.from_pretrained(
        model_id, transformer=transformer, tokenizer_4=tok4, text_encoder_4=te4,
        torch_dtype=dtype, cache_dir=cache_dir)
    # Quantized submodules are already placed by accelerate; move only the rest.
    pipe.enable_model_cpu_offload()
    return pipe


def _load_janus_pro(model_id: str, device: str, cache_dir: str | None):
    """Janus-Pro: NOT a diffusers pipeline — an autoregressive any-to-any transformer.

    Uses the transformers-native port (`deepseek-community/*`), which exposes
    `JanusForConditionalGeneration`; the original `deepseek-ai/*` weights declare
    model_type `multi_modality` and need DeepSeek's own repo on sys.path. Returns
    (model, processor) because image generation needs the processor at call time, and
    `janus_generate` below unpacks that pair.
    """
    from transformers import JanusForConditionalGeneration, AutoProcessor
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = JanusForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, cache_dir=cache_dir)
    return (model.to(device).eval(), processor)


def janus_generate(pipe, prompt: str, steps: int, guidance: float, seed: int, device: str):
    """Janus-Pro image generation. `steps` is unused — the image-token count is fixed
    by the model's tokenizer, so only `guidance` (CFG) and the seed apply.

    Three stages, none of which a diffusers pipeline needs: generate() emits image
    TOKENS, `decode_image_tokens` runs them through the VQGAN to pixel values laid out
    (B, H, W, C), and only then does the processor convert to PIL. The magic string for
    that last step is `return_tensors="PIL.Image.Image"`, and it returns a BatchFeature
    keyed on "pixel_values" rather than a bare list.
    """
    model, processor = pipe
    torch.manual_seed(seed)
    chat = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        add_generation_prompt=True)
    inputs = processor(text=[chat], generation_mode="image",
                       return_tensors="pt").to(device)
    with torch.inference_mode():
        tokens = model.generate(**inputs, generation_mode="image",
                                do_sample=True, guidance_scale=guidance)
        pixels = model.decode_image_tokens(tokens)
    # return_tensors="PIL.Image.Image" is documented but broken in transformers 5.5:
    # postprocess forwards it into self.preprocess() BEFORE the line that strips it,
    # so the string reaches BatchFeature and raises "not a valid TensorType". Ask for
    # "pt" (a uint8 CHW tensor) and convert here instead.
    out = processor.postprocess(list(pixels.float()), return_tensors="pt")
    arr = out["pixel_values"][0]                     # (C, H, W) uint8
    return Image.fromarray(arr.permute(1, 2, 0).cpu().numpy())


def zimage_generate(pipe, prompt: str, steps: int, guidance: float, seed: int, device: str):
    """Z-Image Turbo generation.

    The official card recommends 1024x1024 outputs, 9 inference steps (8 DiT
    forwards), and guidance_scale=0.0 for the turbo checkpoint.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
    return result.images[0]


def _ensure_blip3o_importable(root: Path) -> None:
    """Make `import blip3o` resolve to a BLIP3o-NEXT checkout.

    The original BLIP3-o (repo `main`) ships a package with the SAME import name, so
    the NEXT branch is recognised by a directory only it has
    (blip3o/model/multimodal_decoder). A package that is present but missing a
    DEPENDENCY is reported separately from a missing checkout.
    """
    def _try_import() -> ModuleNotFoundError | None:
        try:
            import blip3o  # noqa: F401
        except ModuleNotFoundError as e:
            return e
        return None

    err = _try_import()
    if err is not None and (err.name or "").split(".")[0] == "blip3o":
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        err = _try_import()
    if err is not None:
        missing_pkg = (err.name or "").split(".")[0] == "blip3o"
        raise SystemExit(
            (f"[GEN] blip3o-next needs a checkout of the BLIP3o-NEXT branch at {root}\n"
             f"       (set BLIP3O_NEXT_ROOT); see README.md.\n" if missing_pkg else
             f"[GEN] blip3o is importable but its dependency {err.name!r} is missing:\n"
             f"       activate the BLIP3o-NEXT env (README.md).\n")
            + f"       underlying error: {type(err).__name__}: {err}") from err

    import blip3o
    pkg_dir = Path(blip3o.__file__).resolve().parent
    if not (pkg_dir / "model" / "multimodal_decoder").is_dir():
        raise SystemExit(
            f"[GEN] the importable `blip3o` package ({pkg_dir}) is the original BLIP3-o, "
            f"not BLIP3o-NEXT. Point BLIP3O_NEXT_ROOT at a BLIP3o-NEXT checkout.")


def _load_blip3o_next(model_id: str, device: str, cache_dir: str | None):
    """BLIP3o-NEXT (Chen et al., 2025c): Qwen3 AR backbone + SANA1.5 diffusion decoder.

    ONE checkpoint, no separate decoder subfolder, and stock diffusers. A single
    from_pretrained brings up all three parts — build_sana/build_vae instantiate the SANA
    transformer and its DC-AE VAE from the skeleton named in the config, and the
    checkpoint then overwrites those weights (model.sana.* and model.sana_vae.* are in
    the shards alongside the AR layers).

    Two hub repos are pulled beyond the checkpoint, and BOTH must be cached before an
    offline run:
      * `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` (config.diffusion_name_or_path)
        for the transformer/vae/scheduler skeletons;
      * `csuhan/TA-Tok` (2.4 GB), which tok/ta_tok.py downloads at MODULE IMPORT time from
        a module-level hf_hub_download. That call is unconditional and takes no cache_dir,
        so it ignores --cache-dir — point HF_HOME/HF_HUB_CACHE at the shared cache instead.

    The checkpoint's `mm_vision_tower` points at a path that does not exist, but it is
    INERT: TextAlignedTokenizer.from_checkpoint ignores its `ckpt` argument and reads
    that module-level download instead. So nothing needs patching, and the TA-Tok weights
    it loads are immediately overwritten by the checkpoint's own vision-tower shards.
    """
    _ensure_blip3o_importable(BLIP3O_NEXT_ROOT)
    from transformers import AutoTokenizer
    # Importing this is what triggers the TA-Tok download described above.
    from blip3o.model import blip3oQwenForInferenceLM

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = blip3oQwenForInferenceLM.from_pretrained(
        model_id, torch_dtype=dtype, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    return (model.to(device).eval(), tokenizer)


def blip3o_next_generate(pipe, prompt: str, steps: int, guidance: float, seed: int,
                         device: str):
    """BLIP3o-NEXT: the AR model writes a discrete blueprint, then SANA renders it.

    The prompt is the checkpoint's own Qwen3 chat template plus an `<im_start><S0>`
    suffix, which is what switches the model into image mode; <S0> is scale token 0 of
    the three the checkpoint defines, and matches upstream's inference.py. 729 new tokens
    is the 27x27 SigLIP-2 token grid, and top_p/top_k are upstream's values.

    Unlike the diffusion-only models, the first stage SAMPLES (do_sample is forced True
    inside generate_images), so two runs at identical steps/guidance differ unless the
    RNG is pinned. `generate_images` advertises a `generator` argument but hardcodes
    generator=None in its randn_tensor call, so seeding the global RNG is the only hook
    there is. `steps` and `guidance` reach the SANA sampler.
    """
    model, tokenizer = pipe
    _blip3o_next_seed(seed)
    return _blip3o_next_sample(model, tokenizer, [prompt], steps, guidance, device)[0]


def _blip3o_next_seed(seed: int) -> None:
    """Seed every global RNG BLIP3o-NEXT draws from (it honours no `generator`)."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _blip3o_next_sample(model, tokenizer, prompts: list[str], steps: int, guidance: float,
                        device: str) -> list:
    """One forward pass for `prompts`; returns one PIL image per prompt, in order.

    Left padding is what makes a mixed-length batch safe: it right-aligns every row, so
    the `<im_start>` marker lands at the same index in all of them and generate_images'
    `torch.stack` over the per-row hidden-state slices sees equal lengths. Verified on
    this checkpoint that a batch keeps sample diversity (identical prompts still give
    different images) and keeps image[i] matched to prompts[i].
    """
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user",
              "content": f"Please generate image based on the following caption: {p}"}],
            tokenize=False,
            add_generation_prompt=True) + "<im_start><S0>"
        for p in prompts
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       padding_side="left")
    _gen_ids, images = model.generate_images(
        inputs.input_ids.to(device),
        inputs.attention_mask.to(device),
        max_new_tokens=729,
        do_sample=True,
        top_p=0.95,
        top_k=1200,
        num_inference_steps=steps,
        guidance_scale=guidance,
    )
    return list(images)     # decode_latents returns a list of PIL images


def _make_negationclip_loader():
    """Build a loader+generate pair for SD 1.5 + NegationCLIP guidance.

    The custom denoising loop lives in negationclip_generate.py, imported lazily
    because it pulls in OpenAI CLIP. Requires --clip-weights (see diffusers_generate.py).
    """
    def loader(model_id: str, device: str, cache_dir: str | None, *, clip_weights: str,
               clip_guidance_scale: float = 100.0):
        from importlib import import_module
        mod = import_module(".negationclip_generate", __package__)
        clip_model = mod.load_negationclip(clip_weights, device)
        pipe = mod.load_sd_pipeline(model_id, device, cache_dir)

        def generate(_pipe, prompt, steps, guidance, seed, device):
            return mod.generate_image(_pipe, clip_model, prompt, steps, guidance,
                                      clip_guidance_scale, seed, device)

        return pipe, generate

    return loader


def _load_hunyuanimage_4bit(model_id: str, device: str, cache_dir: str | None):
    """HunyuanImage-2.1 with the transformer AND the Qwen2.5-VL text encoder quantized to
    NF4 -- the bf16 transformer alone (34.9 GB) does not fit a 35 GB GPU even with CPU
    offload. NF4 cuts both big components ~4x (same approach as `hidream-4bit`).
    text_encoder_2 (a much smaller ByT5-style OCR encoder) is left unquantized.
    """
    from diffusers import (HunyuanImagePipeline, HunyuanImageTransformer2DModel,
                           BitsAndBytesConfig as DiffusersBnB)
    from transformers import (Qwen2_5_VLForConditionalGeneration,
                              BitsAndBytesConfig as TransformersBnB)

    dtype = torch.bfloat16
    nf4 = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
               bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, subfolder="text_encoder", quantization_config=TransformersBnB(**nf4),
        torch_dtype=dtype, cache_dir=cache_dir)
    transformer = HunyuanImageTransformer2DModel.from_pretrained(
        model_id, subfolder="transformer", quantization_config=DiffusersBnB(**nf4),
        torch_dtype=dtype, cache_dir=cache_dir)
    pipe = HunyuanImagePipeline.from_pretrained(
        model_id, transformer=transformer, text_encoder=text_encoder,
        torch_dtype=dtype, cache_dir=cache_dir)
    # Quantized submodules are already placed by accelerate; move only the rest.
    pipe.enable_model_cpu_offload()
    return pipe


def hunyuanimage_generate(pipe, prompt: str, steps: int, guidance: float, seed: int, device: str):
    """No negation guidance: negative_prompt='red' worked, but 'blue' did nothing (mixed
    effectiveness, verified) -- disabled entirely rather than kept as an unreliable knob.
    The distilled checkpoint ships no `guider` at all, so negative_prompt would be a
    silent no-op regardless (AdaptiveProjectedMixGuidance(enabled=False) in the source).

    HunyuanImagePipeline has no plain `guidance_scale` -- only `distilled_guidance_scale`
    (its CFG is driven by a `Guider` object baked into the checkpoint, not a scalar
    threshold check). `height`/`width` default to 2048x2048 if omitted; pinned to
    1024x1024 here to match every other model in this registry."""
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        result = pipe(
            prompt,
            height=1024,
            width=1024,
            num_inference_steps=steps,
            distilled_guidance_scale=guidance,
            generator=generator,
        )
    return result.images[0]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    default_model_id: str
    loader: Callable
    default_steps: int
    default_guidance: float
    generate: Callable = std_generate
    needs_clip_weights: bool = False
    notes: str = ""


MODELS: dict[str, ModelSpec] = {
    "flux2": ModelSpec("flux2", "black-forest-labs/FLUX.2-klein-9B", _load_flux2, 4, 1.0),
    "flux2-4bit": ModelSpec("flux2-4bit", "diffusers/FLUX.2-dev-bnb-4bit", _load_flux2_4bit, 28, 3.5,
                            notes="4-bit FLUX.2-dev."),
    "sd35": ModelSpec("sd35", "stabilityai/stable-diffusion-3.5-large", _load_sd35, 28, 10.0),
    # HiDream-I1-Full (Cai et al., 2025), NF4 on load. Diffusers-native, but pulls a
    # GATED Llama 3.1-8B as its 4th text encoder — see _load_hidream_4bit.
    "hidream-4bit": ModelSpec("hidream-4bit", "HiDream-ai/HiDream-I1-Full",
                              _load_hidream_4bit, 50, 5.0,
                              notes="NF4-quantized official weights + CPU offload; "
                                    "still needs the gated Llama text encoder"),
    # Janus-Pro (Chen et al., 2025b). Autoregressive, not a diffusers pipeline: the
    # transformers-native port is used so no external repo is needed.
    "janus-pro": ModelSpec("janus-pro", "deepseek-community/Janus-Pro-7B",
                           _load_janus_pro, 1, 5.0, generate=janus_generate,
                           notes="autoregressive; --steps ignored, guidance = CFG"),
    # BLIP3o-NEXT (Chen et al., 2025c). Needs its own env and a checkout of the
    # BLIP3o-NEXT branch (BLIP3O_NEXT_ROOT).
    # Deliberately the SFT checkpoint: the GRPO-GenEval one is RL-tuned against a
    # detector-based GenEval reward, which is the same metric family this benchmark
    # scores with, so it is not comparable with the diffusion models in a results table.
    "blip3o-next": ModelSpec("blip3o-next", "BLIP3o/BLIP3o-NEXT-SFT-3B", _load_blip3o_next,
                             30, 2.0, generate=blip3o_next_generate,
                             notes="Qwen3 AR + SANA1.5; own conda env + NEXT-branch checkout "
                                   "(BLIP3O_NEXT_ROOT). The AR stage samples, so output varies "
                                   "run to run at fixed steps/guidance."),
    "qwen": ModelSpec("qwen", "Qwen/Qwen-Image-2512", _load_qwen, 50, 5.0,
                      notes="Also Wan-AI/Wan2.1-T2I-* via --model-id."),
    "zimage": ModelSpec("zimage", "Tongyi-MAI/Z-Image-Turbo", _load_zimage, 9, 0.0,
                        generate=zimage_generate,
                        notes="Z-Image-Turbo; 1024x1024, guidance 0.0, 8 NFEs."),
    "negationclip": ModelSpec("negationclip", "runwayml/stable-diffusion-v1-5",
                              _make_negationclip_loader(), 50, 7.5,
                              needs_clip_weights=True,
                              notes="SD 1.5 guided by NegationCLIP."),
    # HunyuanImage-2.1 (Tencent) distilled checkpoint, NF4-quantized to fit a 35 GB GPU
    # (see _load_hunyuanimage_4bit). 8 steps vs 50 for the base checkpoint. Ships
    # no `guider` component at all, so negative_prompt is a no-op here regardless --
    # no negation guidance (see hunyuanimage_generate).
    "hunyuanimage-distilled": ModelSpec(
        "hunyuanimage-distilled", "hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers",
        _load_hunyuanimage_4bit, 8, 3.5,
        generate=hunyuanimage_generate,
        notes="distilled + NF4-quantized; 8 steps vs 50 for the base checkpoint"),
}


def list_models() -> list[str]:
    return list(MODELS)


def get_spec(key: str) -> ModelSpec:
    if key not in MODELS:
        raise SystemExit(f"Unknown --model {key!r}. Choose from: {', '.join(MODELS)}")
    return MODELS[key]
