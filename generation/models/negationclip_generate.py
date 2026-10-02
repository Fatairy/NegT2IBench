"""negationclip_generate.py — Stable Diffusion 1.x guided by NegationCLIP (ViT-B/32).

NegationCLIP is a CLIP model fine-tuned to better understand negation in text.
Used here as a gradient-based guidance signal on top of Stable Diffusion: at
each denoising step the predicted clean image is decoded and passed through
NegationCLIP, and the cosine-similarity gradient is added to the noise
prediction (CLIP-guided diffusion), on top of standard classifier-free
guidance.

Vendored directly in this repo (no external repo dependency): loads via
OpenAI's `clip` package (`pip install git+https://github.com/openai/CLIP.git`
— the `clip` package on PyPI is an unrelated clipboard utility) against the
`negationclip_ViT-B32.pth` checkpoint passed as --clip-weights.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image
from torchvision.transforms.functional import normalize

# 'clip' on PyPI is a clipboard utility, NOT OpenAI's CLIP, which must be
# installed via `pip install git+https://github.com/openai/CLIP.git`. Imported
# lazily (and validated) so importing this module doesn't require it unless
# the negationclip model is actually used.
_openai_clip = None


def _get_clip():
    global _openai_clip
    if _openai_clip is not None:
        return _openai_clip
    import importlib
    try:
        m = importlib.import_module("clip")
        if not hasattr(m, "load") or not hasattr(m, "tokenize"):
            raise AttributeError("missing 'load' / 'tokenize'")
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"OpenAI CLIP not found or a conflicting 'clip' package is installed ({exc}).\n"
            "Fix:\n"
            "  pip uninstall clip          # remove the wrong package if present\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        ) from exc
    _openai_clip = m
    return _openai_clip


# ImageNet-style normalisation constants used by CLIP.
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _load_state_dict(weights_path: Path, device: str) -> dict:
    """Load a NegationCLIP state dict, trying several formats and unwrapping
    common dict wrappers ({'state_dict': ...} or {'model': ...})."""
    try:
        obj = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(obj, dict):
            if "state_dict" in obj:
                return obj["state_dict"]
            if "model" in obj:
                return obj["model"]
            return obj
        if hasattr(obj, "state_dict"):
            return obj.state_dict()
        raise ValueError(f"Unexpected checkpoint type: {type(obj)}")
    except RuntimeError as e:
        if "zip archive" not in str(e):
            raise

    try:
        return torch.jit.load(str(weights_path), map_location=device).state_dict()
    except Exception:
        pass

    try:
        with open(weights_path, "rb") as f:
            obj = pickle.load(f, encoding="latin1")
        if isinstance(obj, dict):
            if "state_dict" in obj:
                return obj["state_dict"]
            if "model" in obj:
                return obj["model"]
            return obj
        if hasattr(obj, "state_dict"):
            return obj.state_dict()
    except Exception:
        pass

    raise RuntimeError(
        f"Could not load NegationCLIP weights from {weights_path}.\n"
        "The file may be corrupted or in an unsupported format.\n"
        "Re-download/re-copy and verify: python -c \"import torch; torch.load('<path>')\"."
    )


def load_negationclip(weights_path: str | Path, device: str):
    """Load NegationCLIP ViT-B/32 from a local .pth weights file."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"NegationCLIP weights not found: {weights_path}\nPass the actual path via --clip-weights."
        )
    print(f"[INFO] NegationCLIP weights: {weights_path}")
    # OpenAI clip.load() downloads the base ViT-B/32 checkpoint to
    # ~/.cache/clip by default. CLIP_DOWNLOAD_ROOT redirects that cache;
    # config.sh points it at $CACHE_DIR/clip.
    download_root = os.environ.get("CLIP_DOWNLOAD_ROOT")
    model, _ = _get_clip().load("ViT-B/32", device=device, download_root=download_root)
    state = _load_state_dict(weights_path, device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_sd_pipeline(model_id: str, device: str, cache_dir: str | None):
    """Load Stable Diffusion 1.x with a DDIM scheduler."""
    print(f"[INFO] SD model : {model_id}")
    print(f"[INFO] Device   : {device}")
    print(f"[INFO] Cache    : {cache_dir or 'HF default'}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def _encode_clip_text(clip_model, prompt: str, device: str) -> torch.Tensor:
    """Return L2-normalised NegationCLIP text features for a prompt."""
    tokens = _get_clip().tokenize([prompt], truncate=True).to(device)
    with torch.no_grad():
        features = clip_model.encode_text(tokens).float()
    return features / features.norm(dim=-1, keepdim=True)


def _decode_latents_to_pixels(vae, latents: torch.Tensor) -> torch.Tensor:
    """Decode VAE latents -> float32 pixel tensor in [0, 1].

    The VAE is loaded in float16; latents are cast to half before decoding so
    the dtypes match, then the output is converted to float32 for the
    subsequent CLIP / gradient operations.
    """
    decoded = vae.decode(latents.half() / vae.config.scaling_factor).sample
    return (decoded.float() / 2 + 0.5).clamp(0, 1)


def _pixels_to_clip_input(pixels: torch.Tensor) -> torch.Tensor:
    """Resize pixel tensor to 224x224 and apply CLIP normalisation."""
    resized = F.interpolate(pixels, size=(224, 224), mode="bicubic", align_corners=False)
    return normalize(resized, mean=_CLIP_MEAN, std=_CLIP_STD)


def generate_image(
    pipe,
    clip_model,
    prompt: str,
    steps: int,
    guidance: float,
    clip_guidance_scale: float,
    seed: int,
    device: str,
) -> Image.Image:
    """Run NegationCLIP-guided Stable Diffusion for a single prompt.

    At each denoising step:
      1. SD's UNet predicts noise with classifier-free guidance.
      2. The current x0 estimate is decoded through the VAE.
      3. NegationCLIP encodes the decoded image and computes cosine
         similarity with the text embedding.
      4. The gradient of -similarity w.r.t. the latents is added to the noise
         prediction, steering generation toward better negation fidelity.
    """
    height, width = 512, 512
    generator = torch.Generator(device=device).manual_seed(seed)

    text_input = pipe.tokenizer(
        ["", prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(text_input.input_ids)[0]

    clip_text_features = _encode_clip_text(clip_model, prompt, device)

    latents = torch.randn(
        (1, pipe.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=device,
        dtype=torch.float16,
    )
    pipe.scheduler.set_timesteps(steps)
    latents = latents * pipe.scheduler.init_noise_sigma

    for t in pipe.scheduler.timesteps:
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)

        if clip_guidance_scale > 0:
            latents_grad = latents.detach().float().requires_grad_(True)
            alpha_t = pipe.scheduler.alphas_cumprod[t].to(device)
            pred_x0 = (latents_grad - (1 - alpha_t) ** 0.5 * noise_pred.float()) / (alpha_t**0.5)

            pixels = _decode_latents_to_pixels(pipe.vae, pred_x0)
            clip_input = _pixels_to_clip_input(pixels)

            img_features = clip_model.encode_image(clip_input.half()).float()
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            clip_loss = -torch.cosine_similarity(img_features, clip_text_features).mean()

            grad = torch.autograd.grad(clip_loss, latents_grad)[0]
            # + (not -): verified empirically -- with epsilon-prediction the noise/x0
            # relationship is inverted, so this is the sign that actually raises
            # image/prompt CLIP similarity as clip_guidance_scale increases (checked
            # 0/100/300 -> monotonically higher; the old "-" sign was monotonically
            # *lower*, i.e. guidance was fighting the prompt instead of reinforcing it).
            # sqrt(1-alpha_t) decay (standard in CLIP-guided-diffusion, e.g. the HF
            # community pipeline this technique comes from): without it, a constant
            # scale keeps injecting a full-strength CLIP gradient into the last,
            # near-noise-free steps, which just optimizes raw CLIP score into
            # adversarial static instead of refining a real image.
            noise_pred = noise_pred + clip_guidance_scale * (1 - alpha_t) ** 0.5 * grad.half()

            del latents_grad, pred_x0, pixels, clip_input, img_features
            torch.cuda.empty_cache()

        latents = pipe.scheduler.step(noise_pred, t, latents, generator=generator).prev_sample

    with torch.no_grad():
        image_tensor = _decode_latents_to_pixels(pipe.vae, latents)
    image_np = image_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    image_np = (image_np * 255).round().astype("uint8")
    return Image.fromarray(image_np)
