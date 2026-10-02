#!/usr/bin/env python3
"""
diffusers_generate.py — multi-model image generation for GenEval.

Reads a GenEval/PosEval-style metadata JSONL and generates N samples per prompt
with the chosen text-to-image model, writing GenEval's canonical layout so
evaluation/evaluate_images.py runs unchanged for every model:

    <outdir>/<index:05d>/metadata.jsonl
    <outdir>/<index:05d>/samples/<sample:05d>.png

The model is selected by a short key resolved through generation/models/registry
(the ten benchmark models). The
output layout is GenEval's.

Usage
-----
    # one category file of one setting, FLUX.2
    python generation/diffusers_generate.py negt2ibench_prompts/pos1_neg1/neg_rel_directional.jsonl \\
        --model flux2 --outdir outputs/images/flux2/pos1_neg1/neg_rel_directional --n_samples 4

    python generation/diffusers_generate.py negt2ibench_prompts/pos0_neg1/neg_attr_color.jsonl \\
        --model negationclip --clip-weights cache/negationclip_ViT-B32.pth \\
        --outdir outputs/images/negationclip/pos0_neg1/neg_attr_color
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from tqdm import trange
from pytorch_lightning import seed_everything

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import get_spec, list_models

# Inference-only by default: saves memory for every standard diffusers pipeline.
# NOT unconditional -- the NegationCLIP-guided models backpropagate a CLIP loss to the
# latents each denoising step, and a process-wide grad switch leaves that loss with no
# grad_fn ("element 0 of tensors does not require grad"). Re-enabled in main() for any
# spec that needs CLIP weights.
torch.set_grad_enabled(False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate GenEval images with a selectable T2I model.")
    p.add_argument("metadata_file", type=str,
                   help="JSONL metadata file (one prompt per line).")
    p.add_argument("--model", type=str, default="zimage",
                   help=f"Model key. Choices: {', '.join(list_models())} (default: zimage)")
    p.add_argument("--model-id", type=str, default=None,
                   help="Override the registry's default HF model id for --model.")
    p.add_argument("--cache-dir", type=str, default=None, help="HF cache directory.")
    p.add_argument("--clip-weights", type=str, default=None,
                   help="NegationCLIP weights (.pth); required for the negationclip models.")
    p.add_argument("--clip-guidance-scale", type=float, default=100.0)
    p.add_argument("--outdir", type=str, default="outputs", help="Output directory.")
    p.add_argument("--n_samples", type=int, default=4, help="Images per prompt (default: 4).")
    p.add_argument("--steps", type=int, default=None, help="Inference steps (default: model's).")
    p.add_argument("--scale", "--guidance", dest="scale", type=float, default=None,
                   help="Guidance scale (default: model's).")
    p.add_argument("--seed", type=int, default=42, help="Base seed (default: 42).")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip samples whose PNG already exists.")
    p.add_argument("--skip_grid", action="store_true", help="Do not save the per-prompt grid.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only the first N prompts of the file (smoke tests).")
    return p.parse_args()


def build_pipeline(args, device: str):
    """Resolve the model spec, load the pipeline, and return (generate_fn, steps, guidance)."""
    spec = get_spec(args.model)
    model_id = args.model_id or spec.default_model_id
    steps = args.steps if args.steps is not None else spec.default_steps
    guidance = args.scale if args.scale is not None else spec.default_guidance
    print(f"[GEN] model={spec.key}  id={model_id}  steps={steps}  guidance={guidance}")
    if spec.notes:
        print(f"[GEN] note: {spec.notes}")

    if spec.needs_clip_weights:
        if not args.clip_weights:
            raise SystemExit(f"--model {spec.key} requires --clip-weights")
        # This model guides sampling with a CLIP gradient, so autograd must be live.
        torch.set_grad_enabled(True)
        pipe, generate = spec.loader(
            model_id, device, args.cache_dir,
            clip_weights=args.clip_weights, clip_guidance_scale=args.clip_guidance_scale)
    else:
        pipe = spec.loader(model_id, device, args.cache_dir)
        generate = spec.generate
    return generate, pipe, steps, guidance


def main() -> None:
    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    with open(args.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp if line.strip()]
    if args.limit is not None:
        metadatas = metadatas[: args.limit]

    generate, pipe, steps, guidance = build_pipeline(args, device)
    outroot = Path(args.outdir)

    for index, metadata in enumerate(metadatas):
        outpath = outroot / f"{index:05d}"
        sample_path = outpath / "samples"
        sample_path.mkdir(parents=True, exist_ok=True)
        (outpath / "metadata.jsonl").write_text(json.dumps(metadata))

        prompt = metadata["prompt"]
        print(f"[{index + 1:>4}/{len(metadatas)}] {prompt}")

        seed_everything(args.seed)
        grid_tensors = []
        for sample_idx in trange(args.n_samples, desc="sampling", leave=False):
            out_png = sample_path / f"{sample_idx:05d}.png"
            if args.skip_existing and out_png.exists():
                continue
            image = generate(pipe, prompt, steps, guidance, args.seed + sample_idx, device)
            image.save(out_png)
            if not args.skip_grid:
                grid_tensors.append(ToTensor()(image))
            torch.cuda.empty_cache() if device == "cuda" else None

        if not args.skip_grid and grid_tensors:
            grid = make_grid(torch.stack(grid_tensors, 0), nrow=args.n_samples)
            grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
            Image.fromarray(grid.astype(np.uint8)).save(outpath / "grid.png")

    print("[GEN] done.")


if __name__ == "__main__":
    main()
