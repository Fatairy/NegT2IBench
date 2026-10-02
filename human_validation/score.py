#!/usr/bin/env python3
"""Score image/caption pairs from JSONL with a registered metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL containing image and caption fields")
    parser.add_argument("output", type=Path, help="Output JSONL (one score per input row)")
    parser.add_argument("--metric", required=True, help="Registered metric key")
    parser.add_argument("--checkpoint", help="HF model id or local weights directory")
    parser.add_argument("--model-name", help="Model architecture (for weight-compatible variants)")
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--image-field", default="image_path")
    parser.add_argument("--caption-field", default="prompt")
    parser.add_argument("--id-field", default="image_id")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--cache-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Append to an existing output, skipping rows whose "
                             "--id-field value is already scored there")
    return parser.parse_args()


def read_rows(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc


def scored_ids(path: Path, id_field: str) -> set[str]:
    """Ids already present in a partial output, so --resume can skip them.

    A run over the full DATA_ROOT sweep is long enough that it will be
    interrupted at least once; re-scoring from scratch each time is the only
    thing that makes that expensive. A trailing truncated line (the process was
    killed mid-write) is dropped and simply rescored.
    """
    ids: set[str] = set()
    if not path.is_file():
        return ids
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                ids.add(str(json.loads(line)[id_field]))
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def truncate_partial_line(path: Path) -> None:
    """Drop an incomplete trailing line before appending.

    A run killed mid-write leaves a half-written record with no newline; opening
    in append mode would glue the next result onto it and lose both rows.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open("rb+") as handle:
        handle.seek(-1, 2)
        if handle.read(1) == b"\n":
            return
        data = path.read_bytes()
        cut = data.rfind(b"\n")
        handle.truncate(cut + 1)


def batches(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def main() -> None:
    args = parse_args()
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required: install the metric evaluation dependencies") from exc
    try:
        from .models import METRICS, load_metric
    except ImportError as exc:
        raise SystemExit(
            "PyTorch and the metric evaluation dependencies must be installed") from exc
    if args.metric not in METRICS:
        raise SystemExit(f"Unknown --metric {args.metric!r}; choose from {', '.join(METRICS)}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.output.exists() and not (args.overwrite or args.resume):
        raise SystemExit(f"Refusing to overwrite {args.output}; pass --overwrite")

    if args.resume:
        truncate_partial_line(args.output)
    done = scored_ids(args.output, args.id_field) if args.resume else set()
    rows = list(read_rows(args.input))
    prepared = []
    skipped = 0
    for line_number, row in rows:
        if done and str(row.get(args.id_field, line_number)) in done:
            skipped += 1
            continue
        missing = [field for field in (args.image_field, args.caption_field) if field not in row]
        if missing:
            raise SystemExit(f"{args.input}:{line_number} lacks field(s): {', '.join(missing)}")
        image_path = Path(row[args.image_field])
        if not image_path.is_absolute():
            image_path = args.image_root / image_path
        if not image_path.is_file():
            raise SystemExit(f"Image not found for {args.input}:{line_number}: {image_path}")
        prepared.append((line_number, row, image_path))

    if args.resume:
        print(f"Resuming {args.output}: {skipped} already scored, {len(prepared)} to go",
              flush=True)
        if not prepared:
            return

    scorer = load_metric(
        args.metric, checkpoint=args.checkpoint, model_name=args.model_name, device=args.device,
        dtype=args.dtype, cache_dir=args.cache_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as output:
        for batch in batches(prepared, args.batch_size):
            images = []
            try:
                for _, _, path in batch:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                scores = scorer.score(
                    images,
                    [row[args.caption_field] for _, row, _ in batch],
                    image_paths=[path for _, _, path in batch],
                )
            finally:
                for image in images:
                    image.close()
            for (line_number, row, image_path), score in zip(batch, scores):
                # Preserve setup/category/generator/element metadata for the later
                # statistics stage instead of forcing it to join another file.
                result = dict(row)
                result.update({
                    "image_id": row.get(args.id_field, str(line_number)),
                    "resolved_image_path": str(image_path),
                    "metric": args.metric,
                    "checkpoint": scorer.checkpoint,
                    "score": score,
                })
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
            # Flush per batch so an interrupted run stays resumable to within
            # one batch instead of losing whatever the OS still buffered.
            output.flush()


if __name__ == "__main__":
    main()
