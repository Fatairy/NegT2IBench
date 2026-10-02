#!/usr/bin/env python3
"""Emit NegT2IBench's own verdict (`neggenbench`) for the full DATA_ROOT sweep.

`export_human_eval_files.write_neggenbench_both()` does this for the 288
human-annotated datapoints. This is the same thing over every row of
`FULL_SWEEP_DIR/manifest.jsonl`, so `neggenbench` can be compared with
the CLIP-family and VQA metrics on the whole sweep instead of only where humans
annotated.

No model and no GPU: the evaluator already stored `correct`/`reason`/`details`
per image, and `neggenbench_elements.elements_for_row()` reconstructs one
boolean per element from them. `score` is the fraction of elements satisfied,
matching the human-eval file's definition.

Like the GPU metrics this takes --resume, so re-running only fills in docIds the
output does not already have.

Usage:
    python -m human_validation.build_neggenbench_full --resume
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_geneval_manifest import (
    DATA_ROOT,
    doc_id_for,
    relative_image_path,
)
from .neggenbench_elements import elements_for_row
from .paths import FULL_SWEEP_DIR, result_files

DEFAULT_MANIFEST = FULL_SWEEP_DIR / "manifest.jsonl"
DEFAULT_OUTPUT = FULL_SWEEP_DIR / "neggenbench_both.jsonl"

# Only the manifest fields this output echoes back, so 165k rows stay small.
CARRIED = ("image_path", "prompt", "model", "posNeg", "category", "tag")


def load_manifest(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["docId"]] = {key: row.get(key) for key in CARRIED}
    return rows


def scored_ids(path: Path) -> set[str]:
    """docIds already written, so --resume can skip them (mirrors score.py)."""
    ids: set[str] = set()
    if not path.is_file():
        return ids
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                ids.add(json.loads(line)["docId"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def truncate_partial_line(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open("rb+") as handle:
        handle.seek(-1, 2)
        if handle.read(1) == b"\n":
            return
        cut = path.read_bytes().rfind(b"\n")
        handle.truncate(cut + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true",
                        help="Append, skipping docIds already in the output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not (args.overwrite or args.resume):
        raise SystemExit(f"Refusing to overwrite {args.output}; pass --overwrite or --resume")

    manifest = load_manifest(args.manifest)
    if args.resume:
        truncate_partial_line(args.output)
    done = scored_ids(args.output) if args.resume else set()
    print(f"Manifest: {len(manifest)} rows; already scored: {len(done)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.output.exists() else "w"
    written = skipped_done = skipped_unknown = skipped_tag = 0
    # An AND-reconstruction mismatch means a reason template drifted from
    # evaluate_images.py, or a tag family this module does not decompose. Those
    # rows are counted and listed rather than silently dropped or fatal.
    mismatches: list[str] = []

    with args.output.open(mode, encoding="utf-8") as out:
        for results_path in result_files(DATA_ROOT):
            with results_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    doc_id = doc_id_for(relative_image_path(raw["filename"]))
                    carried = manifest.get(doc_id)
                    if carried is None:
                        skipped_unknown += 1
                        continue
                    if doc_id in done:
                        skipped_done += 1
                        continue
                    try:
                        elements = elements_for_row(raw)
                    except AssertionError as exc:
                        mismatches.append(f"{doc_id}: {exc}")
                        continue
                    if not elements:
                        skipped_tag += 1
                        continue
                    record = {"docId": doc_id, "image_id": doc_id}
                    record.update(carried)
                    record.update({
                        "correct": bool(raw["correct"]),
                        "score": sum(e["satisfied"] for e in elements) / len(elements),
                        "reason": raw.get("reason", ""),
                        "elements": elements,
                        "metric": "neggenbench",
                    })
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    if written % 20000 == 0:
                        out.flush()
                        print(f"  wrote {written}", flush=True)

    print(f"Wrote {args.output}: {written} new rows "
          f"({skipped_done} already scored, {skipped_unknown} not in manifest, "
          f"{skipped_tag} undecomposable tag, {len(mismatches)} reconstruction mismatches)")
    for line in mismatches[:5]:
        print(f"  mismatch {line}")
    if len(mismatches) > 5:
        print(f"  ... and {len(mismatches) - 5} more")


if __name__ == "__main__":
    main()
