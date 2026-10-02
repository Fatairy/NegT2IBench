#!/usr/bin/env python3
"""
finalize_prompts.py

Aggregate per-category prompt JSONLs (produced in parallel with --no-finalize)
into the merged `_all` files and the manifests, for one setting folder.

Run AFTER all per-category generator tasks finish (see jobs/01_prompts.sbatch).
Reads every neg_rel_*.jsonl / neg_attr_*.jsonl in the folder (excluding _all),
writes neg_rel_all.jsonl + neg_attr_all.jsonl and the two manifests with stats.
No GPU / no LLM.

Usage:
    python prompt_generation/finalize_prompts.py --dir outputs/prompts/neg/pos2_neg2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neg_ext import manifest, vocab


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _finalize_group(dir_path: Path, prefix: str, generator: str, stats_fn,
                    config_extra: dict, num_prompts: int) -> int:
    files = sorted(f for f in dir_path.glob(f"{prefix}_*.jsonl") if not f.name.endswith("_all.jsonl"))
    if not files:
        return 0
    file_counts, file_stats, merged = {}, {}, []
    for f in files:
        rows = _read_jsonl(f)
        file_counts[f.name] = len(rows)
        if rows:
            file_stats[f.name] = stats_fn(rows)
        merged.extend(rows)

    merged_file = dir_path / f"{prefix}_all.jsonl"
    with merged_file.open("w") as fp:
        for row in merged:
            fp.write(json.dumps(row) + "\n")
    file_counts[merged_file.name] = len(merged)
    if merged:
        file_stats["_overall"] = stats_fn(merged)

    # Infer pos/neg from the data so the manifest reflects what's on disk.
    num_pos = sorted({r.get("num_pos", 0) for r in merged})
    num_neg = sorted({r.get("num_neg", 0) for r in merged})
    manifest.write_manifest(
        dir_path, generator=generator, seed=-1, num_prompts=num_prompts,
        config={**config_extra, "num_pos_seen": num_pos, "num_neg_seen": num_neg,
                "finalized": True},
        files=file_counts, llm_model=None, used_llm=False, stats=file_stats)
    print(f"[finalize] {prefix}: {len(files)} files, {len(merged)} rows -> {merged_file.name}")
    return len(merged)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build merged _all files + manifests for a setting folder.")
    ap.add_argument("--dir", required=True, help="Setting folder, e.g. outputs/prompts/neg/pos2_neg2")
    ap.add_argument("--num-prompts", type=int, default=-1, help="Recorded in the manifest (optional)")
    args = ap.parse_args()
    dir_path = Path(args.dir)
    if not dir_path.is_dir():
        raise SystemExit(f"not a directory: {dir_path}")

    _finalize_group(dir_path, "neg_rel", "create_neg_rel_prompts",
                    manifest.relation_stats,
                    {"taxonomy": {c.name: c.predicates for c in vocab.RELATION_TAXONOMY}},
                    args.num_prompts)
    _finalize_group(dir_path, "neg_attr", "create_neg_attr_prompts",
                    manifest.attribute_stats,
                    {"vocab": dict(vocab.ATTRIBUTE_VOCAB)},
                    args.num_prompts)
    print(f"[finalize] done: {dir_path}")


if __name__ == "__main__":
    main()
