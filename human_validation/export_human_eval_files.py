#!/usr/bin/env python3
"""Materialize two condensed JSONL files under HUMAN_EVAL_DIR, matching
the format the VQA judges already use (one row per docId, `elements[]`
mirroring the human annotation schema), so NegT2IBench and the human ground
truth can be inspected/shared the same way as `internvl_both.jsonl` etc.
instead of only ever existing as a live computation inside
compare_human_metrics.py.

Writes:

- `neggenbench_both.jsonl` -- one row per docId in HUMAN_EVAL_INPUT that has a matching
  NegT2IBench result under DATA_ROOT/. Per row: `correct` (binary, the
  evaluator's own verdict), `score` (continuous, fraction of elements
  satisfied -- see neggenbench_elements.py), and `elements[]` (one entry per
  object/relation/attribute check, each with `satisfied`). No `whole_*`
  fields -- NegT2IBench has no separate whole-caption mode to report; it is
  inherently element-based.

- `human_scores.jsonl` -- one row per annotated datapoint: the same
  aggregate fields compare_human_metrics.py computes on the fly
  (human_score, human_label, unanimous, exclusions) PLUS the raw
  per-annotator responses (objectResponses/conditionResponses, one entry per
  annotator) straight from annotations.jsonl -- so this file is both the
  ground truth used everywhere else AND traceable back to the individual
  clicks that produced it, without needing to re-open the raw annotation
  export.

Usage:
    python -m human_validation.export_human_eval_files
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .compare_human_metrics import (
    HUMAN_ANNOTATIONS,
    DATA_ROOT,
    METRIC_DIR,
    REPO_ROOT,
    aggregate_datapoint,
    read_jsonl,
)
from .neggenbench_elements import elements_for_row
from .paths import HUMAN_EVAL_INPUT as INPUT_JSONL, result_files



def _neggenbench_doc_id(row: dict) -> str:
    filename = Path(row["filename"])
    sample = filename.stem
    idx = filename.parent.parent.name
    category = filename.parent.parent.parent.name
    setup = filename.parent.parent.parent.parent.name
    model = filename.parent.parent.parent.parent.parent.name
    return f"{model}-{setup}-{category}-{idx}-samples-{sample}"


def write_neggenbench_both() -> Path:
    by_doc_id: dict[str, dict] = {}
    for path in result_files(DATA_ROOT):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                by_doc_id[_neggenbench_doc_id(row)] = row

    manifest = {row["docId"]: row for row in read_jsonl(INPUT_JSONL)}

    out_path = METRIC_DIR / "neggenbench_both.jsonl"
    written = skipped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for doc_id, manifest_row in manifest.items():
            raw = by_doc_id.get(doc_id)
            if raw is None:
                skipped += 1
                continue
            elements = elements_for_row(raw)
            if elements is None:
                skipped += 1
                continue
            score = sum(e["satisfied"] for e in elements) / len(elements)
            out.write(json.dumps({
                "docId": doc_id,
                "image_id": doc_id,
                "image_path": manifest_row.get("image_path"),
                "prompt": manifest_row.get("prompt"),
                "model": manifest_row.get("model"),
                "posNeg": manifest_row.get("posNeg"),
                "category": manifest_row.get("category"),
                "tag": manifest_row.get("tag"),
                "correct": bool(raw["correct"]),
                "score": score,
                "reason": raw.get("reason", ""),
                "elements": elements,
                "metric": "neggenbench",
            }) + "\n")
            written += 1
    print(f"Wrote {out_path} ({written} datapoints, {skipped} skipped -- no matching NegT2IBench result)")
    return out_path


def write_human_scores() -> Path:
    rows = read_jsonl(HUMAN_ANNOTATIONS)
    by_dp: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_dp[row["datapointId"]].append(row)

    out_path = METRIC_DIR / "human_scores.jsonl"
    written = skipped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for dp_id, dp_rows in by_dp.items():
            agg = aggregate_datapoint(dp_id, dp_rows)
            if agg is None:
                skipped += 1
                continue
            out.write(json.dumps({
                "docId": dp_id,
                "datapointId": dp_id,
                "prompt": agg["prompt"],
                "model": agg["model"],
                "posNeg": agg["posNeg"],
                "category": agg["category"],
                "tag": agg["tag"],
                "n_annotators": agg["n_annotators"],
                "human_score": agg["human_score"],
                "human_label": agg["human_label"],
                "unanimous": agg["unanimous"],
                "exclusions": agg["exclusions"],
                "annotators": [
                    {
                        "annotatorId": r.get("annotatorId"),
                        "objectResponses": r.get("objectResponses", []),
                        "conditionResponses": r.get("conditionResponses", []),
                        "durationMs": r.get("durationMs"),
                        "submittedAt": r.get("submittedAt"),
                    }
                    for r in dp_rows
                ],
            }) + "\n")
            written += 1
    print(f"Wrote {out_path} ({written} datapoints, {skipped} skipped -- no metadata found)")
    return out_path


def main() -> None:
    write_neggenbench_both()
    write_human_scores()


if __name__ == "__main__":
    main()