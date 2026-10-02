"""Strip personal data from a raw annotation export before it is shared.

Keeps exactly what the human-validation code reads (the datapoint, its prompt
metadata, and every object/condition answer) and replaces the annotator identity
with a stable pseudonym (`annotator_01`, ...), so rows of the same annotator stay
grouped. Drops emails, nicknames, annotation/assignment ids and timestamps.

    python -m human_validation.anonymize_annotations raw_annotations.jsonl \
        human_annotations/annotations.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEEP = (
    "datapointId", "prompt", "model", "category", "posNeg", "tag",
    "objectFailure", "objectResponses", "conditionResponses", "conditionResponseVersion",
    "durationMs",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    # Row order is preserved: the leave-one-annotator-out folds and tie handling in
    # compare_human_metrics depend on it. Pseudonyms follow first appearance.
    pseudonym: dict[str, str] = {}
    out_rows = []
    for row in rows:
        who = str(row.get("annotatorId") or row.get("annotatorEmail") or "unknown")
        pseudonym.setdefault(who, f"annotator_{len(pseudonym) + 1:02d}")
        clean = {k: row[k] for k in KEEP if k in row}
        clean["annotator"] = pseudonym[who]
        out_rows.append(clean)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}: {len(out_rows)} annotations, {len(pseudonym)} annotators, "
          f"{len({r['datapointId'] for r in out_rows})} datapoints")


if __name__ == "__main__":
    main()
