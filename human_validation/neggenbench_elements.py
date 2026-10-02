"""Per-element reconstruction of NegT2IBench's (`neggenbench`) own verdict.

`DATA_ROOT/<model>/<setup>/results/<category>.jsonl` only stores the row-level
`correct` boolean plus a free-text `reason`. This module reconstructs, per row,
one boolean per *element* -- one per required object (`metadata["include"]`
entries) and one per relation/attribute condition on that object -- so
NegT2IBench can be compared to humans and the VQA judges on the same per-element
footing as `{vqa}_elem`, instead of only ever contributing one number per item.

Two element kinds need two different reconstruction strategies, because the
evaluator (`evaluation/evaluate_images.py`) persists different
amounts of detail for each:

- **Relation** elements (`neg_rel_*` tags): the evaluator already stores the
  per-edge verdict directly in `details["edges"]` (subject/target/predicate/
  polarity/computed), recorded whether the edge passed or failed
  (`evaluate_neg_relation`, `edges_out`). No reconstruction needed beyond
  reading it back out.
- **Attribute** elements (`neg_attr_*` tags, color/material): the verdict
  comes from a *live* CLIP confusion-set call (`attribute_classification()`)
  that is never persisted -- only the final `correct`/`reason` survive. This
  module instead parses `reason` against the exact templates
  `evaluate_neg_attribute`/`_resolve_include` emit (verified against
  evaluate_images.py source, not guessed), matching lines to `include`
  entries positionally and in the same order the evaluator appends them. This
  is deterministic log-parsing of the evaluator's own diagnostic output, not
  free-text NLP.

Object-presence/count elements (one per `include` entry, both tag families,
via the shared `_resolve_include`/`counted_instances`/`over_count_reason`
helpers) are reconstructed the same templated-reason-parsing way.

Dependency rule, matching the human/VQA side: if an object's count check
fails (under-detected or over-counted), any relation/attribute condition that
depends on it is marked unsatisfied too, without independently "blaming" the
condition logic -- exactly mirroring `evaluate_neg_relation`'s own
missing-object handling and `evaluate_neg_attribute`'s early `continue`.

Sanity check: ANDing every element for a row must reproduce that row's
`correct` field exactly. `elements_for_row()` asserts this; a mismatch means
either a template drifted from evaluate_images.py or a genuinely new tag
family this module doesn't know about yet -- not a per-element average to
be trusted silently.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import result_files

# Mirrors evaluation/evaluate_images.py -- keep in sync if those
# change (COUNTING_THRESHOLD, EXACTLY_ONE, COUNT_EXEMPT_CLASSES; L374-425).
COUNT_EXEMPT_CLASSES = {"skis"}
EXACTLY_ONE = True


def under_detect_reason(classname: str, count: int, n_found: int) -> str:
    return f"expected {classname}>={count}, found {n_found}"


def over_count_reason_text(classname: str, count: int, n_counted: int) -> str | None:
    if classname in COUNT_EXEMPT_CLASSES:
        return None
    if EXACTLY_ONE and n_counted > count:
        return f"expected exactly {count} {classname}, found {n_counted} (counted instances)"
    return None


def neg_attr_reason(classname: str, value: str, attr_type: str) -> str:
    return f"{classname} should NOT be {value} ({attr_type}), but "  # + "N is (in CLIP confusion set)"


def pos_attr_reason(classname: str, value: str) -> str:
    return f"expected {value} {classname}, found "  # + "N (confusion set)"


def relation_missing_reason(subj_cls: str, expected_rel: str, target_group: int) -> str:
    return f"missing object for '{subj_cls} {expected_rel} #{target_group}'"


def element_for_edge(edge: dict) -> bool:
    """Direct readout, no reconstruction: `details['edges']` already carries the verdict."""
    computed = edge.get("computed")
    if computed is None:
        return False  # dependency rule: endpoint undetected/over-counted
    predicate = edge["predicate"]
    holds = predicate in computed
    return (not holds) if edge["polarity"] == "neg" else holds


def elements_for_row(row: dict) -> list[dict] | None:
    """Return one dict per element for a single results-jsonl row, or None if
    this row's tag isn't one this module knows how to decompose (legacy tags).

    Each element dict: {"element_id", "element_type" ("object"/"relation"/
    "color"/"material"), "satisfied": bool}.
    """
    metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
    details = json.loads(row["details"]) if isinstance(row["details"], str) else row["details"]
    tag = metadata.get("tag", "")
    include = metadata.get("include", [])
    if not tag.startswith("neg_rel_") and not tag.startswith("neg_attr_"):
        return None

    reason_lines = [l for l in row.get("reason", "").split("\n") if l]
    consumed = [False] * len(reason_lines)

    def consume_matching(prefix: str) -> bool:
        """Consume+return True if some unconsumed reason line starts with `prefix`."""
        for i, line in enumerate(reason_lines):
            if not consumed[i] and line.startswith(prefix):
                consumed[i] = True
                return True
        return False

    elements: list[dict] = []
    object_ok: list[bool] = []
    for idx, req in enumerate(include):
        classname = req["class"]
        count = req["count"]
        # Under-detected (too few) or over-counted (too many, exactly-one rule)
        # -- both are logged with fixed prefixes; a match consumes that line so
        # a later, differently-typed check on the same classname isn't confused
        # by it (see docstring: two failure modes must not share a phrase).
        under = consume_matching(under_detect_reason(classname, count, 0).rsplit("found ", 1)[0] + "found ")
        over = consume_matching(f"expected exactly {count} {classname}, found ")
        ok = not (under or over)
        object_ok.append(ok)
        elements.append({
            "element_id": f"object:{idx}:{classname}",
            "element_type": "object",
            "satisfied": ok,
        })

        for attr_type in ("color", "material"):
            if attr_type not in req:
                continue
            value = req[attr_type]
            polarity = req.get(f"{attr_type}_polarity")
            if not ok:
                # evaluate_neg_attribute's `continue` after an object failure
                # means the attribute was never independently checked --
                # dependency rule, not a free pass.
                elements.append({
                    "element_id": f"attr:{idx}:{attr_type}",
                    "element_type": attr_type,
                    "satisfied": False,
                })
                continue
            if polarity == "neg":
                failed = consume_matching(neg_attr_reason(classname, value, attr_type))
            else:
                failed = consume_matching(pos_attr_reason(classname, value))
            elements.append({
                "element_id": f"attr:{idx}:{attr_type}",
                "element_type": attr_type,
                "satisfied": not failed,
            })

    # Relation edges: direct readout from details["edges"] (already computed
    # by the evaluator, whether the edge passed or failed).
    for edge in details.get("edges", []) or []:
        elements.append({
            "element_id": f"relation:{edge['subject']}->{edge['target']}:{edge['predicate']}:{edge['polarity']}",
            "element_type": "relation",
            "satisfied": element_for_edge(edge),
        })

    reconstructed = all(e["satisfied"] for e in elements)
    if reconstructed != bool(row["correct"]):
        raise AssertionError(
            f"AND-reconstruction mismatch for {row.get('filename')}: "
            f"reconstructed={reconstructed} stored correct={row['correct']} "
            f"unconsumed_reason_lines={[l for i, l in enumerate(reason_lines) if not consumed[i]]}"
        )
    return elements


def neggenbench_score(row: dict) -> float | None:
    """Continuous NegT2IBench score: fraction of this row's own elements satisfied
    -- the single-run analogue of human_score (no averaging over annotators,
    since there is one detector pass, not three raters)."""
    elements = elements_for_row(row)
    if not elements:
        return None
    return sum(e["satisfied"] for e in elements) / len(elements)


def iter_results_rows(data_root: Path):
    for path in result_files(data_root):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
