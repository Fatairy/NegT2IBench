#!/usr/bin/env python3
"""Build a metric-evaluation manifest covering every model in ``DATA_ROOT``.

Rows are built straight from the evaluator's own outputs
(``DATA_ROOT/<model>/<setup>/results/<tag>.jsonl``). With ``--only-annotated`` the
manifest is restricted to the human-annotated images (HUMAN_EVAL_INPUT, the input of
the human-validation comparison); without it, it covers the full generation sweep.

Per row we emit:

- ``image_path`` / ``filename``: path relative to ``DATA_ROOT`` (the last six
  components of the stored ``filename``).
- ``docId``/``image_id``: ``<model>-<setup>-<tag>-<idx>-samples-<sample>``, the
  identifier ``export_human_eval_files._neggenbench_doc_id`` already uses, so
  these rows join against ``neggenbench_both.jsonl`` and the human files.
- ``elements``: one entry per required object plus one per attribute/relation
  condition, with the exact question wording the annotation UI showed, so
  per-element VQA scores are directly comparable with the human labels.
- ``autoCorrect``/``autoReason``: NegT2IBench's own verdict, carried along so the
  statistics stage does not need to re-open DATA_ROOT.

Human-only fields (``human_score``, ``answers``, ...) are absent by design; the
human answers are read from HUMAN_ANNOTATIONS by compare_human_metrics.py.

Usage:
    python -m human_validation.build_geneval_manifest \
        --output outputs/human_validation/full_sweep/manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .paths import DATA_ROOT, NON_MODEL_DIRS, result_files


def build_objects(metadata: dict) -> list[str]:
    """The objects an annotator was asked about (one per distinct class)."""
    objects: list[str] = []
    for include in metadata.get("include", []):
        class_name = include.get("class")
        if class_name and class_name not in objects:
            objects.append(class_name)
    return objects


def build_conditions(metadata: dict) -> list[dict]:
    """The attribute/relation conditions an annotator was asked about, with the
    exact question wording of the annotation UI (``--check-against`` asserts it
    against an annotated manifest)."""
    category = metadata.get("category")
    slots = metadata.get("slots", [])
    conditions: list[dict] = []

    if category in ("color", "material"):
        for index, slot in enumerate(slots):
            obj = slot["object"]
            value = slot["value"]
            polarity = slot["polarity"]
            question = (
                f"Does the {obj} appear {value}?"
                if category == "material"
                else f"Is the {obj} {value}?"
            )
            conditions.append({
                "id": f"cond_attr_{index}",
                "type": "attribute",
                "category": category,
                "subject": obj,
                "predicate": value,
                "polarity": polarity,
                "expected": polarity == "pos",
                "question": question,
            })
        return conditions

    subject_entry = next(
        (inc for inc in metadata.get("include", []) if "position" in inc), None)
    if subject_entry is None:
        return conditions

    subject = subject_entry["class"]
    for index, (relation_phrase, slot_index) in enumerate(subject_entry["position"]):
        target = slots[slot_index]["target"]
        is_negated = relation_phrase.lower().startswith("not ")
        predicate = relation_phrase[4:] if is_negated else relation_phrase
        polarity = "neg" if is_negated else "pos"
        conditions.append({
            "id": f"cond_rel_{index}",
            "type": "relation",
            "subject": subject,
            "predicate": predicate,
            "polarity": polarity,
            "expected": polarity == "pos",
            "target": target,
            "question": f"Is the {subject} {predicate} the {target}?",
        })
    return conditions


def build_elements(metadata: dict) -> list[dict]:
    """Same element list export_metric_dataset.build_elements produces, minus the
    human answers (this manifest covers unannotated images)."""
    conditions = build_conditions(metadata)
    condition_object_ids = {
        str(condition.get("id", ""))[4:]
        for condition in conditions
        if str(condition.get("id", "")).startswith("obj_")
    }

    elements: list[dict] = []
    for index, name in enumerate(build_objects(metadata)):
        name = str(name).strip()
        if name in condition_object_ids:
            continue
        elements.append({
            "element_id": f"object:{index}:{name}",
            "element_type": "object",
            "polarity": "affirmative",
            "question": f"Is there exactly one {name} in the image?",
            "source_id": name,
            "expected": True,
            "active": True,
        })

    for index, condition in enumerate(conditions):
        condition_id = condition.get("id") or str(index)
        required = [condition.get("subject")]
        if condition.get("target"):
            required.append(condition["target"])
        expected = bool(condition.get("expected", True))
        category = condition.get("category")
        element_type = (
            category if category in {"color", "material"}
            else condition.get("type", "condition"))
        if element_type == "attribute":
            element_type = category or "attribute"
        elements.append({
            "element_id": f"condition:{condition_id}",
            "element_type": element_type,
            "polarity": "affirmative" if expected else "negated",
            "question": condition.get("question", ""),
            "source_id": condition_id,
            "expected": expected,
            # Without human object labels there is nothing to deactivate on;
            # score_vqa asks every active element and the dependency rule is
            # applied downstream from the object answers it returns.
            "active": True,
            "requires": [name for name in required if name],
        })
    return elements


def relative_image_path(stored_filename: str) -> Path:
    """`filename` ends in <model>/<setup>/<tag>/<idx>/samples/<n>.png; keep those six
    trailing components, which are exactly the layout under DATA_ROOT/."""
    return Path(*Path(stored_filename).parts[-6:])


def doc_id_for(relative: Path) -> str:
    """Same identifier as export_human_eval_files._neggenbench_doc_id."""
    model, setup, tag, index, _samples, sample = relative.parts
    return f"{model}-{setup}-{tag}-{index}-samples-{Path(sample).stem}"


def iter_rows(models: list[str] | None, setups: list[str] | None):
    for results_path in result_files(DATA_ROOT):
        model = results_path.parents[2].name
        setup = results_path.parents[1].name
        if models and model not in models:
            continue
        if setups and setup not in setups:
            continue
        with results_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield model, setup, json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Restrict to these DATA_ROOT model directories")
    parser.add_argument("--setups", nargs="*", default=None,
                        help="Restrict to these pos*_neg* setups")
    parser.add_argument("--samples-per-prompt", type=int, default=None,
                        help="Keep at most N of the 4 samples per prompt "
                             "(lowest sample index first, deterministic)")
    parser.add_argument("--prompts-per-tag", type=int, default=None,
                        help="Keep at most N prompt indices per (model, setup, tag), "
                             "sampled with --seed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-against", type=Path, default=None,
                        help="Annotated manifest (input.jsonl) to verify question "
                             "wording against for every shared docId")
    parser.add_argument("--only-annotated", type=Path, default=None,
                        help="Keep only images that appear in this human-annotation JSONL "
                             "(its `datapointId`s). This builds the human-validation "
                             "input manifest (HUMAN_EVAL_INPUT).")
    parser.add_argument("--allow-missing-images", action="store_true",
                        help="Skip rows whose image is absent instead of failing")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    keep_ids = None
    if args.only_annotated is not None:
        with args.only_annotated.open(encoding="utf-8") as handle:
            keep_ids = {json.loads(line)["datapointId"] for line in handle if line.strip()}
        print(f"Restricting to {len(keep_ids)} annotated datapoints from {args.only_annotated}")
    by_group: dict[tuple[str, str, str], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list))
    missing_images = 0
    total_seen = 0

    for model, setup, row in iter_rows(args.models, args.setups):
        total_seen += 1
        relative = relative_image_path(row["filename"])
        if keep_ids is not None and doc_id_for(relative) not in keep_ids:
            continue
        image_path = DATA_ROOT / relative
        if not image_path.is_file():
            missing_images += 1
            continue
        metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        tag = row.get("tag") or relative.parts[2]
        prompt_index = relative.parts[3]
        by_group[(model, setup, tag)][prompt_index].append({
            "relative": relative,
            "row": row,
            "metadata": metadata,
        })

    if missing_images and not args.allow_missing_images:
        raise SystemExit(
            f"{missing_images}/{total_seen} result rows have no image under {DATA_ROOT} "
            f"(models with no local images: pass --models to exclude them, or "
            f"--allow-missing-images to skip)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    per_model: dict[str, int] = defaultdict(int)
    with args.output.open("w", encoding="utf-8") as out:
        for (model, setup, tag) in sorted(by_group):
            prompts = by_group[(model, setup, tag)]
            indices = sorted(prompts)
            if args.prompts_per_tag is not None and len(indices) > args.prompts_per_tag:
                indices = sorted(rng.sample(indices, args.prompts_per_tag))
            for prompt_index in indices:
                items = sorted(prompts[prompt_index], key=lambda item: item["relative"].name)
                if args.samples_per_prompt is not None:
                    items = items[: args.samples_per_prompt]
                for item in items:
                    relative = item["relative"]
                    row = item["row"]
                    metadata = item["metadata"]
                    doc_id = doc_id_for(relative)
                    record = {
                        "image_id": doc_id,
                        "docId": doc_id,
                        "datapointId": doc_id,
                        "image_path": str(relative),
                        # score.py/score_vqa.py in the existing runner script are
                        # invoked with --image-field filename.
                        "filename": str(relative),
                        "prompt": row["prompt"],
                        "setup": setup,
                        "posNeg": setup,
                        "category": metadata.get("category"),
                        "tag": tag,
                        "model": model,
                        "promptGroupId": f"{setup}-{tag}-{relative.parts[3]}",
                        "sourcePath": str(relative),
                        "autoCorrect": bool(row["correct"]),
                        "autoReason": row.get("reason", ""),
                        "elements": build_elements(metadata),
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    per_model[model] += 1

    print(f"Wrote {args.output} ({written} rows from {total_seen} result rows"
          + (f", {missing_images} skipped for missing images" if missing_images else "")
          + ")")
    for model in sorted(per_model):
        print(f"  {model:16s} {per_model[model]}")

    if args.check_against:
        check_questions(args.output, args.check_against)


def check_questions(manifest: Path, annotated: Path) -> None:
    def load(path: Path) -> dict[str, dict]:
        rows = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows[row["docId"]] = row
        return rows

    ours = load(manifest)
    theirs = load(annotated)
    shared = sorted(set(ours) & set(theirs))
    mismatches = []
    for doc_id in shared:
        a = {e["element_id"]: e["question"] for e in ours[doc_id]["elements"]}
        b = {e["element_id"]: e["question"] for e in theirs[doc_id]["elements"]}
        if a != b:
            mismatches.append((doc_id, a, b))
    print(f"Question check: {len(shared)} shared docIds, {len(mismatches)} mismatched")
    for doc_id, a, b in mismatches[:5]:
        print(f"  {doc_id}\n    built:     {a}\n    annotated: {b}")
    if mismatches:
        raise SystemExit("Question wording drifted from the annotated manifest")


if __name__ == "__main__":
    main()
