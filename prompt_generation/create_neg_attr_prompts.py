#!/usr/bin/env python3
"""
create_neg_attr_prompts.py

Generate negated-attribute benchmark prompts, one JSONL file per attribute type
(color, material), in GenEval's evaluation_metadata schema.

Each row pairs a plain positive object with one attribute-negated object; the
object is always present and only its attribute is denied:

    "a photo of a {pos_obj} and a non-{value} {neg_obj}"

The LLM selects which attribute values make a CHALLENGING slot for each object, in
both directions (the same list serves asserted and negated slots):

    canonical values           hard to DENY   — "a non-metal pair of scissors"
    surprising but depictable  hard to ASSERT — "a purple broccoli"

Values the object essentially never has are excluded, because denying those is
trivially true: "a non-wooden pair of scissors" or "a non-metal giraffe" is satisfied
by any ordinary photo of the object, so the prompt measures nothing. A rule-based
template builds the prompt string. 'person' is excluded as an attribute object (a
"wooden person" is nonsensical).

With --balance-attributes (default) each slot takes the LEAST-USED value among the
challenging ones, so the vocab is sampled ~equally instead of concentrating on the
values the LLM happens to rank first. Verify via `stats.*.balance.value_spread` in
the manifest (std ≈ 0 when balanced).

Output  (each (pos, neg) setting nests in its own folder)
------
    <out_dir>/pos<P>_neg<N>/neg_attr_color.jsonl
    <out_dir>/pos<P>_neg<N>/neg_attr_material.jsonl
    <out_dir>/pos<P>_neg<N>/neg_attr_all.jsonl     (merged, optional)
    <out_dir>/pos<P>_neg<N>/create_neg_attr_prompts.manifest.json
    <out_dir>/llm_cache.json                       (shared across settings)

Usage
-----
    python prompt_generation/create_neg_attr_prompts.py --out-dir outputs/prompts/neg --num-prompts 50
    python prompt_generation/create_neg_attr_prompts.py --no-llm
    python prompt_generation/create_neg_attr_prompts.py --types material
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neg_ext import builders, llm, manifest, vocab


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate negated-attribute prompts (per type).")
    ap.add_argument("--out-dir", default="outputs/prompts/neg",
                    help="Directory for the per-type JSONL files (default: outputs/prompts/neg)")
    ap.add_argument("--num-prompts", "-n", type=int, default=50,
                    help="Target rows per attribute type (default: 50)")
    ap.add_argument("--num-neg", type=int, default=1,
                    help="Negated-attribute objects per row (default: 1)")
    ap.add_argument("--num-pos", type=int, default=1,
                    help="Positive (asserted) attribute objects per row (default: 1)")
    ap.add_argument("--types", nargs="*", default=None,
                    help="Subset of attribute types (default: color material)")
    ap.add_argument("--extend", action="store_true",
                    help="EXTEND an existing file up to --num-prompts instead of "
                         "rewriting it: existing rows are kept BYTE-IDENTICAL and only "
                         "the shortfall is generated and appended. A prompt's "
                         "row index is its image directory name, so a moved row silently "
                         "re-points generated images. Idempotent.")
    ap.add_argument("--balance-objects", action=argparse.BooleanOptionalAction, default=True,
                    help="Pick the LEAST-USED objects for each row instead of sampling "
                         "uniformly. Without it objects were drawn with rng.sample and "
                         "the corpus spread ran 0-70 appearances (CV 0.689) with some "
                         "objects never used at all.")
    ap.add_argument("--balance-attributes", action=argparse.BooleanOptionalAction, default=True,
                    help="Pick the UNDER-used attribute value among an object's plausible "
                         "ones so values are ~equally sampled, instead of the LLM's first "
                         "choice (default: on; --no-balance-attributes for the old behaviour)")
    ap.add_argument("--seed", type=int, default=44, help="Sampling seed (default: 44)")
    ap.add_argument("--merge", action="store_true", help="Also write a merged neg_attr_all.jsonl")
    ap.add_argument("--cache-file", default=None,
                    help="Override the LLM verdict cache path (per-task file for parallel runs)")
    ap.add_argument("--no-finalize", action="store_true",
                    help="Skip the merged _all file and manifest (finalize step aggregates them)")
    # LLM
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip the LLM; use the full vocab for every object (smoke test)")
    ap.add_argument("--llm-model", default=llm.DEFAULT_MODEL_ID)
    ap.add_argument("--llm-cache-dir", default=None)
    ap.add_argument("--no-4bit", action="store_true")
    return ap.parse_args()


def _read_existing(out_file: Path):
    """(raw_lines, parsed_rows) for an existing prompt file, else ([], []).

    Raw lines are kept so extend mode writes them back untouched — re-serialising is
    equivalent in content but not guaranteed byte-identical, and row identity is what
    keeps already-generated images pointing at the right prompt.
    """
    if not out_file.exists():
        return [], []
    raw = [ln for ln in out_file.read_text().splitlines() if ln.strip()]
    rows = []
    for ln in raw:
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return raw, rows


def _tally_attr(rows, obj_usage, val_usage) -> None:
    """Count object and value usage over existing attribute rows."""
    for row in rows:
        for slot in row.get("slots", []):
            if slot.get("object"):
                obj_usage[slot["object"]] += 1
            if slot.get("value"):
                val_usage[slot["value"]] += 1


def _least_used_objects(rng, objects, usage, k):
    """The k least-used objects, ties broken randomly.

    Strict rather than probabilistic: over 100 rows and 79 objects strict selection
    reaches the arithmetic floor with every object covered, where a uniform or weighted
    draw leaves a long tail unused. The random tie-break stops every
    attribute type from walking the object list in the same order.
    """
    pool = list(objects)
    rng.shuffle(pool)
    return sorted(pool, key=lambda o: usage[o])[:k]


def main() -> None:
    args = parse_args()
    # Per-setting folder so different (pos, neg) runs don't overwrite each other;
    # the LLM verdict cache is shared at the base dir.
    base_dir = Path(args.out_dir)
    out_dir = base_dir / f"pos{args.num_pos}_neg{args.num_neg}"
    out_dir.mkdir(parents=True, exist_ok=True)

    objects = vocab.load_coco_objects(include_person=False)  # no attributed person
    types = args.types or list(vocab.ATTRIBUTE_VOCAB)
    for t in types:
        if t not in vocab.ATTRIBUTE_VOCAB:
            raise SystemExit(f"Unknown attribute type {t!r}; choose from {list(vocab.ATTRIBUTE_VOCAB)}")

    pipe = None if args.no_llm else llm.load_selector(
        args.llm_model, args.llm_cache_dir, load_4bit=not args.no_4bit)
    cache = llm.VerdictCache(args.cache_file or base_dir / "llm_cache.json")
    if args.no_llm:
        # Choosing values that make a negation hard needs world knowledge; the --no-llm
        # sampler returns the whole vocab, so negations can land on values the object
        # never has ("non-wooden scissors") and be trivially true. Fine for smoke
        # tests, not for a real dataset — say so rather than silently shipping them.
        print("[ATTR] WARNING: --no-llm cannot judge which values are challenging, so "
              "negations may be trivially true. Use the LLM for real datasets.")

    rng = random.Random(args.seed)
    file_counts: dict[str, int] = {}
    file_stats: dict[str, dict] = {}
    merged: list[dict] = []

    for attr_type in types:
        attr_vocab = vocab.ATTRIBUTE_VOCAB[attr_type]
        out_file = out_dir / f"{vocab.attr_tag(attr_type)}.jsonl"
        keep_raw, keep_rows = _read_existing(out_file) if args.extend else ([], [])
        rows: list[dict] = []
        seen: set[tuple] = set()
        usage = collections.Counter()          # value -> times used (this attr_type)
        obj_usage = collections.Counter()      # object -> times used (this attr_type)
        # Extend mode seeds both counters from the rows being KEPT, so the new rows fill
        # their deficits and cannot duplicate them.
        _tally_attr(keep_rows, obj_usage, usage)
        for r in keep_rows:
            seen.add(tuple((sl.get("object"), sl.get("value"), sl.get("polarity"))
                           for sl in r.get("slots", [])))
        if keep_rows:
            print(f"[ATTR] {attr_type:10s} extend: keeping {len(keep_rows)} existing rows, "
                  f"generating {max(args.num_prompts - len(keep_rows), 0)} more")
        need = args.num_pos + args.num_neg
        if need > len(objects):
            raise SystemExit(f"--num-pos + --num-neg = {need} exceeds {len(objects)} objects")
        want = args.num_prompts - len(keep_rows)
        attempts, max_attempts = 0, max(want, 1) * 40
        while len(rows) < want and attempts < max_attempts:
            attempts += 1
            # Distinct objects, one per slot. Least-used-first rather than uniform, so
            # every object gets used before any is reused.
            row_objs = (_least_used_objects(rng, objects, obj_usage, need)
                        if args.balance_objects else rng.sample(objects, need))
            rng.shuffle(row_objs)                  # role/polarity order stays random
            slots = []
            for j, obj in enumerate(row_objs):
                # The selector returns the values that make a CHALLENGING slot for this
                # object (see llm.select_attributes) — canonical values, whose denial
                # fights the model's prior, and surprising-but-depictable ones, whose
                # assertion fights it too. Implausible values are excluded there, so a
                # negation is never trivially true.
                vals = llm.select_attributes(pipe, obj, attr_vocab, n=len(attr_vocab),
                                             seed=args.seed + attempts * 10 + j, cache=cache)
                if not vals:
                    break
                # With balancing, take the least-used of those so values stay ~equal.
                val = min(vals, key=lambda v: usage[v]) if args.balance_attributes else vals[0]
                slots.append({"object": obj, "value": val,
                              "polarity": "pos" if j < args.num_pos else "neg"})
            if len(slots) < need:
                continue
            key = tuple((s["object"], s["value"], s["polarity"]) for s in slots)
            if key in seen:
                continue
            seen.add(key)
            for s in slots:
                usage[s["value"]] += 1
                obj_usage[s["object"]] += 1
            rows.append(builders.build_attribute_row(slots, attr_type))

        with out_file.open("w") as fp:
            for ln in keep_raw:
                fp.write(ln + "\n")
            for row in rows:
                fp.write(json.dumps(row) + "\n")
        all_rows = keep_rows + rows
        file_counts[out_file.name] = len(all_rows)
        if all_rows:
            file_stats[out_file.name] = manifest.attribute_stats(all_rows)
        merged.extend(all_rows)
        _new = f" (+{len(rows)} new)" if keep_rows else ""
        print(f"[ATTR] {attr_type:10s} -> {len(all_rows):3d} rows{_new}  {out_file}")

    cache.flush()
    if args.no_finalize:
        print(f"[ATTR] done (no-finalize) — {len(types)} type(s); "
              f"run finalize_prompts.py to build _all + manifest")
        return

    if args.merge and merged:
        merged_file = out_dir / "neg_attr_all.jsonl"
        with merged_file.open("w") as fp:
            for row in merged:
                fp.write(json.dumps(row) + "\n")
        file_counts[merged_file.name] = len(merged)
    if merged:
        file_stats["_overall"] = manifest.attribute_stats(merged)

    manifest.write_manifest(
        out_dir, generator="create_neg_attr_prompts", seed=args.seed,
        num_prompts=args.num_prompts,
        config={"types": types, "num_pos": args.num_pos, "num_neg": args.num_neg,
                "balance_attributes": args.balance_attributes,
                "balance_objects": args.balance_objects,
                "vocab": {t: vocab.ATTRIBUTE_VOCAB[t] for t in types}},
        files=file_counts, llm_model=None if args.no_llm else args.llm_model,
        used_llm=not args.no_llm, stats=file_stats)
    print("[ATTR] done")


if __name__ == "__main__":
    main()
