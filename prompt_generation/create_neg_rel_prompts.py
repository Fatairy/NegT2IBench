#!/usr/bin/env python3
"""
create_neg_rel_prompts.py

Generate negated-relation benchmark prompts, one JSONL file per relation
category, in GenEval's evaluation_metadata schema.

Two dataset shapes:

  Every relation in a row shares one subject A. Per row the LLM is
  shown a fresh RANDOM subset of --candidate-pool objects (default 5) and picks
  suitable (predicate, target) pairs from them; a rule-based template emits the row
  ("a photo of a A ... where the A is not <pred> the B"). The random pool keeps
  object usage from concentrating on the objects the LLM favours over all 80.

Output
------
    <out_dir>/pos<P>_neg<N>/neg_rel_<category>.jsonl   (one row per combination)
    <out_dir>/pos<P>_neg<N>/neg_rel_all.jsonl          (merged, optional)
    <out_dir>/pos<P>_neg<N>/create_neg_rel_prompts.manifest.json

Usage
-----
    python prompt_generation/create_neg_rel_prompts.py --num-prompts 50
    python prompt_generation/create_neg_rel_prompts.py --no-llm                 # smoke test, no GPU
    python prompt_generation/create_neg_rel_prompts.py --categories directional comparative
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
    ap = argparse.ArgumentParser(description="Generate negated-relation prompts (per category).")
    ap.add_argument("--out-dir", default=None,
                    help="Base dir for the JSONL files (default: outputs/prompts/neg)")
    ap.add_argument("--candidate-pool", type=int, default=5,
                    help="Give the LLM a subset of this many objects per "
                         "row to pick targets from, instead of all 80 (default: 5)")
    ap.add_argument("--balance-objects", action=argparse.BooleanOptionalAction, default=True,
                    help="STAR mode: bias the per-row candidate pool + subject + target "
                         "choice toward UNDER-used objects so usage is ~equal, instead "
                         "of uniform-random (default: on; --no-balance-objects for random)")
    ap.add_argument("--num-prompts", "-n", type=int, default=50,
                    help="Target rows per category (default: 50)")
    ap.add_argument("--num-neg", type=int, default=1,
                    help="Negated relation edges per row (default: 1)")
    ap.add_argument("--num-pos", type=int, default=0,
                    help="Positive (asserted) relation edges per row (default: 0)")
    ap.add_argument("--categories", nargs="*", default=None,
                    help="Subset of relation categories (default: all)")
    ap.add_argument("--seed", type=int, default=43, help="Sampling seed (default: 43)")
    ap.add_argument("--extend", action="store_true",
                    help="EXTEND an existing file up to --num-prompts instead of "
                         "rewriting it: existing rows are kept BYTE-IDENTICAL and only "
                         "the shortfall is generated and appended. Required for growing "
                         "a dataset whose images already exist, because a prompt's row "
                         "index is its image directory name, so a row that moves "
                         "silently re-points generated images at a different prompt "
                         ". Idempotent: re-running adds nothing.")
    ap.add_argument("--strict-balance", action=argparse.BooleanOptionalAction, default=True,
                    help="Pick the LEAST-USED object rather than sampling weighted "
                         "toward under-used ones. Reaches the arithmetic floor (CV 0.349 "
                         "for 100 rows over 79 objects, all covered) where weighted "
                         "sampling leaves ~0.665 and misses ~13 objects. "
                         "--no-strict-balance restores the weighted draw.")
    ap.add_argument("--merge", action="store_true", help="Also write a merged neg_rel_all.jsonl")
    # Parallel-mode helpers (one category per process)
    ap.add_argument("--cache-file", default=None,
                    help="Override the LLM verdict cache path (use a per-task file to "
                         "avoid write races when running categories in parallel)")
    ap.add_argument("--no-finalize", action="store_true",
                    help="Skip writing the merged _all file and the manifest "
                         "(a separate finalize step aggregates them)")
    # LLM
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip the LLM; sample combinations randomly (for smoke tests)")
    ap.add_argument("--llm-model", default=llm.DEFAULT_MODEL_ID)
    ap.add_argument("--llm-cache-dir", default=None, help="HF cache dir for the LLM")
    ap.add_argument("--no-4bit", action="store_true", help="Load the LLM in full precision")
    return ap.parse_args()


def _as_score(score):
    """Accept either a {item: value} mapping or a callable, so the strict and weighted
    pickers are drop-in interchangeable at the call site."""
    return score if callable(score) else (lambda o: score[o])


def _least_used(rng, items, score, k):
    """The k items with the LOWEST score, ties broken randomly.

    Strict, unlike _weighted_sample's probabilistic draw: it always takes the rarest
    object, which is what reaches the arithmetic floor. Over 100 rows and 79 objects
    strict selection gives CV 0.349 with every object covered, where the weighted draw
    gives ~0.665 and leaves ~13 objects unused.

    The random tie-break matters: without it every category would walk the object list
    in the same order, so the same object would be subject #1 everywhere and the
    categories would correlate even though each is balanced.
    """
    items = list(items)
    key = _as_score(score)
    rng.shuffle(items)                       # randomises ties; score ordering dominates
    return sorted(items, key=key)[:k]


def _weighted_sample(rng, items, score, k):
    """Sample k distinct items, favouring LOW score (inverse-frequency weighting).
    Seeded via `rng` so it stays reproducible."""
    items = list(items)
    picked = []
    for _ in range(min(k, len(items))):
        weights = [1.0 / (1.0 + _as_score(score)(o)) for o in items]
        x = rng.choices(items, weights=weights, k=1)[0]
        picked.append(x)
        items.remove(x)
    return picked


def _tally_row(row, subj, tgt, rel) -> None:
    """Add one prompt row's subject / target / predicate usage into the counters."""
    include = row.get("include", [])
    for entry in include:
        pos = entry.get("position") or []
        if pos and not isinstance(pos[0], list):
            pos = [pos]                      # legacy flat ["not left of", 0]
        for expected_rel, tgt_idx in pos:
            subj[entry.get("class")] += 1
            base = expected_rel[4:] if str(expected_rel).startswith("not ") else expected_rel
            rel[base] += 1
            if isinstance(tgt_idx, int) and 0 <= tgt_idx < len(include):
                tgt[include[tgt_idx].get("class")] += 1


def _read_existing(out_file: Path):
    """(raw_lines, parsed_rows) for an existing prompt file, else ([], []).

    The raw lines are kept so extend mode can write them back untouched. Re-serialising
    parsed rows would be equivalent in content but not guaranteed byte-identical, and
    row identity is what keeps already-generated images pointing at the right prompt.
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


def _star_rows(category, args, objects, pipe, cache, rng, keep=None) -> list[dict]:
    """STAR: rows sharing one subject. Per row the LLM is shown a subset of
    --candidate-pool objects and picks suitable relations to them.

    With --balance-objects (default) three things are balanced, all via running
    Counters over COMMITTED rows only (so rejected attempts don't skew the weights):

      subject + candidate pool  drawn UNDER-used-first (inverse-frequency weighting),
                                so object usage is ~equal rather than merely
                                uniform-random — which lumps in finite samples, and
                                which the LLM re-biases toward salient objects.
      predicate, then target    among the LLM's suitable pairs the choice is
                                PREDICATE-first with the target as tiebreak, so the
                                category's predicates (e.g. left/right/above/below)
                                come out ~uniform. Objects are already evened upstream
                                at pool sampling, hence the lower priority here.

    A per-subject failure counter down-weights objects the LLM keeps rejecting,
    avoiding starvation loops. Balance is bounded by plausibility: an object the LLM
    never finds suitable in this category stays rare — a real semantic limit, not
    fixable bias. Check the outcome in the manifest's `stats.*.balance` block.
    """
    cat = vocab.RELATION_CATEGORIES[category]
    need = args.num_pos + args.num_neg
    pool_size = max(need, min(args.candidate_pool, len(objects) - 1))
    balance = args.balance_objects
    # Subject and target usage are counted SEPARATELY. They are different roles with
    # different pressures — a target is only reachable if the LLM finds it suitable
    # for some subject, while a subject is chosen freely — so sharing one counter let
    # a heavy target suppress that object as a subject and vice versa.
    usage_subj = collections.Counter()
    usage_tgt = collections.Counter()
    pred_usage = collections.Counter()
    fails = collections.Counter()          # rows that failed to fill, per subject
    score = lambda o: usage_subj[o] + 0.5 * fails[o]

    # Extend mode seeds from the rows being KEPT, so the new ones fill their deficits
    # and cannot duplicate them.
    for r in (keep or []):
        _tally_row(r, usage_subj, usage_tgt, pred_usage)
    rows, seen = [], set()
    for r in (keep or []):
        inc = r.get("include", [])
        subj_cls = next((e.get("class") for e in inc if e.get("position")), None)
        slots_key = []
        for e in inc:
            pos = e.get("position") or []
            if pos and not isinstance(pos[0], list):
                pos = [pos]
            for er, ti in pos:
                base = er[4:] if str(er).startswith("not ") else er
                pol = "neg" if str(er).startswith("not ") else "pos"
                t = inc[ti].get("class") if isinstance(ti, int) and 0 <= ti < len(inc) else None
                slots_key.append((base, t, pol))
        if subj_cls:
            seen.add((subj_cls, tuple(slots_key)))
    subjects = objects[:]
    rng.shuffle(subjects)
    # NOT named `target`: the edge loop below binds `target` to an object name, which
    # would silently clobber a row counter of that name after the first iteration.
    want = args.num_prompts - len(keep or [])
    attempts, max_attempts = 0, max(want, 1) * 80
    while len(rows) < want and attempts < max_attempts:
        attempts += 1
        if balance:
            # subject drawn against SUBJECT usage; the candidate pool against TARGET
            # usage, since that is the role those objects are about to play.
            pick = _least_used if getattr(args, "strict_balance", True) else _weighted_sample
            subject = pick(rng, objects, lambda o: score(o), 1)[0]
            others = [o for o in objects if o != subject]
            pool = pick(rng, others, lambda o: usage_tgt[o], pool_size)
        else:
            subject = subjects[attempts % len(subjects)]
            pool = rng.sample([o for o in objects if o != subject], pool_size)

        pairs = llm.select_relation_targets(
            pipe, subject, cat.predicates, pool,
            n=pool_size * len(cat.predicates), seed=args.seed + attempts, cache=cache)
        # Prefer under-used PREDICATES first, then under-used targets, among the
        # suitable pairs — so predicates come out ~uniform within the category and
        # objects stay evened (counters the LLM's re-bias toward salient choices).
        if balance:
            pairs = sorted(pairs, key=lambda pt: (pred_usage[pt[0]], usage_tgt[pt[1]]))
        chosen, seen_t = [], set()
        for predicate, target in pairs:
            if target == subject or target in seen_t:
                continue
            seen_t.add(target)
            chosen.append((predicate, target))
            if len(chosen) >= need:
                break
        if len(chosen) < need:
            fails[subject] += 1
            continue
        slots = [{"relation": p, "target": t, "polarity": "pos" if j < args.num_pos else "neg"}
                 for j, (p, t) in enumerate(chosen)]
        key = (subject, tuple((s["relation"], s["target"], s["polarity"]) for s in slots))
        if key in seen:
            continue
        seen.add(key)
        usage_subj[subject] += 1
        for p, t in chosen:
            usage_tgt[t] += 1
            pred_usage[p] += 1
        rows.append(builders.build_relation_row(subject, slots, category))
    return rows


def main() -> None:
    args = parse_args()
    if args.num_pos + args.num_neg < 1:
        raise SystemExit("--num-pos + --num-neg must be >= 1")
    # Each (pos, neg) setting nests in its own folder.
    base_dir = Path(args.out_dir or "outputs/prompts/neg")
    out_dir = base_dir / f"pos{args.num_pos}_neg{args.num_neg}"
    out_dir.mkdir(parents=True, exist_ok=True)

    objects = vocab.load_coco_objects(include_person=True)   # person allowed in relations
    categories = args.categories or list(vocab.RELATION_CATEGORIES)
    all_categories = list(vocab.RELATION_CATEGORIES)
    for c in categories:
        if c not in vocab.RELATION_CATEGORIES:
            raise SystemExit(f"Unknown category {c!r}; choose from {all_categories}")

    pipe = None if args.no_llm else llm.load_selector(
        args.llm_model, args.llm_cache_dir, load_4bit=not args.no_4bit)
    cache = llm.VerdictCache(args.cache_file or base_dir / "llm_cache.json")

    rng = random.Random(args.seed)
    file_counts: dict[str, int] = {}
    file_stats: dict[str, dict] = {}
    merged: list[dict] = []

    for category in categories:
        out_file = out_dir / f"{vocab.rel_tag(category)}.jsonl"
        keep_raw, keep_rows = _read_existing(out_file) if args.extend else ([], [])
        if keep_rows:
            print(f"[REL] {category:12s} extend: keeping {len(keep_rows)} existing rows, "
                  f"generating {max(args.num_prompts - len(keep_rows), 0)} more")

        # Per-file balance: the counters start from THIS file's kept rows.
        # One RNG per category so categories are independent: a single shared RNG made
        # each category's output depend on how many draws the previous one consumed, so
        # regenerating one category silently changed every later one.
        cat_rng = random.Random(args.seed * 7919 + all_categories.index(category))
        rows = _star_rows(category, args, objects, pipe, cache, cat_rng, keep=keep_rows)

        # Existing lines are written back verbatim, then the new rows appended.
        with out_file.open("w") as fp:
            for ln in keep_raw:
                fp.write(ln + "\n")
            for row in rows:
                fp.write(json.dumps(row) + "\n")
        all_rows = keep_rows + rows
        file_counts[out_file.name] = len(all_rows)
        if all_rows:
            file_stats[out_file.name] = manifest.relation_stats(all_rows)
        merged.extend(all_rows)
        _new = f" (+{len(rows)} new)" if keep_rows else ""
        print(f"[REL] {category:12s} -> "
              f"{len(all_rows):3d} rows{_new}  {out_file}")

    cache.flush()
    if args.no_finalize:
        print(f"[REL] done (no-finalize) — {len(categories)} categor(y/ies); "
              f"run finalize_prompts.py to build _all + manifest")
        return

    if args.merge and merged:
        merged_file = out_dir / "neg_rel_all.jsonl"
        with merged_file.open("w") as fp:
            for row in merged:
                fp.write(json.dumps(row) + "\n")
        file_counts[merged_file.name] = len(merged)
    if merged:
        file_stats["_overall"] = manifest.relation_stats(merged)

    used_llm = not args.no_llm
    manifest.write_manifest(
        out_dir, generator="create_neg_rel_prompts", seed=args.seed,
        num_prompts=args.num_prompts,
        config={"categories": categories,
                "candidate_pool": args.candidate_pool,
                "balance_objects": args.balance_objects,
                "num_pos": args.num_pos, "num_neg": args.num_neg,
                "taxonomy": {c.name: c.predicates for c in vocab.RELATION_TAXONOMY}},
        files=file_counts, llm_model=args.llm_model if used_llm else None,
        used_llm=used_llm, stats=file_stats)
    print(f"[REL] done — {sum(v for k, v in file_counts.items() if k.startswith('neg_rel_') and 'all' not in k)} rows across {len(categories)} categories")


if __name__ == "__main__":
    main()
