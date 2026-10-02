#!/usr/bin/env python3
"""Compare automatic metric scores against human annotations.

Runs three deliberately separate evaluations (see the comment block below the
imports, and each section's own banner in main()):

1. Binary vs `human_label` -- every judge's own yes/no call, plus NegT2IBench's
   native correct/incorrect. No CLIP-family row (no principled cutoff).
2. Continuous vs `human_score` -- every judge's raw score, NegT2IBench's own
   continuous score (fraction of ITS elements satisfied -- see
   neggenbench_elements.py), and the CLIP-family scores. Spearman/Kendall/
   pairwise accuracy over all pairs.
3. Same-prompt ranking -- pairwise accuracy restricted to images sharing the
   exact same prompt (different generator models), decisive where
   `human_label` differs. Threshold-free, so this is how the CLIP-family
   metrics get a legitimate binary-style comparison at all.

See HUMAN_METRIC_COMPARISON.md for the original spec and PROGRESS.md for run
history / current annotation coverage.

Usage:
    python -m human_validation.compare_human_metrics
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from .build_geneval_manifest import build_conditions, build_objects
from .neggenbench_elements import (
    elements_for_row as _neggenbench_elements_for_row,
    iter_results_rows as _iter_neggenbench_rows,
)

from .paths import DATA_ROOT, HUMAN_ANNOTATIONS, HUMAN_EVAL_DIR as METRIC_DIR
from .paths import REPO_ROOT, result_files

# --- Three separate evaluations, deliberately kept apart (see main()): ---
# 1. Binary vs human_label: every judge's own yes/no call (whole_binary or
#    _elem's AND-across-elements) plus NegT2IBench's native correct/incorrect.
#    No CLIP-family row -- there is no principled 0.5 cutoff for a raw
#    embedding-similarity score (HUMAN_METRIC_COMPARISON.md Sec.4).
# 2. Continuous vs human_score, with Spearman/Kendall/pairwise accuracy over
#    ALL pairs: every judge's raw score, NegT2IBench's new continuous score
#    (fraction of its own elements satisfied -- see neggenbench_elements.py),
#    and the CLIP-family scores.
# 3. Same-prompt ranking: a threshold-free way to score the CLIP-family
#    metrics (and, for comparison, everyone else) that sidesteps the "no
#    principled cutoff" problem entirely -- restricts pairwise accuracy to
#    pairs sharing the exact same prompt (multiple generator models per
#    prompt), so it never needs an absolute cutoff, only "which of these two
#    is scored higher."

CLIP_METRICS = ["clipscore", "blipscore", "negclip", "imagereward", "pickscore", "hpsv2"]
VQA_METRICS = ["vqascore", "mplug", "qwen25vl", "internvl"]
# "_elem" variants score each datapoint by averaging the model's own answer to
# every element's dedicated question (mirroring how human_score/human_label are
# built from conditionResponses/objectResponses), instead of asking one
# whole-caption question. Same source file as the bare metric (`{m}_both.jsonl`
# has both `whole_score` and `elements`); only the aggregation differs.
VQA_ELEMENT_METRICS = [f"{m}_elem" for m in VQA_METRICS]
CONTINUOUS_METRICS = CLIP_METRICS + VQA_METRICS + VQA_ELEMENT_METRICS
BINARY_METRICS = ["neggenbench"] + VQA_METRICS + VQA_ELEMENT_METRICS
METRICS = CONTINUOUS_METRICS + ["neggenbench"]


def _vqa_base(metric: str) -> str:
    """Strip the "_elem" suffix to get the underlying VQA output file's metric name."""
    return metric[: -len("_elem")] if metric.endswith("_elem") else metric




def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_datapoint_id(datapoint_id: str, model: str, pos_neg: str, tag: str) -> tuple[str, str]:
    prefix = f"{model}-{pos_neg}-{tag}-"
    if not datapoint_id.startswith(prefix):
        raise ValueError(f"docId {datapoint_id!r} does not start with {prefix!r}")
    rest = datapoint_id[len(prefix):]
    idx, _, sample = rest.partition("-samples-")
    if not sample:
        raise ValueError(f"docId {datapoint_id!r} has no -samples- suffix")
    return idx, sample


_METADATA_CACHE: dict[tuple[str, str, str, str], dict] = {}


def load_metadata(model: str, pos_neg: str, tag: str, idx: str) -> dict | None:
    key = (model, pos_neg, tag, idx)
    if key in _METADATA_CACHE:
        return _METADATA_CACHE[key]
    path = DATA_ROOT / model / pos_neg / tag / idx / "metadata.jsonl"
    metadata = None
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            line = handle.readline()
            if line.strip():
                metadata = json.loads(line)
    _METADATA_CACHE[key] = metadata
    return metadata


def elements_for(metadata: dict) -> tuple[list[dict], list[dict]]:
    """Return (object_elements, condition_elements) with dependency info."""
    objects = build_objects(metadata)
    conditions = build_conditions(metadata)
    obj_elements = [{"kind": "object", "id": f"obj:{name}", "name": name} for name in objects]

    for cond in conditions:
        required = {cond["subject"]}
        if cond.get("target"):
            required.add(cond["target"])
        cond["_requires"] = required
    return obj_elements, conditions


def aggregate_datapoint(datapoint_id: str, rows: list[dict]) -> dict | None:
    row0 = rows[0]
    model, pos_neg, tag = row0["model"], row0["posNeg"], row0["tag"]
    try:
        idx, _sample = split_datapoint_id(datapoint_id, model, pos_neg, tag)
    except ValueError:
        return None
    metadata = load_metadata(model, pos_neg, tag, idx)
    if metadata is None:
        return None

    obj_elements, cond_elements = elements_for(metadata)

    # object presence per annotator: annotator_idx -> {object_name: bool}
    obj_by_annotator = []
    cond_by_annotator = []
    for row in rows:
        obj_answers = {r["object"]: r["response"] for r in row.get("objectResponses", [])}
        cond_answers = {r["conditionId"]: r["response"] for r in row.get("conditionResponses", [])}
        obj_by_annotator.append(obj_answers)
        cond_by_annotator.append(cond_answers)

    n_annot = len(rows)

    # majority presence per object, among annotators who answered it
    obj_majority_present = {}
    for obj in obj_elements:
        name = obj["name"]
        answers = [oa[name] for oa in obj_by_annotator if name in oa]
        obj_majority_present[name] = (sum(answers) / len(answers) > 0.5) if answers else False

    per_annotator_scores = []
    element_satisfied_matrix = []  # list of dicts: element_key -> bool, one per annotator (for majority)
    exclusions = 0

    for a in range(n_annot):
        satisfied_flags = []
        elem_map = {}
        for obj in obj_elements:
            name = obj["name"]
            if name in obj_by_annotator[a]:
                sat = obj_by_annotator[a][name]
                satisfied_flags.append(sat)
                elem_map[obj["id"]] = sat
        for cond in cond_elements:
            cid = cond["id"]
            if any(not obj_majority_present.get(r, False) for r in cond["_requires"]):
                sat = False  # dependency rule: required object majority-absent
                satisfied_flags.append(sat)
                elem_map[cid] = sat
            elif cid in cond_by_annotator[a]:
                sat = cond_by_annotator[a][cid] == cond["expected"]
                satisfied_flags.append(sat)
                elem_map[cid] = sat
            else:
                exclusions += 1  # answer missing despite object present (unexpected)
        if satisfied_flags:
            per_annotator_scores.append(sum(satisfied_flags) / len(satisfied_flags))
        element_satisfied_matrix.append(elem_map)

    if not per_annotator_scores:
        return None

    human_score = sum(per_annotator_scores) / len(per_annotator_scores)

    # human_label: majority per element (tie -> unsatisfied), AND across all elements
    all_keys = set()
    for m in element_satisfied_matrix:
        all_keys.update(m.keys())
    human_label = 1
    for key in all_keys:
        vals = [m[key] for m in element_satisfied_matrix if key in m]
        majority = sum(vals) / len(vals) > 0.5
        if not majority:
            human_label = 0
            break

    # Unanimous = every element that any annotator answered got the same
    # satisfied/unsatisfied verdict from all annotators who answered it.
    # (Echoes GenEval's unanimous-subset agreement: a noise-free slice to
    # sanity-check metric agreement against, alongside the majority-vote one.)
    unanimous = all(
        len(set(m[key] for m in element_satisfied_matrix if key in m)) == 1
        for key in all_keys
    )

    return {
        "datapoint_id": datapoint_id,
        "model": model,
        "posNeg": pos_neg,
        "tag": tag,
        "category": row0.get("category"),
        "prompt": row0.get("prompt"),
        "n_annotators": n_annot,
        "human_score": human_score,
        "human_label": human_label,
        "unanimous": unanimous,
        "exclusions": exclusions,
    }


def build_human_table() -> dict[str, dict]:
    rows = read_jsonl(HUMAN_ANNOTATIONS)
    by_dp = defaultdict(list)
    for row in rows:
        by_dp[row["datapointId"]].append(row)

    table = {}
    skipped = []
    for dp_id, dp_rows in by_dp.items():
        agg = aggregate_datapoint(dp_id, dp_rows)
        if agg is None:
            skipped.append(dp_id)
        else:
            table[dp_id] = agg
    if skipped:
        print(f"[warn] skipped {len(skipped)} datapoints (no metadata found): {skipped[:5]}...",
              file=sys.stderr)
    return table


def load_metric_scores(metric: str) -> dict[str, float]:
    if metric in VQA_ELEMENT_METRICS:
        return load_metric_element_scores(_vqa_base(metric))
    if metric in VQA_METRICS:
        rows = read_jsonl(METRIC_DIR / f"{metric}_both.jsonl")
        return {row["docId"]: row["whole_score"] for row in rows}
    rows = read_jsonl(METRIC_DIR / f"{metric}.jsonl")
    return {row["docId"]: row["score"] for row in rows}


def load_vqa_binary_scores(metric: str) -> dict[str, float]:
    if metric in VQA_ELEMENT_METRICS:
        return load_metric_element_binary(_vqa_base(metric))
    rows = read_jsonl(METRIC_DIR / f"{metric}_both.jsonl")
    return {row["docId"]: float(row["whole_binary"]) for row in rows}


def _element_correctness(row: dict) -> list[tuple[bool, float]]:
    """Per element in a VQA `_both.jsonl` row: (is the answer correct?, P(correct answer)).

    `expected` (carried through from the same `elements` input every human
    annotator also answered) is the polarity a satisfying image should produce:
    True for object-presence questions, and for negated/affirmative conditions
    whatever `datapoint_fields.build_conditions` decided. `metric_p_yes` is
    P(Yes); flip it to P(No) when the correct answer is "No" so this is always
    P(the model gave the expected answer), comparable across elements.
    """
    out = []
    for e in row.get("elements", []):
        expected = bool(e["expected"])
        p_yes = float(e["metric_p_yes"])
        out.append((int(e["metric_binary"]) == int(expected), p_yes if expected else 1.0 - p_yes))
    return out


def load_metric_element_scores(metric: str) -> dict[str, float]:
    """Continuous avg_per_element score: mean P(correct answer) over a datapoint's elements."""
    rows = read_jsonl(METRIC_DIR / f"{metric}_both.jsonl")
    scores = {}
    for row in rows:
        pairs = _element_correctness(row)
        if pairs:
            scores[row["docId"]] = sum(p for _, p in pairs) / len(pairs)
    return scores


def load_metric_element_binary(metric: str) -> dict[str, float]:
    """Binary avg_per_element label: 1 iff every element got the expected answer (mirrors human_label)."""
    rows = read_jsonl(METRIC_DIR / f"{metric}_both.jsonl")
    scores = {}
    for row in rows:
        pairs = _element_correctness(row)
        if pairs:
            scores[row["docId"]] = 1.0 if all(correct for correct, _ in pairs) else 0.0
    return scores


def load_neggenbench_scores() -> dict[str, float]:
    """Our detector-based automatic evaluator (GenEval + Depth + DINO).

    Written per (model, setup, category) by `evaluate_images.py` under
    `DATA_ROOT/<model>/<setup>/results/<category>.jsonl`. Whole-item binary
    `correct` verdict, no continuous score and no per-element breakdown here.
    """
    scores: dict[str, float] = {}
    for path in result_files(DATA_ROOT):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                filename = Path(row["filename"])
                sample = filename.stem
                idx = filename.parent.parent.name
                category = filename.parent.parent.parent.name
                setup = filename.parent.parent.parent.parent.name
                model = filename.parent.parent.parent.parent.parent.name
                doc_id = f"{model}-{setup}-{category}-{idx}-samples-{sample}"
                scores[doc_id] = 1.0 if row["correct"] else 0.0
    return scores


def load_neggenbench_continuous_scores() -> dict[str, float]:
    """NegT2IBench's continuous score: fraction of ITS OWN elements (per-object
    count checks, per-relation edges, per-attribute confusion-set checks --
    see neggenbench_elements.py) satisfied for that row. The single-run
    analogue of human_score: one detector pass, not three annotators, so no
    averaging-over-raters step, just mean(satisfied) over that row's elements.

    Reconstructing this from the stored `correct`/`reason`/`details` fields
    (rather than needing a live rerun) is validated exhaustively in
    neggenbench_elements.py: ANDing every reconstructed element reproduces
    the stored `correct` boolean on all rows, or the row is skipped with a
    loud AssertionError rather than silently producing a wrong average.
    """
    scores: dict[str, float] = {}
    for row in _iter_neggenbench_rows(DATA_ROOT):
        elements = _neggenbench_elements_for_row(row)
        if not elements:
            continue
        filename = Path(row["filename"])
        sample = filename.stem
        idx = filename.parent.parent.name
        category = filename.parent.parent.parent.name
        setup = filename.parent.parent.parent.parent.name
        model = filename.parent.parent.parent.parent.parent.name
        doc_id = f"{model}-{setup}-{category}-{idx}-samples-{sample}"
        scores[doc_id] = sum(e["satisfied"] for e in elements) / len(elements)
    return scores


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rank(xs), rank(ys))


def kendall_tau(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 or dy == 0:
                continue
            if (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


def pairwise_accuracy(xs: list[float], ys: list[float], min_gap: float = 0.5) -> tuple[float, int]:
    n = len(xs)
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(ys[i] - ys[j]) < min_gap:
                continue
            total += 1
            metric_order = xs[i] > xs[j]
            human_order = ys[i] > ys[j]
            if metric_order == human_order:
                correct += 1
    if total == 0:
        return float("nan"), 0
    return correct / total, total


def same_prompt_pairwise_accuracy(human: dict[str, dict], metric_scores: dict[str, float]) -> tuple[float, int]:
    """Evaluation 3: threshold-free ranking comparison, restricted to pairs of
    images generated from the EXACT SAME prompt (different generator models,
    same text) that both have human annotations.

    For every such pair whose `human_label` differs (0 vs 1 -- the only
    decisive case for a binary label), correct = the metric scores the
    human_label=1 image higher. No absolute cutoff is ever consulted, only
    which of the two scores is bigger -- so this gives CLIP-family metrics
    (raw cosine similarities with no principled 0.5 point) a legitimate,
    non-arbitrary way into a binary-style comparison, mirroring
    HUMAN_METRIC_COMPARISON.md's own forced-choice design (Sec.5) but using
    naturally-occurring same-prompt variation across models instead of a
    constructed caption-polarity flip. Applies equally to every metric with a
    continuous score, not just the CLIP-family ones, as a same-prompt-controlled
    cross-check against Evaluation 2's pairwise accuracy (computed over ALL
    pairs, not just same-prompt ones).
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for i, d in human.items():
        if i in metric_scores and d.get("prompt"):
            groups[d["prompt"]].append(i)

    correct = total = 0
    for ids in groups.values():
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                li, lj = human[i]["human_label"], human[j]["human_label"]
                if li == lj:
                    continue
                total += 1
                si, sj = metric_scores[i], metric_scores[j]
                metric_prefers_i = si > sj
                human_prefers_i = li > lj
                if metric_prefers_i == human_prefers_i:
                    correct += 1
    if total == 0:
        return float("nan"), 0
    return correct / total, total


def accuracy(preds: list[int], labels: list[int]) -> float:
    n = len(preds)
    return sum(p == y for p, y in zip(preds, labels)) / n


def cohen_kappa(preds: list[int], labels: list[int]) -> float:
    n = len(preds)
    po = sum(p == y for p, y in zip(preds, labels)) / n
    p_pred1 = sum(preds) / n
    p_label1 = sum(labels) / n
    pe = p_pred1 * p_label1 + (1 - p_pred1) * (1 - p_label1)
    if pe == 1:
        return float("nan")
    return (po - pe) / (1 - pe)


def report_table(rows: list[str]) -> None:
    for row in rows:
        print(row)


def compute_measures(human: dict[str, dict], metric_scores: dict[str, float],
                      subset_ids: list[str] | None = None) -> dict | None:
    ids = subset_ids if subset_ids is not None else list(human.keys())
    ids = [i for i in ids if i in metric_scores]
    if len(ids) < 3:
        return None
    xs = [metric_scores[i] for i in ids]
    ys = [human[i]["human_score"] for i in ids]
    acc, n_pairs = pairwise_accuracy(xs, ys)
    return {
        "n": len(ids),
        "spearman": spearman(xs, ys),
        "kendall": kendall_tau(xs, ys),
        "pairwise_acc": acc,
        "n_pairs": n_pairs,
    }


def human_agreement_baseline(human: dict[str, dict]) -> None:
    """Annotator-count coverage, printed ahead of the leave-one-out ceiling."""
    counts = defaultdict(int)
    for d in human.values():
        counts[d["n_annotators"]] += 1
    for n in sorted(counts):
        print(f"Datapoints with {n} annotator(s): {counts[n]} / {len(human)}")


def leave_one_annotator_out_ceiling(min_annotators: int = 3) -> dict | None:
    """Human ceiling via leave-one-annotator-out (HUMAN_METRIC_COMPARISON.md line 42):
    "for each annotator, compute their agreement against the majority of the
    other two. Average over the 3." Every metric in Table 1/2 is ultimately
    trying to predict `human_score`/`human_label`; this asks how well a human
    predicts them too, using only the *other* humans -- the ceiling no metric
    should be expected to beat.

    Reuses `aggregate_datapoint` itself for both sides of the comparison: called
    with a single held-out annotator's row it gives that person's own
    (`human_score`, `human_label`) with no aggregation to speak of; called with
    the other two it gives their majority-vote label and (as the natural
    continuous extension, since "majority" isn't itself a fractional score)
    the mean of their two own fractions -- exactly `human_score`'s usual
    definition, just restricted to 2 raters.

    Only datapoints with >= `min_annotators` are eligible; ties are broken by
    keeping the first 3 annotation rows so "the 3" is well-defined even if a
    datapoint somehow has more. Datapoints with fewer are silently excluded,
    same as everywhere else in this script for items missing an input.
    """
    rows = read_jsonl(HUMAN_ANNOTATIONS)
    by_dp: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_dp[row["datapointId"]].append(row)

    continuous_folds: list[list[tuple[float, float]]] = [[], [], []]
    binary_folds: list[list[tuple[int, int]]] = [[], [], []]
    for dp_id, dp_rows in by_dp.items():
        if len(dp_rows) < min_annotators:
            continue
        dp_rows = dp_rows[:3]
        for fold in range(3):
            individual = aggregate_datapoint(dp_id, [dp_rows[fold]])
            majority = aggregate_datapoint(dp_id, [r for k, r in enumerate(dp_rows) if k != fold])
            if individual is None or majority is None:
                continue
            continuous_folds[fold].append((individual["human_score"], majority["human_score"]))
            binary_folds[fold].append((individual["human_label"], majority["human_label"]))

    if not continuous_folds[0]:
        return None

    def avg(values: list[float]) -> float:
        finite = [v for v in values if not math.isnan(v)]
        return sum(finite) / len(finite) if finite else float("nan")

    spearmans, kendalls, pair_accs, accs, kappas = [], [], [], [], []
    for fold in range(3):
        xs = [x for x, _ in continuous_folds[fold]]
        ys = [y for _, y in continuous_folds[fold]]
        acc_pa, _ = pairwise_accuracy(xs, ys)
        spearmans.append(spearman(xs, ys))
        kendalls.append(kendall_tau(xs, ys))
        pair_accs.append(acc_pa)
        preds = [p for p, _ in binary_folds[fold]]
        labels = [l for _, l in binary_folds[fold]]
        accs.append(accuracy(preds, labels))
        kappas.append(cohen_kappa(preds, labels))

    return {
        "n_datapoints": len(continuous_folds[0]),
        "spearman": avg(spearmans),
        "kendall": avg(kendalls),
        "pairwise_acc": avg(pair_accs),
        "accuracy": avg(accs),
        "kappa": avg(kappas),
    }


def main() -> None:
    # Materialize the two condensed JSONL files (neggenbench_both.jsonl,
    # human_scores.jsonl) alongside the VQA judges' *_both.jsonl in
    # HUMAN_EVAL_DIR, so every run of this script keeps them current --
    # not just a separate manual step. Lazy import: export_human_eval_files
    # imports FROM this module, so importing it back at module load time
    # would be circular; deferring to inside main() is safe because by the
    # time main() runs this module is already fully loaded.
    from .export_human_eval_files import (
        write_neggenbench_both, write_human_scores,
    )
    write_neggenbench_both()
    write_human_scores()
    print()

    human = build_human_table()
    print(f"Human-annotated datapoints usable: {len(human)}")
    total_exclusions = sum(d["exclusions"] for d in human.values())
    print(f"Element answers excluded (missing despite object present): {total_exclusions}")
    print()
    human_agreement_baseline(human)
    ceiling = leave_one_annotator_out_ceiling()
    if ceiling is None:
        print("NOTE: HUMAN_METRIC_COMPARISON.md specifies 3 annotators/item for a")
        print("leave-one-annotator-out ceiling. No datapoint has 3 yet, so no")
        print("ceiling is reported below.")
    else:
        print(f"Leave-one-annotator-out human ceiling (n={ceiling['n_datapoints']} datapoints, "
              f"averaged over the 3 held-out folds): spearman={ceiling['spearman']:.3f} "
              f"kendall={ceiling['kendall']:.3f} pair_acc={ceiling['pairwise_acc']:.3f} "
              f"accuracy={ceiling['accuracy']:.3f} kappa={ceiling['kappa']:.3f}")
    print()

    is_vqa = set(VQA_METRICS) | set(VQA_ELEMENT_METRICS)
    available_continuous = [m for m in CONTINUOUS_METRICS if (
        METRIC_DIR / (f"{_vqa_base(m)}_both.jsonl" if m in is_vqa else f"{m}.jsonl")
    ).is_file()]
    # Continuous scores for Evaluations 2 & 3: every judge's raw score, plus
    # NegT2IBench's NEW continuous score (fraction of its own elements
    # satisfied) -- not the binary "correct" field, which is Evaluation 1's.
    continuous_data = {m: load_metric_scores(m) for m in available_continuous}
    continuous_data["neggenbench"] = load_neggenbench_continuous_scores()
    continuous_metrics = available_continuous + ["neggenbench"]

    # Binary calls for Evaluation 1: every judge's own yes/no (whole_binary or
    # _elem AND), plus NegT2IBench's native correct/incorrect. No CLIP-family
    # row here -- see module docstring / HUMAN_METRIC_COMPARISON.md Sec.4.
    binary_data = {"neggenbench": load_neggenbench_scores()}
    binary_data.update({m: load_vqa_binary_scores(m)
                        for m in VQA_METRICS + VQA_ELEMENT_METRICS if m in continuous_data})

    for m, scores in continuous_data.items():
        overlap = len(set(scores) & set(human))
        print(f"{m}: {len(scores)} scored (continuous), {overlap} overlap with human annotations")
    print()

    # ============================================================
    # Evaluation 1 -- Binary vs human_label
    # ============================================================
    print("=" * 70)
    print("EVALUATION 1 -- Binary calls vs human_label (accuracy, Cohen's kappa)")
    print("VQA judges (whole_binary / _elem AND-across-elements) + NegT2IBench's")
    print("native correct/incorrect. No CLIP-family metrics: no principled 0.5")
    print("cutoff exists for a raw cosine-similarity score -- see Evaluation 3.")
    print("=" * 70)
    print(f"{'metric':<14}{'n':>5}{'accuracy':>10}{'kappa':>9}")
    for m in (m for m in BINARY_METRICS if m in binary_data):
        ids = [i for i in human if i in binary_data[m]]
        preds = [int(binary_data[m][i]) for i in ids]
        labels = [human[i]["human_label"] for i in ids]
        print(f"{m:<14}{len(ids):>5}{accuracy(preds, labels):>10.3f}{cohen_kappa(preds, labels):>9.3f}")
    if ceiling is not None:
        print(f"{'human_ceiling':<14}{ceiling['n_datapoints']:>5}{ceiling['accuracy']:>10.3f}{ceiling['kappa']:>9.3f}")
    print()

    # ============================================================
    # Evaluation 2 -- Continuous vs human_score (+ pairwise accuracy, all pairs)
    # ============================================================
    print("=" * 70)
    print("EVALUATION 2 -- Continuous scores vs human_score")
    print("VQA judges (whole_score / _elem mean P(correct)) + NegT2IBench's new")
    print("continuous score (fraction of its own elements satisfied) + the")
    print("CLIP-family metrics. Spearman / Kendall-tau / pairwise accuracy")
    print("(over ALL pairs -- see Evaluation 3 for the same-prompt-only variant).")
    print("=" * 70)
    header = f"{'metric':<14}{'n':>5}{'spearman':>11}{'kendall':>10}{'pair_acc':>10}{'n_pairs':>9}"
    print(header)
    for m in continuous_metrics:
        res = compute_measures(human, continuous_data[m])
        if res is None:
            print(f"{m:<14}  insufficient overlap")
            continue
        print(f"{m:<14}{res['n']:>5}{res['spearman']:>11.3f}{res['kendall']:>10.3f}"
              f"{res['pairwise_acc']:>10.3f}{res['n_pairs']:>9}")
    if ceiling is not None:
        print(f"{'human_ceiling':<14}{ceiling['n_datapoints']:>5}{ceiling['spearman']:>11.3f}"
              f"{ceiling['kendall']:>10.3f}{ceiling['pairwise_acc']:>10.3f}{'--':>9}")
    print()

    unanimous_ids = [i for i, d in human.items() if d["unanimous"]]
    print(f"-- Evaluation 2, unanimous subset only (n={len(unanimous_ids)}/{len(human)}) --")
    print(header)
    for m in continuous_metrics:
        res = compute_measures(human, continuous_data[m], unanimous_ids)
        if res is None:
            print(f"{m:<14}  insufficient overlap")
            continue
        print(f"{m:<14}{res['n']:>5}{res['spearman']:>11.3f}{res['kendall']:>10.3f}"
              f"{res['pairwise_acc']:>10.3f}{res['n_pairs']:>9}")
    print()

    # ============================================================
    # Evaluation 3 -- Same-prompt ranking (threshold-free, ALL continuous metrics)
    # ============================================================
    print("=" * 70)
    print("EVALUATION 3 -- Same-prompt ranking accuracy (threshold-free)")
    print("Pairwise accuracy restricted to pairs of images sharing the EXACT")
    print("SAME prompt (different generator models), decisive where human_label")
    print("differs. Never consults an absolute cutoff -- gives the CLIP-family")
    print("metrics a legitimate binary-style comparison, and doubles as a")
    print("same-prompt-controlled cross-check for everyone else vs Evaluation 2.")
    print("=" * 70)
    print(f"{'metric':<14}{'n_pairs':>9}{'accuracy':>10}")
    for m in continuous_metrics:
        acc, n_pairs = same_prompt_pairwise_accuracy(human, continuous_data[m])
        if n_pairs == 0:
            print(f"{m:<14}  insufficient same-prompt overlap")
            continue
        print(f"{m:<14}{n_pairs:>9}{acc:>10.3f}")
    print()

    # ============================================================
    # Breakdowns (Evaluation 2's continuous measures, by subgroup)
    # ============================================================
    def breakdown(key_fn, title):
        print(f"=== {title} ===")
        groups = defaultdict(list)
        for dp_id, d in human.items():
            groups[key_fn(d)].append(dp_id)
        for group_key in sorted(groups):
            ids = groups[group_key]
            print(f"-- {group_key} (n={len(ids)}) --")
            for m in continuous_metrics:
                res = compute_measures(human, continuous_data[m], ids)
                if res is None:
                    continue
                print(f"   {m:<14}{res['n']:>5}{res['spearman']:>11.3f}{res['kendall']:>10.3f}"
                      f"{res['pairwise_acc']:>10.3f}{res['n_pairs']:>9}")
        print()

    breakdown(lambda d: d["model"], "Table 5-like: by generator model")
    breakdown(lambda d: d["category"], "Table 6: by category")
    breakdown(lambda d: d["posNeg"], "Table 5: by setup (posNeg)")
    breakdown(lambda d: "pos*_neg0" if d["posNeg"].endswith("_neg0") else "has_negation",
              "By pos*_neg0 vs rest")


if __name__ == "__main__":
    main()
