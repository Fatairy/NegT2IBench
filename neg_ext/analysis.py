"""
analysis.py — load and reshape prompt datasets for the analysis notebook.

Pure pandas over the prompt JSONL (no GPU, model-independent), used by
summary_scores.py and visualize_results.py; keeping the logic here makes it
unit-testable.

The main entry points return tidy "long" DataFrames, one row per appearance, so
the notebook can `.value_counts()` / `groupby` freely:

    load_prompt_rows(root)      raw rows + {setting, file}
    explode_objects(df)         (setting, category, mode, object)          — presence
    explode_roles(df)           (setting, category, object, role, relation, polarity)
    explode_relations(df)       (setting, category, relation, polarity)
    explode_attributes(df)      (setting, attr_type, object, value, polarity)

A second group reads the EVALUATION results (which value / relation the model
actually gets wrong), joining each result row back to its statements:

    load_result_rows(root)      results + parsed metadata   (root = DATA_ROOT)
    result_rows_from_jsonl(p)   the same shape from ONE merged jsonl file
    explode_result_slots(df)    one row per slot, with per-slot blame
    key_failure_rates(slots)    per value/predicate: blame_rate, row_accuracy, …

A third group reports at the level the paper needs:

    failure_causes(res)         one row per FAILING row, with its single cause
    failure_cause_shares(c)     causes as shares that sum to 1 within a group
    prompt_success_counts(res)  k-of-n samples correct — systematic vs stochastic
    paired_polarity(slots)      negated vs positive, paired within the same image
    relation_confusion(res)     expected predicate vs the one that actually held
    wilson_interval(k, n)       95% CI on every reported accuracy
"""

from __future__ import annotations

import json
import os
import math
import re
from pathlib import Path

import pandas as pd

from . import vocab

# Per-setting verdict folder: <data_root>/<model>/<setting>/<RESULTS_SUBDIR>/<category>.jsonl.
# Set by config.sh; every stage (evaluate, summarize, report, human validation) reads it.
RESULTS_SUBDIR = os.environ.get("RESULTS_SUBDIR", "results")


# Run-level scalars that ride in the same `category` column as the real categories,
# so one long frame can carry both. Defined here rather than in a consumer because
# the plot layer and the table layer both key off them.
OVERALL = "__overall__"        # mean over per-category (image) accuracies
PROMPTS = "__prompts__"        # % of prompts with any correct sample
IMAGES = "__images__"          # % of individual images correct
NONCAT = {OVERALL, PROMPTS, IMAGES}


def _setting_key(name: str):
    m = re.match(r"pos(\d+)_neg(\d+)", name)
    return (int(m.group(1)), int(m.group(2))) if m else (99, 99)


def load_prompt_rows(root: str | Path, settings=None) -> pd.DataFrame:
    """Load every per-category prompt row under a root (e.g. negt2ibench_prompts/).

    Scans <root>/<setting>/neg_*.jsonl (excluding merged *_all). Returns one row
    per prompt with added 'setting' and 'file' columns. `settings` optionally
    restricts to a subset of setting names.
    """
    root = Path(root)
    rows = []
    setting_dirs = sorted((p for p in root.glob("pos*_neg*") if p.is_dir()),
                          key=lambda p: _setting_key(p.name))
    for sd in setting_dirs:
        if settings and sd.name not in settings:
            continue
        for f in sorted(sd.glob("neg_*.jsonl")):
            if f.name.endswith("_all.jsonl"):
                continue
            for line in f.read_text().splitlines():
                if line.strip():
                    rows.append({"setting": sd.name, "file": f.name, **json.loads(line)})
    return pd.DataFrame(rows)


def explode_objects(df: pd.DataFrame) -> pd.DataFrame:
    """One row per object appearance in a scene (from `include`)."""
    out = []
    for _, r in df.iterrows():
        for inc in r.get("include", []):
            out.append({"setting": r["setting"], "category": r["category"],
                        "mode": r.get("mode", "star"), "object": inc["class"]})
    return pd.DataFrame(out)


def explode_roles(df: pd.DataFrame) -> pd.DataFrame:
    """One row per relation endpoint: object + role (subject/target) + relation."""
    out = []
    for _, r in df.iterrows():
        if not str(r.get("tag", "")).startswith("neg_rel_"):
            continue
        for s in r.get("slots", []):
            subj = s.get("subject") or (r["include"][-1]["class"] if r.get("include") else None)
            out.append({"setting": r["setting"], "category": r["category"],
                        "object": subj, "role": "subject",
                        "relation": s["relation"], "polarity": s["polarity"]})
            out.append({"setting": r["setting"], "category": r["category"],
                        "object": s["target"], "role": "target",
                        "relation": s["relation"], "polarity": s["polarity"]})
    return pd.DataFrame(out)


def explode_relations(df: pd.DataFrame) -> pd.DataFrame:
    """One row per relation statement: (setting, category, relation, polarity)."""
    out = []
    for _, r in df.iterrows():
        if not str(r.get("tag", "")).startswith("neg_rel_"):
            continue
        for s in r.get("slots", []):
            out.append({"setting": r["setting"], "category": r["category"],
                        "relation": s["relation"], "polarity": s["polarity"]})
    return pd.DataFrame(out)


def explode_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per attribute statement: (setting, attr_type, object, value, polarity)."""
    out = []
    for _, r in df.iterrows():
        if not str(r.get("tag", "")).startswith("neg_attr_"):
            continue
        for s in r.get("slots", []):
            out.append({"setting": r["setting"], "attr_type": r["category"],
                        "object": s["object"], "value": s["value"],
                        "polarity": s["polarity"]})
    return pd.DataFrame(out)


# ── Evaluation results: which value / relation actually fails ─────────────────
#
# A result row carries ONE boolean `correct` for the whole prompt, but a row can
# hold several statements (a pos2_neg2 row has four). So row-level accuracy cannot say
# WHICH statement failed. The evaluator's `reason` string can: it names the
# offending statement verbatim. We reconstruct the exact phrase each statement would produce and test
# for it, rather than regex-parsing free text — deterministic, and it tracks the
# evaluator instead of guessing at it.
#
# Two failure kinds are kept apart, because conflating them is misleading:
#   object_missing  the object was never detected ("expected cup>=1, found 0").
#                   Nothing to do with the colour/relation — the model just did not
#                   draw the thing, so this must not be blamed on "red" or "left of".
#   blamed          the attribute/relation itself was violated. This is the signal.

def load_result_rows(root: str | Path, models=None, settings=None,
                     categories=None) -> pd.DataFrame:
    """Load evaluation results from <root>/<model>/<setting>/results/<category>.jsonl.

    Skips the merged all_results.jsonl (it duplicates the per-category files) and
    keeps only negation tags. `metadata` is parsed back into a dict so the statements are
    available.

    Model directories are listed one level at a time rather than swept with
    rglob("**/results/*.jsonl"). `**` does not descend into SYMLINKED directories, and
    a model directory may well be a symlink into another project -- such a model would
    otherwise vanish from every table with no warning. Listing also avoids walking the
    hundreds of thousands of per-prompt image directories under the data root.
    """
    root = Path(root)
    rows = []

    def take(f: Path, model: str, setting: str) -> None:
        if f.name == "all_results.jsonl":
            return
        if categories and f.stem not in categories:
            return
        if models and model not in models:
            return
        if settings and setting not in settings:
            return
        rows.extend(_parse_result_lines(f.read_text(), model, setting, f.stem))

    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sdir in sorted(p for p in mdir.iterdir()
                           if p.is_dir() and p.name != RESULTS_SUBDIR):
            for f in sorted((sdir / RESULTS_SUBDIR).glob("*.jsonl")):
                take(f, mdir.name, sdir.name)
    return pd.DataFrame(rows)


def _parse_result_lines(text: str, model: str, setting: str, category=None) -> list[dict]:
    """Parse jsonl text into result-row dicts, keeping only negation tags.

    `category` defaults to the row's own tag, so a MERGED file spanning several
    categories still gets a correct per-row category (the per-category files pass
    their filename stem instead, which is the same value).
    """
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tag = str(r.get("tag", ""))
        if not tag.startswith("neg_"):
            continue
        meta = r.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        out.append({"model": model, "setting": setting, "category": category or tag,
                    "tag": tag, "correct": bool(r.get("correct")),
                    "reason": r.get("reason") or "", "prompt": r.get("prompt", ""),
                    "filename": r.get("filename", ""), "metadata": meta or {}})
    return out


def result_rows_from_jsonl(path: str | Path, model: str = ".",
                           setting: str = ".") -> pd.DataFrame:
    """Result rows from ONE merged jsonl (what summary_scores.py is handed).

    Same columns as load_result_rows, so every downstream function works on either.
    `category` comes from each row's own tag, since a merged file mixes categories.
    Model and setting are unknown to a bare file path, hence the caller-supplied
    defaults.
    """
    return pd.DataFrame(_parse_result_lines(Path(path).read_text(), model, setting))


def _object_missing(reason: str, obj: str) -> bool:
    """Did the row fail because `obj` was not detected at all?
    Matches the evaluator's "expected <obj>>=<n>, found <k>" phrasing."""
    return f"expected {obj}>=" in reason


def _over_count(reason: str, obj: str) -> bool:
    """Did the row fail because MORE than the requested number of `obj` was drawn?

    Matches "expected exactly <n> <obj>, found <k> (counted instances)".
    Deliberately a separate phrase from the undetected one: both make the statement
    unscoreable, but they are opposite generator behaviours, and the undetected matcher
    above would otherwise swallow an over-count and report it as an object that was
    never drawn.
    """
    return f"expected exactly " in reason and f" {obj}, found " in reason


def _attr_blamed(reason: str, obj: str, value: str, attr_type: str, polarity: str) -> bool:
    """Was THIS attribute statement the one the evaluator complained about?

    Reconstructs the evaluator's phrasing (evaluate_neg_attribute). Both the current
    and the older wording are accepted, so results from earlier runs still parse:
        neg  "<obj> should NOT be <value> (<attr_type>), but one is"        (old)
             "<obj> should NOT be <value> (<attr_type>), but 1 is (in ...)" (current)
        pos  "expected <value> <obj>, found 0"                             (old)
             "expected <value> <obj>, found 0 (confusion set)"             (current)
    """
    if polarity == "neg":
        return f"{obj} should NOT be {value} ({attr_type})" in reason
    return f"expected {value} {obj}, found" in reason


def _rel_blamed(reason: str, subject: str, relation: str, polarity: str) -> bool:
    """Was THIS relation edge the one the evaluator complained about?

    evaluate_neg_relation emits, for expected_rel = "<rel>" or "not <rel>":
        "expected <subject> <expected_rel> target, but it IS <rel>"   (negated violated)
        "expected <subject> <expected_rel> target, found <rels>"      (positive unmet)
        "missing object for '<subject> <expected_rel> #<i>'"           (endpoint absent)
    """
    expected_rel = f"not {relation}" if polarity == "neg" else relation
    phrase = f"{subject} {expected_rel} target"
    return phrase in reason or f"'{subject} {expected_rel} #" in reason


def explode_result_slots(results_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (result row x statement), with blame attributed per statement.

    Columns: row_id, model, setting, category, tag, kind (attr|rel), subject,
    target, key (the attribute value or the relation predicate), polarity,
    row_correct, blamed, object_missing, violated.

    `row_id` is the source row's index in `results_df`, so statements can be regrouped
    back into rows (failure_causes needs this).

    `row_correct` is the whole prompt's verdict (shared by the row's statements);
    `blamed` marks the statement the evaluator actually named as failing. Use `blamed`
    for "which value/relation does the model get wrong", and `row_correct` only for
    coarse participation rates.

    `violated` = `blamed & ~object_missing` is the metric ALL reporting should use. `blamed` alone also fires when a relation edge merely lost an
    endpoint ("missing object for 'cup not left of #0'"), which inflates every
    relation predicate's difficulty by 5-15pp and reorders which looks hardest.
    `blamed` is kept unchanged so existing notebook numbers stay reproducible.
    """
    out = []
    for row_id, r in results_df.iterrows():
        meta, reason = r["metadata"], r["reason"]
        base = {k: r[k] for k in ("model", "setting", "category", "tag")}
        base["row_id"] = row_id
        base["row_correct"] = r["correct"]
        if r["tag"].startswith("neg_attr_"):
            attr_type = meta.get("category", r["category"].replace("neg_attr_", ""))
            for s in meta.get("slots", []):
                obj, val, pol = s["object"], s["value"], s["polarity"]
                out.append({**base, "kind": "attr", "subject": obj, "target": None,
                            "key": val, "polarity": pol,
                            "blamed": _attr_blamed(reason, obj, val, attr_type, pol),
                            "object_missing": _object_missing(reason, obj),
                            "over_count": _over_count(reason, obj)})
        elif r["tag"].startswith("neg_rel_"):
            # Read the edges the way evaluate_neg_relation does: from `include`, whose
            # entries carry `position` = [[expected_rel, target_index], ...]. There is
            # no top-level "subject" — the subject IS the entry holding the position,
            # which is what the evaluator names in `reason`.
            include = meta.get("include", [])
            for entry in include:
                req_pos = entry.get("position") or []
                subj = entry.get("class")
                for expected_rel, tgt_idx in req_pos:
                    pol = "neg" if str(expected_rel).startswith("not ") else "pos"
                    rel = expected_rel[4:] if pol == "neg" else expected_rel
                    tgt = (include[tgt_idx].get("class")
                           if isinstance(tgt_idx, int) and 0 <= tgt_idx < len(include) else None)
                    out.append({**base, "kind": "rel", "subject": subj, "target": tgt,
                                "key": rel, "polarity": pol,
                                "blamed": _rel_blamed(reason, subj, rel, pol),
                                "object_missing": _object_missing(reason, subj),
                                "over_count": (_over_count(reason, subj)
                                               or _over_count(reason, tgt or "\0"))})
    df = pd.DataFrame(out)
    if df.empty:
        return pd.DataFrame(columns=["row_id", "model", "setting", "category", "tag",
                                     "kind", "subject", "target", "key", "polarity",
                                     "row_correct", "blamed", "object_missing",
                                     "over_count", "violated"])
    df["violated"] = df["blamed"] & ~df["object_missing"] & ~df["over_count"]
    return df


def key_failure_rates(slots_df: pd.DataFrame, group=("key", "polarity")) -> pd.DataFrame:
    """Per attribute value / relation predicate: how often it is the named failure.

    Returns one row per group with:
        statements      how many prompt statements used it
        blamed          times the evaluator named it as the failure
        blame_rate      blamed / statements    <- the headline "does the model fail here"
        row_accuracy    share of its rows that passed overall (coarse: a row can fail
                        for another statement, so this understates a good value)
        object_missing  times the row failed only because the object was not detected

    Sorted worst-first by blame_rate, so the hardest values/relations come first.

    `violated` / `violated_rate` are the same counts with missing-endpoint edges
    excluded — prefer these for any "which predicate is hardest" claim;
    `blame_rate` is retained because the existing notebook sections report it.
    """
    if slots_df.empty:
        return pd.DataFrame(columns=["statements", "blamed", "blame_rate", "violated",
                                     "violated_rate", "row_accuracy", "object_missing"])
    group = list(group)
    g = slots_df.groupby(group, dropna=False)
    out = pd.DataFrame({
        "statements": g.size(),
        "blamed": g["blamed"].sum(),
        "violated": g["violated"].sum(),
        "row_accuracy": g["row_correct"].mean().round(3),
        "object_missing": g["object_missing"].sum(),
    })
    out["blame_rate"] = (out["blamed"] / out["statements"]).round(3)
    out["violated_rate"] = (out["violated"] / out["statements"]).round(3)
    out = out[["statements", "blamed", "blame_rate", "violated", "violated_rate",
               "row_accuracy", "object_missing"]]
    return out.sort_values("violated_rate", ascending=False)


# ── Failure causes: what broke, as shares of the failures ──────────
#
# The unit is the FAILING ROW, and the four buckets are mutually exclusive and
# exhaustive, so "N% of failures are undetected objects" is literally true.
#
# Deliberately NO precedence between negated and positive: in pos2_neg2, 36.4% of
# failing rows violate both, and a tie-break either way would move negation's
# headline share by that same 36 points. They get their own bucket instead, which
# makes "negated only" a clean lower bound on negation-specific failure.
#
# An undetected object pre-empts only when NO detected statement was violated — it
# cannot have its own statements judged, but a violated negation elsewhere in the same
# row is still real evidence. Hence violations are tested first, undetected last.

UNDETECTED = "undetected object"
OVER_COUNT = "over-count"
BOTH_VIOLATED = "both violated"
NEG_ONLY = "negated only"
POS_ONLY = "positive only"
UNCLASSIFIED = "unclassified"
CAUSE_ORDER = [UNDETECTED, OVER_COUNT, BOTH_VIOLATED, NEG_ONLY, POS_ONLY, UNCLASSIFIED]


def failure_causes(results_df: pd.DataFrame, slots_df=None) -> pd.DataFrame:
    """One row per FAILING result row, carrying its single exclusive cause.

    Columns: row_id, model, setting, category, tag, cause (one of CAUSE_ORDER),
    plus the raw neg_violated / pos_violated / undetected booleans behind it.

    UNCLASSIFIED should stay empty — it fires only if a row failed while none of
    its statements was named, e.g. an `exclude`-clause failure with no statement to blame. It
    is reported rather than dropped so a silent gap cannot masquerade as 100%.
    """
    cols = ["row_id", "model", "setting", "category", "tag", "cause",
            "neg_violated", "pos_violated", "undetected", "over_counted"]
    fail = results_df[~results_df["correct"].astype(bool)]
    if fail.empty:
        return pd.DataFrame(columns=cols)
    if slots_df is None:
        slots_df = explode_result_slots(results_df)

    s = slots_df[slots_df["row_id"].isin(fail.index)] if not slots_df.empty else slots_df
    if s.empty:
        agg = pd.DataFrame(index=fail.index,
                           data={"neg_violated": False, "pos_violated": False,
                                 "undetected": False, "over_counted": False})
    else:
        s = s.assign(_nv=(s["polarity"] == "neg") & s["violated"],
                     _pv=(s["polarity"] == "pos") & s["violated"])
        agg = s.groupby("row_id").agg(neg_violated=("_nv", "any"),
                                      pos_violated=("_pv", "any"),
                                      undetected=("object_missing", "any"),
                                      over_counted=("over_count", "any"))
        agg = agg.reindex(fail.index).fillna(False).astype(bool)

    nv, pv, ud = agg["neg_violated"], agg["pos_violated"], agg["undetected"]
    oc = agg["over_counted"]
    cause = pd.Series(UNCLASSIFIED, index=agg.index, dtype=object)
    cause[oc] = OVER_COUNT                      # fallbacks: no violation found
    cause[ud] = UNDETECTED
    cause[pv & ~nv] = POS_ONLY                  # a violation always outranks both
    cause[nv & ~pv] = NEG_ONLY
    cause[nv & pv] = BOTH_VIOLATED

    out = agg.copy()
    out["cause"] = cause
    for k in ("model", "setting", "category", "tag"):
        out[k] = fail[k]
    return out.reset_index(names="row_id")[cols]


def failure_cause_shares(causes_df: pd.DataFrame,
                         group=("model", "setting", "category")) -> pd.DataFrame:
    """Causes as counts AND shares that sum to 1 within each group.

    Long form (one row per group x cause) so it survives a CSV round-trip and
    plots directly as a stacked bar. Causes absent from a group are omitted rather
    than zero-filled — a setting with no positive statements genuinely has no
    'positive only' bucket, and a zero row would imply it was measured.
    """
    cols = list(group) + ["cause", "rows", "share"]
    if causes_df.empty:
        return pd.DataFrame(columns=cols)
    group = list(group)
    counts = causes_df.groupby(group + ["cause"], dropna=False).size().rename("rows")
    totals = counts.groupby(group, dropna=False).transform("sum")
    out = counts.to_frame()
    out["share"] = (counts / totals).round(4)
    out = out.reset_index()
    out["cause"] = pd.Categorical(out["cause"], CAUSE_ORDER, ordered=True)
    return out.sort_values(group + ["cause"])[cols]


# ── Systematicity: is a failure consistent or a coin flip? ────────────────────

_PROMPT_ID = r"/(\d+)/samples/"      # str, not compiled: Series.str.extract wants a pattern string
_SAMPLES_TAIL = r"/samples/[^/]*$"


def prompt_success_counts(results_df: pd.DataFrame) -> pd.DataFrame:
    """Per prompt: how many of its samples passed (k) out of how many exist (n).

    A prompt at 0/n is a systematic failure — the model cannot do it. A prompt at
    n/2 is stochastic. Row accuracy and 'any sample correct' both hide this
    distinction, and they are currently the only two numbers reported.

    A prompt is identified by its output DIRECTORY (the path above `samples/`),
    not by the numeric index: the index only counts within one model/setting/
    category, so on a merged file spanning several settings — which is exactly what
    summary_scores.py is handed for the all-settings aggregate, where every row
    carries setting "." — index 00045 from each setting would collapse into one
    group and report n = 6x the real sample count. The directory is unique
    regardless of what the model/setting columns say.
    """
    cols = ["model", "setting", "category", "prompt_id", "prompt_dir", "k", "n"]
    if results_df.empty:
        return pd.DataFrame(columns=cols)
    df = results_df.copy()
    fn = df["filename"].astype(str)
    df["prompt_id"] = fn.str.extract(_PROMPT_ID)[0]
    df["prompt_dir"] = fn.str.replace(_SAMPLES_TAIL, "", regex=True)
    df = df[df["prompt_id"].notna()]
    if df.empty:
        return pd.DataFrame(columns=cols)
    g = df.groupby(["model", "setting", "category", "prompt_dir"], dropna=False)
    out = g.agg(k=("correct", "sum"), n=("correct", "size"),
                prompt_id=("prompt_id", "first")).reset_index()
    out["k"] = out["k"].astype(int)
    return out[cols]


def systematicity_summary(counts_df: pd.DataFrame,
                          group=("model", "setting", "category")) -> pd.DataFrame:
    """Per group: prompts, all_failed, all_passed, partial — and their shares.

    Defined by `k == 0` and `k == n`, which is meaningful whatever n is. Prefer this
    over pivoting `systematicity()` on k: the moment one model or setting is
    generated with a different N_SAMPLES, a k-indexed pivot silently pools n=2 with
    n=4 and "passes all 4" quietly starts counting prompts that only passed 2 of 2.
    """
    cols = list(group) + ["prompts", "all_failed", "all_passed", "partial",
                          "all_failed_share", "all_passed_share", "partial_share",
                          "samples"]
    if counts_df.empty:
        return pd.DataFrame(columns=cols)
    group = list(group)
    df = counts_df.assign(_fail=counts_df["k"] == 0,
                          _pass=counts_df["k"] == counts_df["n"])
    g = df.groupby(group, dropna=False) if group else df.assign(_all=1).groupby("_all")
    out = g.agg(prompts=("_fail", "size"), all_failed=("_fail", "sum"),
                all_passed=("_pass", "sum"),
                samples=("n", lambda s: s.iloc[0] if s.nunique() == 1 else -1))
    out = out.reset_index(drop=not group)
    out["partial"] = out["prompts"] - out["all_failed"] - out["all_passed"]
    for k in ("all_failed", "all_passed", "partial"):
        out[f"{k}_share"] = (out[k] / out["prompts"]).round(4)
    return out[cols]


def systematicity(counts_df: pd.DataFrame,
                  group=("model", "setting", "category")) -> pd.DataFrame:
    """Share of prompts at each k-of-n, long form, summing to 1 within a group.

    `n` is carried in the output rather than assumed constant, so a consumer can
    label each bar with its own denominator (see systematicity_summary).
    """
    cols = list(group) + ["k", "n", "prompts", "share"]
    if counts_df.empty:
        return pd.DataFrame(columns=cols)
    group = list(group)
    counts = counts_df.groupby(group + ["k", "n"], dropna=False).size().rename("prompts")
    totals = counts.groupby(group, dropna=False).transform("sum")
    out = counts.to_frame()
    out["share"] = (counts / totals).round(4)
    return out.reset_index().sort_values(group + ["k"])[cols]


# ── Paired negation-vs-assertion, within the same image ──────────────────────

def _mcnemar(b: int, c: int) -> float:
    """Two-sided p for a paired binary contrast, from the discordant counts only.

    Continuity-corrected chi-square with 1 df; the survival function for 1 df is
    exactly erfc(sqrt(x/2)), so no scipy dependency is needed. Counts here are in
    the hundreds, well clear of where the exact binomial form would differ.
    """
    n = b + c
    if n == 0:
        return float("nan")
    chi2 = (abs(b - c) - 1) ** 2 / n if n > 1 else 0.0
    return math.erfc(math.sqrt(chi2 / 2))


def paired_polarity(slots_df: pd.DataFrame,
                    group=("model", "setting", "category")) -> pd.DataFrame:
    """Negated vs positive violation rate, PAIRED within each image.

    A pos<P>_neg<N> row with P>0 and N>0 scores both polarities on the SAME image,
    so this is a paired contrast, not two independent samples — which is the
    strongest form of the benchmark's central claim. Rows are reduced to two
    booleans (any negated statement violated / any positive statement violated) and compared
    with McNemar's test.

    Returns rows only for groups that actually contain both polarities, so the
    pos0_neg* settings (zero positive statements by construction) are simply absent.

    Columns: <group>, rows, neg_rate, pos_rate, delta (neg-pos), b (neg-only
    violated), c (pos-only violated), p.
    """
    cols = list(group) + ["rows", "neg_rate", "pos_rate", "delta", "b", "c", "p"]
    if slots_df.empty:
        return pd.DataFrame(columns=cols)
    group = list(group)
    s = slots_df.assign(_nv=(slots_df["polarity"] == "neg") & slots_df["violated"],
                        _pv=(slots_df["polarity"] == "pos") & slots_df["violated"],
                        _n=(slots_df["polarity"] == "neg"),
                        _p=(slots_df["polarity"] == "pos"))
    per_row = s.groupby(group + ["row_id"], dropna=False).agg(
        nv=("_nv", "any"), pv=("_pv", "any"),
        has_neg=("_n", "any"), has_pos=("_p", "any")).reset_index()
    per_row = per_row[per_row["has_neg"] & per_row["has_pos"]]
    if per_row.empty:
        return pd.DataFrame(columns=cols)

    out = []
    for keys, g in per_row.groupby(group, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        b = int((g["nv"] & ~g["pv"]).sum())
        c = int((g["pv"] & ~g["nv"]).sum())
        neg_rate, pos_rate = g["nv"].mean(), g["pv"].mean()
        out.append({**dict(zip(group, keys)), "rows": len(g),
                    "neg_rate": round(neg_rate, 4), "pos_rate": round(pos_rate, 4),
                    "delta": round(neg_rate - pos_rate, 4), "b": b, "c": c,
                    "p": _mcnemar(b, c)})
    return pd.DataFrame(out)[cols].sort_values(group)


# ── Relation confusion: what held instead of what was asked ──────────────────

# Anchor on the predicate vocabulary rather than splitting on spaces: both the
# subject ("dining table") and the predicate ("in front of") contain spaces, so a
# positional parse is ambiguous. Longest-first prevents "near" matching inside a
# longer predicate.
_PREDICATES = sorted(vocab.all_predicates(), key=len, reverse=True)
_NO_RELATION = "no relation"


def _split_expected(segment: str):
    """'<subject> <predicate>' -> predicate, or None if none of the vocab matches.

    The subject is discarded: it is already a column on the result row, and it is
    the part that makes a positional parse ambiguous ("dining table left of").
    """
    for p in _PREDICATES:
        if segment.endswith(" " + p):
            return p
    return None


def relation_confusion(results_df: pd.DataFrame,
                       group=("model", "setting", "category"),
                       same_category_only: bool = True) -> pd.DataFrame:
    """Expected relation predicate vs the predicate(s) that actually held.

    Parsed from the evaluator's own wording for an unmet POSITIVE edge:

        "expected <subj> <pred> target, found <pred> and <pred>"
        "expected <subj> <pred> target, found no relation"

    Only positive edges produce this line — a violated NEGATED edge says "but it
    IS <pred>", where the observed predicate is just the denied one and carries no
    information.

    `same_category_only` (default) keeps only observed predicates from the EXPECTED
    predicate's own category. `compute_relations` reports every predicate that holds
    across all four categories, so an unmet "behind" also reports "near", "left of"
    and so on — true, but it says nothing about why the depth edge failed, and at
    ~3x the volume of the informative cells it swamps the matrix. Filtering leaves
    the readable question: when "left of" was asked and missed, was it "right of"
    (an axis flip) or nothing at all?

    An edge with no surviving within-category observation lands in 'no relation',
    a distinct outcome from a wrong predicate: for the dead-zone categories it means
    the evaluator abstained rather than disagreed, so a negation there passes for
    free. Pass same_category_only=False for the raw cross-category counts.

    One output row per (group, expected, observed) with a count. An edge that fired
    several predicates contributes one row per observed predicate, so columns may
    sum above the number of edges. The diagonal is structurally zero — an edge whose
    own predicate held would not have failed.
    """
    cols = list(group) + ["expected", "observed", "count"]
    if results_df.empty:
        return pd.DataFrame(columns=cols)
    group = list(group)
    recs = []
    rel = results_df[results_df["tag"].astype(str).str.startswith("neg_rel_")]
    for _, r in rel.iterrows():
        keys = {k: r[k] for k in group}
        for line in str(r["reason"]).splitlines():
            if not line.startswith("expected ") or ", found " not in line:
                continue
            head, found = line[len("expected "):].split(", found ", 1)
            if not head.endswith(" target"):
                continue
            pred = _split_expected(head[: -len(" target")])
            if pred is None or pred.startswith("not "):
                continue                      # negated edges carry no information
            observed = ([] if found.strip() == _NO_RELATION
                        else [o for o in found.split(" and ") if o in _PREDICATES])
            if same_category_only:
                own = vocab.category_of(pred)
                observed = [o for o in observed if vocab.category_of(o) == own]
            for obs in observed or [_NO_RELATION]:
                recs.append({**keys, "expected": pred, "observed": obs})
    if not recs:
        return pd.DataFrame(columns=cols)
    return (pd.DataFrame(recs).groupby(group + ["expected", "observed"])
            .size().rename("count").reset_index()[cols])


# ── Statement-level failure buckets, with the dead zone ────────────────
#
# A second partition, for "of everything that went wrong, what KIND of thing was
# it". The row partition above cannot answer it in four buckets: 36.4% of pos2_neg2
# failing rows violate both polarities, so four buckets there would need exactly the
# tie-break the row partition forbids. A statement carries one polarity, so at statement level the four
# partition cleanly.

UNDETECTED_SLOT = "undetected object"
OVER_COUNT_SLOT = "over-count"
NEG_VIOLATED = "negated violated"
POS_VIOLATED = "positive violated"
DEAD_ZONE = "dead-zone failure"
INNOCENT = "innocent"
BUCKET_ORDER = [UNDETECTED_SLOT, OVER_COUNT_SLOT, NEG_VIOLATED, POS_VIOLATED, DEAD_ZONE]

# Categories where a dead zone cannot occur, and why. Attributes: a CLIP confusion
# set always contains its argmax, so there is no empty verdict. Proximity:
# gave it a single boundary, making near and far exact complements.
NO_DEAD_ZONE_REASON = {
    "attr": "attributes always yield a confusion set",
    "proximity": "near/far are exact complements",
}


def dead_zone_capable(category: str) -> bool:
    """Can a dead-zone failure occur in this category at all?

    Derived from the tag and the relation taxonomy rather than a hardcoded list, so
    adding a relation category cannot silently mislabel its structural zero.
    """
    cat = str(category)
    if not cat.startswith("neg_rel_"):
        return False                       # attributes: no empty verdict exists
    return cat.replace("neg_rel_", "") != "proximity"


def _dead_zone_edges(reason: str) -> set:
    """(subject, predicate) pairs whose POSITIVE edge failed with an empty verdict.

    An edge is in the dead zone when none of its OWN category's predicates appear in
    the evaluator's `found` list. compute_relations reports every predicate holding
    across all four categories, so an unmet "behind" also reports "near" and
    "left of" — true, but silent on why the depth edge failed. The evaluator's
    literal "no relation" wording needs the set empty across ALL categories and
    therefore almost never fires, which is why it is not used here.
    """
    out = set()
    for line in str(reason).splitlines():
        if not line.startswith("expected ") or ", found " not in line:
            continue
        head, found = line[len("expected "):].split(", found ", 1)
        if not head.endswith(" target"):
            continue
        segment = head[: -len(" target")]
        pred = _split_expected(segment)
        if pred is None or pred.startswith("not "):
            continue                       # negated edges cannot be dead-zone failures
        own = vocab.category_of(pred)
        observed = [o for o in found.split(" and ")
                    if o in _PREDICATES and vocab.category_of(o) == own]
        if not observed:
            out.add((segment[: -len(pred) - 1], pred))
    return out


def add_dead_zone(results_df: pd.DataFrame, slots_df: pd.DataFrame) -> pd.DataFrame:
    """Add a `dead_zone` column to statements: this POSITIVE edge failed with no verdict."""
    if slots_df.empty:
        out = slots_df.copy()
        out["dead_zone"] = []
        return out
    per_row = {rid: _dead_zone_edges(r) for rid, r in
               zip(results_df.index, results_df["reason"])}
    out = slots_df.copy()
    out["dead_zone"] = [
        r.polarity == "pos" and (r.subject, r.key) in per_row.get(r.row_id, ())
        for r in out.itertuples()
    ]
    return out


def slot_buckets(results_df: pd.DataFrame, slots_df=None) -> pd.DataFrame:
    """Slots with a single exclusive `bucket` each.

    Innocent statements — not violated and their object present — are labelled INNOCENT
    and should be filtered out before plotting: their row failed on a different
    statement, so they are not a failure cause. They are labelled rather than dropped so
    a caller can report how many were set aside.

    Order matters: undetected first (an absent object pre-empts any verdict on its
    own statement), then polarity, then dead zone as a refinement of positive.
    """
    if slots_df is None:
        slots_df = explode_result_slots(results_df)
    df = add_dead_zone(results_df, slots_df)
    if df.empty:
        df["bucket"] = []
        return df
    bucket = pd.Series(INNOCENT, index=df.index, dtype=object)
    viol = df["violated"].astype(bool)
    bucket[viol & (df["polarity"] == "pos")] = POS_VIOLATED
    bucket[viol & (df["polarity"] == "pos") & df["dead_zone"]] = DEAD_ZONE
    bucket[viol & (df["polarity"] == "neg")] = NEG_VIOLATED
    # Both pre-empt the verdict: an absent object has nothing to judge, an
    # over-counted one has no single referent. Mutually exclusive by construction, so
    # the order of these two assignments cannot matter.
    bucket[df["over_count"].astype(bool)] = OVER_COUNT_SLOT
    bucket[df["object_missing"].astype(bool)] = UNDETECTED_SLOT
    df["bucket"] = bucket
    return df


def slot_bucket_shares(buckets_df: pd.DataFrame,
                       group=("model", "setting", "category")) -> pd.DataFrame:
    """Failed statements per bucket, as counts and shares summing to 1 within each group.

    Innocent statements are excluded here, so `failed` is the group's denominator and the
    shares answer "of what went wrong in this cell, how much was each kind".
    """
    cols = list(group) + ["bucket", "statements", "share", "failed"]
    if buckets_df.empty:
        return pd.DataFrame(columns=cols)
    group = list(group)
    failed = buckets_df[buckets_df["bucket"] != INNOCENT]
    if failed.empty:
        return pd.DataFrame(columns=cols)
    counts = failed.groupby(group + ["bucket"], dropna=False).size().rename("statements")
    totals = counts.groupby(group, dropna=False).transform("sum")
    out = counts.to_frame()
    out["share"] = (counts / totals).round(4)
    out["failed"] = totals
    out = out.reset_index()
    out["bucket"] = pd.Categorical(out["bucket"], BUCKET_ORDER, ordered=True)
    return out.sort_values(group + ["bucket"])[cols]


def free_pass_exposure(buckets_df: pd.DataFrame,
                       group=("model", "setting", "category")) -> pd.DataFrame:
    """Negated statements that HELD in a dead-zone-capable category — the upper bound on
    unearned negation successes.

    A dead zone makes a negation pass, and the evaluator records nothing for a
    passing edge, so the free-pass rate itself is not measurable from these results.
    What IS measurable is the exposure: how many negated statements held where a free
    pass is even possible. Reported so the caption can bound the claim rather than
    imply the negation was earned.
    """
    cols = list(group) + ["negated_statements", "held", "held_share"]
    if buckets_df.empty:
        return pd.DataFrame(columns=cols)
    group = list(group)
    neg = buckets_df[(buckets_df["polarity"] == "neg")
                     & buckets_df["category"].map(dead_zone_capable)]
    if neg.empty:
        return pd.DataFrame(columns=cols)
    held = ~neg["violated"].astype(bool) & ~neg["object_missing"].astype(bool)
    if not group:                          # group=() -> one row for the whole set
        out = pd.DataFrame([{"negated_statements": len(neg), "held": int(held.sum())}])
    else:
        g = neg.assign(_held=held).groupby(group, dropna=False)
        out = g.agg(negated_statements=("_held", "size"), held=("_held", "sum")).reset_index()
    out["held_share"] = (out["held"] / out["negated_statements"]).round(4)
    return out[cols]


# ── Confidence intervals ─────────────────────────────────────────────────────

def wilson_interval(k, n, z: float = 1.96):
    """Wilson score interval for k successes in n trials (default 95%).

    Wilson rather than the normal approximation because these accuracies sit near
    0 — several category cells are under 0.05, where a Wald interval runs negative.
    """
    if not n:
        return (float("nan"), float("nan"))
    k, n = float(k), float(n)
    p = k / n
    d = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / d), min(1.0, (centre + half) / d)


def add_wilson(df: pd.DataFrame, correct="correct", total="total",
               prefix="ci") -> pd.DataFrame:
    """Add <prefix>_lo / <prefix>_hi columns from a correct/total pair."""
    out = df.copy()
    bounds = [wilson_interval(k, n) for k, n in zip(out[correct], out[total])]
    out[f"{prefix}_lo"] = [round(lo, 4) for lo, _ in bounds]
    out[f"{prefix}_hi"] = [round(hi, 4) for _, hi in bounds]
    return out


def object_concentration(objects_df: pd.DataFrame) -> pd.DataFrame:
    """Per setting: how concentrated object usage is — #unique objects, and the
    share taken by the top-8 objects (informs whether to restrict the LLM pool)."""
    rows = []
    for setting, g in objects_df.groupby("setting"):
        counts = g["object"].value_counts()
        top8 = counts.head(8).sum()
        rows.append({"setting": setting, "unique_objects": counts.size,
                     "total_appearances": int(counts.sum()),
                     "top8_share": round(top8 / counts.sum(), 3) if counts.sum() else 0.0})
    return pd.DataFrame(rows).sort_values("setting")
