"""
tables.py — render the analysis frames as Markdown report tables.

Companion to analysis.py (which computes) and visualize_results.py (which plots).
Kept separate and dependency-free beyond pandas so the tables are unit-testable and
so a figure change cannot break a number in the write-up.

    setting_report(...)     one Markdown doc per setting
    comparison_report(...)  one doc comparing every setting
    index_report(...)       README linking them

**Highlighting.** The bold cell in a comparison is the HIGHEST number, always — that
is what "highlight the highest" means and it stays predictable. Highest is not always
best, so every table header carries a direction marker: `↑` where higher is better
(accuracy) and `↓` where higher is worse (failure shares, undetected rate). A bold
`↓` cell is therefore the worst performer, and the legend on each page says so. The
alternative — bolding "the best", flipping meaning per column — was rejected: a
reader scanning for bold would have to re-derive the direction for every column.
"""

from __future__ import annotations

import re

import pandas as pd

from . import analysis as A

UP = "↑"      # higher is better
DOWN = "↓"    # higher is worse
DAGGER = "†"
LEGEND = (f"**Bold** = highest value in that comparison. "
          f"{UP} higher is better, {DOWN} higher is worse — so a bold {DOWN} cell is "
          f"the *worst* performer.")


# ── Markdown primitives ──────────────────────────────────────────────────────

def _md_table(headers, rows, aligns=None) -> str:
    """A GitHub-flavoured Markdown table. Cells are already-formatted strings."""
    if not rows:
        return "_no data_\n"
    aligns = aligns or ["--:" if i else ":--" for i in range(len(headers))]
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join(aligns) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def _pct(x, digits=1) -> str:
    return "—" if x is None or pd.isna(x) else f"{float(x) * 100:.{digits}f}%"


def _bold(s) -> str:
    return f"**{s}**"


def _fmt_highlight(values, fmt=_pct) -> list[str]:
    """Format a comparison group, bolding the single highest value.

    Genuine ties among two or more distinct values are all bolded — silently picking
    one would misreport the tie. But nothing is bolded when the group cannot
    discriminate: all values equal (every cell would be bold, which is just noise) or
    the maximum is zero (there is no "highest" worth pointing at). NaN never wins and
    renders as an em dash.
    """
    clean = [v for v in values if v is not None and not pd.isna(v)]
    if len(set(clean)) <= 1 or (clean and max(clean) == 0):
        return [fmt(v) for v in values]
    top = max(clean)
    return [_bold(fmt(v)) if v is not None and not pd.isna(v) and v == top else fmt(v)
            for v in values]


def _cat_label(category: str, mark_dead_zone=False) -> str:
    label = str(category).replace("neg_", "", 1)
    if mark_dead_zone and not A.dead_zone_capable(category):
        label += DAGGER
    return label


def _cats(df) -> list[str]:
    """Real categories only, relations before attributes."""
    cats = [c for c in df["category"].unique() if c not in A.NONCAT]
    return sorted(cats, key=lambda c: (0 if "rel" in c else 1, c))


def _slots_of(setting: str):
    """(positive, negated) statement counts, or None if the name is not pos<P>_neg<N>.

    One optional tuple rather than a tuple of optionals: callers then guard once and
    both values are known good, instead of every arithmetic use having to re-check.
    """
    m = re.match(r"pos(\d+)_neg(\d+)", str(setting))
    return (int(m.group(1)), int(m.group(2))) if m else None


DEAD_ZONE_NOTE = (
    f"{DAGGER} A dead zone cannot occur in this category: attributes always yield a "
    f"CLIP confusion set, and proximity has a single boundary. A blank "
    f"cell there means *impossible*, not *none measured*.")


def _free_pass_note(exposure: pd.DataFrame) -> str:
    if exposure.empty:
        return ""
    held, total = int(exposure["held"].sum()), int(exposure["negated_statements"].sum())
    if not total:
        return ""
    return (f"\n> **Negated success here is an upper bound.** A dead zone makes a "
            f"negation *pass*, and the evaluator records a reason only for failures, "
            f"so a **free pass** leaves no trace. {held:,} of {total:,} negated statements "
            f"({held / total:.1%}) held in the three dead-zone-capable categories, an "
            f"unknown share of them unearned.\n")


# ── Per-setting report ───────────────────────────────────────────────────────

def setting_report(setting: str, acc: pd.DataFrame, buckets: pd.DataFrame,
                   counts: pd.DataFrame, paired: pd.DataFrame,
                   models: list[str]) -> str:
    """Markdown doc for one setting: accuracy, failure causes, systematicity, paired.

    `acc` is the long accuracy frame (model, setting, category, accuracy, ci_*);
    the rest are the analysis frames, each already filtered to this setting.
    """
    shape = _slots_of(setting)
    models = [m for m in models if m in set(acc["model"])] or models
    doc = [f"# {setting}", ""]
    if shape:
        n_pos, n_neg = shape
        doc.append(f"`{n_pos}` positive statement(s), `{n_neg}` negated statement(s) per prompt "
                   f"— {n_pos + n_neg} total, the compositional load.")
        doc.append("")
    doc += [LEGEND, ""]

    # 1. Accuracy ------------------------------------------------------------
    doc += ["## 1. Accuracy", "",
            f"Share of images scored correct, with a Wilson 95% interval. {UP}", ""]
    cats = _cats(acc)
    headers = ["category"] + models
    rows = []
    for c in cats + [A.OVERALL, A.PROMPTS]:
        sub = acc[acc["category"] == c].set_index("model")
        vals = [sub["accuracy"].get(m) for m in models]
        cells = _fmt_highlight(vals)
        for i, m in enumerate(models):
            lo, hi = sub["ci_lo"].get(m), sub["ci_hi"].get(m)
            if lo is not None and not pd.isna(lo):
                cells[i] += f" <sub>[{lo * 100:.0f}–{hi * 100:.0f}]</sub>"
        name = {A.OVERALL: "**OVERALL** (mean over categories)",
                A.PROMPTS: "**prompts** (any sample correct)"}.get(c, _cat_label(c))
        rows.append([name] + cells)
    doc += [_md_table(headers, rows), ""]

    # 2. Failure causes ------------------------------------------------------
    doc += ["## 2. Where the failures come from", "",
            "One bucket per **failed statement**, so the four are exclusive and "
            "sum to 100%. Bold marks the dominant cause in each row. Innocent statements "
            "— present in a row that failed on a *different* statement — are excluded.", ""]
    shares = A.slot_bucket_shares(buckets, group=("model", "category"))
    if shares.empty:
        doc += ["_no failed statements_", ""]
    else:
        shares["bucket"] = shares["bucket"].astype(str)
        headers = ["category", "model", "failed statements"] + \
                  [f"{b} {DOWN}" for b in A.BUCKET_ORDER]
        rows = []
        for c in cats:
            for m in models:
                cell = shares[(shares["category"] == c) & (shares["model"] == m)]
                if cell.empty:
                    continue
                by = cell.set_index("bucket")["share"]
                vals = [by.get(b) for b in A.BUCKET_ORDER]
                # A structurally impossible bucket is blank, never 0.0 — see note.
                fmted = _fmt_highlight(vals)
                if not A.dead_zone_capable(c):
                    fmted[A.BUCKET_ORDER.index(A.DEAD_ZONE)] = "n/a"
                rows.append([_cat_label(c, mark_dead_zone=True), m,
                             f"{int(cell['failed'].iloc[0]):,}"] + fmted)
        doc += [_md_table(headers, rows), "", DEAD_ZONE_NOTE, ""]

    # 3. Systematicity -------------------------------------------------------
    sy = A.systematicity_summary(counts, group=("model", "category")) \
        if not counts.empty else pd.DataFrame()
    if not sy.empty:
        # k==0 / k==n are well defined whatever n is, so this stays correct if a model
        # is ever generated with a different N_SAMPLES. `samples` is -1 when a group
        # itself mixes denominators, which the column then reports rather than hides.
        ns = sorted({int(v) for v in sy["samples"] if v > 0})
        n_txt = str(ns[0]) if len(ns) == 1 else "n"
        doc += ["## 3. Systematic or stochastic?", "",
                f"Share of prompts failing **every** sample versus passing every "
                f"sample. A prompt at 0/{n_txt} is a model that cannot do the task; "
                f"the middle is a coin flip. Neither headline accuracy number shows "
                f"this.", ""]
        headers = ["category", "model", "samples", f"fails all {DOWN}",
                   f"passes all {UP}", "in between"]
        rows = []
        for c in cats:
            present, zero_vals, all_vals, mid_vals, samples = [], [], [], [], []
            for m in models:
                cell = sy[(sy["category"] == c) & (sy["model"] == m)]
                if cell.empty:
                    continue
                r = cell.iloc[0]
                present.append(m)
                zero_vals.append(float(r["all_failed_share"]))
                all_vals.append(float(r["all_passed_share"]))
                mid_vals.append(float(r["partial_share"]))
                samples.append(int(r["samples"]))
            if not present:
                continue
            zf, af = _fmt_highlight(zero_vals), _fmt_highlight(all_vals)
            for i, m in enumerate(present):
                rows.append([_cat_label(c) if i == 0 else "", m,
                             "mixed" if samples[i] < 0 else str(samples[i]),
                             zf[i], af[i], _pct(mid_vals[i])])
        doc += [_md_table(headers, rows), ""]

    # 4. Paired negated vs positive -----------------------------------------
    if not paired.empty:
        doc += ["## 4. Negated vs positive, paired within the same image", "",
                "This setting scores both polarities on the *same* image, so the "
                "contrast is paired, not two independent samples. `delta > 0` means "
                "negation is harder. `p` is McNemar on the discordant rows.", ""]
        headers = ["category", "model", f"negated violated {DOWN}",
                   f"positive violated {DOWN}", "delta", "p"]
        rows = []
        for c in _cats(paired.rename(columns={"category": "category"})):
            for m in models:
                cell = paired[(paired["category"] == c) & (paired["model"] == m)]
                if cell.empty:
                    continue
                r = cell.iloc[0]
                nf, pf = _fmt_highlight([r["neg_rate"], r["pos_rate"]])
                p = r["p"]
                stars = ("***" if p < 1e-3 else "**" if p < 1e-2
                         else "*" if p < 0.05 else "ns")
                rows.append([_cat_label(c), m, nf, pf,
                             f"{r['delta'] * 100:+.1f} pp",
                             f"{p:.1e} {stars}"])
        doc += [_md_table(headers, rows), "",
                "`***` p<.001 `**` p<.01 `*` p<.05 `ns` not significant", ""]

    doc.append(_free_pass_note(A.free_pass_exposure(buckets, group=())))
    return "\n".join(doc).rstrip() + "\n"


# ── Cross-setting comparison ─────────────────────────────────────────────────

def comparison_report(acc: pd.DataFrame, buckets: pd.DataFrame,
                      counts: pd.DataFrame, models: list[str],
                      settings: list[str]) -> str:
    """Markdown doc comparing every setting: accuracy, failure mix, the confound."""
    doc = ["# Setting comparison", "", LEGEND, ""]

    # 1. Headline accuracy ---------------------------------------------------
    doc += ["## 1. Overall accuracy by setting", "",
            f"Mean over per-category accuracies. {UP} Bold = each model's best "
            f"setting.", ""]
    headers = ["setting", "statements", "negated"] + models
    per_model_vals = {m: [] for m in models}
    for s in settings:
        for m in models:
            sub = acc[(acc["setting"] == s) & (acc["model"] == m)
                      & (acc["category"] == A.OVERALL)]
            per_model_vals[m].append(None if sub.empty else float(sub["accuracy"].iloc[0]))
    cols = {m: _fmt_highlight(per_model_vals[m]) for m in models}   # bold down a column
    rows = []
    for i, s in enumerate(settings):
        shape = _slots_of(s)
        rows.append([f"`{s}`", str(sum(shape)) if shape else "—",
                     str(shape[1]) if shape else "—"] + [cols[m][i] for m in models])
    doc += [_md_table(headers, rows), ""]

    # 2. Category accuracy per model ----------------------------------------
    doc += ["## 2. Accuracy by category and setting", "",
            f"One table per model; bold = that category's best setting. {UP}", ""]
    cats = _cats(acc)
    for m in models:
        sub = acc[acc["model"] == m]
        if sub.empty:
            continue
        present = [s for s in settings if s in set(sub["setting"])]
        if not present:
            continue
        doc += [f"### {m}", ""]
        rows = []
        for c in cats:
            vals = []
            for s in present:
                cell = sub[(sub["setting"] == s) & (sub["category"] == c)]
                vals.append(None if cell.empty else float(cell["accuracy"].iloc[0]))
            rows.append([_cat_label(c)] + _fmt_highlight(vals))
        doc += [_md_table(["category"] + [f"`{s}`" for s in present], rows), ""]

    # 3. Failure mix ---------------------------------------------------------
    shares = A.slot_bucket_shares(buckets, group=("model", "setting"))
    if not shares.empty:
        shares["bucket"] = shares["bucket"].astype(str)
        doc += ["## 3. Failure mix by setting", "",
                "Share of failed statements in each bucket, pooled over categories. Bold = "
                "the dominant cause for that row.", ""]
        headers = ["setting", "model", "failed statements"] + \
                  [f"{b} {DOWN}" for b in A.BUCKET_ORDER]
        rows = []
        for s in settings:
            for m in models:
                cell = shares[(shares["setting"] == s) & (shares["model"] == m)]
                if cell.empty:
                    continue
                by = cell.set_index("bucket")["share"]
                rows.append([f"`{s}`", m, f"{int(cell['failed'].iloc[0]):,}"]
                            + _fmt_highlight([by.get(b) for b in A.BUCKET_ORDER]))
        doc += [_md_table(headers, rows), ""]

    # 4. The confound --------------------------------------------------------
    doc += ["## 4. Negation load at matched compositional load", "",
            "Total statements and negated statements move together across the default settings, "
            "so a raw accuracy drop cannot separate *negation is hard* from *more "
            "statements is hard*. Rows sharing a total statement count isolate it. Only "
            "counts holding two or more negation levels are shown — a single level "
            "identifies nothing.", ""]
    groups = {}
    for s in settings:
        shape = _slots_of(s)
        if not shape:
            continue
        groups.setdefault(sum(shape), []).append((shape[1], s))
    any_group = False
    for total in sorted(groups):
        members = sorted(groups[total])
        if len({n for n, _ in members}) < 2:
            continue
        any_group = True
        doc += [f"### {total} statement(s) total", ""]
        rows = []
        cols = {}
        for m in models:
            vals = []
            for _, s in members:
                cell = acc[(acc["setting"] == s) & (acc["model"] == m)
                           & (acc["category"] == A.OVERALL)]
                vals.append(None if cell.empty else float(cell["accuracy"].iloc[0]))
            cols[m] = _fmt_highlight(vals)
        for i, (n, s) in enumerate(members):
            rows.append([f"`{s}`", str(n)] + [cols[m][i] for m in models])
        doc += [_md_table(["setting", f"negated statements"] + models, rows), ""]
    if not any_group:
        doc += ["_No matched pair yet — every total statement count has a single negation "
                "level. Generate a positive-only control (`pos<P>_neg0`) at a statement "
                "count you already have._", ""]

    # 5. Systematicity across settings --------------------------------------
    if not counts.empty:
        sy = A.systematicity_summary(counts, group=("model", "setting"))
        if not sy.empty:
            doc += ["## 5. Systematic failure by setting", "",
                    f"Share of prompts failing **every** sample. {DOWN}", ""]
            cols = {}
            for m in models:
                vals = []
                for s in settings:
                    cell = sy[(sy["setting"] == s) & (sy["model"] == m)]
                    vals.append(None if cell.empty
                                else float(cell["all_failed_share"].iloc[0]))
                cols[m] = _fmt_highlight(vals)
            rows = [[f"`{s}`"] + [cols[m][i] for m in models]
                    for i, s in enumerate(settings)]
            doc += [_md_table(["setting"] + models, rows), ""]

    # 6. Free-pass exposure --------------------------------------------------
    exposure = A.free_pass_exposure(buckets, group=("model", "setting"))
    if not exposure.empty:
        doc += ["## 6. Free-pass exposure", "",
                "Negated statements that **held** in the three categories where a dead zone "
                "is possible. A dead zone makes a negation pass without the image "
                "demonstrating anything, and the evaluator leaves no trace of a "
                "passing edge — so this is the ceiling on how much of the negated "
                f"success in those categories may be unearned. {DOWN}", ""]
        cols = {}
        for m in models:
            vals = []
            for s in settings:
                cell = exposure[(exposure["setting"] == s) & (exposure["model"] == m)]
                vals.append(None if cell.empty else float(cell["held_share"].iloc[0]))
            cols[m] = _fmt_highlight(vals)
        rows = [[f"`{s}`"] + [cols[m][i] for m in models]
                for i, s in enumerate(settings)]
        doc += [_md_table(["setting"] + models, rows), ""]

    return "\n".join(doc).rstrip() + "\n"


def index_report(settings: list[str], models: list[str], figures: list[str]) -> str:
    """README linking every generated doc, so the output dir is self-describing."""
    doc = ["# Evaluation report", "",
           f"Models: {', '.join(f'`{m}`' for m in models)}  ",
           f"Settings: {', '.join(f'`{s}`' for s in settings)}", "",
           "## Documents", "",
           "- [Setting comparison](report__comparison.md) — every setting side by "
           "side, plus the matched-load analysis that separates negation difficulty "
           "from compositional load", ""]
    for s in settings:
        shape = _slots_of(s)
        note = f" — {shape[0]} positive, {shape[1]} negated" if shape else ""
        doc.append(f"- [`{s}`](report__{s}.md){note}")
    doc += ["", "## Figures", ""]
    doc.append(", ".join(f"`{f}`" for f in figures) if figures
               else "_none generated (`--figures none`)_")
    doc += ["", "## Reading the tables", "", LEGEND, "", DEAD_ZONE_NOTE, ""]
    return "\n".join(doc).rstrip() + "\n"


# ── Public helpers for other reports (notebooks/prompt_analysis.py) ──────────

def frame_to_markdown(df: pd.DataFrame, index=True, index_name=None,
                      float_fmt="{:.3f}") -> str:
    """Render any DataFrame as a Markdown table using this module's renderer.

    Kept here rather than reaching for pandas' `.to_markdown()` so every table in
    the project goes through one implementation and neg_ext keeps its
    pandas-only dependency (`.to_markdown()` needs tabulate).
    """
    if df is None or df.empty:
        return "_no data_\n"
    body = df.reset_index() if index else df.copy()
    # Only rename for a single-level index. reset_index() on a MultiIndex emits one
    # column per level, all already correctly named, and renaming just the first would
    # relabel it while leaving the rest — printing e.g. "model / category | category".
    if index and index_name and getattr(df.index, "nlevels", 1) == 1:
        body = body.rename(columns={body.columns[0]: index_name})

    def cell(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        if isinstance(v, float):
            return float_fmt.format(v)
        return str(v)

    headers = [str(c) for c in body.columns]
    rows = [[cell(v) for v in row] for row in body.itertuples(index=False)]
    return _md_table(headers, rows)
