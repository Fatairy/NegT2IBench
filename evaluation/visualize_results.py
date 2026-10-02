#!/usr/bin/env python3
"""
visualize_results.py

Plot how models and settings compare, and WHY the failures happen.

Two data paths, one per question:
    <data_root>/<model>/<setting>/results/all_results_summary.csv   how accurate
    <data_root>/<model>/<setting>/results/*.jsonl                   what broke

The CSV carries one accuracy per category and nothing statement-level, so the failure
analysis reads the raw results through neg_ext.analysis. Both are re-scanned every
run, so adding a new setting (or model) and re-running just picks it up. Restrict
with --models / --settings, or plot all.

Outputs go under <plots-dir>/<subdir>/, where <subdir> is derived from the models
AND settings plotted (e.g. 'flux2+sd35__pos0_neg1+pos1_neg0') so different
selections never overwrite each other; override with --subdir.

What is emitted is chosen with ONE parameter, --figures, taking a comma-separated
list of figure names or named sets (default: `report`, a single PNG). Emitting
everything is ~70 PNGs per run and is not reviewable, so it is opt-in via
`--figures all`. Slice with --models / --settings as before.

    figures  report        failure_report.png — THE report: one grid, rows = models,
                            cols = settings, a stacked bar per category over the four
                            statement-level buckets (undetected object / negated violated /
                            positive violated / dead-zone failure)
             overall       overall__models_by_setting.png, prompt_accuracy__*.png
             bars          settings_within__<model>.png, models_within__<setting>.png
             radar         radar__settings_within__<model>.png, radar__models_within__*
             matched       slot_count__matched.png — accuracy vs negation count at
                            MATCHED statement count; emitted only once a positive-only
                            control (pos<P>_neg0) exists
             causes        failure_causes__<setting>.png  (row-level)
             systematicity systematicity__<setting>.png   k-of-n samples correct
             paired        paired_polarity__<setting>.png negated vs positive, paired
             keys          keys__<category>__<setting>.png which value/predicate fails
             confusion     confusion__<category>.png      expected vs observed predicate

    sets     none | report | core (report+overall+matched) | accuracy | failure | all

Markdown reports are written by default (--no-tables to skip); they are the primary
deliverable, with figures as illustration:

    README.md               index linking every doc
    report__comparison.md   every setting side by side + the matched-load analysis
    report__<setting>.md    accuracy, failure causes, systematicity, paired polarity

Tables (tidy CSV, for the paper appendix):

    summary_long.csv        every (model, setting, category, accuracy, CI)
    failure_statements.csv  statement-bucket shares per (model, setting, category)
    free_pass_exposure.csv  negated statements that HELD where a dead zone is possible —
                            the upper bound on unearned negation successes
    statements_long.csv     every statement, with violated / object_missing
    failure_causes.csv      cause shares per (model, setting, category)
    systematicity.csv       k-of-n distribution per (model, setting, category)
    paired_polarity.csv     paired rates + McNemar p per (model, setting, category)
    relation_confusion.csv  expected x observed predicate counts

Usage:
    python evaluation/visualize_results.py                     # report + Markdown
    python evaluation/visualize_results.py --figures all       # everything
    python evaluation/visualize_results.py --figures none      # tables only
    python evaluation/visualize_results.py --figures core --models flux2 sd35
    python evaluation/visualize_results.py --figures report,confusion
    python evaluation/visualize_results.py --settings pos2_neg2 --subdir negation_only
    python evaluation/visualize_results.py --data-root outputs/images --out-dir outputs/report

Normally driven by scripts/5_report.sh, which runs stage 4 first and forwards
FIGURES / MODELS / SETTINGS.
"""

from __future__ import annotations

import argparse
import os
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")            # headless: write PNGs, no display
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# `python evaluation/visualize_results.py` puts evaluation/ on sys.path, NOT the
# repo root, so neg_ext is not importable without help.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from neg_ext import analysis as A
from neg_ext import tables as T
from neg_ext import vocab

# Okabe–Ito: colour-blind-safe by construction, assigned in FIXED order so a given
# entity (setting or model) keeps its colour across every figure. Yellow (#F0E442)
# is dropped — too low-contrast for fills on white.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
           "#CC79A7", "#56B4E9", "#999999", "#000000"]
GRID = "#D9D9D9"
INK = "#222222"
MUTED = "#666666"
OVERALL, PROMPTS, IMAGES, NONCAT = A.OVERALL, A.PROMPTS, A.IMAGES, A.NONCAT
OVERALL_LABEL = "OVERALL"

# One colour per failure cause, held fixed across every figure. Undetected is grey
# because it is the "not about negation" bucket; the two single-polarity buckets
# take the two most distinct hues, and 'both violated' sits between them.
CAUSE_COLORS = {
    A.UNDETECTED: "#999999",
    A.OVER_COUNT: "#666666",      # a sibling of undetected: both are count failures
    A.BOTH_VIOLATED: "#CC79A7",
    A.NEG_ONLY: "#D55E00",
    A.POS_ONLY: "#0072B2",
    A.UNCLASSIFIED: "#000000",
}
POLARITY_COLORS = {"neg": "#D55E00", "pos": "#0072B2"}
# Statement-level buckets. Dead zone takes the light blue deliberately: it is a
# refinement of 'positive violated' (dark blue), and reading them as related is
# correct — both are positive-edge failures, one disagreement and one abstention.
BUCKET_COLORS = {
    A.UNDETECTED_SLOT: "#999999",
    A.OVER_COUNT_SLOT: "#666666",   # darker grey: sibling of undetected, not a violation
    A.NEG_VIOLATED: "#D55E00",
    A.POS_VIOLATED: "#0072B2",
    A.DEAD_ZONE: "#56B4E9",
}
DAGGER = "†"
HEAT = LinearSegmentedColormap.from_list("heat", ["#FFFFFF", "#0072B2"])


def shorten(category: str) -> str:
    """neg_rel_directional -> rel_directional; neg_attr_color -> attr_color."""
    return category.replace("neg_", "", 1)


def setting_key(setting: str):
    """Sort pos<P>_neg<N> by (P, N); unknown formats sort last alphabetically."""
    m = re.match(r"pos(\d+)_neg(\d+)", setting)
    return (0, int(m.group(1)), int(m.group(2))) if m else (1, 0, setting)


def setting_slots(setting: str):
    """(total statements, negated statements) for pos<P>_neg<N>, or (None, None).

    Total statements is P+N — the compositional load — and N alone is the negation load.
    Separating them is the whole point of the matched-control figure: in the
    original three settings the two were perfectly collinear.
    """
    m = re.match(r"pos(\d+)_neg(\d+)", setting)
    if not m:
        return None, None
    p, n = int(m.group(1)), int(m.group(2))
    return p + n, n


def nice_ceil(x: float) -> float:
    """Round an accuracy up to the next 0.1, floored at 0.1."""
    if not np.isfinite(x) or x <= 0:
        return 0.1
    return max(0.1, math.ceil(x * 10 - 1e-9) / 10)


# ── Discovery ────────────────────────────────────────────────────────────────

def _model_dirs(data_root: Path, models):
    """Model dirs under the root, or exactly the ones asked for.

    Auto-discovery requires EVALUATED settings (`pos*_neg*/results/*.jsonl`), not
    merely a `pos*_neg*` child: the sibling `prompts/` holds per-setting prompt dirs
    and `plots/` can hold a view dir called `pos_sweep__neg0`, so the looser test
    picked both up as models.
    """
    if models:
        return [data_root / m for m in models]
    return sorted(p for p in data_root.iterdir()
                  if p.is_dir() and any(p.glob(f"pos*_neg*/{A.RESULTS_SUBDIR}/*.jsonl")))


def _setting_dirs(md: Path, settings):
    if settings:
        return [md / s for s in settings]
    return sorted(md.glob("pos*_neg*"), key=lambda p: setting_key(p.name))


def discover(data_root: Path, models, settings) -> pd.DataFrame:
    """Accuracy per (model, setting, category) from the summary CSVs.

    `correct`/`total` come along so Wilson intervals can be computed; they are
    blank for the derived overall_avg_over_tasks row, which therefore gets no CI.
    Unrecognised `type` values (failure_cause, systematicity, …) are skipped, so
    CSVs written by a newer summary_scores.py load unchanged.
    """
    records = []
    for md in _model_dirs(data_root, models):
        if not md.is_dir():
            print(f"[viz] skip missing model dir: {md}")
            continue
        for sd in _setting_dirs(md, settings):
            csv = sd / A.RESULTS_SUBDIR / "all_results_summary.csv"
            if not csv.exists():
                print(f"[viz] no summary CSV: {csv} (skipped)")
                continue
            df = pd.read_csv(csv)
            for _, row in df.iterrows():
                if row["type"] == "task":
                    cat = str(row["name"])
                elif row["type"] == "summary":
                    cat = {"overall_avg_over_tasks": OVERALL,
                           "prompts": PROMPTS, "images": IMAGES}.get(row["name"])
                    if cat is None:
                        continue
                else:
                    continue
                records.append({"model": md.name, "setting": sd.name,
                                "category": cat, "accuracy": float(row["accuracy"]),
                                "correct": pd.to_numeric(row.get("correct"),
                                                         errors="coerce"),
                                "total": pd.to_numeric(row.get("total"),
                                                       errors="coerce")})
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    bounds = [A.wilson_interval(k, n) if np.isfinite(k) and np.isfinite(n) and n
              else (np.nan, np.nan)
              for k, n in zip(df["correct"], df["total"])]
    df["ci_lo"] = [b[0] for b in bounds]
    df["ci_hi"] = [b[1] for b in bounds]
    return df


def discover_results(data_root: Path, models, settings) -> pd.DataFrame:
    """Raw result rows for every selected (model, setting), for failure analysis.

    Goes per (model, setting) rather than one rglob over the root so that --models
    / --settings restrict identically to discover(), and so a missing directory is
    reported the same way.

    Callers pass the models/settings discover() ACTUALLY found, not the raw CLI
    args: a model can have results on disk while its summary CSV is still missing
    (stage 3 done, stage 4 not run), and then the two discovery paths disagree about
    which models exist — which crashed the figures that order series by the
    accuracy-side model list.
    """
    frames = []
    for md in _model_dirs(data_root, models):
        if not md.is_dir():
            continue
        for sd in _setting_dirs(md, settings):
            rdir = sd / A.RESULTS_SUBDIR
            if not rdir.is_dir():
                continue
            for f in sorted(rdir.glob("*.jsonl")):
                if f.name == "all_results.jsonl":
                    continue          # duplicates the per-category files
                rows = A._parse_result_lines(f.read_text(), md.name, sd.name, f.stem)
                if rows:
                    frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Plot helpers ─────────────────────────────────────────────────────────────

def _style(ax, ylabel="accuracy", ymax=1.08):
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, length=0)


_PALETTE_WARNED = set()


def _color_at(i: int):
    """Colour for the i-th entity, extending past the hand-picked palette.

    PALETTE's colour-blind safety is a property of those 8 specific colours, so it
    cannot be preserved by generating more. Rather than silently wrapping — which
    gives two models the identical colour in one legend — indices past the end walk a
    perceptually uniform map in golden-ratio steps, staying visually distinct, and the
    caller warns once that the guarantee has lapsed.
    """
    if i < len(PALETTE):
        return PALETTE[i]
    return matplotlib.colormaps["turbo"](((i - len(PALETTE) + 1) * 0.618034) % 1.0)


def _colors_for(names: list[str], order: list[str]) -> dict:
    """Stable colour per entity, by its position in the global ordered list.

    Unknown names are appended rather than raising: discovery can legitimately hand a
    figure an entity the caller's ordered list has not seen (a model evaluated but not
    yet summarized), and a KeyError there is a worse outcome than an extra colour.
    """
    order = list(order) + [n for n in names if n not in order]
    if len(order) > len(PALETTE) and id(PALETTE) not in _PALETTE_WARNED:
        _PALETTE_WARNED.add(id(PALETTE))
        print(f"[viz] NOTE: {len(order)} entities exceed the {len(PALETTE)}-colour "
              f"colour-blind-safe palette; the extras are distinct but no longer "
              f"guaranteed safe. Consider --models/--settings to compare fewer.")
    return {n: _color_at(order.index(n)) for n in names}


def _order_key(order: list[str]):
    """Sort key over a reference list that tolerates names missing from it."""
    return lambda n: (order.index(n) if n in order else len(order), str(n))


def grouped_bars(ax, x_labels, series_names, lookup, color_of, label_rot=90,
                 ci_lookup=None):
    """Grouped bars with a value label on every present bar.

    lookup: dict[(series, x_label)] -> accuracy (float). Missing keys draw NO bar
    (a gap), so 'missing' is visually distinct from a true 0 (which is labelled).
    ci_lookup: dict[(series, x_label)] -> (lo, hi) draws a Wilson whisker.
    """
    n = len(series_names)
    width = 0.8 / max(n, 1)
    xs = list(range(len(x_labels)))
    for si, s in enumerate(series_names):
        for xi, xl in zip(xs, x_labels):
            v = lookup.get((s, xl))
            if v is None:
                continue
            pos = xi + (si - (n - 1) / 2) * width
            ax.bar(pos, v, width=width * 0.9, color=color_of[s],
                   edgecolor="white", linewidth=0.5)
            top = v
            if ci_lookup:
                ci = ci_lookup.get((s, xl))
                if ci and all(np.isfinite(c) for c in ci):
                    ax.plot([pos, pos], list(ci), color=INK, lw=0.8,
                            solid_capstyle="butt", zorder=3)
                    top = max(top, ci[1])
            ax.text(pos, top + 0.012, f"{v * 100:.0f}", ha="center", va="bottom",
                    fontsize=6.5, rotation=label_rot, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8.5, color=INK)
    ax.set_xlim(-0.6, len(x_labels) - 0.4)
    handles = [Patch(facecolor=color_of[s], label=s) for s in series_names]
    return handles


def _ink_on(color) -> str:
    """Readable label colour for text drawn on `color`.

    The stack palettes span near-white to saturated, so a fixed white label is
    invisible on the light end. Uses relative luminance rather than a hand-kept
    list so a palette change cannot silently reintroduce white-on-white.
    """
    r, g, b = matplotlib.colors.to_rgb(color)[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if lum > 0.55 else "white"


def stacked_bars(ax, x_labels, stack_order, lookup, color_of, ylabel="share"):
    """Stacked bars summing to 1 per x, one segment per stack_order entry present.

    lookup: dict[(stack, x_label)] -> share. Segments below 4% carry no inline
    label (it would not fit) but remain in the legend.
    """
    xs = np.arange(len(x_labels), dtype=float)
    bottom = np.zeros(len(x_labels))
    used = []
    for name in stack_order:
        vals = np.array([lookup.get((name, xl), 0.0) for xl in x_labels], dtype=float)
        if not vals.any():
            continue
        used.append(name)
        ax.bar(xs, vals, bottom=bottom, width=0.66, color=color_of[name],
               edgecolor="white", linewidth=0.5)
        label_ink = _ink_on(color_of[name])
        for x, v, b in zip(xs, vals, bottom):
            if v >= 0.04:
                ax.text(x, b + v / 2, f"{v * 100:.0f}", ha="center", va="center",
                        fontsize=6.5, color=label_ink)
        bottom += vals
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8.5, color=INK)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, length=0)
    return [Patch(facecolor=color_of[n], label=n) for n in used]


def radar_axis(fig, nrows, ncols, idx, labels, rmax, rticks=None):
    """A polar axis with one spoke per label, 12 o'clock start, clockwise."""
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ax = fig.add_subplot(nrows, ncols, idx, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(ang)
    ax.set_xticklabels(labels, fontsize=8, color=INK)
    ax.set_ylim(0, rmax)
    ticks = rticks if rticks is not None else np.linspace(0, rmax, 4)[1:]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.2f}" for t in ticks], fontsize=6.5, color=MUTED)
    # Park the radial labels BETWEEN two spokes: on a spoke they sit directly on
    # top of the polygons and become unreadable exactly where the data is.
    ax.set_rlabel_position(180.0 / max(len(labels), 1))
    ax.tick_params(pad=2)
    ax.grid(color=GRID, linewidth=0.7)
    ax.spines["polar"].set_color(GRID)
    return ax, ang


def radar_series(ax, ang, values, color, label):
    """Close the polygon and draw it. NaN leaves a gap rather than a false 0."""
    v = np.append(np.asarray(values, dtype=float), values[0])
    a = np.append(ang, ang[0])
    ax.plot(a, v, color=color, lw=1.6, label=label, zorder=3)
    ax.fill(a, v, color=color, alpha=0.12, zorder=2)


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {path}")


# ── Data shaping ─────────────────────────────────────────────────────────────

def category_order(df: pd.DataFrame) -> list[str]:
    """Category display labels, relations first then attributes, OVERALL last."""
    cats = [c for c in df["category"].unique() if c not in NONCAT]
    cats = sorted(cats, key=lambda c: (0 if "rel" in c else 1, c))
    return [shorten(c) for c in cats] + [OVERALL_LABEL]


def _disp(category: str) -> str:
    return OVERALL_LABEL if category == OVERALL else shorten(category)


# ── Accuracy figures ─────────────────────────────────────────────────────────

def plot_scalar(df, out_dir, all_settings, all_models, *, metric, ylabel, title, fname):
    """Headline scalar (one value per model x setting): x=setting, one bar per model."""
    model_colors = _colors_for(all_models, all_models)
    settings = sorted(df["setting"].unique(), key=setting_key)
    models = sorted(df["model"].unique(), key=_order_key(all_models))
    sub = df[df["category"] == metric]
    if sub.empty:
        print(f"[viz] no '{metric}' rows; skipping {fname}")
        return
    lookup = {(r["model"], r["setting"]): r["accuracy"] for _, r in sub.iterrows()}
    ci = {(r["model"], r["setting"]): (r["ci_lo"], r["ci_hi"])
          for _, r in sub.iterrows()}

    fig, ax = plt.subplots(figsize=(max(6, len(settings) * 1.6), 4.6))
    handles = grouped_bars(ax, settings, models, lookup, model_colors, label_rot=0,
                           ci_lookup=ci)
    _style(ax, ylabel)
    ax.set_title(title, color=INK, fontsize=12)
    if len(models) >= 2:
        ax.legend(handles=handles, title="model", frameon=False, fontsize=9,
                  loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, out_dir / fname)


def plot_settings_within_model(df, out_dir, all_settings, all_models):
    """Per model: compare SETTINGS across categories (+ overall)."""
    set_colors = _colors_for(all_settings, all_settings)
    x_labels = category_order(df)
    for model in sorted(df["model"].unique(), key=_order_key(all_models)):
        md = df[df["model"] == model]
        settings = sorted(md["setting"].unique(), key=setting_key)
        lookup = {(r["setting"], _disp(r["category"])): r["accuracy"]
                  for _, r in md.iterrows()}
        ci = {(r["setting"], _disp(r["category"])): (r["ci_lo"], r["ci_hi"])
              for _, r in md.iterrows()}
        fig, ax = plt.subplots(figsize=(max(9, len(x_labels) * 1.25),
                                        4.8 + 0.15 * len(settings)))
        handles = grouped_bars(ax, x_labels, settings, lookup, set_colors,
                               ci_lookup=ci)
        _style(ax)
        ax.set_title(f"{model} — settings compared across categories "
                     f"(bars: Wilson 95% CI)", color=INK, fontsize=12)
        ax.legend(handles=handles, title="setting", frameon=False, fontsize=9,
                  loc="upper left", bbox_to_anchor=(1.01, 1.0))
        save(fig, out_dir / f"settings_within__{model}.png")


def plot_models_within_setting(df, out_dir, all_settings, all_models):
    """Per setting (EVERY one present): compare MODELS across categories (+ overall)."""
    model_colors = _colors_for(all_models, all_models)
    x_labels = category_order(df)
    for setting in sorted(df["setting"].unique(), key=setting_key):
        sd = df[df["setting"] == setting]
        models = sorted(sd["model"].unique(), key=_order_key(all_models))
        lookup = {(r["model"], _disp(r["category"])): r["accuracy"]
                  for _, r in sd.iterrows()}
        ci = {(r["model"], _disp(r["category"])): (r["ci_lo"], r["ci_hi"])
              for _, r in sd.iterrows()}
        fig, ax = plt.subplots(figsize=(max(9, len(x_labels) * 1.25),
                                        4.8 + 0.15 * len(models)))
        handles = grouped_bars(ax, x_labels, models, lookup, model_colors,
                               ci_lookup=ci)
        _style(ax)
        ax.set_title(f"{setting} — models compared across categories "
                     f"(bars: Wilson 95% CI)", color=INK, fontsize=12)
        if len(models) >= 2:
            ax.legend(handles=handles, title="model", frameon=False, fontsize=9,
                      loc="upper left", bbox_to_anchor=(1.01, 1.0))
        save(fig, out_dir / f"models_within__{setting}.png")


# ── Accuracy radars ──────────────────────────────────────────────────────────
#
# Every accuracy radar shares ONE radial range, so the figures are comparable side
# by side. A setting whose polygon collapses toward the centre really is near-zero
# on every category — that is the finding, not a plotting defect. Per-figure
# autoscaling was rejected precisely because it would make such a setting look as
# full as a good one.

def _radar_cats(df) -> list[str]:
    """Category spokes: the real categories only, OVERALL excluded.

    OVERALL is the mean of the other spokes, so including it would let one axis
    restate the whole polygon and inflate the enclosed area.
    """
    cats = [c for c in df["category"].unique() if c not in NONCAT]
    return sorted(cats, key=lambda c: (0 if "rel" in c else 1, c))


def plot_radar_settings_within_model(df, out_dir, all_settings, all_models, rmax):
    set_colors = _colors_for(all_settings, all_settings)
    cats = _radar_cats(df)
    if len(cats) < 3:
        print(f"[viz] only {len(cats)} categories; skipping accuracy radars")
        return
    labels = [shorten(c) for c in cats]
    for model in sorted(df["model"].unique(), key=_order_key(all_models)):
        md = df[df["model"] == model]
        settings = sorted(md["setting"].unique(), key=setting_key)
        fig = plt.figure(figsize=(6.4, 6.0))
        ax, ang = radar_axis(fig, 1, 1, 1, labels, rmax)
        for s in settings:
            sub = md[md["setting"] == s].set_index("category")["accuracy"]
            radar_series(ax, ang, [sub.get(c, np.nan) for c in cats],
                         set_colors[s], s)
        ax.set_title(f"{model} — accuracy by category, settings compared\n"
                     f"(radial axis 0–{rmax:.1f}, shared across figures)",
                     color=INK, fontsize=11, pad=24)
        ax.legend(frameon=False, fontsize=9, loc="upper left",
                  bbox_to_anchor=(1.04, 1.05), title="setting")
        save(fig, out_dir / f"radar__settings_within__{model}.png")


def plot_radar_models_within_setting(df, out_dir, all_settings, all_models, rmax):
    model_colors = _colors_for(all_models, all_models)
    cats = _radar_cats(df)
    if len(cats) < 3:
        return
    labels = [shorten(c) for c in cats]
    for setting in sorted(df["setting"].unique(), key=setting_key):
        sd = df[df["setting"] == setting]
        models = sorted(sd["model"].unique(), key=_order_key(all_models))
        fig = plt.figure(figsize=(6.4, 6.0))
        ax, ang = radar_axis(fig, 1, 1, 1, labels, rmax)
        for m in models:
            sub = sd[sd["model"] == m].set_index("category")["accuracy"]
            radar_series(ax, ang, [sub.get(c, np.nan) for c in cats],
                         model_colors[m], m)
        ax.set_title(f"{setting} — accuracy by category, models compared\n"
                     f"(radial axis 0–{rmax:.1f}, shared across figures)",
                     color=INK, fontsize=11, pad=24)
        if len(models) >= 2:
            ax.legend(frameon=False, fontsize=9, loc="upper left",
                      bbox_to_anchor=(1.04, 1.05), title="model")
        save(fig, out_dir / f"radar__models_within__{setting}.png")


# ── The confound: negation load at matched compositional load ────────────────

def plot_slot_count_matched(df, out_dir, all_models):
    """Accuracy vs NEGATION count, faceted by TOTAL statement count.

    In the original three settings, total statements and negated statements moved
    together (1/2/4 statements at 1/2/2 negations), so "negation is hard" could not be separated
    from "more statements is hard". A positive-only control at the same total statement count
    breaks the collinearity. The figure is therefore emitted only when some statement
    count has at least two settings differing in negation count — otherwise it
    would draw single points and imply an identification it does not have.
    """
    model_colors = _colors_for(all_models, all_models)
    sub = df[df["category"] == OVERALL].copy()
    if sub.empty:
        return
    sub[["slots", "negs"]] = sub["setting"].apply(
        lambda s: pd.Series(setting_slots(s)))
    sub = sub.dropna(subset=["slots", "negs"])
    if sub.empty:
        return
    groups = [(int(n), g) for n, g in sub.groupby("slots")
              if g["negs"].nunique() >= 2]
    if not groups:
        print("[viz] no matched slot-count pair (need a pos<P>_neg0 control); "
              "skipping slot_count__matched.png")
        return

    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 4.2),
                             squeeze=False, sharey=True)
    for k, (slots, g) in enumerate(groups):
        ax = axes[0][k]
        for m in sorted(g["model"].unique(), key=_order_key(all_models)):
            gm = g[g["model"] == m].sort_values("negs")
            ax.plot(gm["negs"], gm["accuracy"], marker="o", ms=5,
                    color=model_colors[m], lw=1.6, label=m)
        ax.set_title(f"{slots} slot{'s' if slots != 1 else ''} total",
                     color=INK, fontsize=10)
        ax.set_xlabel("negated statements", color=INK, fontsize=9)
        # Name the setting under its own tick rather than beside each point: within
        # a facet the negation count identifies the setting, so per-point labels
        # would just restate it once per model, overlapping.
        negs = sorted(g["negs"].unique().astype(int))
        ax.set_xticks(negs)
        ax.set_xticklabels(
            [f"{n}\n{g[g['negs'] == n]['setting'].iloc[0]}" for n in negs],
            fontsize=8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, None)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK, length=0)
        if k == 0:
            ax.set_ylabel("overall accuracy", color=INK, fontsize=10)
            ax.legend(frameon=False, fontsize=9, title="model")
    fig.suptitle("Negation load at matched compositional load", color=INK,
                 fontsize=12)
    save(fig, out_dir / "slot_count__matched.png")


# ── Failure analysis ─────────────────────────────────────────────────────────

def plot_failure_causes(causes, out_dir, all_models):
    """Per setting: what broke, as shares of that category's FAILING rows."""
    shares = A.failure_cause_shares(causes, group=("model", "setting", "category"))
    if shares.empty:
        return
    shares["cause"] = shares["cause"].astype(str)
    for setting, sd in shares.groupby("setting"):
        models = sorted(sd["model"].unique(), key=_order_key(all_models))
        cats = sorted(sd["category"].unique(), key=lambda c: (0 if "rel" in c else 1, c))
        labels = [shorten(c) for c in cats]
        fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.6),
                                 squeeze=False, sharey=True)
        handles = []
        for k, m in enumerate(models):
            ax = axes[0][k]
            md = sd[sd["model"] == m]
            lookup = {(r["cause"], shorten(r["category"])): r["share"]
                      for _, r in md.iterrows()}
            h = stacked_bars(ax, labels, A.CAUSE_ORDER, lookup, CAUSE_COLORS,
                             ylabel="share of failing rows" if k == 0 else "")
            handles = h if len(h) > len(handles) else handles
            n = int(causes[(causes["setting"] == setting)
                           & (causes["model"] == m)].shape[0])
            ax.set_title(f"{m}  (n={n} failing rows)", color=INK, fontsize=10)
        fig.suptitle(f"{setting} — causes of failure (mutually exclusive, sum to 100%)",
                     color=INK, fontsize=12)
        fig.legend(handles=handles, frameon=False, fontsize=9, title="cause",
                   loc="center left", bbox_to_anchor=(1.0, 0.5))
        save(fig, out_dir / f"failure_causes__{setting}.png")


def plot_systematicity(counts, out_dir, all_models):
    """Per setting: share of prompts at each k-of-n samples correct.

    0-of-n is a model that cannot do the task; the middle is a coin flip. Row
    accuracy and 'any sample correct' both average this distinction away.
    """
    sy = A.systematicity(counts, group=("model", "setting", "category"))
    if sy.empty:
        return
    # Label each stack by its OWN k/n. Assuming one global n breaks the moment a model
    # or setting is generated with a different N_SAMPLES: "3/4" would silently absorb
    # prompts that actually scored 3 of 2. Shade by the fraction correct, which is
    # comparable across denominators.
    pairs = sorted({(int(k), int(n)) for k, n in zip(sy["k"], sy["n"])},
                   key=lambda kn: (kn[1], kn[0]))
    stack_order = [f"{k}/{n}" for k, n in pairs]
    # Sequential ramp: dark = fails everything, light = passes everything.
    k_colors = {f"{k}/{n}": HEAT(0.15 + 0.8 * (1 - k / n if n else 0))
                for k, n in pairs}
    n_max = max(n for _, n in pairs)
    for setting, sd in sy.groupby("setting"):
        models = sorted(sd["model"].unique(), key=_order_key(all_models))
        cats = sorted(sd["category"].unique(), key=lambda c: (0 if "rel" in c else 1, c))
        labels = [shorten(c) for c in cats]
        fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.6),
                                 squeeze=False, sharey=True)
        handles = []
        for i, m in enumerate(models):
            ax = axes[0][i]
            md = sd[sd["model"] == m]
            lookup = {(f"{int(r['k'])}/{int(r['n'])}", shorten(r["category"])):
                      r["share"] for _, r in md.iterrows()}
            h = stacked_bars(ax, labels, stack_order, lookup, k_colors,
                             ylabel="share of prompts" if i == 0 else "")
            handles = h if len(h) > len(handles) else handles
            ax.set_title(m, color=INK, fontsize=10)
        fig.suptitle(f"{setting} — samples correct per prompt "
                     f"(0/{n_max} = systematic failure)", color=INK, fontsize=12)
        fig.legend(handles=handles, frameon=False, fontsize=9,
                   title="samples correct", loc="center left",
                   bbox_to_anchor=(1.0, 0.5))
        save(fig, out_dir / f"systematicity__{setting}.png")


def _stars(p) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def plot_paired_polarity(paired, out_dir, all_models):
    """Per setting: negated vs positive violation rate, PAIRED within one image.

    Only settings holding both polarities appear — pos<P>_neg0 and pos0_neg<N>
    have nothing to pair. Significance marks come from McNemar on the discordant
    rows, which is the correct test for a paired binary contrast.
    """
    if paired.empty:
        print("[viz] no setting holds both polarities; skipping paired_polarity")
        return
    for setting, sd in paired.groupby("setting"):
        models = sorted(sd["model"].unique(), key=_order_key(all_models))
        cats = sorted(sd["category"].unique(), key=lambda c: (0 if "rel" in c else 1, c))
        labels = [shorten(c) for c in cats]
        fig, axes = plt.subplots(1, len(models), figsize=(5.0 * len(models), 4.6),
                                 squeeze=False, sharey=True)
        for i, m in enumerate(models):
            ax = axes[0][i]
            md = sd[sd["model"] == m].set_index("category")
            xs = np.arange(len(cats), dtype=float)
            for j, pol in enumerate(("neg", "pos")):
                vals = [md["neg_rate"].get(c, np.nan) if pol == "neg"
                        else md["pos_rate"].get(c, np.nan) for c in cats]
                ax.bar(xs + (j - 0.5) * 0.34, vals, width=0.3,
                       color=POLARITY_COLORS[pol], edgecolor="white", linewidth=0.5,
                       label={"neg": "negated", "pos": "positive"}[pol])
                for x, v in zip(xs, vals):
                    if np.isfinite(v):
                        ax.text(x + (j - 0.5) * 0.34, v + 0.012, f"{v * 100:.0f}",
                                ha="center", va="bottom", fontsize=6.5, color=MUTED)
            for x, c in zip(xs, cats):
                if c in md.index:
                    ax.text(x, 1.0, _stars(md["p"].get(c, np.nan)), ha="center",
                            va="bottom", fontsize=7.5, color=INK)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5,
                               color=INK)
            ax.set_ylim(0, 1.12)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.yaxis.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors=INK, length=0)
            ax.set_title(m, color=INK, fontsize=10)
            if i == 0:
                ax.set_ylabel("slot violation rate", color=INK, fontsize=10)
                ax.legend(frameon=False, fontsize=9)
        fig.suptitle(f"{setting} — negated vs positive statements, paired within the "
                     f"same image (McNemar: *** p<.001, ** p<.01, * p<.05)",
                     color=INK, fontsize=11)
        save(fig, out_dir / f"paired_polarity__{setting}.png")


def plot_keys(slots, out_dir, all_models):
    """Per (category, setting): which value / predicate is violated most.

    A radar over the category's keys when it has >= 3 of them; paired horizontal
    bars when it has 2 (depth: behind / in front of, proximity: near / far from),
    because a two-spoke radar is a line segment and shows nothing. Radius is the
    violation rate, so BIGGER IS WORSE here — the opposite of the accuracy radars,
    which is why the title says so.
    """
    if slots.empty:
        return
    for (category, setting), sub in slots.groupby(["category", "setting"]):
        rates = A.key_failure_rates(sub, group=("model", "key", "polarity"))
        if rates.empty:
            continue
        rates = rates.reset_index()
        keys = sorted(sub["key"].dropna().unique())
        pols = [p for p in ("neg", "pos") if p in set(sub["polarity"])]
        models = sorted(sub["model"].unique(), key=_order_key(all_models))
        cat_short = shorten(str(category))
        note = ("" if len(pols) == 2 else
                f"\n(only {pols[0]} slots exist in {setting})")

        if len(keys) >= 3:
            fig = plt.figure(figsize=(5.6 * len(models) + 1.4, 5.8))
            for i, m in enumerate(models, start=1):
                ax, ang = radar_axis(fig, 1, len(models), i, keys, 1.0,
                                     rticks=[0.25, 0.5, 0.75, 1.0])
                for pol in pols:
                    r = rates[(rates["model"] == m) & (rates["polarity"] == pol)] \
                        .set_index("key")["violated_rate"]
                    radar_series(ax, ang, [r.get(k, np.nan) for k in keys],
                                 POLARITY_COLORS[pol],
                                 {"neg": "negated", "pos": "positive"}[pol])
                ax.set_title(m, color=INK, fontsize=10, pad=18)
                if i == len(models):
                    ax.legend(frameon=False, fontsize=9, loc="upper left",
                              bbox_to_anchor=(1.06, 1.05), title="polarity")
            fig.suptitle(f"{cat_short} / {setting} — violation rate per key "
                         f"(BIGGER = WORSE){note}", color=INK, fontsize=11)
        else:
            fig, axes = plt.subplots(1, len(models),
                                     figsize=(4.4 * len(models), 2.6),
                                     squeeze=False, sharex=True)
            ys = np.arange(len(keys), dtype=float)
            for i, m in enumerate(models):
                ax = axes[0][i]
                for j, pol in enumerate(pols):
                    r = rates[(rates["model"] == m) & (rates["polarity"] == pol)] \
                        .set_index("key")["violated_rate"]
                    vals = [r.get(k, np.nan) for k in keys]
                    off = (j - (len(pols) - 1) / 2) * 0.34
                    ax.barh(ys + off, vals, height=0.3,
                            color=POLARITY_COLORS[pol], edgecolor="white",
                            linewidth=0.5,
                            label={"neg": "negated", "pos": "positive"}[pol])
                    for y, v in zip(ys, vals):
                        if np.isfinite(v):
                            ax.text(v + 0.012, y + off, f"{v * 100:.0f}",
                                    va="center", fontsize=6.5, color=MUTED)
                ax.set_yticks(ys)
                ax.set_yticklabels(keys if i == 0 else [], fontsize=8.5, color=INK)
                ax.set_xlim(0, 1.12)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.xaxis.grid(True, color=GRID, linewidth=0.8)
                ax.set_axisbelow(True)
                for s in ("top", "right", "left"):
                    ax.spines[s].set_visible(False)
                ax.tick_params(colors=INK, length=0)
                ax.set_title(m, color=INK, fontsize=10)
                if i == 0:
                    ax.legend(frameon=False, fontsize=8)
            fig.suptitle(f"{cat_short} / {setting} — violation rate per key "
                         f"(BIGGER = WORSE; 2 keys, so bars not a radar){note}",
                         color=INK, fontsize=10)
        save(fig, out_dir / f"keys__{cat_short}__{setting}.png")


def plot_failure_report(buckets, out_dir, all_models, all_settings):
    """THE report figure: one grid, rows = models, cols = settings, a stacked bar per
    category showing what kind of thing went wrong.

    Statement-level, so the four buckets need no tie-break between negated and positive —
    at row level 36% of pos2_neg2 failures violate both and would have to be
    arbitrarily assigned. Shares sum to 1 within each category so compositions are
    comparable across categories and panels; the failed-statement count sits above each
    bar so volume is not lost to normalisation.

    Categories where a dead zone cannot occur are daggered, since a reader cannot
    otherwise distinguish "impossible here" from "none measured".
    """
    shares = A.slot_bucket_shares(buckets, group=("model", "setting", "category"))
    if shares.empty:
        print("[viz] no failed statements; skipping failure_report.png")
        return
    shares["bucket"] = shares["bucket"].astype(str)

    models = sorted(shares["model"].unique(), key=_order_key(all_models))
    settings = sorted(shares["setting"].unique(), key=setting_key)
    cats = sorted(shares["category"].unique(),
                  key=lambda c: (0 if "rel" in c else 1, c))
    labels = [shorten(c) + ("" if A.dead_zone_capable(c) else DAGGER) for c in cats]

    # Not every model is evaluated on every setting, so the x labels cannot simply go
    # on the bottom row — that panel may be empty. Put them on the lowest panel in
    # each column that actually has data.
    populated = {(r["model"], r["setting"]) for _, r in shares.iterrows()}
    label_row = {j: max((i for i, m in enumerate(models) if (m, s) in populated),
                        default=None)
                 for j, s in enumerate(settings)}

    # Panels shrink as the grid fills so the figure stays a readable page rather than
    # growing without bound — 4x8 is already ~3800px wide. Below the floor the labels
    # stop being legible at all, so say so instead of emitting an unusable PNG.
    panel_w = max(1.9, min(2.9, 26.0 / max(len(settings), 1)))
    panel_h = max(2.1, min(3.0, 16.0 / max(len(models), 1)))
    if panel_w <= 1.9 or panel_h <= 2.1:
        print(f"[viz] NOTE: failure_report is {len(models)}x{len(settings)} panels and "
              f"is getting cramped; narrow it with --models / --settings.")
    fig, axes = plt.subplots(len(models), len(settings), squeeze=False, sharey=True,
                             figsize=(max(3.0, panel_w * len(settings)),
                                      panel_h * len(models) + 1.6))
    handles = []
    for i, m in enumerate(models):
        for j, s in enumerate(settings):
            ax = axes[i][j]
            if i == 0:
                ax.set_title(s, color=INK, fontsize=10)
            if j == 0:
                ax.text(-0.42, 0.5, m, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", color=INK, fontsize=11)
            cell = shares[(shares["model"] == m) & (shares["setting"] == s)]
            if cell.empty:
                # An empty axis frame would read as "measured, nothing failed".
                ax.text(0.5, 0.5, "not evaluated", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8.5, color=MUTED)
                ax.set_axis_off()
                continue
            lookup = {(r["bucket"], shorten(r["category"])
                       + ("" if A.dead_zone_capable(r["category"]) else DAGGER)):
                      r["share"] for _, r in cell.iterrows()}
            h = stacked_bars(ax, labels, A.BUCKET_ORDER, lookup, BUCKET_COLORS,
                             ylabel="share of failed statements" if j == 0 else "")
            handles = h if len(h) > len(handles) else handles
            # Failed-statement count per category, so a 100%-negated bar built on 12 slots
            # cannot be mistaken for one built on 400.
            failed = {shorten(r["category"])
                      + ("" if A.dead_zone_capable(r["category"]) else DAGGER):
                      int(r["failed"]) for _, r in cell.iterrows()}
            for x, lab in enumerate(labels):
                if lab in failed:
                    ax.text(x, 1.015, str(failed[lab]), ha="center", va="bottom",
                            fontsize=6, color=MUTED)
            ax.set_ylim(0, 1.10)
            if i != label_row[j]:
                ax.set_xticklabels([])

    fig.suptitle("Where the failures come from — one bucket per failed statement",
                 color=INK, fontsize=13)
    fig.legend(handles=handles, frameon=False, fontsize=9, title="failure kind",
               loc="center left", bbox_to_anchor=(1.0, 0.5))

    note = [f"{DAGGER} dead zone cannot occur here: "
            "attributes always yield a CLIP confusion set; "
            "proximity has a single boundary.",
            "Numbers above bars = failed statements in that category. "
            "Shares are of failed statements, so innocent statements "
            "(their row failed on a different statement) are excluded."]
    exposure = A.free_pass_exposure(buckets, group=())
    if not exposure.empty:
        r = exposure.iloc[0]
        note.append(
            f"Dead zone is observable on POSITIVE edges only: on a negated edge an "
            f"empty verdict makes the negation pass. {int(r['held'])} of "
            f"{int(r['negated_statements'])} negated statements "
            f"({r['held_share']:.1%}) held in the "
            f"three dead-zone-capable categories, an unknown share of them unearned — "
            f"so negated success there is an UPPER BOUND.")
    fig.text(0.0, -0.02, "\n".join(note), fontsize=7.5, color=MUTED,
             ha="left", va="top", wrap=True)
    save(fig, out_dir / "failure_report.png")


def plot_relation_confusion(confusion, out_dir, all_models):
    """Per relation category: expected predicate vs the predicate that held.

    Aggregated over settings, since the question is about the model and the
    evaluator rather than the prompt shape. 'no relation' is a column of its own:
    for the dead-zone categories it means the evaluator abstained rather than
    disagreed, which is a different failure from naming the wrong predicate.
    """
    if confusion.empty:
        return
    for category, cd in confusion.groupby("category"):
        models = sorted(cd["model"].unique(), key=_order_key(all_models))
        cat_name = str(category).replace("neg_rel_", "")
        preds = [p for p in vocab.RELATION_CATEGORIES[cat_name].predicates
                 if p in set(cd["expected"])] if cat_name in vocab.RELATION_CATEGORIES \
            else sorted(cd["expected"].unique())
        obs_order = [p for p in vocab.all_predicates() if p in set(cd["observed"])]
        if "no relation" in set(cd["observed"]):
            obs_order.append("no relation")
        if not preds or not obs_order:
            continue
        fig, axes = plt.subplots(1, len(models),
                                 figsize=(1.05 * len(obs_order) * len(models) + 2.6,
                                          0.62 * len(preds) + 2.4),
                                 squeeze=False)
        for i, m in enumerate(models):
            ax = axes[0][i]
            md = cd[cd["model"] == m]
            mat = (md.pivot_table(index="expected", columns="observed",
                                  values="count", aggfunc="sum", fill_value=0)
                   .reindex(index=preds, columns=obs_order, fill_value=0))
            row_tot = mat.sum(axis=1).replace(0, np.nan)
            frac = mat.div(row_tot, axis=0).fillna(0.0)
            ax.imshow(frac.values, cmap=HEAT, vmin=0, vmax=1, aspect="auto")
            for y in range(mat.shape[0]):
                for x in range(mat.shape[1]):
                    v = int(mat.values[y, x])
                    ax.text(x, y, str(v), ha="center", va="center", fontsize=7,
                            color="white" if frac.values[y, x] > 0.55 else INK)
            ax.set_xticks(range(len(obs_order)))
            ax.set_xticklabels(obs_order, rotation=30, ha="right", fontsize=8,
                               color=INK)
            ax.set_yticks(range(len(preds)))
            ax.set_yticklabels(preds if i == 0 else [], fontsize=8, color=INK)
            ax.set_title(m, color=INK, fontsize=10)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_ylabel("expected", color=INK, fontsize=9)
            ax.set_xlabel("actually held", color=INK, fontsize=9)
        fig.suptitle(f"{cat_name} — expected vs observed predicate on FAILED "
                     f"positive edges (counts; shade = row share)",
                     color=INK, fontsize=11)
        save(fig, out_dir / f"confusion__{cat_name}.png")


# ── What to emit ─────────────────────────────────────────────────────────────
#
# One registry instead of a flag per figure. Each entry is (needs_raw_jsonl, fn),
# where fn takes the context dict below. `needs_raw` decides whether the slow
# JSONL pass runs at all, so --figures overall costs nothing extra.
#
# Emitting everything produced ~70 PNGs per run, which is not reviewable; the
# default is the single report grid. Slicing by model/setting is still done with
# --models / --settings, so nothing is lost by dropping the old multi-view sweep.

FIGURES = {
    "report":        (True,  lambda c: plot_failure_report(
                                 c["buckets"], c["out"], c["models"], c["settings"])),
    "overall":       (False, lambda c: (
        plot_scalar(c["df"], c["out"], c["settings"], c["models"], metric=OVERALL,
                    ylabel="overall accuracy",
                    title="Overall accuracy — models by setting",
                    fname="overall__models_by_setting.png"),
        plot_scalar(c["df"], c["out"], c["settings"], c["models"], metric=PROMPTS,
                    ylabel="prompt accuracy (any sample correct)",
                    title="Prompt accuracy — models by setting",
                    fname="prompt_accuracy__models_by_setting.png"))),
    "bars":          (False, lambda c: (
        plot_settings_within_model(c["df"], c["out"], c["settings"], c["models"]),
        plot_models_within_setting(c["df"], c["out"], c["settings"], c["models"]))),
    "radar":         (False, lambda c: (
        plot_radar_settings_within_model(c["df"], c["out"], c["settings"],
                                         c["models"], c["rmax"]),
        plot_radar_models_within_setting(c["df"], c["out"], c["settings"],
                                         c["models"], c["rmax"]))),
    "matched":       (False, lambda c: plot_slot_count_matched(
                                 c["df"], c["out"], c["models"])),
    "causes":        (True,  lambda c: plot_failure_causes(
                                 c["causes"], c["out"], c["models"])),
    "systematicity": (True,  lambda c: plot_systematicity(
                                 c["counts"], c["out"], c["models"])),
    "paired":        (True,  lambda c: plot_paired_polarity(
                                 c["paired"], c["out"], c["models"])),
    "keys":          (True,  lambda c: plot_keys(c["slots"], c["out"], c["models"])),
    "confusion":     (True,  lambda c: plot_relation_confusion(
                                 c["confusion"], c["out"], c["models"])),
}

# Named bundles, so the common cases are one short word.
FIGURE_SETS = {
    "none": [],
    "report": ["report"],
    "core": ["report", "overall", "matched"],
    "accuracy": ["overall", "bars", "radar", "matched"],
    "failure": ["report", "causes", "systematicity", "paired", "keys", "confusion"],
    "all": list(FIGURES),
}


def resolve_figures(spec: str) -> list[str]:
    """'core' | 'report,confusion' | 'none' -> an ordered, de-duplicated list."""
    names = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if token in FIGURE_SETS:
            names += FIGURE_SETS[token]
        elif token in FIGURES:
            names.append(token)
        else:
            raise SystemExit(
                f"[viz] unknown --figures value {token!r}. "
                f"Figures: {', '.join(FIGURES)}. Sets: {', '.join(FIGURE_SETS)}.")
    return list(dict.fromkeys(names))       # de-dup, keep order


def write_tables(ctx, out_dir: Path, figures: list[str]) -> None:
    """Markdown reports: one per setting, one comparison, one index."""
    acc, buckets, counts, paired = (ctx["df"], ctx["buckets"], ctx["counts"],
                                    ctx["paired"])
    models, settings = ctx["models"], ctx["settings"]
    for s in settings:
        doc = T.setting_report(
            s,
            acc[acc["setting"] == s],
            buckets[buckets["setting"] == s] if not buckets.empty else buckets,
            counts[counts["setting"] == s] if not counts.empty else counts,
            paired[paired["setting"] == s] if not paired.empty else paired,
            models)
        (out_dir / f"report__{s}.md").write_text(doc)
    (out_dir / "report__comparison.md").write_text(
        T.comparison_report(acc, buckets, counts, models, settings))
    (out_dir / "README.md").write_text(T.index_report(settings, models, figures))
    print(f"[viz] wrote {len(settings) + 2} Markdown tables -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Plot GenEval negation results across settings/models.")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "outputs/images"))
    ap.add_argument("--models", nargs="*", default=None, help="subset of models (default: all found)")
    ap.add_argument("--settings", nargs="*", default=None, help="subset of pos<P>_neg<N> (default: all found)")
    ap.add_argument("--out-dir", default=os.environ.get("REPORT_DIR"),
                    help="base report dir (default: $REPORT_DIR, else <data-root>/plots)")
    ap.add_argument("--subdir", default=None,
                    help="subfolder under the plots dir (default: derived from the plotted "
                         "models and settings, e.g. 'flux2+sd35__pos0_neg1+pos1_neg0')")
    ap.add_argument("--figures", default="report",
                    help="comma-separated figures or sets (default: report). "
                         f"Figures: {', '.join(FIGURES)}. "
                         f"Sets: {', '.join(FIGURE_SETS)}.")
    ap.add_argument("--no-tables", action="store_true",
                    help="skip the Markdown reports (they are written by default)")
    args = ap.parse_args()
    figures = resolve_figures(args.figures)

    data_root = Path(args.data_root).expanduser()
    base_dir = Path(args.out_dir).expanduser() if args.out_dir else data_root / "plots"

    df = discover(data_root, args.models, args.settings)
    if df.empty:
        raise SystemExit(f"[viz] no summary CSVs found under {data_root} "
                         f"(run stage 3+4 first). models={args.models} settings={args.settings}")

    plotted_settings = sorted(df["setting"].unique(), key=setting_key)
    plotted_models = sorted(df["model"].unique())
    subdir = args.subdir or f"{'+'.join(plotted_models)}__{'+'.join(plotted_settings)}"
    out_dir = base_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    df.sort_values(["model", "setting", "category"]).to_csv(out_dir / "summary_long.csv", index=False)
    print(f"[viz] {len(df)} rows | models={plotted_models} settings={plotted_settings}")

    # The raw JSONL pass is the slow one, so only take it when a selected figure
    # or the Markdown tables actually need it.
    need_raw = (not args.no_tables) or any(FIGURES[f][0] for f in figures)
    # One shared radial range for every accuracy radar, from the plotted data, so the
    # radars stay comparable side by side.
    cat_acc = df[~df["category"].isin(NONCAT)]["accuracy"]
    ctx = {"df": df, "out": out_dir, "models": plotted_models,
           "settings": plotted_settings, "rmax": nice_ceil(
               float(cat_acc.max()) if len(cat_acc) else 1.0)}
    empty = pd.DataFrame()
    ctx.update(res=empty, slots=empty, causes=empty, counts=empty, paired=empty,
               confusion=empty, buckets=empty)

    if need_raw:
        res = discover_results(data_root, plotted_models, plotted_settings)
        if res.empty:
            print("[viz] no raw result rows found; skipping the failure analysis "
                  "and the failure sections of the tables")
        else:
            slots = A.explode_result_slots(res)
            buckets = A.slot_buckets(res, slots)
            ctx.update(res=res, slots=slots, buckets=buckets,
                       causes=A.failure_causes(res, slots),
                       counts=A.prompt_success_counts(res),
                       paired=A.paired_polarity(slots),
                       confusion=A.relation_confusion(res))
            n_failed = int((buckets["bucket"] != A.INNOCENT).sum())
            print(f"[viz] {len(res)} result rows -> {len(slots)} slots, "
                  f"{n_failed} failed ({len(slots) - n_failed} innocent)")
            # The two data paths can disagree: raw results exist as soon as stage 3
            # finishes, the summary CSV only after stage 4. Left unsaid, the report
            # shows a populated failure table beside an empty accuracy row, which
            # reads as a contradiction rather than a missing pipeline step.
            have_acc = {(r["model"], r["setting"]) for _, r in df.iterrows()}
            have_raw = {(r["model"], r["setting"]) for _, r in res.iterrows()}
            missing = sorted(have_raw - have_acc)
            if missing:
                pairs = ", ".join(f"{m}/{s}" for m, s in missing[:6])
                more = f" (+{len(missing) - 6} more)" if len(missing) > 6 else ""
                print(f"[viz] NOTE: evaluated but not summarized — {pairs}{more}. "
                      f"Their failure tables are populated but accuracy is blank; "
                      f"run stage 4 (scripts/4_summarize.sh) to fill it in.")
            for name, frame in (
                    ("failure_statements.csv", A.slot_bucket_shares(
                        buckets, group=("model", "setting", "category"))),
                    ("free_pass_exposure.csv", A.free_pass_exposure(
                        buckets, group=("model", "setting", "category"))),
                    ("statements_long.csv", slots),
                    ("failure_causes.csv", A.failure_cause_shares(
                        ctx["causes"], group=("model", "setting", "category"))),
                    ("systematicity.csv", A.systematicity(
                        ctx["counts"], group=("model", "setting", "category"))),
                    ("paired_polarity.csv", ctx["paired"]),
                    ("relation_confusion.csv", ctx["confusion"])):
                frame.to_csv(out_dir / name, index=False)

    for name in figures:
        needs_raw, fn = FIGURES[name]
        if needs_raw and ctx["res"].empty:
            print(f"[viz] skipping figure {name!r}: needs the raw results")
            continue
        fn(ctx)

    if not args.no_tables:
        write_tables(ctx, out_dir, figures)
    print(f"[viz] done -> {out_dir}  "
          f"(figures: {', '.join(figures) if figures else 'none'})")


if __name__ == "__main__":
    main()
