"""Every path the human-validation code reads or writes, in one place.

Values come from the environment (set by ../config.sh) with repository-relative
fallbacks, so `python -m human_validation.<module>` works from the repo root with or
without sourcing config.sh first.

    DATA_ROOT          <model>/<setting>/<category>/<idx>/samples/*.png and
                       <model>/<setting>/<RESULTS_SUBDIR>/<category>.jsonl
                       (stages 2-3 of the benchmark pipeline)
    HUMAN_ANNOTATIONS  one JSON row per (annotator, image); see README.md for the schema
    HUMAN_EVAL_DIR     metric scores, manifests and the comparison report
    HUMAN_EVAL_INPUT   the manifest of annotated images that every metric is scored on
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("NEGT2I_ROOT", Path(__file__).resolve().parents[1]))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", REPO_ROOT / "outputs"))

DATA_ROOT = Path(os.environ.get("DATA_ROOT", OUTPUT_ROOT / "images"))
RESULTS_SUBDIR = os.environ.get("RESULTS_SUBDIR", "results")
HUMAN_EVAL_DIR = Path(os.environ.get("HUMAN_EVAL_DIR", OUTPUT_ROOT / "human_validation"))
HUMAN_ANNOTATIONS = Path(os.environ.get(
    "HUMAN_ANNOTATIONS", REPO_ROOT / "human_annotations" / "annotations.jsonl"))
HUMAN_EVAL_INPUT = Path(os.environ.get("HUMAN_EVAL_INPUT", HUMAN_EVAL_DIR / "input.jsonl"))
# Full (unannotated) sweep over every generated image.
FULL_SWEEP_DIR = Path(os.environ.get("FULL_SWEEP_DIR", HUMAN_EVAL_DIR / "full_sweep"))

# Top-level DATA_ROOT entries that are not model directories.
NON_MODEL_DIRS = {"prompts", "plots", "reports"}


def result_files(data_root: Path = DATA_ROOT):
    """Every per-category verdict file: <model>/<setting>/<RESULTS_SUBDIR>/<category>.jsonl."""
    for path in sorted(data_root.glob(f"*/*/{RESULTS_SUBDIR}/*.jsonl")):
        if path.parents[2].name in NON_MODEL_DIRS:
            continue
        if path.stem in {"all_results", "all_results_summary"}:
            continue
        yield path
