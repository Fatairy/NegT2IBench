#!/usr/bin/env bash
# NegT2IBench — the ONE place every path, environment and benchmark parameter is set.
#
# Every script in scripts/ and jobs/ sources this file first, so editing a value here
# changes it everywhere. Each line is `VAR="${VAR:-default}"`: a value you export
# before running a script wins over the default written here.
#
#   source config.sh          # also usable interactively; prints nothing
#   NEGT2I_VERBOSE=1 source config.sh   # prints the resolved configuration
#
# Output layout (all under OUTPUT_ROOT):
#
#   outputs/
#     images/<model>/<setting>/<category>/<idx>/metadata.jsonl      prompt row
#     images/<model>/<setting>/<category>/<idx>/samples/0000N.png   generated images
#     images/<model>/<setting>/results/<category>.jsonl             per-image verdicts
#     images/<model>/<setting>/results/all_results.jsonl            per-setting merge
#     images/<model>/<setting>/results/all_results_summary.csv      per-setting scores
#     images/<model>/results/all_results.jsonl                      all settings merged
#     reports/<name>/                                               Markdown/CSV/PNG report
#     human_validation/                                             metric scores + comparison
#     logs/                                                         SLURM / local logs
#
# <setting> is pos<P>_neg<N> (P positive, N negated statements per prompt).
# <category> is neg_attr_{color,material} or neg_rel_{directional,proximity,comparative,depth}.

# Repository root = the directory holding this file (works when sourced from anywhere).
NEGT2I_ROOT="${NEGT2I_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export NEGT2I_ROOT

# ── Paths ────────────────────────────────────────────────────────────────────
export PROMPT_DIR="${PROMPT_DIR:-$NEGT2I_ROOT/negt2ibench_prompts}"   # released benchmark prompts
export OUTPUT_ROOT="${OUTPUT_ROOT:-$NEGT2I_ROOT/outputs}"             # everything the pipeline writes
export DATA_ROOT="${DATA_ROOT:-$OUTPUT_ROOT/images}"                  # generated images + verdicts
export RESULTS_SUBDIR="${RESULTS_SUBDIR:-results}"                    # per-setting verdict folder name
export REPORT_DIR="${REPORT_DIR:-$OUTPUT_ROOT/reports}"               # stage-5 reports (one subfolder per run)
export HUMAN_EVAL_DIR="${HUMAN_EVAL_DIR:-$OUTPUT_ROOT/human_validation}"
export LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"

# Model caches. Hugging Face weights (T2I models, RF-DETR-XL, SigLIP2, SAM 2,
# Depth-Anything) total several hundred GB if you run every model: point CACHE_DIR at
# a large disk.
export CACHE_DIR="${CACHE_DIR:-$NEGT2I_ROOT/cache}"
export HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_DIR/torch}"
export CLIP_DOWNLOAD_ROOT="${CLIP_DOWNLOAD_ROOT:-$CACHE_DIR/clip}"    # OpenAI clip.load() (NegationCLIP)
export RF_HOME="${RF_HOME:-$CACHE_DIR/rfdetr}"                        # RF-DETR-XL checkpoint

# ── Python environment ───────────────────────────────────────────────────────
# ENV_MANAGER: conda | micromamba | none. "none" uses whatever python is on PATH
# (e.g. a venv you activated yourself). Default: auto-detect micromamba, then conda.
export ENV_NAME="${ENV_NAME:-negt2ibench}"
export ENV_MANAGER="${ENV_MANAGER:-auto}"
# RF-DETR-XL needs a newer torch than the base env, so it lives in its own
# `pip install --target` directory that is prepended to PYTHONPATH for evaluation only
# (scripts/0_setup_env.sh creates it). Set to "" if rfdetr is installed in the env itself.
export RFDETR_OVERLAY="${RFDETR_OVERLAY-$NEGT2I_ROOT/envs/rfdetr_overlay}"

# ── Generation (stage 2) ─────────────────────────────────────────────────────
export MODEL="${MODEL:-zimage}"          # registry key, see README "Text-to-image models"
export N_SAMPLES="${N_SAMPLES:-4}"       # images per prompt (paper: 4)
export SEED="${SEED:-42}"                # base seed; sample i uses SEED+i (paper: 42)
export CLIP_WEIGHTS="${CLIP_WEIGHTS:-$CACHE_DIR/negationclip_ViT-B32.pth}"  # negationclip models only

# ── Evaluation (stage 3) ─────────────────────────────────────────────────────
# The evaluator has no runtime options: every parameter of the benchmark is a
# constant in evaluation/evaluate_images.py, evaluation/relation_eval.py and
# evaluation/sam_refine.py, and each run writes them to results/_config.json.
#   RF-DETR-XL (threshold 0.03, counted instances >= 0.5, exactly one per object)
#   SigLIP2 SO400M/14-378 colour/material (confusion buffer 0.5 colour, 0 material)
#   SAM 2.1-small relation masks, Depth-Anything-V2-Small facing-surface depth

# Determinism (the evaluator also enforces these itself; see evaluation/repro.py).
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# ── Helpers ──────────────────────────────────────────────────────────────────
negt2i_activate() {
    # Activate ENV_NAME with the chosen manager. `negt2i_activate eval` also layers the
    # RF-DETR overlay onto PYTHONPATH. Safe to call more than once.
    local mgr="$ENV_MANAGER"
    if [ "$mgr" = auto ]; then
        if command -v micromamba >/dev/null 2>&1; then mgr=micromamba
        elif command -v conda >/dev/null 2>&1; then mgr=conda
        else mgr=none; fi
    fi
    case "$mgr" in
        micromamba) eval "$(micromamba shell hook --shell bash)"; micromamba activate "$ENV_NAME" ;;
        conda)      eval "$(conda shell.bash hook)"; conda activate "$ENV_NAME" ;;
        none)       : ;;
        *) echo "[config] unknown ENV_MANAGER=$mgr (conda|micromamba|none|auto)" >&2; return 1 ;;
    esac
    if [ "${1:-}" = eval ] && [ -n "$RFDETR_OVERLAY" ]; then
        [ -d "$RFDETR_OVERLAY" ] || echo "[config] WARNING: RFDETR_OVERLAY=$RFDETR_OVERLAY does not exist (run scripts/0_setup_env.sh)" >&2
        case ":${PYTHONPATH:-}:" in *":$RFDETR_OVERLAY:"*) ;;
            *) export PYTHONPATH="$RFDETR_OVERLAY${PYTHONPATH:+:$PYTHONPATH}" ;; esac
    fi
    export PYTHONPATH="$NEGT2I_ROOT${PYTHONPATH:+:$PYTHONPATH}"
}

negt2i_show_config() {
    local v
    for v in NEGT2I_ROOT PROMPT_DIR OUTPUT_ROOT DATA_ROOT RESULTS_SUBDIR REPORT_DIR \
             HUMAN_EVAL_DIR LOG_DIR CACHE_DIR HF_HOME ENV_MANAGER ENV_NAME RFDETR_OVERLAY \
             MODEL N_SAMPLES SEED; do
        printf '  %-16s %s\n' "$v" "${!v}"
    done
}

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$HF_HOME" "$TORCH_HOME" "$CLIP_DOWNLOAD_ROOT" "$RF_HOME"
[ "${NEGT2I_VERBOSE:-0}" = 1 ] && { echo "[config] NegT2IBench configuration:"; negt2i_show_config; }
true
