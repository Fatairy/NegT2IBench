#!/usr/bin/env bash
# Human validation: how well do NegT2IBench and other metrics agree with human labels?
#
#   1. manifest   HUMAN_EVAL_INPUT  <- every annotated image that has an evaluator verdict
#   2. metrics    $HUMAN_EVAL_DIR/<metric>.jsonl / <metric>_both.jsonl   (optional, GPU)
#   3. compare    $HUMAN_EVAL_DIR/comparison.txt  (+ neggenbench_both.jsonl, human_scores.jsonl)
#
# Needs: HUMAN_ANNOTATIONS (see README "Human annotations"), and stages 2-3 already run
# for the annotated models so DATA_ROOT holds their images and verdicts.
#
# Env vars (paths from config.sh):
#   CLIP_METRICS   e.g. "clipscore blipscore negclip imagereward pickscore hpsv2"
#   VQA_METRICS    e.g. "vqascore mplug qwen25vl internvl"
#   CLIP_PYTHON    python of the negt2i-clip-metrics env      (default: python)
#   VQA_PYTHON     python for qwen25vl / internvl            (default: python)
#   VQASCORE_PYTHON / MPLUG_PYTHON   python for vqascore / mplug (default: python)
#   MPLUG_ROOT     checkout of the official mPLUG repo       (mplug only)
#   OVERWRITE=1    rescore metrics whose output already exists
#
# Examples:
#   bash scripts/human_validation/run_human_validation.sh            # NegT2IBench vs humans
#   CLIP_METRICS=clipscore CLIP_PYTHON=envs/negt2i-clip-metrics/bin/python \
#       bash scripts/human_validation/run_human_validation.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
cd "$REPO"

[ -f "$HUMAN_ANNOTATIONS" ] || { echo "[hv] HUMAN_ANNOTATIONS not found: $HUMAN_ANNOTATIONS" >&2; exit 1; }
mkdir -p "$HUMAN_EVAL_DIR"
export HUMAN_EVAL_INPUT="${HUMAN_EVAL_INPUT:-$HUMAN_EVAL_DIR/input.jsonl}"

echo "[hv] 1/3 manifest -> $HUMAN_EVAL_INPUT"
python -m human_validation.build_geneval_manifest \
    --only-annotated "$HUMAN_ANNOTATIONS" --output "$HUMAN_EVAL_INPUT" --allow-missing-images

ow=(); [ "${OVERWRITE:-0}" = 1 ] && ow=(--overwrite) || ow=(--resume)
for m in ${CLIP_METRICS:-}; do
    echo "[hv] 2/3 $m"
    "${CLIP_PYTHON:-python}" -m human_validation.score "$HUMAN_EVAL_INPUT" "$HUMAN_EVAL_DIR/$m.jsonl" \
        --metric "$m" --image-root "$DATA_ROOT" --cache-dir "$HF_HUB_CACHE" "${ow[@]}"
done
for m in ${VQA_METRICS:-}; do
    echo "[hv] 2/3 $m (whole caption + per element)"
    case "$m" in
        vqascore) py="${VQASCORE_PYTHON:-python}" ;;
        mplug)    py="${MPLUG_PYTHON:-python}" ;;
        *)        py="${VQA_PYTHON:-python}" ;;
    esac
    extra=(); [ "$m" = mplug ] && extra=(--backend-root "${MPLUG_ROOT:?set MPLUG_ROOT for mplug}")
    "$py" -m human_validation.score_vqa "$HUMAN_EVAL_INPUT" "$HUMAN_EVAL_DIR/${m}_both.jsonl" \
        --metric "$m" --image-root "$DATA_ROOT" --mode both --cache-dir "$HF_HUB_CACHE" \
        --device cuda "${ow[@]}" "${extra[@]}"
done

echo "[hv] 3/3 compare -> $HUMAN_EVAL_DIR/comparison.txt"
python -m human_validation.compare_human_metrics | tee "$HUMAN_EVAL_DIR/comparison.txt"
