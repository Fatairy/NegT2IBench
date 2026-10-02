#!/usr/bin/env bash
# Stage 3 — score one model's images with the detector-based evaluator.
#
# Reads   $DATA_ROOT/$MODEL/<setting>/<category>/<idx>/samples/*.png
# Writes  $DATA_ROOT/$MODEL/<setting>/$RESULTS_SUBDIR/<category>.jsonl  (one row per image:
#         `correct`, `reason`, detections) and .../_config.json (options + environment)
#
# Models: RF-DETR-XL (detection + masks), SigLIP2 SO400M/14-378 (colour/material),
# SAM 2.1-small (relation masks), Depth-Anything-V2-Small (depth). Weights download to
# the cache on first use. Every parameter is fixed in the evaluator code.
#
# Env vars:
#   MODEL        registry key whose images to score   (default: config.sh)
#   SETTINGS     e.g. "pos1_neg1"                      (default: all found)
#   CATEGORIES   e.g. "neg_attr_color"                 (default: all found)
#   SKIP_DONE=1  skip categories already fully scored (resume an interrupted run)
#
# Example:
#   MODEL=sd35 bash scripts/3_evaluate_images.sh
set -euo pipefail
NEGT2I_STAGE=eval source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MODEL_DIR="$DATA_ROOT/$MODEL"
[ -d "$MODEL_DIR" ] || { echo "[3] no images at $MODEL_DIR (run scripts/2_generate_images.sh)" >&2; exit 1; }

echo "[3] model=$MODEL  images=$MODEL_DIR"

n=0
while IFS= read -r d; do
    rel="${d#"$MODEL_DIR"/}"             # pos1_neg1/neg_rel_directional
    setting="$(dirname "$rel")"
    category="$(basename "$rel")"
    keep_setting "$setting" && keep_category "$category" || continue
    rdir="$MODEL_DIR/$setting/$RESULTS_SUBDIR"
    out="$rdir/$category.jsonl"
    mkdir -p "$rdir"
    if [ "${SKIP_DONE:-0}" = 1 ] && [ -f "$out" ]; then
        have=$(grep -vc '^[[:space:]]*$' "$out" 2>/dev/null || echo 0)
        want=$(find "$d" -path '*/samples/*.png' | wc -l)
        if [ "$have" -ge "$want" ] && [ "$want" -gt 0 ]; then
            echo "  - $rel SKIP (already scored: $have rows for $want images)"; continue
        fi
    fi
    echo "  - $rel -> $out"
    python "$REPO/evaluation/evaluate_images.py" "$d" --outfile "$out"
    n=$((n + 1))
done < <(find "$MODEL_DIR" -mindepth 2 -maxdepth 2 -type d -name 'neg_*' | sort)

echo "[3] done: $n category dirs scored. Next: MODEL=$MODEL bash scripts/4_summarize.sh"
