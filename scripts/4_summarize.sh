#!/usr/bin/env bash
# Stage 4 — merge one model's per-category verdicts and print the scores, per setting
# and overall. CPU only.
#
# Writes  $DATA_ROOT/$MODEL/<setting>/$RESULTS_SUBDIR/all_results.jsonl          per-setting merge
#         $DATA_ROOT/$MODEL/<setting>/$RESULTS_SUBDIR/all_results_summary.csv    per-setting scores
#         $DATA_ROOT/$MODEL/$RESULTS_SUBDIR/all_results.jsonl                    all settings
#         $DATA_ROOT/$MODEL/$RESULTS_SUBDIR/all_results_summary.csv              overall scores
#
# Example:
#   MODEL=sd35 bash scripts/4_summarize.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MODEL_DIR="$DATA_ROOT/$MODEL"
SUMMARY="$REPO/evaluation/summary_scores.py"
[ -d "$MODEL_DIR" ] || { echo "[4] no model dir: $MODEL_DIR" >&2; exit 1; }

mapfile -t SETTING_RES < <(find "$MODEL_DIR" -mindepth 2 -maxdepth 2 -type d -name "$RESULTS_SUBDIR" | sort)
[ "${#SETTING_RES[@]}" -gt 0 ] || { echo "[4] no $RESULTS_SUBDIR/ dirs under $MODEL_DIR (run stage 3)" >&2; exit 1; }

OVERALL="$MODEL_DIR/$RESULTS_SUBDIR"
mkdir -p "$OVERALL"
: > "$OVERALL/all_results.jsonl"
for rdir in "${SETTING_RES[@]}"; do
    setting="$(basename "$(dirname "$rdir")")"
    keep_setting "$setting" || continue
    find "$rdir" -maxdepth 1 -name 'neg_*.jsonl' -exec cat {} + > "$rdir/all_results.jsonl"
    echo
    echo "=================  $MODEL / $setting  ================="
    python "$SUMMARY" "$rdir/all_results.jsonl"
    cat "$rdir/all_results.jsonl" >> "$OVERALL/all_results.jsonl"
done

echo
echo "=================  $MODEL / OVERALL  ================="
python "$SUMMARY" "$OVERALL/all_results.jsonl"
echo
echo "[4] per-setting: $MODEL_DIR/<setting>/$RESULTS_SUBDIR/   overall: $OVERALL/"
