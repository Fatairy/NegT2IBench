#!/usr/bin/env bash
# Stage 5 — the cross-model report: runs stage 4 for every model, then writes Markdown
# tables, CSVs and figures. CPU only.
#
# Writes  $REPORT_DIR/$SUBDIR/README.md                index
#         $REPORT_DIR/$SUBDIR/report__comparison.md    every setting side by side
#         $REPORT_DIR/$SUBDIR/report__<setting>.md     per-setting accuracy, failure causes
#         $REPORT_DIR/$SUBDIR/*.csv / *.png            tables and figures
#
# Env vars:
#   MODELS        space-separated models   (default: every evaluated model found)
#   SETTINGS      space-separated settings (default: every evaluated setting found)
#   FIGURES       report (default) | none | core | accuracy | failure | all | a,b,...
#   SUBDIR        output subfolder         (default: main)
#   RUN_SUMMARY=0 skip stage 4 (reuse the summary CSVs on disk)
#   CLEAN=1       delete this script's previous .md/.csv/.png in the output dir first
#
# Example:
#   bash scripts/5_report.sh
#   MODELS="sd35 flux2" FIGURES=all CLEAN=1 bash scripts/5_report.sh
set -euo pipefail
shopt -s nullglob
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

SUBDIR="${SUBDIR:-main}"
FIGURES="${FIGURES:-report}"
[ -d "$DATA_ROOT" ] || { echo "[5] no data root: $DATA_ROOT" >&2; exit 1; }

discover_models() {
    local d
    for d in "$DATA_ROOT"/*/; do
        local found=("$d"pos*_neg*/"$RESULTS_SUBDIR"/neg_*.jsonl)
        [ "${#found[@]}" -gt 0 ] && basename "$d"
    done
}
MODELS="${MODELS:-$(discover_models | tr '\n' ' ')}"
MODELS="$(echo $MODELS)"
[ -n "$MODELS" ] || { echo "[5] no evaluated models under $DATA_ROOT" >&2; exit 1; }

echo "[5] data root : $DATA_ROOT"
echo "[5] report    : $REPORT_DIR/$SUBDIR"
echo "[5] models    : $MODELS"
echo "[5] settings  : ${SETTINGS:-all}"

if [ "${RUN_SUMMARY:-1}" = 1 ]; then
    for m in $MODELS; do
        MODEL="$m" NEGT2I_SKIP_ACTIVATE=1 bash "$REPO/scripts/4_summarize.sh" > "$LOG_DIR/summary_$m.txt"
        echo "[5] stage 4 for $m -> $LOG_DIR/summary_$m.txt"
    done
fi

if [ "${CLEAN:-0}" = 1 ] && [ -d "$REPORT_DIR/$SUBDIR" ]; then
    find "$REPORT_DIR/$SUBDIR" -maxdepth 1 -type f \( -name '*.png' -o -name '*.csv' -o -name '*.md' \) -delete
fi

# shellcheck disable=SC2086
cmd=(python "$REPO/evaluation/visualize_results.py" --data-root "$DATA_ROOT"
     --out-dir "$REPORT_DIR" --subdir "$SUBDIR" --models $MODELS --figures "$FIGURES")
# shellcheck disable=SC2206
[ -n "${SETTINGS:-}" ] && cmd+=(--settings $SETTINGS)
"${cmd[@]}"
echo "[5] start here: $REPORT_DIR/$SUBDIR/README.md"
