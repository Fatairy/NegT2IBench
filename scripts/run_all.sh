#!/usr/bin/env bash
# End-to-end local run on the released prompts: generate -> evaluate -> summarize for
# each model, then one report across all of them.
#
#   MODELS="sd35 flux2" bash scripts/run_all.sh
#   MODELS=zimage SETTINGS=pos1_neg1 LIMIT=1 N_SAMPLES=1 bash scripts/run_all.sh   # quick
#
# Honours every variable of the stage scripts (SETTINGS, CATEGORIES, LIMIT, N_SAMPLES,
# SKIP_DONE, ...). Outputs: see config.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../config.sh"
MODELS="${MODELS:-$MODEL}"

for model in $MODELS; do
    echo "================  MODEL: $model  ================"
    MODEL="$model" bash "$HERE/2_generate_images.sh"
    MODEL="$model" bash "$HERE/3_evaluate_images.sh"
    MODEL="$model" bash "$HERE/4_summarize.sh"
done
MODELS="$MODELS" RUN_SUMMARY=0 bash "$HERE/5_report.sh"
echo "all done."
