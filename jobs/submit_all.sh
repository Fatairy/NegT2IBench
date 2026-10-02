#!/usr/bin/env bash
# Submit the full benchmark to SLURM with dependencies:
#   generate[array] -> evaluate[array]   for each model in MODELS
#   report                               after every evaluate array
#
#   MODELS="sd35 flux2" bash jobs/submit_all.sh
#   MODELS=sd35 SETTINGS="pos1_neg1 pos0_neg1" bash jobs/submit_all.sh
#   MODELS=sd35 SKIP_GENERATE=1 bash jobs/submit_all.sh      # images already exist
#
# Extra sbatch flags go in SBATCH_ARGS, e.g.
#   SBATCH_ARGS="--time=24:00:00" MODELS=sd35 bash jobs/submit_all.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source config.sh
MODELS="${MODELS:-$MODEL}"
SBATCH_ARGS="${SBATCH_ARGS:-}"
export SETTINGS="${SETTINGS:-}" CATEGORIES="${CATEGORIES:-}"

# Size the array to the (setting, category) pairs left after the filters.
n=$(SLURM_ARRAY_TASK_ID=0 MODEL=x bash -c 'source config.sh; source jobs/_task.sh >/dev/null; echo ${#_PAIRS[@]}')
[ "$n" -gt 0 ] || { echo "no prompt files match SETTINGS='$SETTINGS' CATEGORIES='$CATEGORIES'" >&2; exit 1; }
ARRAY="0-$((n - 1))"
echo "submitting $n (setting, category) tasks per model: array=$ARRAY  models=$MODELS"

deps=()
for m in $MODELS; do
    dep=()
    if [ "${SKIP_GENERATE:-0}" != 1 ]; then
        # shellcheck disable=SC2086
        gj=$(MODEL="$m" sbatch --parsable --export=ALL --array="$ARRAY" $SBATCH_ARGS jobs/generate.sbatch)
        dep=(--dependency=afterok:"$gj")
        echo "  $m: generate $gj"
    fi
    # shellcheck disable=SC2086
    ej=$(MODEL="$m" sbatch --parsable --export=ALL --array="$ARRAY" "${dep[@]}" $SBATCH_ARGS jobs/evaluate.sbatch)
    echo "  $m: evaluate $ej"
    deps+=("$ej")
done
# shellcheck disable=SC2086
rj=$(MODELS="$MODELS" sbatch --parsable --export=ALL \
     --dependency=afterok:"$(IFS=:; echo "${deps[*]}")" $SBATCH_ARGS jobs/report.sbatch)
echo "  report $rj"
echo "logs: $LOG_DIR   report: $REPORT_DIR/${SUBDIR:-main}/README.md"
