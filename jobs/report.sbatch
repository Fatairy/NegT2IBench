#!/bin/bash
#SBATCH --job-name=negt2i_report
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#
# Stages 4+5 on SLURM (CPU only): per-model summaries, then the cross-model report in
# $REPORT_DIR/$SUBDIR. MODELS defaults to every evaluated model.
#
#   sbatch jobs/report.sbatch
#   MODELS="sd35 flux2" FIGURES=all sbatch jobs/report.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source config.sh
negt2i_activate
bash scripts/5_report.sh
