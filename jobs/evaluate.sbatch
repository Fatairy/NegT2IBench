#!/bin/bash
#SBATCH --job-name=negt2i_evaluate
#SBATCH --array=0-47
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/logs/%x_%A_%a.out
#SBATCH --error=outputs/logs/%x_%A_%a.err
#
# Stage 3 on SLURM: score one (setting, category) per array task with the benchmark
# evaluator (RF-DETR-XL + SigLIP2 + SAM 2 + Depth-Anything; parameters in config.sh).
# Needs ~20 GB of GPU memory; a 35 GB MIG slice is enough.
#
#   MODEL=sd35 sbatch jobs/evaluate.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source config.sh
negt2i_activate eval
source jobs/_task.sh
bash scripts/3_evaluate_images.sh
