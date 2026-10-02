#!/bin/bash
#SBATCH --job-name=negt2i_generate
#SBATCH --array=0-47
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/%x_%A_%a.out
#SBATCH --error=outputs/logs/%x_%A_%a.err
#
# Stage 2 on SLURM: one (setting, category) prompt file per array task. The default
# array 0-47 covers the full benchmark (8 settings x 6 categories). Submit from the
# repository root (or use jobs/submit_all.sh, which sizes the array for you):
#
#   MODEL=sd35 sbatch jobs/generate.sbatch
#   MODEL=flux2 SETTINGS=pos1_neg1 sbatch --array=0-5 jobs/generate.sbatch
#
# All paths and environment settings come from config.sh.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source config.sh
negt2i_activate
source jobs/_task.sh
bash scripts/2_generate_images.sh
