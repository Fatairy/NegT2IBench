#!/bin/bash
#SBATCH --job-name=negt2i_human_validation
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#
# Human validation on SLURM: manifest -> (optional) metric scoring -> comparison.
# Without CLIP_METRICS / VQA_METRICS it only compares NegT2IBench with the humans
# (CPU is enough then; drop --gpus).
#
#   sbatch jobs/human_validation.sbatch
#   CLIP_METRICS="clipscore pickscore" CLIP_PYTHON=envs/negt2i-clip-metrics/bin/python \
#       sbatch jobs/human_validation.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source config.sh
negt2i_activate
bash scripts/human_validation/run_human_validation.sh
