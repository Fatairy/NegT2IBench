#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${VQA_ENV_NAME:-negt2i-vqa-metrics}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba env create -f "$ROOT/vqa_environment.yml" -n "$ENV_NAME" 2>/dev/null || micromamba env update -f "$ROOT/vqa_environment.yml" -n "$ENV_NAME"
  micromamba run -n "$ENV_NAME" python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.7.1' 'torchvision==0.22.1'
  micromamba run -n "$ENV_NAME" python -m pip install -r "$ROOT/vqa_requirements.txt"
  micromamba run -n "$ENV_NAME" python -m pip uninstall -y torchaudio >/dev/null 2>&1 || true
elif command -v mamba >/dev/null 2>&1; then
  mamba env create -f "$ROOT/vqa_environment.yml" -n "$ENV_NAME" 2>/dev/null || mamba env update -f "$ROOT/vqa_environment.yml" -n "$ENV_NAME"
  mamba run -n "$ENV_NAME" python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.7.1' 'torchvision==0.22.1'
  mamba run -n "$ENV_NAME" python -m pip install -r "$ROOT/vqa_requirements.txt"
  mamba run -n "$ENV_NAME" python -m pip uninstall -y torchaudio >/dev/null 2>&1 || true
else
  echo "mamba or micromamba is required" >&2
  exit 1
fi
echo "[$ENV_NAME] ready"
