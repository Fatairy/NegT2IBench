#!/usr/bin/env bash
# Create and verify the isolated CLIP-like metric evaluation environment.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/environment.yml"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
ENV_NAME="negt2i-clip-metrics"

if command -v micromamba >/dev/null 2>&1; then
    ENV_TOOL=micromamba
elif command -v mamba >/dev/null 2>&1; then
    ENV_TOOL=mamba
elif command -v conda >/dev/null 2>&1; then
    ENV_TOOL=conda
else
    echo "ERROR: micromamba, mamba, or conda must be available." >&2
    exit 1
fi

echo "[metrics-env] creating $ENV_NAME from $ENV_FILE with $ENV_TOOL"
if "$ENV_TOOL" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "[metrics-env] $ENV_NAME already exists; updating it in place"
    "$ENV_TOOL" env update --name "$ENV_NAME" --file "$ENV_FILE" --prune
else
    "$ENV_TOOL" env create --file "$ENV_FILE"
fi

echo "[metrics-env] installing CUDA 12.1 PyTorch wheels"
"$ENV_TOOL" run --name "$ENV_NAME" python -m pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1

echo "[metrics-env] installing metric dependencies"
"$ENV_TOOL" run --name "$ENV_NAME" python -m pip install \
    --requirement "$REQUIREMENTS_FILE"

echo "[metrics-env] verifying imports and versions"
"$ENV_TOOL" run --name "$ENV_NAME" python - <<'PY'
import accelerate
import hpsv2
import ImageReward
import open_clip
import pandas
import PIL
import scipy
import sklearn
import torch
import torchvision
import transformers

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("open_clip:", open_clip.__version__)
print("accelerate:", accelerate.__version__)
print("hpsv2: import OK")
print("ImageReward: import OK")
print("CUDA runtime:", torch.version.cuda)
print("CUDA available on this node:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "[metrics-env] ready; activate with: $ENV_TOOL activate $ENV_NAME"
