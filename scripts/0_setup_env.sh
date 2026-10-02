#!/usr/bin/env bash
# Stage 0 — create the Python environment for prompt generation, image generation and
# the detector-based evaluator: one conda env plus an RF-DETR overlay directory.
#
#   bash scripts/0_setup_env.sh
#
# Uses ENV_MANAGER / ENV_NAME from config.sh (conda or micromamba). With
# ENV_MANAGER=none it installs into the python currently on PATH (e.g. your venv).
#
# Tested with Python 3.11 + CUDA 12 on H100/H200. The human-validation metrics need
# their own envs, see human_validation/envs/ and README.md.
set -euo pipefail
NEGT2I_SKIP_ACTIVATE=1 source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

mgr="$ENV_MANAGER"
if [ "$mgr" = auto ]; then
    if command -v micromamba >/dev/null 2>&1; then mgr=micromamba
    elif command -v conda >/dev/null 2>&1; then mgr=conda
    else mgr=none; fi
fi

case "$mgr" in
    micromamba) micromamba create -y -n "$ENV_NAME" -c conda-forge python=3.11 pip ;;
    conda)      conda create -y -n "$ENV_NAME" python=3.11 pip ;;
    none)       echo "[0] ENV_MANAGER=none: installing into $(command -v python)" ;;
esac
ENV_MANAGER="$mgr" negt2i_activate

python -m pip install --upgrade pip
python -m pip install -r "$REPO/requirements.txt"
if [ -n "$RFDETR_OVERLAY" ]; then
    echo "[0] installing RF-DETR overlay into $RFDETR_OVERLAY"
    python -m pip install --target "$RFDETR_OVERLAY" -r "$REPO/requirements-rfdetr.txt"
fi

echo "[0] verifying the generation layer..."
python - <<'PY'
import torch, diffusers, transformers, open_clip, sam2, pandas  # noqa: F401
from clip_benchmark.metrics import zeroshot_classification  # noqa: F401
print("  torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| diffusers", diffusers.__version__)
PY
echo "[0] verifying the evaluation layer (with the RF-DETR overlay)..."
negt2i_activate eval
python - <<'PY'
import torch, rfdetr, open_clip, sam2  # noqa: F401
print("  torch", torch.__version__, "| rfdetr ok | cuda", torch.cuda.is_available())
PY
echo "[0] done. Next: python sanity_check.py --skip-heavy"
