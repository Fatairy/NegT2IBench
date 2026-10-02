#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${NEGT2I_ROOT:-$(cd "$ROOT/../.." && pwd)}"   # repository root
ENVS="${METRIC_ENVS_DIR:-$WORKSPACE/envs}"
CACHE="${CACHE_DIR:-$WORKSPACE/cache}"
mkdir -p "$ENVS" "$CACHE/pip" "$CACHE/conda-pkgs" "$CACHE/mamba-root" "$WORKSPACE/tmp"

export PIP_CACHE_DIR="$CACHE/pip"
export CONDA_PKGS_DIRS="$CACHE/conda-pkgs"
export MAMBA_ROOT_PREFIX="$CACHE/mamba-root"
export TMPDIR="$WORKSPACE/tmp"
export XDG_CACHE_HOME="$CACHE/xdg"
# Everything (envs, caches, tmp) must stay under $WORKSPACE; fall back to the
# a micromamba binary under $WORKSPACE/bin if one is not on PATH.
export PATH="$WORKSPACE/bin:$PATH"

if command -v micromamba >/dev/null 2>&1; then
  RUNNER=micromamba
elif command -v mamba >/dev/null 2>&1; then
  RUNNER=mamba
else
  RUNNER=""
  BASE_PYTHON="${BASE_PYTHON:-$ENVS/negt2i-vqa-metrics/bin/python}"
  [[ -x "$BASE_PYTHON" ]] || {
    echo "mamba/micromamba or BASE_PYTHON is required" >&2
    exit 1
  }
fi

create_base() {
  local prefix="$1"
  local pyver="${2:-3.10}"
  if [[ ! -x "$prefix/bin/python" ]]; then
    if [[ -n "$RUNNER" ]]; then
      "$RUNNER" create -y -p "$prefix" "python=$pyver" pip=24.3 setuptools=75.8
    else
      echo "venv fallback cannot pin a python version other than the base interpreter" >&2
      "$BASE_PYTHON" -m venv --copies "$prefix"
      "$prefix/bin/python" -m pip install --upgrade pip==24.3 setuptools==75.8
    fi
  fi
}

VQASCORE_ENV="$ENVS/negt2i-vqascore"
create_base "$VQASCORE_ENV" 3.10
"$VQASCORE_ENV/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1
"$VQASCORE_ENV/bin/python" -m pip install -r "$ROOT/vqascore_requirements.txt"
# Installed from a local worktree (not the bare git+URL): the pinned upstream
# commit's pyproject.toml has an unquoted PEP 508 URL dependency that trips
# pip's TOML parser, so we patch a checkout instead of the remote revision.
T2V_METRICS_SRC="${T2V_METRICS_SRC:-$WORKSPACE/external/t2v_metrics-clip-flant5-xxl}"
[[ -f "$T2V_METRICS_SRC/pyproject.toml" ]] || {
  echo "Missing patched t2v_metrics checkout at $T2V_METRICS_SRC" >&2
  exit 1
}
"$VQASCORE_ENV/bin/python" -m pip install --no-deps "$T2V_METRICS_SRC"

MPLUG_ENV="$ENVS/negt2i-mplug"
# Python 3.9: transformers==4.14.1 pins tokenizers<0.11, and that old Rust
# extension only ships prebuilt wheels up to cp39 (cp310+ has none, forcing a
# from-source build that needs a Rust toolchain we don't have here).
create_base "$MPLUG_ENV" 3.9
"$MPLUG_ENV/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.0.1 torchvision==0.15.2
"$MPLUG_ENV/bin/python" -m pip install -r "$ROOT/mplug_requirements.txt"

echo "VQAScore environment: $VQASCORE_ENV"
echo "mPLUG environment:    $MPLUG_ENV"
