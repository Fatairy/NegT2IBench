"""Make an evaluation run reproducible, and record what it ran on.

Two separate jobs, deliberately in one place:

1. `enforce()` pins every knob that can make the same image score differently on a
   second run -- algorithm autotuning, reduced-precision matmuls, RNG state.
2. `manifest()` records the versions, GPU model and flags that the numbers depend on,
   so a result file can be traced to the stack that produced it. It records no
   user names, host names, paths or device ids.

Why this is not paranoia: the pipeline is inference-only, so it has no sampling to
seed -- but `cudnn.benchmark` picks convolution algorithms by timing them, which
depends on what else is on the GPU, and TF32 silently drops matmul mantissa bits.
Attribute verdicts turn on logit gaps as small as 0.5, so a run is only
comparable to another run if both fixed these the same way.

CUBLAS_WORKSPACE_CONFIG must be set BEFORE the first CUDA op; config.sh exports it.
"""
from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

DEFAULT_SEED = 0
REPO = Path(__file__).resolve().parent.parent

# TF32 off: an attribute verdict compares logits within 0.5 of each other, and the
# thresholds were fitted against human labels without it.
ALLOW_TF32 = False
SEED = DEFAULT_SEED

_STATE: dict = {"enforced": False, "warnings": []}


def _git(*args: str) -> str:
    try:
        return subprocess.run(("git", "-C", str(REPO)) + args, capture_output=True,
                              text=True, timeout=15).stdout.strip()
    except Exception:                                             # noqa: BLE001
        return ""


def enforce(seed: int | None = None, allow_tf32: bool | None = None) -> dict:
    """Pin RNG, algorithm selection and matmul precision. Idempotent.

    Returns the flag dict it applied, which goes straight into `manifest()`.
    """
    import numpy as np
    import torch

    seed = SEED if seed is None else seed
    tf32 = ALLOW_TF32 if allow_tf32 is None else allow_tf32

    # First, before anything can create a cuBLAS handle: the variable is read once at
    # handle creation and ignored afterwards.
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        if torch.cuda.is_initialized():
            _STATE["warnings"].append(
                "CUBLAS_WORKSPACE_CONFIG was unset and CUDA is already initialised, "
                "so setting it now cannot take effect; some cuBLAS reductions may "
                "vary run to run. Export it before starting python "
                "(config.sh does).")
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Time-based convolution autotuning: the algorithm chosen depends on GPU load at
    # that moment, so two runs of the same image can take different code paths.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")

    # warn_only: this stack (DINOv2 + SAM 2 + Depth-Anything + CLIP) is forward-only,
    # but a single op without a deterministic kernel would otherwise abort a
    # multi-hour run. A warning is recorded in the manifest instead.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:                                      # noqa: BLE001
        _STATE["warnings"].append(f"use_deterministic_algorithms unavailable: {exc}")

    if os.environ.get("PYTHONHASHSEED") is None:
        # Set iteration order decides the order of predicates listed in a relation
        # `reason`; the verdict does not depend on it. config.sh exports it.
        _STATE["warnings"].append(
            "PYTHONHASHSEED unset; the predicate order inside `reason` strings may "
            "vary between runs (verdicts do not).")

    # Recorded, not pinned: the thread count does not change results.
    flags = {"torch_num_threads": torch.get_num_threads(),
             "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
             "seed": seed, "allow_tf32": bool(tf32),
             "cudnn_benchmark": False, "cudnn_deterministic": True,
             "float32_matmul_precision": torch.get_float32_matmul_precision(),
             "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
             "pythonhashseed": os.environ.get("PYTHONHASHSEED", "")}
    _STATE["enforced"] = True
    _STATE["flags"] = flags
    return flags


def _versions() -> dict:
    import importlib.metadata as md
    out = {}
    for pkg in ("torch", "torchvision", "rfdetr", "transformers", "open_clip_torch",
                "numpy", "pillow", "supervision", "timm", "sam2", "opencv-python",
                "opencv-python-headless", "scipy", "pandas"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:                                         # noqa: BLE001
            continue
    return out


def manifest(extra: dict | None = None) -> dict:
    """Everything the numbers in a result file depend on, as JSON-able data."""
    import torch
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({"index": i, "name": p.name,
                         "capability": f"{p.major}.{p.minor}",
                         "total_mem_gib": round(p.total_memory / 1024 ** 3, 1)})
    # A dirty tree makes the commit hash alone insufficient to identify the code, so
    # fingerprint the uncommitted diff too: commit + diff digest pins the working tree
    # exactly, whether or not the change was committed before the run.
    diff = _git("diff", "HEAD")
    m = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "git_diff_sha256": (hashlib.sha256(diff.encode()).hexdigest() if diff else ""),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_cuda": getattr(__import__("torch").version, "cuda", None),
        "cudnn": __import__("torch").backends.cudnn.version(),
        "gpus": gpus,
        "versions": _versions(),
        "determinism": _STATE.get("flags", {}) if _STATE["enforced"]
                       else {"enforced": False},
        "determinism_warnings": list(_STATE["warnings"]),
    }
    if extra:
        m.update(extra)
    return m


def report() -> str:
    """One-line-per-fact summary for the run log."""
    m = manifest()
    v = m["versions"]
    lines = [f"[REPRO] git {m['git_commit'][:10]}"
             f"{' +dirty' if m['git_dirty'] else ''} on {m['git_branch']}",
             f"[REPRO] torch {v.get('torch','?')} cuda {m['torch_cuda']} | "
             f"rfdetr {v.get('rfdetr','?')} | open_clip {v.get('open_clip_torch','?')}"
             f" | transformers {v.get('transformers','?')}",
             f"[REPRO] determinism: seed={m['determinism'].get('seed')} "
             f"tf32={m['determinism'].get('allow_tf32')} "
             f"matmul={m['determinism'].get('float32_matmul_precision')} "
             f"cudnn_benchmark={m['determinism'].get('cudnn_benchmark')}"]
    if m["gpus"]:
        g = m["gpus"][0]
        lines.append(f"[REPRO] gpu {g['name']} (sm{g['capability']})")
    for w in m["determinism_warnings"]:
        lines.append(f"[REPRO] WARNING: {w}")
    return "\n".join(lines)
