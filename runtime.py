"""
runtime.py — shared process infrastructure.

Owns the things every model/pipeline module needs but that must be initialised
exactly once: the `.env` load, the device string, the VRAM cap + TF32 tuning,
the malloc-trim helper, and a small registry so heavy models can be freed by
name without any module reaching into another module's globals.

Import layering:  config → runtime → models/* → pipeline/* → server/client.
runtime imports only config (+ torch / dotenv); never models or pipeline.
"""

from __future__ import annotations

import ctypes
import gc
import os
import sys
import types
from pathlib import Path
from typing import Callable

# ── pkg_resources shim ────────────────────────────────────────────────────────
# Some torch / pytorch_lightning checkpoint pickles call
# __import__("pkg_resources").declare_namespace() at import time. uv venvs omit
# setuptools by default, so inject a minimal stub before anything triggers it.
if "pkg_resources" not in sys.modules:
    try:
        import pkg_resources as _pr  # noqa: F401
        if not hasattr(_pr, "declare_namespace"):
            _pr.declare_namespace = lambda name: None  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        _stub = types.ModuleType("pkg_resources")
        _stub.DistributionNotFound = Exception  # type: ignore[attr-defined]
        _stub.get_distribution = lambda name: type("_D", (), {"version": "0.0.0"})()  # type: ignore[attr-defined]
        _stub.declare_namespace = lambda name: None  # type: ignore[attr-defined]
        _stub.require = lambda *a, **k: []  # type: ignore[attr-defined]
        sys.modules["pkg_resources"] = _stub

import torch
from dotenv import load_dotenv

import config

# ── Environment ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY   = os.environ.get("OPENAI_API_KEY")
GPT_MODEL = os.environ.get("OPENAI_MODEL")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ── Auto-detect Flux offload mode + GPU cap from VRAM ─────────────────
# Picks the right strategy regardless of whether we're on a 12 GB RTX 3060, a
# 22 GB L4, an A100, etc. Set config.AUTO_DETECT_OFFLOAD = False to keep the
# manual FLUX_FILL_*_OFFLOAD values from config.py.
if DEVICE == "cuda":
    _total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if bool(getattr(config, "AUTO_DETECT_OFFLOAD", True)):
        #   <14 GB  → stream layer-by-layer (sequential_cpu_offload)
        #   14–38 GB → swap component-by-component (model_cpu_offload)
        #   ≥38 GB  → keep everything resident (no offload, fastest)
        if _total_gb < 14.0:
            mode = "sequential"
            config.FLUX_FILL_SEQUENTIAL_OFFLOAD = True
            config.FLUX_FILL_CPU_OFFLOAD = True
            cap = max(_total_gb - 2.0, 6.0)
        elif _total_gb < 38.0:
            mode = "model_offload"
            config.FLUX_FILL_SEQUENTIAL_OFFLOAD = False
            config.FLUX_FILL_CPU_OFFLOAD = True
            cap = max(_total_gb - 2.0, 12.0)
        else:
            mode = "fully_on_device"
            config.FLUX_FILL_SEQUENTIAL_OFFLOAD = False
            config.FLUX_FILL_CPU_OFFLOAD = False
            cap = _total_gb - 4.0
        config.GPU_MEMORY_LIMIT_GB = cap
        print(f"GPU      : {torch.cuda.get_device_name(0)}  ({_total_gb:.1f} GB)")
        print(f"Auto-cfg : flux_offload={mode}  gpu_cap={cap:.1f} GB")

    if config.GPU_MEMORY_LIMIT_GB:
        _fraction = min(config.GPU_MEMORY_LIMIT_GB / _total_gb, 1.0)
        torch.cuda.set_per_process_memory_fraction(_fraction, device=0)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        print(f"GPU cap  : {config.GPU_MEMORY_LIMIT_GB:.1f} GB / {_total_gb:.1f} GB  ({_fraction:.0%})")

    # Free Ampere perf wins — mathematically benign on bf16/fp32 inference.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _release_ram() -> None:
    """gc.collect + malloc_trim: forces Python to return freed memory to the OS."""
    gc.collect()
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ── Model-free registry ───────────────────────────────────────────────────────
# Each model module registers its free() at import time via register_free_fn.
# free_all_except() then frees every loaded model except the named keepers,
# without runtime needing to import any model module.
_FREE_FNS: "dict[str, Callable[[], None]]" = {}


def register_free_fn(name: str, fn: "Callable[[], None]") -> None:
    _FREE_FNS[name] = fn


def free_all_except(*keep_names: str) -> None:
    """Free every registered GPU-resident model EXCEPT those named in
    `keep_names` (e.g. 'sam3', 'flux', 'clip'). Hard barrier before loading a
    heavy model so it can grab peak allocation without fragmentation."""
    keep = set(keep_names)
    for name, fn in list(_FREE_FNS.items()):
        if name in keep:
            continue
        try:
            fn()
        except Exception:  # noqa: BLE001
            pass
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
