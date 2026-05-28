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

# ── GPU setup ─────────────────────────────────────────────────────────────────
if DEVICE == "cuda":
    _total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    _seq = bool(getattr(config, "SEQUENTIAL_OFFLOAD", False))

    if _seq:
        # Small GPU: cap VRAM, enable sequential offload flags for flux.py
        _cap = max(_total_gb - 2.0, 6.0)
        config.FLUX_FILL_SEQUENTIAL_OFFLOAD = True
        config.FLUX_FILL_CPU_OFFLOAD        = True
        config.GPU_MEMORY_LIMIT_GB          = _cap
        _fraction = min(_cap / _total_gb, 1.0)
        torch.cuda.set_per_process_memory_fraction(_fraction, device=0)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        print(f"GPU      : {torch.cuda.get_device_name(0)}  ({_total_gb:.1f} GB)  "
              f"[sequential offload, cap={_cap:.1f} GB]")
    else:
        # Large GPU: no cap, no offload, all models resident
        config.FLUX_FILL_SEQUENTIAL_OFFLOAD = False
        config.FLUX_FILL_CPU_OFFLOAD        = False
        config.GPU_MEMORY_LIMIT_GB          = None
        print(f"GPU      : {torch.cuda.get_device_name(0)}  ({_total_gb:.1f} GB)  "
              f"[fully on device]")

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
#
# SERVER_MODE: when True (set by server.py at startup), all model free/unload
# calls become no-ops. On a large-VRAM GPU (L40S, A100) all models fit resident
# simultaneously — freeing and reloading between pipeline steps only wastes time.
SERVER_MODE: bool = False

_FREE_FNS: "dict[str, Callable[[], None]]" = {}


def register_free_fn(name: str, fn: "Callable[[], None]") -> None:
    _FREE_FNS[name] = fn


def free_all_except(*keep_names: str) -> None:
    """Free every registered GPU-resident model EXCEPT those named in
    `keep_names`. No-op in SERVER_MODE — all models stay resident."""
    if SERVER_MODE:
        return
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
