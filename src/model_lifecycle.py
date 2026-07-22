"""GPU-model lifecycle management: frees every other GPU-resident singleton
before loading a new heavy model, so SAM3 / SAM3-text / Flux-Fill can share
a single GPU sequentially without fighting each other for VRAM.

Lives in its own module (rather than runtime.py) because models/flux_fill.py
needs to call `_free_all_gpu_models_except` from inside `_get_flux_fill_pipe`,
which would otherwise create a circular import; this module is the one place
that imports every models/* submodule, and consumers of it do the reverse
import lazily (see the `from model_lifecycle import ...` inside
models/flux_fill.py) to break the cycle.
"""
import torch

from runtime import DEVICE
from models.sam3 import _free_sam3, _free_sam3_text
from models.flux_fill import _free_flux_fill

def _free_all_gpu_models_except(*keep_names: str):
    """Free every loaded GPU-resident singleton EXCEPT those named in
    `keep_names` (one of: 'sam3', 'sam3_text', 'flux').

    Used as a hard barrier before loading a heavy model (Flux's 12B-param
    transformer in particular) — clears CUDA caches so the new model can
    grab its full peak allocation without fighting fragmentation.
    """
    keep = set(keep_names)
    if "sam3"        not in keep: _free_sam3()
    if "sam3_text"   not in keep: _free_sam3_text()
    if "flux"        not in keep: _free_flux_fill()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
