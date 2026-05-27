"""
models — per-model modules for the amodal pipeline.

SAM3 segmentation, GPT vision, CLIP grounding, and FLUX.1-Fill inpainting.
Each submodule lazy-loads its model on first use and registers a free()
with runtime so `runtime.free_all_except(...)` can reclaim VRAM by name.
"""

from . import clip, flux, gpt, sam3  # noqa: F401

__all__ = ["gpt", "sam3", "clip", "flux"]
