"""Optional pre-stage for Agent 2: depth estimation + FLUX.1-Depth-dev.

Two-stage idea: instead of handing Flux-Fill a hidden region that's just
neutral gray (weak/no visual evidence of what belongs there), first (1)
estimate a depth map of the cutout and (2) run FLUX.1-Depth-dev — a
depth-conditioned *full-image* generator, not a masked inpainter — to
produce a plausible complete version of the subject following that depth
structure. That guide image's content is then pasted into the hidden
region (replacing the gray placeholder) before Flux-Fill runs, so Fill-dev
has real geometry to refine/blend rather than starting from blank gray.

Gated off by default via config.USE_DEPTH_GUIDED_FILL — this only helps
cases where the mask is placed over genuinely-occluded geometry that needs
extra structural grounding to complete well; it does nothing for a mask
that's wrong in the first place (see the vase case's actual root cause).
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config
from runtime import DEVICE, _release_ram

_DEPTH_ESTIMATOR = None
_FLUX_DEPTH_PIPE = None


def _get_depth_estimator():
    """Lazy-load the Depth-Anything-V2 depth-estimation pipeline."""
    global _DEPTH_ESTIMATOR
    if _DEPTH_ESTIMATOR is not None:
        return _DEPTH_ESTIMATOR
    from transformers import pipeline as hf_pipeline
    model_id = getattr(config, "DEPTH_ESTIMATOR_MODEL_ID", "depth-anything/Depth-Anything-V2-Large-hf")
    print(f"  [Depth] Loading depth estimator ({model_id})…")
    device_id = 0 if DEVICE == "cuda" else -1
    _release_ram()
    _DEPTH_ESTIMATOR = hf_pipeline("depth-estimation", model=model_id, device=device_id)
    return _DEPTH_ESTIMATOR


def _free_depth_estimator():
    global _DEPTH_ESTIMATOR
    if _DEPTH_ESTIMATOR is None:
        return
    try:
        _DEPTH_ESTIMATOR.model.cpu()
    except Exception:
        pass
    del _DEPTH_ESTIMATOR
    _DEPTH_ESTIMATOR = None
    _release_ram()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print("  [Depth] Estimator released from GPU")


def _estimate_depth(pil_image: Image.Image) -> Image.Image:
    """Return an RGB depth-map image (near=bright) sized to match the input."""
    estimator = _get_depth_estimator()
    result = estimator(pil_image)
    depth = result["depth"]  # PIL 'L' or 'F' image, same size as input
    return depth.convert("RGB")


def _get_flux_depth_pipe():
    """Lazy-load FLUX.1-Depth-dev via FluxControlPipeline."""
    global _FLUX_DEPTH_PIPE
    if _FLUX_DEPTH_PIPE is not None:
        return _FLUX_DEPTH_PIPE
    try:
        from diffusers import FluxControlPipeline
    except ImportError as exc:
        print(f"  [Flux-Depth] diffusers does not expose FluxControlPipeline "
              f"({exc!r}) — upgrade diffusers to use this stage")
        return None
    model_id = getattr(config, "FLUX_DEPTH_MODEL_ID", "black-forest-labs/FLUX.1-Depth-dev")
    print(f"  [Flux-Depth] Loading {model_id}…")
    from model_lifecycle import _free_all_gpu_models_except
    _free_all_gpu_models_except("flux")
    _release_ram()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    _FLUX_DEPTH_PIPE = FluxControlPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    _FLUX_DEPTH_PIPE.to(DEVICE)
    return _FLUX_DEPTH_PIPE


def _free_flux_depth():
    global _FLUX_DEPTH_PIPE
    if _FLUX_DEPTH_PIPE is None:
        return
    try:
        _FLUX_DEPTH_PIPE.to("cpu")
    except Exception:
        pass
    del _FLUX_DEPTH_PIPE
    _FLUX_DEPTH_PIPE = None
    _release_ram()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print("  [Flux-Depth] Released from GPU")


def build_depth_guided_base(cutout_bgr: np.ndarray, hidden_mask: np.ndarray,
                             prompt: str, seed: int = 42) -> np.ndarray:
    """Run the two-stage depth-guide step and return a new base image (BGR)
    with the hidden region replaced by Depth-dev's hallucinated content,
    ready to hand to Flux-Fill at partial strength for refinement.

    Falls back to returning `cutout_bgr` unchanged (no-op) on any failure —
    this is an optional quality boost, never a hard requirement.
    """
    import cv2

    try:
        h, w = cutout_bgr.shape[:2]
        pil_cutout = Image.fromarray(cv2.cvtColor(cutout_bgr, cv2.COLOR_BGR2RGB))

        depth_map = _estimate_depth(pil_cutout)
        _free_depth_estimator()

        pipe = _get_flux_depth_pipe()
        if pipe is None:
            return cutout_bgr

        gen_w = min(round(w / 16) * 16, 1024)
        gen_h = min(round(h / 16) * 16, 1024)
        depth_resized = depth_map.resize((gen_w, gen_h))

        generator = torch.Generator(device="cpu").manual_seed(seed)
        guide = pipe(
            prompt=prompt,
            control_image=depth_resized,
            height=gen_h,
            width=gen_w,
            num_inference_steps=getattr(config, "FLUX_DEPTH_STEPS", 28),
            guidance_scale=getattr(config, "FLUX_DEPTH_GUIDANCE_SCALE", 10.0),
            generator=generator,
        ).images[0]
        if guide.size != (w, h):
            guide = guide.resize((w, h), Image.LANCZOS)
        guide_bgr = cv2.cvtColor(np.array(guide), cv2.COLOR_RGB2BGR)

        new_base = cutout_bgr.copy()
        m = hidden_mask.astype(bool)
        new_base[m] = guide_bgr[m]
        print(f"  [Flux-Depth] Pasted depth-guided content into "
              f"{int(m.sum())} hidden px")
        return new_base
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [Flux-Depth] guide stage failed ({exc!r}) — "
              f"falling back to plain gray-filled cutout")
        return cutout_bgr
    finally:
        _free_flux_depth()
