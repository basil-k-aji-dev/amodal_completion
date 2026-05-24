"""
Amodal Completion Pipeline — LangGraph (pix2gestalt)
=====================================================

Agent 1 — Occlusion Agent
    Runs SAM2 automatic mask generation to segment every object in the image.
    Sends segments to GPT which identifies:
        - selected_segment_ids : the occluder (what to remove)
        - visible_segment_ids  : the visible portion of the occluded object
                                 (modal mask that pix2gestalt conditions on)
    Produces binary occluder mask + visible-object modal mask.
    Can be re-invoked on mask failures (MASK_INACCURATE / OCCLUDER_REMNANT).

Agent 2 — Inpainting Agent (pix2gestalt)
    Resizes image and modal mask to 256×256.
    Runs pix2gestalt to synthesise the COMPLETE amodal object on a white background.
    Caches all n_samples completions to disk; on retry cycles to the next sample
    instead of re-running inference.
    Composites result back into the scene:
        hidden region  (occluder ∩ amodal object) ← pix2gestalt pixels
        background     (occluder ∖ amodal object) ← cv2 Navier-Stokes inpaint
    Poisson-blends the composite for seamless boundary integration.

Agent 3 — Reviewer
    GPT compares original vs. inpainted result and scores completeness/seamlessness.
    Routes:
        MASK_INACCURATE / OCCLUDER_REMNANT → re-run occlusion agent
        Other failures                     → cycle to next pix2gestalt sample

Graph:
    occlusion_agent → inpainting_agent → reviewer
          ↑                   ↑                ↓
          │ (retry_mask)      └─(retry_sample)─┤
          └──────────────────────────────────  ↓
                                              END
"""
import base64
import json
import os
import re
import sys
import traceback
import types
from pathlib import Path
from typing import Optional, TypedDict

# ── pkg_resources shim ────────────────────────────────────────────────────────
# pytorch_lightning (used inside the pix2gestalt checkpoint pickle) calls
# __import__("pkg_resources").declare_namespace() at import time.
# uv venvs exclude setuptools by default, so we inject a minimal stub before
# any downstream imports can trigger the missing-module error.
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

import ctypes
import gc
import cv2
import numpy as np
import torch
from PIL import Image
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import OpenAI

import config
from mixed_context import build_mixed_context_inputs, make_mc_latent_callback


def _release_ram():
    """gc.collect + malloc_trim: forces Python to return freed memory to the OS immediately."""
    gc.collect()
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")

API_KEY   = os.environ.get("OPENAI_API_KEY")
GPT_MODEL = os.environ.get("OPENAI_MODEL")
if not GPT_MODEL:
    raise EnvironmentError("OPENAI_MODEL is not set in .env")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR  = Path(__file__).parent

# Hard-cap VRAM so model swaps don't OOM.
if DEVICE == "cuda" and config.GPU_MEMORY_LIMIT_GB:
    _total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    _fraction = min(config.GPU_MEMORY_LIMIT_GB / _total_gb, 1.0)
    torch.cuda.set_per_process_memory_fraction(_fraction, device=0)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"GPU cap  : {config.GPU_MEMORY_LIMIT_GB:.1f} GB / {_total_gb:.1f} GB  ({_fraction:.0%})")

# ── Free Ampere perf wins (no quality impact) ────────────────────────
# All three are mathematically benign on bf16/fp32 inference workloads.
# Combined effect on RTX 3060 + Flux: ~5–10 % per-step.
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True          # auto-tune conv kernels for static shapes
    torch.backends.cuda.matmul.allow_tf32 = True   # TF32 matmul (Ampere)
    torch.backends.cudnn.allow_tf32 = True         # TF32 in cuDNN convolutions
    torch.set_float32_matmul_precision("high")     # newer API for the same TF32 path

if not API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in .env")

gpt = OpenAI(api_key=API_KEY)
print(f"Model : {GPT_MODEL}  |  Device: {DEVICE}")


# ── Model singletons (lazy-loaded, cached for process lifetime) ───────────────

_SAM3_PIPE         = None
_PIX2GESTALT_MODEL = None
_LAMA_MODEL        = None
_CONTROLNET_PIPE   = None
_FLUX_FILL_PIPE    = None
_AISFORMER         = None
_CLIP_MODEL        = None
_CLIP_PREPROCESS   = None
_GDINO_MODEL       = None
_GDINO_PROCESSOR   = None
_DEPTH_PIPELINE    = None
_DEPTH_MAP_CACHE: dict = {}   # image_path → np.ndarray (predicted_depth)


def _get_depth_pipeline():
    """Lazy-load a Hugging Face depth-estimation pipeline.

    Returns a transformers pipeline that maps PIL image → {'depth': PIL,
    'predicted_depth': torch.Tensor}.  Higher predicted_depth = closer to
    camera (i.e. inverse depth) for the default Depth-Anything-V2 model.
    Returns None on failure so callers can degrade gracefully.
    """
    global _DEPTH_PIPELINE
    if _DEPTH_PIPELINE is not None:
        return _DEPTH_PIPELINE
    try:
        from transformers import pipeline
        model_id = getattr(config, "DEPTH_MODEL_ID",
                           "depth-anything/Depth-Anything-V2-Small-hf")
        _DEPTH_PIPELINE = pipeline(
            "depth-estimation",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1,
        )
        print(f"  [Depth] Loaded {model_id}")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [Depth] pipeline load failed: {exc!r}")
        _DEPTH_PIPELINE = None
    return _DEPTH_PIPELINE


def _free_depth_pipeline():
    """Release the Depth-Anything pipeline from GPU."""
    global _DEPTH_PIPELINE, _DEPTH_MAP_CACHE
    if _DEPTH_PIPELINE is None:
        return
    try:
        mdl = getattr(_DEPTH_PIPELINE, "model", None)
        if isinstance(mdl, torch.nn.Module):
            mdl.cpu()
    except Exception:
        pass
    del _DEPTH_PIPELINE
    _DEPTH_PIPELINE = None
    _DEPTH_MAP_CACHE.clear()
    _release_ram()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  [Depth] Released from GPU")


def _free_all_gpu_models_except(*keep_names: str):
    """Free every loaded GPU-resident singleton EXCEPT those named in
    `keep_names` (one of: 'sam3', 'flux', 'pix2gestalt', 'lama',
    'controlnet', 'aisformer', 'clip', 'gdino', 'depth').

    Used as a hard barrier before loading a heavy model (Flux's 12B-param
    transformer in particular) — clears CUDA caches so the new model can
    grab its full peak allocation without fighting fragmentation.
    """
    keep = set(keep_names)
    if "sam3"        not in keep: _free_sam3()
    if "flux"        not in keep: _free_flux_fill()
    if "pix2gestalt" not in keep: _free_pix2gestalt()
    if "lama"        not in keep: _free_lama()
    if "controlnet"  not in keep: _free_controlnet()
    if "aisformer"   not in keep: _free_aisformer()
    if "clip"        not in keep: _free_clip()
    if "gdino"       not in keep: _free_grounding_dino()
    if "depth"       not in keep: _free_depth_pipeline()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _estimate_depth(image_path: str) -> Optional[np.ndarray]:
    """Return a H×W float32 predicted-depth map (higher = closer).

    Cached per image_path so we only pay the inference cost once.
    """
    if image_path in _DEPTH_MAP_CACHE:
        return _DEPTH_MAP_CACHE[image_path]
    pipe = _get_depth_pipeline()
    if pipe is None:
        return None
    try:
        pil = Image.open(image_path).convert("RGB")
        out = pipe(pil)
        # Most HF depth pipelines return {"depth": PIL, "predicted_depth": Tensor}.
        # PIL .depth is 8-bit normalised; predicted_depth is float — prefer the
        # float tensor when available for finer comparisons.
        depth = out.get("predicted_depth")
        if depth is not None:
            arr = depth.squeeze().detach().to("cpu").float().numpy()
        else:
            arr = np.array(out["depth"], dtype=np.float32)
        # Resize to image dimensions if pipeline downsampled.
        w, h = pil.size
        if arr.shape != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
        _DEPTH_MAP_CACHE[image_path] = arr
        print(f"  [Depth] map computed for {Path(image_path).name} "
              f"(shape={arr.shape}, range=[{arr.min():.2f}, {arr.max():.2f}])")
        return arr
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [Depth] inference failed: {exc!r}")
        return None


def _depth_rank_masks(image_path: str, masks: list) -> Optional[list]:
    """Rank a list of binary masks by depth (closest first).

    Returns a list of {'index': i, 'median_depth': float, 'mean_depth': float}
    sorted descending by median_depth (closer to camera first, since the
    pipeline returns inverse depth).  Returns None when depth is unavailable
    or any mask is empty.
    """
    if not masks or len(masks) < 2:
        return None
    depth = _estimate_depth(image_path)
    if depth is None:
        return None
    H, W = depth.shape
    rows = []
    for i, m in enumerate(masks):
        mm = (m > 0)
        if mm.shape != (H, W):
            mm = cv2.resize(mm.astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        if not mm.any():
            return None
        vals = depth[mm]
        rows.append({
            "index": i,
            "median_depth": float(np.median(vals)),
            "mean_depth":   float(np.mean(vals)),
            "n_px":         int(mm.sum()),
        })
    rows.sort(key=lambda r: r["median_depth"], reverse=True)
    return rows


def _get_sam3():
    global _SAM3_PIPE
    if _SAM3_PIPE is None:
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError("transformers is not installed. Run: pip install transformers")
        print(f"  [SAM3] Loading pipeline ({config.SAM3_MODEL_ID})…")
        device_id = 0 if DEVICE == "cuda" else -1
        _release_ram()
        _SAM3_PIPE = hf_pipeline(
            "mask-generation",
            model=config.SAM3_MODEL_ID,
            device=device_id,
            model_kwargs={"low_cpu_mem_usage": True},
        )
    return _SAM3_PIPE


def _free_sam3():
    """Move every SAM3 pipeline component off GPU then clear cache."""
    global _SAM3_PIPE
    if _SAM3_PIPE is None:
        return
    import gc
    # Walk all pipeline attributes and move any nn.Module to CPU.
    # Using isinstance(_, torch.nn.Module) is more reliable than duck-typing.
    for attr in list(vars(_SAM3_PIPE).keys()):
        component = getattr(_SAM3_PIPE, attr, None)
        if isinstance(component, torch.nn.Module):
            try:
                component.cpu()
            except Exception:
                pass
    # Fallback: if the pipeline itself is an nn.Module (some wrappers are)
    if isinstance(_SAM3_PIPE, torch.nn.Module):
        try:
            _SAM3_PIPE.cpu()
        except Exception:
            pass
    del _SAM3_PIPE
    _SAM3_PIPE = None
    _release_ram()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  [SAM3] Released from GPU")


def _save_sam3_features(pipe, pil_img, feat_path: Path) -> None:
    """Extra image-encoder pass to save SAM3 multi-scale features for AISFormer.

    Runs only when AISFORMER_ENABLED=True and the file doesn't already exist.
    Non-fatal: any failure is caught and printed.
    """
    if feat_path.exists():
        return
    try:
        model     = pipe.model
        processor = pipe.image_processor
        with torch.no_grad():
            inputs = processor(images=pil_img, return_tensors="pt")
            px     = inputs["pixel_values"].to(DEVICE)
            result = model.get_image_embeddings(px)

        # Normalise to a flat list of NCHW tensors — handles list, tuple, or
        # dataclass-style outputs (e.g. Sam2ImageEncoderOutput).
        if isinstance(result, (list, tuple)):
            tensors = [f for f in result if isinstance(f, torch.Tensor)]
        elif isinstance(result, torch.Tensor):
            tensors = [result]
        elif hasattr(result, "__dict__"):
            tensors = [v for v in vars(result).values() if isinstance(v, torch.Tensor)]
        else:
            tensors = []

        # SAM2/SAM3 may return NHWC tensors — convert to NCHW for Conv2d
        nchw = []
        for t in tensors:
            if t.ndim == 4:
                # NHWC heuristic: spatial dims (dim 1,2) larger than channel dim (dim 3)
                if t.shape[1] > t.shape[3] and t.shape[2] > t.shape[3]:
                    t = t.permute(0, 3, 1, 2).contiguous()
            nchw.append(t.cpu())

        if not nchw:
            print("  [SAM3] No tensor features found — skipping AISFormer feature save")
            return

        torch.save(nchw, str(feat_path))
        print(f"  [SAM3] Features ({len(nchw)} tensors) → {feat_path.name}")
        for i, f in enumerate(nchw):
            print(f"    [{i}] shape={tuple(f.shape)}")
    except Exception as exc:
        print(f"  [SAM3] Feature extraction failed (non-fatal): {exc}")


def _free_pix2gestalt():
    """Move pix2gestalt off GPU and clear cache."""
    global _PIX2GESTALT_MODEL
    if _PIX2GESTALT_MODEL is not None:
        try:
            _PIX2GESTALT_MODEL.cpu()
        except Exception:
            pass
        del _PIX2GESTALT_MODEL
        _PIX2GESTALT_MODEL = None
        _release_ram()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("  [pix2gestalt] Released from GPU")


def _get_lama():
    global _LAMA_MODEL
    if _LAMA_MODEL is None:
        try:
            from simple_lama_inpainting import SimpleLama
        except ImportError:
            raise ImportError("simple-lama-inpainting not installed. Run: pip install simple-lama-inpainting")
        lama_device = config.LAMA_DEVICE or DEVICE
        print(f"  [LaMa] Loading model (device={lama_device})…")
        _LAMA_MODEL = SimpleLama(device=lama_device)
    return _LAMA_MODEL


def _free_lama():
    global _LAMA_MODEL
    if _LAMA_MODEL is not None:
        try:
            if hasattr(_LAMA_MODEL, "model") and isinstance(_LAMA_MODEL.model, torch.nn.Module):
                _LAMA_MODEL.model.cpu()
        except Exception:
            pass
        del _LAMA_MODEL
        _LAMA_MODEL = None
        _release_ram()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("  [LaMa] Released")


def _get_controlnet_pipe():
    global _CONTROLNET_PIPE
    if _CONTROLNET_PIPE is None:
        try:
            from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
        except ImportError:
            raise ImportError("diffusers not installed. Run: pip install diffusers transformers accelerate")
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        # low_cpu_mem_usage=True loads weights layer-by-layer via accelerate,
        # keeping CPU RAM peak to roughly one layer at a time instead of the
        # full ~4 GB model buffer.
        kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
        print(f"  [ControlNet] Loading {config.CONTROLNET_MODEL_ID}…")
        _release_ram()
        controlnet = ControlNetModel.from_pretrained(config.CONTROLNET_MODEL_ID, **kwargs)
        print(f"  [ControlNet] Loading SD base {config.SD_INPAINT_MODEL_ID}…")
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            config.SD_INPAINT_MODEL_ID, controlnet=controlnet, **kwargs,
        )
        pipe = pipe.to(DEVICE)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        pipe.safety_checker = None
        _CONTROLNET_PIPE = pipe
    return _CONTROLNET_PIPE


def _free_controlnet():
    global _CONTROLNET_PIPE
    if _CONTROLNET_PIPE is not None:
        try:
            _CONTROLNET_PIPE.to("cpu")
        except Exception:
            pass
        del _CONTROLNET_PIPE
        _CONTROLNET_PIPE = None
        _release_ram()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("  [ControlNet] Released from GPU")


# ── Flux-Fill inpainter (optional alternative to ControlNet+SD-1.5) ───────────

def _get_flux_fill_pipe():
    """Lazy-load FluxFillPipeline.

    Flux-Fill is a 12B-param DiT-based inpainter with much stronger anatomy
    priors than SD-1.5+ControlNet.  Drawback: ~16-24 GB VRAM, no UNet so
    MCDS hooks DON'T transfer.

    Returns None when the diffusers version is too old to support
    `FluxFillPipeline` or the model checkpoint can't be loaded — the
    caller falls back to ControlNet-Inpaint in that case.
    """
    global _FLUX_FILL_PIPE
    if _FLUX_FILL_PIPE is not None:
        return _FLUX_FILL_PIPE
    try:
        from diffusers import FluxFillPipeline
    except ImportError as exc:
        print(f"  [Flux-Fill] diffusers does not expose FluxFillPipeline "
              f"({exc!r}) — upgrade diffusers ≥ 0.31 to use this backend")
        return None
    try:
        model_id = config.FLUX_FILL_MODEL_ID
        print(f"  [Flux-Fill] Loading {model_id}…")
        # Hard barrier — free every other GPU-resident singleton (depth,
        # SAM3, controlnet, AISFormer, etc.) so Flux's 12B-param weight
        # load doesn't fight for the last MB. Without this, `cpu_offload`
        # mode (~10 GB peak) OOMs on 12 GB cards at the SAM3→Flux handoff.
        _free_all_gpu_models_except("flux")
        _release_ram()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            free_mib = torch.cuda.mem_get_info()[0] / 1024 ** 2
            print(f"  [Flux-Fill] Pre-load free VRAM: {free_mib:.0f} MiB")
        kwargs: dict = {"torch_dtype": torch.bfloat16}
        # low_cpu_mem_usage streams sharded weights at load time instead of
        # constructing the full bf16 tensors in CPU RAM first → avoids the
        # ~24 GB load-time RAM peak that triggered systemd-oomd on this
        # 14 GB box.
        if bool(getattr(config, "FLUX_FILL_LOW_CPU_MEM_USAGE", False)):
            kwargs["low_cpu_mem_usage"] = True
        pipe = FluxFillPipeline.from_pretrained(model_id, **kwargs)
        # Pick offload mode based on config (priority order):
        #   mmgp       → "GPU Poor" memory manager: smarter lifecycle + adaptive
        #                slicing + async transfers. Takes precedence when on.
        #   group      → block_level group offload, N adjacent blocks resident
        #                + async stream-prefetch of the next group.
        #   sequential → leaf-level, every layer round-trips PCIe per step.
        #                Slowest but lowest VRAM (~6 GB).
        #   model      → whole pipeline on device during inference (~24 GB
        #                for Flux — OOMs on 12 GB cards, only for SD-1.5).
        #   none       → fully on device, requires 24 GB+ headroom (A100).
        use_mmgp  = bool(getattr(config, "USE_MMGP_OFFLOAD", False))
        use_group = bool(getattr(config, "FLUX_FILL_GROUP_OFFLOAD", False))
        use_seq   = bool(getattr(config, "FLUX_FILL_SEQUENTIAL_OFFLOAD", False))
        if use_mmgp:
            try:
                from mmgp import offload as mmgp_offload
                profile_no   = int(getattr(config, "MMGP_PROFILE", 5))
                pinned       = bool(getattr(config, "MMGP_PINNED_MEMORY", False))
                mmgp_offload.profile(pipe, profile_no, pinnedMemory=pinned)
                mode = f"mmgp(profile={profile_no}, pinned={pinned})"
                _FLUX_FILL_PIPE = pipe
                print(f"  [Flux-Fill] Ready (offload_mode={mode})")
                return _FLUX_FILL_PIPE
            except Exception as exc:                          # noqa: BLE001
                print(f"  [Flux-Fill] mmgp setup failed: {exc!r} — falling back to sequential")
                use_seq = True
        if use_group:
            n_blocks   = int(getattr(config, "FLUX_FILL_GROUP_BLOCKS", 4))
            use_stream = bool(getattr(config, "FLUX_FILL_GROUP_USE_STREAM", False))
            disk_path  = getattr(config, "FLUX_FILL_OFFLOAD_DISK_PATH", None)
            go_kwargs: dict = {
                "onload_device":        torch.device(DEVICE),
                "offload_device":       torch.device("cpu"),
                "offload_type":         "block_level",
                "num_blocks_per_group": n_blocks,
                "use_stream":           use_stream,
                "record_stream":        use_stream,
            }
            if disk_path:
                Path(disk_path).mkdir(parents=True, exist_ok=True)
                go_kwargs["offload_to_disk_path"] = disk_path
            pipe.enable_group_offload(**go_kwargs)
            mode = (f"group_offload(blocks={n_blocks}, stream={use_stream}"
                    f"{', disk=' + disk_path if disk_path else ''})")
        elif use_seq:
            pipe.enable_sequential_cpu_offload()
            mode = "sequential_cpu_offload"
        elif getattr(config, "FLUX_FILL_CPU_OFFLOAD", True):
            pipe.enable_model_cpu_offload()
            mode = "model_cpu_offload"
        else:
            pipe = pipe.to(DEVICE)
            mode = "fully_on_device"
        # VAE tiling + slicing — frees ~400 MB during the VAE encode/decode
        # phases. Free win with no quality impact on inpainting at our
        # resolution (≤1280 longest side). Both methods are best-effort:
        # newer diffusers versions have them on the pipeline directly,
        # older ones expose them on pipe.vae.
        try:
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
            elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
        except Exception as exc:                              # noqa: BLE001
            print(f"  [Flux-Fill] VAE tiling/slicing unavailable: {exc!r}")
        _FLUX_FILL_PIPE = pipe
        print(f"  [Flux-Fill] Ready (offload_mode={mode})")
    except Exception as exc:                                      # noqa: BLE001
        print(f"  [Flux-Fill] load failed: {exc!r} — falling back to ControlNet")
        _FLUX_FILL_PIPE = None
    return _FLUX_FILL_PIPE


def _free_flux_fill():
    global _FLUX_FILL_PIPE
    if _FLUX_FILL_PIPE is not None:
        try:
            _FLUX_FILL_PIPE.to("cpu")
        except Exception:
            pass
        del _FLUX_FILL_PIPE
        _FLUX_FILL_PIPE = None
        _release_ram()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("  [Flux-Fill] Released from GPU")


# ── AISFormer — SAM3 backbone + amodal head ───────────────────────────────────

def _get_aisformer(feat_path: Path):
    """Lazy-load AISFormerHead.  Infers feat_channels from saved SAM3 features.
    Returns None when AISFORMER_CKPT is unset (no inference, features saved only).
    """
    global _AISFORMER
    if _AISFORMER is not None:
        return _AISFORMER
    if not getattr(config, "AISFORMER_CKPT", ""):
        return None  # no checkpoint — feature extraction only

    try:
        features = torch.load(str(feat_path), map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"  [AISFormer] Cannot load features from {feat_path}: {exc}")
        return None

    if not isinstance(features, list) or not features:
        print("  [AISFormer] Empty or invalid feature file — skipping")
        return None

    feat_channels = [f.shape[1] for f in features if f.ndim == 4]
    if not feat_channels:
        print("  [AISFormer] No 4-D feature tensors found — skipping")
        return None

    from aisformer.model import AISFormerHead
    head = AISFormerHead(
        feat_channels    = feat_channels,
        hidden_dim       = config.AISFORMER_HIDDEN_DIM,
        num_heads        = config.AISFORMER_NUM_HEADS,
        num_decoder_layers = config.AISFORMER_LAYERS,
    )

    ckpt = Path(config.AISFORMER_CKPT)
    if ckpt.exists():
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        head.load_state_dict(state)
        print(f"  [AISFormer] Checkpoint loaded: {ckpt.name}")
    else:
        print(f"  [AISFormer] WARNING: checkpoint not found at {ckpt} — skipping inference")
        return None

    _AISFORMER = head.eval().to(DEVICE)
    return _AISFORMER


def _free_aisformer():
    global _AISFORMER
    if _AISFORMER is not None:
        try:
            _AISFORMER.cpu()
        except Exception:
            pass
        del _AISFORMER
        _AISFORMER = None
        _release_ram()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("  [AISFormer] Released from GPU")


# ── Class-label noun extraction (for CLIP / Grounding DINO prompts) ───────────
# GPT often returns long descriptive class names like "large pale weathered
# wooden log/stump in front of the bear".  CLIP text-image scoring works far
# better with 1–3 word labels ("a log", "wood"), so we strip qualifiers down
# to the head noun(s).

_QUALIFIER_WORDS = {
    "large", "small", "big", "huge", "tiny", "short", "long", "tall",
    "thin", "thick", "wide", "narrow", "round", "rounded", "square",
    "pale", "dark", "light", "bright", "dull", "weathered", "old", "new",
    "clean", "dirty", "rough", "smooth", "shiny", "matte",
    "horizontal", "vertical", "front", "back", "side", "main", "primary",
    "foreground", "background", "central", "centre", "center",
    # colours
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "black", "white", "grey", "gray", "brown", "tan", "beige", "cream",
    "golden", "silver", "amber", "ivory",
    # texture words
    "wooden", "metallic", "plastic", "glass", "leather", "fabric",
    "soft", "hard", "fluffy", "shaggy",
    # filler
    "the", "a", "an", "of", "in", "on", "with", "and", "or",
    "very", "quite", "rather", "somewhat",
}

_PHRASE_BREAKERS = re.compile(
    r"\b(?:"
    r"in front of|behind|covered by|partially|"
    r"running (?:across|along|through)|"
    r"lying (?:on|under|across)|"
    r"draped (?:over|across)|"
    r"resting (?:on|against)|"
    r"protruding|extending|sticking|standing|"
    r"covering|blocking|hiding|"
    r"across the|along the|near the"
    r")\b",
    re.I,
)


def _extract_short_label(phrase: str) -> str:
    """Reduce a long descriptive phrase to its head noun(s).

    Examples:
      "large pale weathered wooden log/stump in front of the bear" → "log"
      "horizontal driftwood log running across the bottom"        → "driftwood log"
      "brown bear (Ursus arctos)"                                  → "brown bear"
    """
    if not phrase:
        return ""
    # Drop parenthesised qualifiers like "(Ursus arctos)"
    s = re.sub(r"\([^)]*\)", "", phrase).strip()
    # Cut at relational phrase breakers ("in front of …" etc.)
    s = _PHRASE_BREAKERS.split(s)[0].strip().rstrip(",")
    # Slash-separated alternatives → take the first option ("log/stump" → "log")
    s = s.split("/")[0]
    # Tokenise + drop qualifiers
    tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z\-]*", s)
              if t.lower() not in _QUALIFIER_WORDS]
    if not tokens:
        # Fallback: keep last 1-2 words of the original phrase
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]*", phrase)[-2:]
    # Keep at most the last 2 head words ("driftwood log" or just "log")
    return " ".join(tokens[-2:]).lower()


# ── CLIP — vision-grounded segment labeling (CVPR'25 amodal-style) ────────────

def _get_clip():
    """Lazy-load OpenAI CLIP. Falls back to (None, None) if the package isn't
    importable so the rest of the pipeline can still run."""
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS
    try:
        import clip as openai_clip
    except ImportError:
        print("  [CLIP] openai/CLIP not installed — skipping CLIP segment labeling")
        return None, None
    print(f"  [CLIP] Loading {config.CLIP_MODEL_NAME}…")
    _release_ram()
    try:
        _CLIP_MODEL, _CLIP_PREPROCESS = openai_clip.load(
            config.CLIP_MODEL_NAME, device=DEVICE,
        )
        _CLIP_MODEL.eval()
    except Exception as exc:
        print(f"  [CLIP] Load failed (non-fatal): {exc}")
        _CLIP_MODEL = None
        _CLIP_PREPROCESS = None
    return _CLIP_MODEL, _CLIP_PREPROCESS


def _free_clip():
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if _CLIP_MODEL is not None:
        try:
            _CLIP_MODEL.cpu()
        except Exception:
            pass
        _CLIP_MODEL = None
        _CLIP_PREPROCESS = None
        _release_ram()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("  [CLIP] Released")


def _clip_label_segments(
    image_rgb: np.ndarray,
    segments: list,
    text_labels: list,
) -> list:
    """For each SAM3 segment: crop its bbox (with small padding), grey out the
    non-segment pixels, run CLIP image-text similarity against text_labels, and
    return the best-matching label per segment.

    Returns list of {"id": int, "label": str | None, "score": float, "scores": dict}.
    """
    try:
        import clip as openai_clip
    except ImportError:
        return [{"id": s["id"], "label": None, "score": 0.0, "scores": {}} for s in segments]

    model, preprocess = _get_clip()
    if model is None:
        return [{"id": s["id"], "label": None, "score": 0.0, "scores": {}} for s in segments]

    # Encode the text labels once
    prompts = [f"a photo of {t}" if t else "a photo" for t in text_labels]
    text_tokens = openai_clip.tokenize(prompts).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    h, w = image_rgb.shape[:2]
    out = []
    for seg in segments:
        x1, y1, x2, y2 = seg["bbox"]
        pad = max(4, int(0.05 * max(x2 - x1, y2 - y1)))
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad + 1), min(h, y2 + pad + 1)
        crop  = image_rgb[cy1:cy2, cx1:cx2].copy()
        smask = seg["mask"][cy1:cy2, cx1:cx2]
        # Grey out non-segment pixels so CLIP focuses on the segment content
        crop[smask == 0] = 128

        try:
            pil = Image.fromarray(crop)
            inp = preprocess(pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feats = model.encode_image(inp)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                sim   = (100.0 * feats @ text_features.T).softmax(dim=-1)[0]
            scores  = {text_labels[i]: float(sim[i].item()) for i in range(len(text_labels))}
            best_i  = int(sim.argmax().item())
            out.append({
                "id":     seg["id"],
                "label":  text_labels[best_i],
                "score":  float(sim[best_i].item()),
                "scores": scores,
            })
        except Exception as exc:
            print(f"  [CLIP] Segment {seg['id']} scoring failed: {exc}")
            out.append({"id": seg["id"], "label": None, "score": 0.0, "scores": {}})

    _free_clip()
    return out


def _clip_grid_locate(
    image_rgb: np.ndarray,
    text_label: str,
    grid_n: int = 16,
    patch_overlap: float = 0.25,
) -> "tuple[np.ndarray, list]":
    """
    Score every grid cell of the image with CLIP against `text_label` and the
    standard distractor classes.  Returns:
      heatmap : (grid_n, grid_n) float — softmax probability of `text_label`
      hits    : list of (row, col, score, cx, cy) sorted by score desc

    Used as a fallback when SAM3 auto-segment missed the occluder entirely:
    the highest-scoring grid cell becomes a SAM3 point-prompt anchor that
    forces SAM3 to produce a mask at that location.
    """
    try:
        import clip as openai_clip
    except ImportError:
        return np.zeros((grid_n, grid_n), dtype=np.float32), []

    model, preprocess = _get_clip()
    if model is None:
        return np.zeros((grid_n, grid_n), dtype=np.float32), []

    h, w = image_rgb.shape[:2]
    cell_h = h / grid_n
    cell_w = w / grid_n
    overlap_h = int(cell_h * patch_overlap)
    overlap_w = int(cell_w * patch_overlap)

    distractors = ["background", "wall", "sky", "ground", "animal fur"]
    prompts = [f"a photo of {text_label}"] + [f"a photo of {d}" for d in distractors]
    text_tokens = openai_clip.tokenize(prompts).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    heatmap = np.zeros((grid_n, grid_n), dtype=np.float32)
    hits    = []

    for r in range(grid_n):
        for c in range(grid_n):
            y1 = max(0, int(r * cell_h) - overlap_h)
            y2 = min(h, int((r + 1) * cell_h) + overlap_h)
            x1 = max(0, int(c * cell_w) - overlap_w)
            x2 = min(w, int((c + 1) * cell_w) + overlap_w)
            if y2 - y1 < 16 or x2 - x1 < 16:
                continue
            patch = image_rgb[y1:y2, x1:x2]
            try:
                pil = Image.fromarray(patch)
                inp = preprocess(pil).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    feats = model.encode_image(inp)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    sim   = (100.0 * feats @ text_features.T).softmax(dim=-1)[0]
                score = float(sim[0].item())
                heatmap[r, c] = score
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                hits.append((r, c, score, cx, cy))
            except Exception:
                continue

    hits.sort(key=lambda t: t[2], reverse=True)
    _free_clip()
    return heatmap, hits


def _save_clip_grid_viz(
    image_bgr: np.ndarray,
    heatmap: np.ndarray,
    out_path: Path,
) -> None:
    """Overlay a CLIP-on-grid heatmap on the image and save."""
    if heatmap.size == 0 or heatmap.max() <= 0:
        return
    h, w = image_bgr.shape[:2]
    norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-9)
    norm_8u = (norm * 255).astype(np.uint8)
    big = cv2.resize(norm_8u, (w, h), interpolation=cv2.INTER_NEAREST)
    color_map = cv2.applyColorMap(big, cv2.COLORMAP_JET)
    blend = cv2.addWeighted(image_bgr, 0.55, color_map, 0.45, 0)
    cv2.imwrite(str(out_path), blend)


# ── Grounding DINO (open-vocab text → bbox via HF transformers) ───────────────

def _get_grounding_dino():
    """Lazy-load IDEA-Research/grounding-dino-tiny via HF transformers."""
    global _GDINO_MODEL, _GDINO_PROCESSOR
    if _GDINO_MODEL is not None:
        return _GDINO_MODEL, _GDINO_PROCESSOR
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except ImportError:
        print("  [GDINO] transformers without zero-shot detection — skipping")
        return None, None
    print(f"  [GDINO] Loading {config.GROUNDING_DINO_MODEL_ID}…")
    _release_ram()
    try:
        _GDINO_PROCESSOR = AutoProcessor.from_pretrained(config.GROUNDING_DINO_MODEL_ID)
        _GDINO_MODEL     = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(config.GROUNDING_DINO_MODEL_ID)
            .to(DEVICE)
            .eval()
        )
    except Exception as exc:
        print(f"  [GDINO] Load failed (non-fatal): {exc}")
        _GDINO_MODEL = None
        _GDINO_PROCESSOR = None
    return _GDINO_MODEL, _GDINO_PROCESSOR


def _free_grounding_dino():
    global _GDINO_MODEL, _GDINO_PROCESSOR
    if _GDINO_MODEL is not None:
        try:
            _GDINO_MODEL.cpu()
        except Exception:
            pass
        _GDINO_MODEL = None
        _GDINO_PROCESSOR = None
        _release_ram()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("  [GDINO] Released")


def _grounding_dino_detect(
    image_pil: Image.Image,
    text_query: str,
) -> list:
    """Run Grounding DINO with `text_query`. Returns list of
    {"box": [x1,y1,x2,y2], "score": float, "label": str}, sorted by score desc.
    Returns [] on any failure so the caller can fall through to other paths."""
    model, processor = _get_grounding_dino()
    if model is None or processor is None:
        return []

    # GDINO expects a period-terminated, lowercased prompt
    text = text_query.strip().lower().rstrip(".") + "."
    try:
        inputs = processor(images=image_pil, text=text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        # API name varies between transformers versions; try the modern then legacy
        post = getattr(processor, "post_process_grounded_object_detection", None) \
               or getattr(processor, "post_process_object_detection", None)
        if post is None:
            print("  [GDINO] Processor has no post-process method")
            return []
        # Modern API takes input_ids; legacy doesn't
        try:
            results = post(
                outputs,
                inputs.input_ids,
                box_threshold=config.GROUNDING_DINO_BOX_THRESH,
                text_threshold=config.GROUNDING_DINO_TEXT_THRESH,
                target_sizes=[image_pil.size[::-1]],
            )[0]
        except TypeError:
            results = post(
                outputs,
                threshold=config.GROUNDING_DINO_BOX_THRESH,
                target_sizes=[image_pil.size[::-1]],
            )[0]
        boxes  = results.get("boxes",  torch.zeros((0, 4)))
        scores = results.get("scores", torch.zeros((0,)))
        labels = results.get("labels", [""] * len(boxes))
        out = []
        for i in range(len(boxes)):
            box = boxes[i].detach().cpu().numpy().tolist()
            s   = float(scores[i].detach().cpu().item())
            lbl = labels[i] if isinstance(labels[i], str) else str(labels[i])
            out.append({"box": [int(v) for v in box], "score": s, "label": lbl})
        out.sort(key=lambda d: d["score"], reverse=True)
        print(f"  [GDINO] '{text_query}' → {len(out)} detections "
              f"(top score={out[0]['score']:.2f})" if out
              else f"  [GDINO] '{text_query}' → 0 detections")
        return out
    except Exception as exc:
        print(f"  [GDINO] Detection failed (non-fatal): {exc}")
        return []
    finally:
        _free_grounding_dino()


# ── PSALM (referring-expression segmentation, subprocess into separate venv) ──
# PSALM uses a custom transformers fork that conflicts with SAM3/diffusers, so
# it lives in its own venv at amodal_test3/.psalm-venv. We invoke a small CLI
# script that loads the model, runs inference, writes the mask, and exits —
# the subprocess shutdown automatically releases all VRAM, so no _free_psalm()
# bookkeeping is needed.

def _psalm_referring_seg(
    image_path: str,
    text: str,
    out_dir: Path,
    out_filename: str = "psalm_mask.png",
) -> Optional[np.ndarray]:
    """Run PSALM referring-expression segmentation in its own venv.

    Returns a binary mask (np.uint8 0/1) at the input image resolution, or None
    when PSALM is disabled, missing, or fails. The mask file is also persisted
    to out_dir/<out_filename> for inspection.
    """
    if not getattr(config, "USE_PSALM", False):
        return None
    if not text or not text.strip():
        return None

    venv_py = Path(config.PSALM_VENV_PYTHON)
    script  = Path(config.PSALM_INFER_SCRIPT)
    repo    = Path(config.PSALM_REPO_DIR)
    ckpt    = Path(config.PSALM_CHECKPOINT_DIR)

    for label, p in (("venv", venv_py), ("script", script), ("repo", repo), ("ckpt", ckpt)):
        if not p.exists():
            print(f"  [PSALM] {label} missing at {p} — skipping (run setup_psalm.sh)")
            return None

    mask_path = out_dir / out_filename
    cmd = [
        str(venv_py), str(script),
        "--image",   str(image_path),
        "--text",    text,
        "--repo-dir", str(repo),
        "--ckpt-dir", str(ckpt),
        "--out",     str(mask_path),
    ]

    print(f"  [PSALM] subprocess on '{text[:60]}{'…' if len(text) > 60 else ''}'")
    import subprocess
    try:
        result = subprocess.run(
            cmd, timeout=config.PSALM_TIMEOUT,
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        print(f"  [PSALM] timeout after {config.PSALM_TIMEOUT}s — skipping")
        return None
    except Exception as exc:
        print(f"  [PSALM] subprocess launch failed: {exc}")
        return None

    if result.returncode != 0:
        print(f"  [PSALM] non-zero exit ({result.returncode}). Last stderr:")
        tail = (result.stderr or "(empty)").strip().splitlines()[-12:]
        for line in tail:
            print(f"    {line}")
        return None

    if not mask_path.exists():
        print(f"  [PSALM] no mask written at {mask_path}")
        return None

    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        print(f"  [PSALM] could not read mask file {mask_path}")
        return None
    binary = (raw > 127).astype(np.uint8)
    area = int(binary.sum())
    if area < int(getattr(config, "PSALM_MIN_AREA", 0)):
        print(f"  [PSALM] mask too small ({area} px < {config.PSALM_MIN_AREA}) — dropping")
        return None
    print(f"  [PSALM] mask area: {area} px → {mask_path.name}")
    # Subprocess has exited — its VRAM is already released.
    return binary


# ── Multi-model occluder-mask fusion ──────────────────────────────────────────

def _fuse_masks(
    candidates: list,
    mode: str = "majority",
    min_agree: int = 2,
    priority: Optional[list] = None,
) -> tuple:
    """Fuse a list of (name, binary_mask uint8 0/1) candidates.

    Returns (fused_mask uint8 0/1, used_names list[str]).
    Empty / None masks are filtered out before fusion.

    Modes:
        majority      — pixel kept when ≥ min_agree candidates agree.
        union         — OR.
        intersection  — AND.
        priority      — first non-empty candidate from `priority` order.
    """
    valid = [(n, m) for n, m in candidates
             if m is not None and isinstance(m, np.ndarray) and int(m.sum()) > 0]

    if not valid:
        # All empty — return zero mask shaped like first candidate, or 1×1
        ref = next((m for _, m in candidates if isinstance(m, np.ndarray)), None)
        z = np.zeros_like(ref) if ref is not None else np.zeros((1, 1), dtype=np.uint8)
        return z, []

    H, W = valid[0][1].shape[:2]
    aligned = []
    for n, m in valid:
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        aligned.append((n, (m > 0).astype(np.uint8)))

    if mode == "priority":
        order = priority or [n for n, _ in aligned]
        for name in order:
            for n, m in aligned:
                if n == name:
                    return m, [n]
        n0, m0 = aligned[0]
        return m0, [n0]

    stack = np.stack([m for _, m in aligned], axis=0)   # (N, H, W)
    n_sources = stack.shape[0]
    if mode == "union":
        fused = (stack.sum(axis=0) >= 1).astype(np.uint8)
    elif mode == "intersection":
        fused = (stack.sum(axis=0) == n_sources).astype(np.uint8)
    else:  # majority
        thr = max(1, min(int(min_agree), n_sources))
        fused = (stack.sum(axis=0) >= thr).astype(np.uint8)

    return fused, [n for n, _ in aligned]


def _save_clip_label_viz(
    image_bgr: np.ndarray,
    segments: list,
    seg_by_id: dict,
    seg_labels: list,
    target_class: str,
    occluder_class: str,
    out_path: Path,
) -> None:
    """Save a coloured visualisation of CLIP segment labels.
    target = green, occluder = red, background = grey, other = yellow."""
    palette = {
        target_class:    (0, 255, 0),
        occluder_class:  (0, 0, 255),
        "background":    (128, 128, 128),
        "other object":  (0, 200, 200),
    }
    viz     = image_bgr.copy()
    overlay = np.zeros_like(viz)
    for sl in seg_labels:
        sid = sl["id"]
        if sid not in seg_by_id or sl["label"] is None:
            continue
        colour = palette.get(sl["label"], (200, 200, 200))
        overlay[seg_by_id[sid]["mask"] == 1] = colour
    viz = cv2.addWeighted(viz, 0.55, overlay, 0.45, 0)
    for sl in seg_labels:
        sid = sl["id"]
        if sid not in seg_by_id:
            continue
        seg_mask = seg_by_id[sid]["mask"]
        M = cv2.moments(seg_mask)
        if M["m00"] <= 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        lbl = (sl["label"] or "?")[:10]
        text = f"{sid}:{lbl}({sl['score']:.2f})"
        cv2.putText(viz, text, (cx - 36, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(viz, text, (cx - 36, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), viz)


def _try_run_aisformer(
    feat_path: Path,
    visible_mask_np: np.ndarray,
    out_hw: tuple,
) -> Optional[np.ndarray]:
    """Run AISFormer if a checkpoint is configured and features exist.

    Returns a uint8 binary mask of shape out_hw, or None on any failure.
    """
    if not getattr(config, "AISFORMER_ENABLED", False):
        return None
    if not getattr(config, "AISFORMER_CKPT", ""):
        return None
    if not feat_path.exists():
        print("  [AISFormer] No SAM3 features found — falling back to pix2gestalt threshold")
        return None

    head = _get_aisformer(feat_path)
    if head is None:
        return None

    try:
        features = torch.load(str(feat_path), map_location="cpu", weights_only=True)
        features = [f.to(DEVICE) for f in features if isinstance(f, torch.Tensor) and f.ndim == 4]
        if not features:
            return None

        vm = (torch.from_numpy(visible_mask_np)
              .float().unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
              .to(DEVICE))

        with torch.no_grad():
            logits = head(features, vm, out_hw=out_hw)

        prob = torch.sigmoid(logits[0, 0]).cpu().numpy()
        mask = (prob > config.AISFORMER_THRESHOLD).astype(np.uint8)
        print(f"  [AISFormer] Amodal mask: {mask.sum()} px at threshold {config.AISFORMER_THRESHOLD}")
        return mask

    except Exception as exc:
        print(f"  [AISFormer] Inference failed (non-fatal): {exc}")
        return None
    finally:
        _free_aisformer()


def _get_pix2gestalt():
    global _PIX2GESTALT_MODEL
    if _PIX2GESTALT_MODEL is None:
        repo = str(config.PIX2GESTALT_REPO)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        try:
            from omegaconf import OmegaConf
            from inference import load_model_from_config
        except ImportError as e:
            raise ImportError(
                f"pix2gestalt dependencies missing: {e}\n"
                "Clone https://github.com/cvlab-columbia/pix2gestalt, install "
                "taming-transformers + CLIP, then set PIX2GESTALT_REPO in config.py."
            )
        # taming-transformers editable install leaves MAPPING empty on some uv versions;
        # add it to sys.path directly so `import taming` works from inside ldm code.
        taming_path = str(BASE_DIR / "taming-transformers")
        if taming_path not in sys.path:
            sys.path.insert(0, taming_path)

        cfg = OmegaConf.load(config.PIX2GESTALT_CFG)

        # ── Pre-baked checkpoint (created on first run) ────────────────────────
        # Original checkpoint: 8.58 GB FP32 (5.15 GB model + 3.44 GB EMA).
        # Loading 8.58 GB directly to a 10 GB-capped GPU always OOMs.
        # We bake EMA into the model weights once, strip model_ema, convert to
        # FP16, and save a 2.57 GB file.  Every subsequent run loads this file
        # straight to GPU in seconds with no conversion or EMA overhead.
        ckpt_path  = Path(config.PIX2GESTALT_CKPT)
        # New filename so the prior EMA-stripped bake doesn't get reused.
        baked_path = ckpt_path.with_name(ckpt_path.stem + "_baked_ema_fp16.ckpt")

        if not baked_path.exists():
            print(f"  [pix2gestalt] Pre-baking checkpoint (one-time, ~30 s)…")
            # mmap=True: state_dict tensors are file-backed — no 8.58 GB heap spike.
            # Apply EMA to the base weights at the state-dict level (no model
            # instantiation, so no 8.58 GB RAM spike), then strip the EMA block
            # and convert to FP16.  LitEma stores params under
            # model_ema.<dot-stripped-name>, so we walk those, find the matching
            # base key, and overwrite it.  Published pix2gestalt quality numbers
            # use EMA weights — dropping them was the prior regression.
            _release_ram()
            raw = torch.load(str(ckpt_path), map_location="cpu",
                             weights_only=False, mmap=True)
            sd = raw["state_dict"]
            n_ema_total = sum(1 for k in sd if k.startswith("model_ema."))

            # LitEma stores keys relative to self.model (e.g. EMA suffix
            # 'diffusion_modelinput_blocks00weight'), but the outer LDM
            # state_dict prefixes its params with 'model.'  (e.g.
            # 'model.diffusion_model.input_blocks.0.0.weight').  We strip that
            # 'model.' before dot-stripping when building the lookup table —
            # the previous run logged "Applied EMA: 0/688 keys" because of
            # this mismatch.
            base_keys = [k for k in sd.keys() if not k.startswith("model_ema.")]
            dotless_to_full: dict = {}
            for k in base_keys:
                if k.startswith("model."):
                    dotless = k[len("model."):].replace(".", "")
                else:
                    dotless = k.replace(".", "")
                dotless_to_full[dotless] = k
            ema_applied = 0
            ema_skipped = 0
            for ema_k in [k for k in sd.keys() if k.startswith("model_ema.")]:
                short = ema_k[len("model_ema."):]
                if short in ("decay", "num_updates"):
                    continue
                target = dotless_to_full.get(short)
                if target is None:
                    ema_skipped += 1
                    continue
                sd[target] = sd[ema_k]
                ema_applied += 1
            print(f"  [pix2gestalt] Applied EMA: {ema_applied}/{n_ema_total} keys "
                  f"(skipped {ema_skipped} unmatched), now stripping EMA and "
                  f"converting to FP16…")

            baked_sd = {
                k: v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                for k, v in sd.items()
                if not k.startswith("model_ema.")
            }
            torch.save({"state_dict": baked_sd}, str(baked_path))
            del raw, sd, baked_sd
            _release_ram()
            print(f"  [pix2gestalt] Saved → {baked_path.name} "
                  f"({baked_path.stat().st_size / 1e9:.2f} GB)")

        # ── Load baked checkpoint directly to GPU ──────────────────────────────
        print(f"  [pix2gestalt] Loading baked checkpoint ({baked_path.name})…")
        _release_ram()
        _PIX2GESTALT_MODEL = load_model_from_config(cfg, str(baked_path), "cpu")

        # Baked checkpoint has no EMA — disable scope to skip any ema_scope() calls.
        _PIX2GESTALT_MODEL.use_ema = False
        if hasattr(_PIX2GESTALT_MODEL, "model_ema"):
            del _PIX2GESTALT_MODEL.model_ema
            _PIX2GESTALT_MODEL.model_ema = None  # type: ignore[assignment]

        if DEVICE == "cuda":
            # Baked weights are FP16; move to GPU — no conversion needed.
            _release_ram()
            _PIX2GESTALT_MODEL = _PIX2GESTALT_MODEL.to(DEVICE)
    return _PIX2GESTALT_MODEL


# ── Graph State ───────────────────────────────────────────────────────────────

class State(TypedDict):
    image_path:         str
    target:             str           # optional user hint — empty = auto-detect
    # Occlusion Agent outputs
    occluded_object:    str
    occluder:           str
    what_to_remove:     str
    bbox:               Optional[list]
    boundary_expansion: int
    region_desc:        str
    subject_description: str          # rich description of visible subject (species, colors, posture)
    visible_parts:      str           # body parts currently visible in frame
    missing_parts:      str           # body parts cut off / expected but absent
    frame_cropped:      bool          # True when subject is cut off at image boundary
    expansion_directions: list        # e.g. ["bottom"] or ["right", "bottom"]
    expansion_pixels:   Optional[dict]  # {top, bottom, left, right} pixels to add
    mask_path:              Optional[str]   # binary occluder mask
    visible_mask_path:      Optional[str]   # binary modal mask (visible part of occluded obj)
    hidden_mask_path:       Optional[str]   # binary amodal target region (missing part to generate)
    occluder_removed_path:  Optional[str]
    occluder_viz_path:      Optional[str]
    hidden_polygon:         Optional[list]   # Fix 2: exact polygon of hidden region to inpaint
    # Inpainting Agent outputs
    pix2gestalt_dir:    Optional[str]   # dir holding completed_N.png samples
    output_path:        str
    output_rgba_path:   str
    # Reviewer
    review_score:       float
    review_feedback:    str
    failure_code:       str
    attempt:            int
    mask_retry_count:   int
    best_attempt:       int
    best_score:         float


# ── GPT vision helper ─────────────────────────────────────────────────────────

def _encode(path) -> dict:
    """Encode an image file as a Responses-API `input_image` content item."""
    p    = Path(path)
    mime = "jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "png"
    b64  = base64.b64encode(p.read_bytes()).decode()
    return {
        "type": "input_image",
        "image_url": f"data:image/{mime};base64,{b64}",
        "detail": "high",
    }


# ── Strict JSON schemas ────────────────────────────────────────────────────────

OCCLUSION_SCHEMA = {
    "name": "occlusion_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "occluded_object": {"type": "string"},
            "occluder":        {"type": "string"},
            "what_to_remove":  {"type": "string"},
            # Rich description used by the reviewer to verify correctness
            "subject_description": {"type": "string"},
            "visible_parts":       {"type": "string"},
            "missing_parts":       {"type": "string"},
            # Frame-crop completion fields
            "frame_cropped": {"type": "boolean"},
            "expansion_directions": {
                "type": "array",
                "items": {"type": "string"},
            },
            "expansion_pixels": {
                "type": "object",
                "properties": {
                    "top":    {"type": "integer"},
                    "bottom": {"type": "integer"},
                    "left":   {"type": "integer"},
                    "right":  {"type": "integer"},
                },
                "required": ["top", "bottom", "left", "right"],
                "additionalProperties": False,
            },
            # Classic occlusion fields
            "selected_segment_ids": {
                "type": "array",
                "items": {"type": "integer"}
            },
            "visible_segment_ids": {
                "type": "array",
                "items": {"type": "integer"}
            },
            "polygon_override": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}}
            },
            "visible_polygon_override": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}}
            },
            "hidden_region": {
                "type": "object",
                "properties": {
                    "x1":          {"type": "integer"},
                    "y1":          {"type": "integer"},
                    "x2":          {"type": "integer"},
                    "y2":          {"type": "integer"},
                    "description": {"type": "string"},
                },
                "required": ["x1", "y1", "x2", "y2", "description"],
                "additionalProperties": False,
            },
            "boundary_expansion": {"type": "integer"},
            # Fix 2: precise hidden-region polygon and SAM3 point-prompt coordinates
            "hidden_polygon": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
            },
            "occluder_click": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
                "additionalProperties": False,
            },
            "subject_click": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
                "additionalProperties": False,
            },
        },
        "required": [
            "occluded_object", "occluder", "what_to_remove",
            "subject_description", "visible_parts", "missing_parts",
            "frame_cropped", "expansion_directions", "expansion_pixels",
            "selected_segment_ids", "visible_segment_ids",
            "polygon_override", "visible_polygon_override", "hidden_region", "boundary_expansion",
            "hidden_polygon", "occluder_click", "subject_click",
        ],
        "additionalProperties": False,
    },
}

REVIEWER_SCHEMA = {
    "name": "review_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "score":                    {"type": "number"},
            "feedback":                 {"type": "string"},
            "failure_code":             {"type": "string"},
            "improved_prompt":          {"type": "string"},
            "improved_negative_prompt": {"type": "string"},
        },
        "required": [
            "score", "feedback", "failure_code",
            "improved_prompt", "improved_negative_prompt",
        ],
        "additionalProperties": False,
    },
}


# ── GPT vision call (Responses API + reasoning=xhigh + prompt caching) ────────
#
# OpenAI Responses API replaces chat.completions for reasoning models. Key
# upgrades vs. the old call site:
#   • `reasoning.effort = "high"` — the maximum the server-side `gpt-5`
#     model accepts (the SDK enum lists `xhigh` too but that's rejected by
#     gpt-5 with a 400; reserve xhigh for gpt-5-codex / 5.1+ if upgraded).
#     Gives Agent 1 / amodal-completion / reviewer more chain-of-thought
#     budget so the polygon predictions are less likely to hallucinate
#     anatomy in the wrong spatial direction.
#   • `prompt_cache_key` — static prompt caching. Each logical call site
#     should pass a stable key so the long instruction prefix gets cached
#     across images. Retention set to 24h.
#   • `text.format` with JSON-schema strict mode — same effective contract
#     as the old `response_format={"type": "json_schema", ...}` but using
#     the flat Responses-API shape.
#   • Image content items use {type: "input_image", image_url, detail}.

def gpt_vision(images: list, prompt: str, schema: dict = None,
               max_retries: int = 2, cache_key: str = None) -> dict:
    current_prompt = prompt
    for attempt in range(max_retries + 1):
        content = [{"type": "input_text", "text": current_prompt}] + [_encode(p) for p in images]

        if schema:
            # Existing schemas are {name, schema, strict}. Responses API wants
            # those flat inside text.format.
            text_cfg = {"format": {
                "type":   "json_schema",
                "name":   schema["name"],
                "schema": schema["schema"],
                "strict": schema.get("strict", True),
            }}
        else:
            text_cfg = {"format": {"type": "json_object"}}

        kwargs: dict = {
            "model":     GPT_MODEL,
            "input":     [{"role": "user", "content": content}],
            "reasoning": {"effort": "high", "summary": "concise"},
            "text":      text_cfg,
        }
        if cache_key:
            kwargs["prompt_cache_key"]       = cache_key
            kwargs["prompt_cache_retention"] = "24h"

        try:
            resp = gpt.responses.create(**kwargs)
        except Exception as exc:                              # noqa: BLE001
            if attempt < max_retries:
                print(f"  [GPT] API error (attempt {attempt + 1}/{max_retries + 1}): {exc!r}, retrying…")
                continue
            raise

        # Walk output items to detect refusals (Responses API surfaces
        # refusals as content items with type='refusal' inside a message).
        refusal_msg = None
        for item in (resp.output or []):
            if getattr(item, "type", None) != "message":
                continue
            for c in (getattr(item, "content", None) or []):
                if getattr(c, "type", None) == "refusal":
                    refusal_msg = getattr(c, "refusal", None) or "refused"
                    break
            if refusal_msg:
                break
        if refusal_msg:
            if attempt < max_retries:
                print(f"  [GPT] Refusal (attempt {attempt + 1}/{max_retries + 1}), retrying…")
                current_prompt = f"Please analyze this image technically and objectively. {current_prompt}"
                continue
            raise RuntimeError(f"[GPT] Model refused after {max_retries + 1} attempts: {refusal_msg}")

        raw = (resp.output_text or "").strip()
        if not raw:
            if attempt < max_retries:
                print(f"  [GPT] Empty response (status={resp.status!r}, attempt {attempt + 1}/{max_retries + 1}), retrying…")
                continue
            raise RuntimeError(
                f"[GPT] Empty response (status={resp.status!r}). "
                f"Try increasing GPT_MAX_TOKENS in config.py (currently {config.GPT_MAX_TOKENS})."
            )

        if resp.status == "incomplete":
            reason = getattr(getattr(resp, "incomplete_details", None), "reason", None)
            print(f"  [GPT] WARNING: response incomplete (reason={reason!r}) — partial output:\n{raw[:300]}")

        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            if attempt < max_retries:
                print(f"  [GPT] JSON parse error (attempt {attempt + 1}/{max_retries + 1}), retrying…")
                continue
            raise RuntimeError(f"[GPT] JSON parse error: {e}\nRaw response:\n{raw[:500]}")

    raise RuntimeError("[GPT] Exhausted all retries without a valid response")


# ── SAM3 segment-everything helper ────────────────────────────────────────────

def _sam_segment_all(image_path: str, out_dir: Path) -> tuple:
    """
    Run SAM3 automatic mask generation via the Transformers mask-generation pipeline.
    Returns (segments, viz_path).
    segments: list of {id, mask (H×W uint8), bbox [x1,y1,x2,y2], polygon, area, iou}
    """
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_bgr.shape[:2]

    pipe = _get_sam3()
    pil_img = Image.fromarray(img_rgb)

    print("  [SAM3] Generating masks…")

    def _run_sam3(extra_kwargs: dict) -> tuple:
        """Call SAM3 with the given inference kwargs.  Silently drops kwargs
        the pipeline doesn't accept (older transformers versions)."""
        try:
            out = pipe(pil_img,
                       points_per_batch=config.SAM3_POINTS_PER_BATCH,
                       **extra_kwargs)
        except TypeError:
            # One of the kwargs isn't supported — drop them and retry with
            # only the supported subset, probing each one.
            cleaned = {}
            for k, v in extra_kwargs.items():
                try:
                    pipe(pil_img,
                         points_per_batch=config.SAM3_POINTS_PER_BATCH,
                         **cleaned, **{k: v})
                    cleaned[k] = v
                except TypeError:
                    pass
            out = pipe(pil_img,
                       points_per_batch=config.SAM3_POINTS_PER_BATCH,
                       **cleaned)
        return list(out["masks"]), list(out.get("scores", [1.0] * len(out["masks"])))

    # First try: default pipeline thresholds.
    outputs    = pipe(pil_img, points_per_batch=config.SAM3_POINTS_PER_BATCH)
    raw_masks  = list(outputs["masks"])
    raw_scores = list(outputs.get("scores", [1.0] * len(raw_masks)))
    del outputs

    # Auto-fallback: SAM3's default pred_iou_thresh (~0.88) and
    # stability_score_thresh (~0.95) reject all proposals on small/low-
    # contrast subjects (bee on flower, butterfly on petal).  When the
    # initial call returns 0 raw masks, RE-RUN SAM3 with looser internal
    # thresholds AND a denser grid.  We try two levels of loosening.
    if len(raw_masks) == 0:
        print(f"  [SAM3] 0 raw proposals — retrying with looser thresholds…")
        try:
            raw_masks, raw_scores = _run_sam3({
                "pred_iou_thresh":        0.5,
                "stability_score_thresh": 0.7,
                "points_per_side":        48,
            })
            print(f"  [SAM3] looser-1 → {len(raw_masks)} raw proposals")
        except Exception as exc:                              # noqa: BLE001
            print(f"  [SAM3] looser-1 retry failed: {exc!r}")

    if len(raw_masks) == 0:
        print(f"  [SAM3] still 0 — retrying with very loose thresholds…")
        try:
            raw_masks, raw_scores = _run_sam3({
                "pred_iou_thresh":        0.3,
                "stability_score_thresh": 0.5,
                "points_per_side":        64,
            })
            print(f"  [SAM3] looser-2 → {len(raw_masks)} raw proposals")
        except Exception as exc:                              # noqa: BLE001
            print(f"  [SAM3] looser-2 retry failed: {exc!r}")

    # Extract image encoder features for AISFormer while the model is still on GPU
    if getattr(config, "AISFORMER_ENABLED", False):
        _save_sam3_features(pipe, pil_img, out_dir / "sam3_features.pt")

    del pipe, pil_img                              # now safe to drop model refs

    masks_dir = out_dir / "sam3_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    def _filter_segments(score_thresh: float, min_area: int) -> list:
        out: list = []
        for mask_bool, score in zip(raw_masks, raw_scores):
            mask = (np.asarray(mask_bool) > 0).astype(np.uint8)
            area = int(mask.sum())
            if area < min_area:
                continue
            if float(score) < score_thresh:
                continue

            ys, xs = np.where(mask > 0)
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygon = []
            if contours:
                largest = max(contours, key=cv2.contourArea)
                epsilon = 0.02 * cv2.arcLength(largest, True)
                approx  = cv2.approxPolyDP(largest, epsilon, True)
                polygon = approx.reshape(-1, 2).tolist()

            out.append({
                "id":      len(out),
                "mask":    mask,
                "bbox":    [x1, y1, x2, y2],
                "polygon": polygon,
                "area":    area,
                "iou":     float(score),
            })
        return out

    # First try strict thresholds.
    segments = _filter_segments(config.SAM3_SCORE_THRESH, config.SAM3_MIN_AREA)

    # Auto-fallback: if SAM3 returned 0 segments under strict thresholds,
    # progressively loosen score+min_area cutoffs.  Small / low-contrast
    # subjects (bee on flower, butterfly on petal) routinely fall under
    # score=0.50 / min_area=100 but are still real instances.  Re-filter
    # the SAME raw masks at score=0.30/min_area=30, then 0.15/10.
    if not segments:
        for fallback_score, fallback_area in [(0.30, 30), (0.15, 10)]:
            segments = _filter_segments(fallback_score, fallback_area)
            if segments:
                print(f"  [SAM3] strict thresholds returned 0 — retrying with "
                      f"score≥{fallback_score} / area≥{fallback_area} → "
                      f"{len(segments)} segments")
                break

    # Sort largest → smallest so important objects get lower IDs
    segments.sort(key=lambda s: s["area"], reverse=True)
    for i, s in enumerate(segments):
        s["id"] = i
        cv2.imwrite(str(masks_dir / f"seg_{i:03d}.png"), s["mask"] * 255)

    print(f"  [SAM3] {len(segments)} segments (after area/score filter)")

    # Colour-coded numbered visualisation
    viz = img_bgr.copy()
    rng = np.random.default_rng(42)
    for seg in segments:
        colour  = rng.integers(60, 210, 3).tolist()
        overlay = np.zeros_like(viz)
        overlay[seg["mask"] == 1] = colour
        viz = cv2.addWeighted(viz, 0.65, overlay, 0.35, 0)
        M = cv2.moments(seg["mask"])
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.putText(viz, str(seg["id"]), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(viz, str(seg["id"]), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    viz_path = out_dir / "sam3_segments_viz.png"
    cv2.imwrite(str(viz_path), viz)
    print(f"  [SAM3] Visualisation → {viz_path}")

    _free_sam3()   # release VRAM before pix2gestalt loads
    return segments, viz_path


# ── SAM3 point-prompt helper (Fix 1) ─────────────────────────────────────────

def _sam_segment_targeted(
    image_path: str,
    click_points: list,   # [{"label": "occluder", "x": int, "y": int}, ...]
    out_dir: Path,
) -> list:
    """Run SAM3 with specific click-point prompts to cleanly isolate individual objects.

    Returns list of {"label": str, "mask": H×W uint8, "score": float}.
    Falls back to [] on any failure so the caller can degrade to auto-segment masks.
    """
    if not click_points:
        return []
    try:
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        pipe = _get_sam3()   # reloads if freed; cached after first load

        # One [[x, y]] set per object → one mask per object
        input_points = [[[int(pt["x"]), int(pt["y"])]] for pt in click_points]
        input_labels = [[1]] * len(click_points)

        print(f"  [SAM3] Point-prompt segmentation ({len(click_points)} targets)…")
        outputs = pipe(pil_img, input_points=input_points, input_labels=input_labels)

        masks_raw  = outputs.get("masks",  [])
        scores_raw = outputs.get("scores", [1.0] * len(masks_raw))

        results = []
        for i, (mask_raw, score) in enumerate(zip(masks_raw, scores_raw)):
            label = click_points[i].get("label", f"target_{i}") if i < len(click_points) else f"target_{i}"
            mask  = (np.asarray(mask_raw) > 0).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"targeted_{label}.png"), mask * 255)
            print(f"    '{label}': area={mask.sum()} px  score={score:.3f}")
            results.append({"label": label, "mask": mask, "score": float(score)})

        _free_sam3()
        return results
    except Exception as exc:
        print(f"  [SAM3] Point-prompt segmentation failed (non-fatal): {exc}")
        _free_sam3()
        return []


def _sam_segment_with_boxes(
    image_path: str,
    boxes: list,         # [{"label": str, "box": [x1,y1,x2,y2]}, ...]
    out_dir: Path,
) -> list:
    """Run SAM3 with bbox prompts. Bboxes give SAM3 a much stronger spatial
    constraint than a single point — the resulting mask covers the whole
    object inside the bbox instead of a small fragment around a click point.
    Use this when GroundingDINO returns clean bboxes for the occluder.

    Returns list of {"label": str, "mask": H×W uint8, "score": float}.
    """
    if not boxes:
        return []
    try:
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        pipe = _get_sam3()
        # SAM3 expects [[ [x1,y1,x2,y2] ]] — outer list = batch, inner list = boxes per image
        input_boxes = [[[int(b["box"][0]), int(b["box"][1]),
                         int(b["box"][2]), int(b["box"][3])] for b in boxes]]

        print(f"  [SAM3] Box-prompt segmentation ({len(boxes)} boxes)…")
        try:
            outputs = pipe(pil_img, input_boxes=input_boxes)
        except TypeError:
            # Older HF API uses 'input_bboxes' or doesn't support boxes — fall back
            print("  [SAM3] pipeline doesn't accept input_boxes — falling back to point centers")
            click_points = [
                {"label": b["label"],
                 "x": (b["box"][0] + b["box"][2]) // 2,
                 "y": (b["box"][1] + b["box"][3]) // 2}
                for b in boxes
            ]
            _free_sam3()
            return _sam_segment_targeted(image_path, click_points, out_dir)

        masks_raw  = outputs.get("masks",  [])
        scores_raw = outputs.get("scores", [1.0] * len(masks_raw))

        results = []
        for i, (mask_raw, score) in enumerate(zip(masks_raw, scores_raw)):
            label = boxes[i].get("label", f"bbox_{i}") if i < len(boxes) else f"bbox_{i}"
            mask  = (np.asarray(mask_raw) > 0).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"targeted_box_{label}.png"), mask * 255)
            print(f"    '{label}' (bbox): area={mask.sum()} px  score={score:.3f}")
            results.append({"label": label, "mask": mask, "score": float(score)})

        _free_sam3()
        return results
    except Exception as exc:
        print(f"  [SAM3] Box-prompt segmentation failed (non-fatal): {exc}")
        _free_sam3()
        # Try the point-prompt path as a last resort
        click_points = [
            {"label": b.get("label", f"bbox_{i}"),
             "x": (b["box"][0] + b["box"][2]) // 2,
             "y": (b["box"][1] + b["box"][3]) // 2}
            for i, b in enumerate(boxes)
        ]
        return _sam_segment_targeted(image_path, click_points, out_dir)


# ── GPT geometry self-check ───────────────────────────────────────────────────
# Catches the case where GPT's text is correct ("occluder is the wood log") but
# the click/polygon coordinates land on the subject instead. Two cheap checks:
#   (1) Point-in-polygon: occluder_click inside visible_polygon_override?
#   (2) SAM3 cross-check: SAM3(occluder_click) overlaps visible polygon > threshold?
#   (3) hidden_polygon overlaps visible polygon > threshold?
# When any check fires, we ask GPT to redo with explicit feedback.

def _validate_gpt_geometry(
    data: dict,
    image_path: str,
    h: int,
    w: int,
    out_dir: Path,
) -> tuple:
    """Returns (is_valid, feedback_text). When invalid, GPT should be re-prompted.
    Skips silently in frame-crop mode (no in-scene occluder to validate)."""
    if bool(data.get("frame_cropped", False)):
        return True, ""

    vis_poly = data.get("visible_polygon_override", [])
    if len(vis_poly) < 3:
        # No visible polygon to cross-check against. Trust GPT.
        return True, ""
    vis_pts  = np.array([[int(p[0]), int(p[1])] for p in vis_poly], dtype=np.int32)
    vis_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(vis_mask, [vis_pts], 1)
    vis_b    = vis_mask.astype(bool)
    vis_area = int(vis_mask.sum())
    if vis_area == 0:
        return True, ""

    issues = []

    # ── Check 1: occluder_click inside visible polygon? ──────────────────────
    occ_click = data.get("occluder_click", {}) or {}
    occ_x = int(occ_click.get("x", 0))
    occ_y = int(occ_click.get("y", 0))
    have_click = (occ_x or occ_y) and 0 <= occ_x < w and 0 <= occ_y < h
    if have_click and vis_mask[occ_y, occ_x] == 1:
        issues.append(
            f"`occluder_click`=({occ_x},{occ_y}) lies INSIDE `visible_polygon_override` — "
            f"that point is on the subject, not the occluder."
        )

    # ── Check 2: SAM3 at occluder_click overlaps visible polygon > threshold? ─
    # Only run SAM3 if the cheap check above passed (otherwise we already know it's bad).
    if have_click and not issues:
        targeted = _sam_segment_targeted(
            image_path,
            [{"label": "geo_check_occluder", "x": occ_x, "y": occ_y}],
            out_dir,
        )
        if targeted:
            occ_v0 = targeted[0]["mask"].astype(bool)
            occ_area = int(occ_v0.sum())
            if occ_area > 0:
                overlap_with_vis = int((occ_v0 & vis_b).sum()) / max(occ_area, 1)
                if overlap_with_vis > config.GEO_OCCLUDER_VS_VISIBLE_MAX:
                    issues.append(
                        f"SAM3 segmentation at `occluder_click`=({occ_x},{occ_y}) is "
                        f"{100 * overlap_with_vis:.0f}% inside the subject — the click "
                        f"is on the subject, not the occluder."
                    )

    # ── Check 3: hidden_polygon mostly inside visible polygon? ────────────────
    hidden_poly = data.get("hidden_polygon", [])
    if len(hidden_poly) >= 3:
        hp_pts = np.array([[int(p[0]), int(p[1])] for p in hidden_poly], dtype=np.int32)
        hp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(hp_mask, [hp_pts], 1)
        hp_area = int(hp_mask.sum())
        if hp_area > 0:
            overlap = int((hp_mask.astype(bool) & vis_b).sum()) / hp_area
            if overlap > config.GEO_HIDDEN_VS_VISIBLE_MAX:
                issues.append(
                    f"`hidden_polygon` is {100 * overlap:.0f}% inside the visible "
                    f"subject region — it should trace area BEHIND the occluder, "
                    f"not the visible body."
                )

    if not issues:
        return True, ""

    occluder_label = data.get("occluder", "the occluder")
    feedback = (
        "GEOMETRY SELF-CHECK FAILED — your previous answer is geographically wrong:\n"
        + "\n".join(f"  • {iss}" for iss in issues)
        + f"\n\nThe occluder is described as: \"{occluder_label}\".\n"
        + "Look at the original image again and provide:\n"
        + "  (1) `occluder_click` = a pixel that is clearly ON THE OCCLUDER\n"
        + "      (NOT on the subject — verify the colour/texture under that pixel "
        + "is the occluder material, e.g. wood, fence, fabric — not fur/skin).\n"
        + "  (2) `polygon_override` = polygon tracing the OCCLUDER OUTLINE.\n"
        + "  (3) `hidden_polygon` = polygon tracing the area BEHIND the occluder where\n"
        + "      the subject's body is hidden — must NOT overlap the visible subject."
    )
    return False, feedback


_GPT_AMODAL_SCHEMA = {
    "name": "amodal_silhouette",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "full_subject_polygon": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "minItems": 8,
                "maxItems": 60,
            },
            "rationale": {"type": "string"},
        },
        "required": ["full_subject_polygon", "rationale"],
        "additionalProperties": False,
    },
}


_GPT_REVIEW_SCHEMA = {
    "name": "amodal_review",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["ok", "redraw"]},
            "rationale": {"type": "string"},
            "corrected_polygon": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "minItems": 0,
                "maxItems": 60,
            },
        },
        "required": ["verdict", "rationale", "corrected_polygon"],
        "additionalProperties": False,
    },
}


def _gpt_review_amodal_mask(
    image_bgr: np.ndarray,
    visible_mask: np.ndarray,
    occluder_mask: np.ndarray,
    pix2gestalt_mask: np.ndarray,
    subject_text: str,
    occluder_text: str,
    out_dir: Path,
) -> np.ndarray:
    """Ask GPT-V to REVIEW pix2gestalt's amodal completion against the
    original image + visible/occluder overlay.

    Pix2gestalt is trained to extend a subject's silhouette to its
    canonical full shape, but it can over-extend or miss the parts that
    are hidden behind a side-adjacent occluder.  GPT-V looks at the
    pix2gestalt mask overlaid on the scene and either:
      • verdict='ok'    → keep pix2gestalt's mask
      • verdict='redraw'→ return a corrected polygon to use instead

    Returns the final H×W uint8 binary mask.
    """
    h, w = image_bgr.shape[:2]

    # Build the review viz: 2-panel side-by-side.
    # Panel A: original + green visible + red occluder overlay
    # Panel B: original + magenta pix2gestalt amodal overlay
    pA = image_bgr.copy()
    ovA = np.zeros_like(image_bgr)
    ovA[visible_mask > 0]  = (0, 255, 0)        # green = visible subject
    ovA[occluder_mask > 0] = (0, 0, 255)        # red   = occluder
    pA = cv2.addWeighted(pA, 0.45, ovA, 0.55, 0)
    cv2.rectangle(pA, (0, 0), (pA.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(pA, "A: GREEN=visible RED=occluder",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    pB = image_bgr.copy()
    ovB = np.zeros_like(image_bgr)
    ovB[pix2gestalt_mask > 0] = (255, 0, 255)   # magenta = pix2gestalt amodal
    pB = cv2.addWeighted(pB, 0.45, ovB, 0.55, 0)
    cv2.rectangle(pB, (0, 0), (pB.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(pB, "B: MAGENTA=pix2gestalt amodal (review this)",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    review_viz = np.hstack([pA, pB])
    review_path = out_dir / "amodal_review_input.png"
    cv2.imwrite(str(review_path), review_viz)

    prompt = f"""You are reviewing an amodal-completion mask.

The SUBJECT in the image is: {subject_text}.
The OCCLUDER is: {occluder_text}.

Two panels are shown side-by-side:
  Panel A — original image with GREEN = visible subject pixels, RED = occluder.
  Panel B — original image with MAGENTA = the candidate AMODAL MASK
            produced by pix2gestalt.

TASK: redraw the magenta mask as the **complete anatomical silhouette
of {subject_text}** — every body part the subject has, INCLUDING parts
that are:
  • currently behind the occluder (red region)
  • folded under the body (e.g. legs tucked under a lying bear)
  • hidden by depth / self-occlusion (the far side of the body)
  • partly cropped at the image edge

KEEP THE SUBJECT'S POSE — do NOT rotate or re-pose the subject.  If
the bear is lying down in the photo, draw a lying-down bear silhouette
with the legs FOLDED UNDER (not standing legs extending downward).
If the person is sitting, keep them seated.  The silhouette must
match what the subject would look like in a clean transparent cutout
of THIS pose, with every anatomical detail included.

For the silhouette outline include FINE DETAIL when present.  Allocate
polygon points to body-part transitions, not smooth interiors:

  PERSON       (30-50 pts)
    head outline   3-4 pts (chin/cheek/forehead/crown)
    ears           1-2 pts each (notches)
    neck           1 pt each side
    shoulder→arm   2-3 pts each side
    each FINGER    2-3 pts when separable (otherwise hand as a lump = 3 pts)
    torso edge     2-3 pts each side
    waist/hip      1 pt each side
    each LEG       3-5 pts (knee bend, calf, ankle, heel, toe-tips)

  QUADRUPED    (35-55 pts)  ← bear / cat / dog / horse
    crown of head  2 pts
    each EAR       2-3 pts (tip + base notch on each side)
    snout          2-3 pts (top + nose tip + lower jaw)
    neck/shoulder  2-3 pts
    back outline   3-5 pts (sloping spine)
    hip            1-2 pts
    TAIL           3-5 pts (base + curl + tip)
    each LEG       3-5 pts (shoulder/hip joint + knee/elbow + paw)
    each PAW       3-5 pts (each toe/claw if visible as a bump)

  BIRD         (30-45 pts)
    head + crown   2 pts
    BEAK           2-3 pts (upper + lower mandible meeting at tip)
    each WING      4-6 pts (shoulder, wingtip, trailing edge bumps)
    body curve     3-4 pts
    TAIL feathers  2-3 pts (often shows as a fan with notches)
    each LEG       2-3 pts (shin + tarsus + foot/toes)

  VEHICLE      (25-40 pts) — chassis + wheels + mirrors + antennas
  OBJECT       (20-40 pts) — full boundary including handles/spouts/etc.

⚠ Don't smooth a wavy edge (paw-bumps along an underside, multiple
toes, finger separations) into a single arc.  Each anatomical "bump"
deserves its own polygon point or pair of points so the binary mask
shows the detail, not a featureless oval.

Lying-down / curled / sitting cases (very common!):
  • A bear lying on a log will have legs FOLDED UNDER its body — the
    silhouette is still a bear shape with bumps/paws on the underside,
    NOT vertical standing legs.
  • A cat curled in a basket will have a rounded body with paws
    tucked in — silhouette is roughly circular, not extended.
  • A person seated has a 90° body bend — silhouette goes head→torso
    →horizontal upper-legs→vertical lower-legs.

Return JSON:
{{
  "verdict":   "ok"     — magenta mask already includes ALL the
                          anatomical detail for the subject's pose
               OR
               "redraw" — supply a corrected polygon
  "rationale": "<one short sentence — what details are being added>",
  "corrected_polygon": [[x, y], [x, y], ...]
       Required ONLY when verdict='redraw'.  Trace the COMPLETE
       anatomical silhouette in the subject's actual pose, with
       **at least 30 polygon points** capturing every body-part
       transition (each paw bump, ear notch, finger separator, tail
       curl, snout tip).  Aim for 35-55 points for a quadruped /
       full person.  Image is {w}×{h} pixels.
       For verdict='ok', return [] or omit the field.
}}

REDRAW RULES:
  • The polygon MUST CONTAIN the green pixels (visible subject).
  • The polygon SHOULD EXTEND into the red region where the body
    continues behind the occluder.
  • The polygon SHOULD EXTEND BEYOND the visible region for body
    parts that are anatomically present but not visible (folded legs,
    tail, off-frame parts).  The extension respects the POSE — don't
    add standing legs to a lying creature.
  • Use 16+ points to capture detail: ear notches, paw shape, tail
    tip, snout, claws.  More points = better outline.

DEFAULT TO 'redraw' when ANY of:
  • magenta mask is a smooth blob with no paw/ear/tail detail
  • a creature/person is missing limbs visible in its current pose
  • the polygon would be < 30 points for a creature
    (insufficient anatomical detail)
  • multiple paws / toes / fingers / wing-tips visible in the photo
    are merged into a single rounded edge
"""

    data = gpt_vision(
        [str(review_path)],
        prompt,
        schema=_GPT_REVIEW_SCHEMA,
        cache_key="amodal_pix2gestalt_review_v1",
    )

    verdict = (data.get("verdict", "ok") if isinstance(data, dict) else "ok").lower()
    rationale = data.get("rationale", "") if isinstance(data, dict) else ""
    print(f"  [Amodal/review] verdict={verdict}  rationale={rationale}")

    if verdict == "ok":
        final = (pix2gestalt_mask > 0).astype(np.uint8)
    else:
        poly = data.get("corrected_polygon", [])
        if not poly or len(poly) < 3:
            print("  [Amodal/review] verdict=redraw but no usable polygon — keeping pix2gestalt mask")
            final = (pix2gestalt_mask > 0).astype(np.uint8)
        else:
            pts = np.array([[int(round(p[0])), int(round(p[1]))] for p in poly],
                           dtype=np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            final = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(final, [pts], 1)
            # Always include the visible subject
            final = np.clip(final | (visible_mask > 0).astype(np.uint8),
                            0, 1).astype(np.uint8)
            print(f"  [Amodal/review] applied corrected polygon ({len(pts)} pts) "
                  f"→ {int(final.sum())} px (raw)")

            # ── Anatomical-bbox clip ──────────────────────────────────────
            # GPT-V sometimes redraws the subject silhouette to cover the
            # ENTIRE occluder shape (horse case: drew the person silhouette
            # spanning across the horse's body).  Clip the polygon to a
            # sane anatomical bbox: visible_mask's bbox dilated by 30%,
            # which gives enough room for hidden body parts but prevents
            # the silhouette from drifting onto the occluder.
            ys, xs = np.where(visible_mask > 0)
            if len(ys) >= 16:
                vx1, vy1 = int(xs.min()), int(ys.min())
                vx2, vy2 = int(xs.max()), int(ys.max())
                bw, bh = vx2 - vx1, vy2 - vy1
                pad_x = int(0.30 * bw)
                pad_y = int(0.30 * bh)
                cx1 = max(0, vx1 - pad_x)
                cy1 = max(0, vy1 - pad_y)
                cx2 = min(w, vx2 + pad_x)
                cy2 = min(h, vy2 + pad_y)
                bbox_mask = np.zeros((h, w), dtype=np.uint8)
                bbox_mask[cy1:cy2, cx1:cx2] = 1
                before = int(final.sum())
                final = (final & bbox_mask).astype(np.uint8)
                # Always keep visible_mask, even if bbox clipped some of it
                # (it shouldn't, but safety).
                final = np.clip(final | (visible_mask > 0).astype(np.uint8),
                                0, 1).astype(np.uint8)
                after = int(final.sum())
                print(f"  [Amodal/review] bbox clip [{cx1}-{cx2} × {cy1}-{cy2}] "
                      f"→ {before} → {after} px")

    # Save a final viz so the user can see the chosen mask in context
    viz = image_bgr.copy()
    ov = np.zeros_like(image_bgr)
    ov[occluder_mask > 0]  = (0, 0, 255)
    ov[visible_mask > 0]   = (0, 255, 0)
    ov[(final > 0) & (visible_mask == 0)] = (255, 0, 255)   # magenta = hidden completion
    viz = cv2.addWeighted(viz, 0.45, ov, 0.55, 0)
    cv2.imwrite(str(out_dir / "subject_full_amodal_mask_viz.png"), viz)
    return final


def _flux_extend_amodal_mask_offframe(
    image_bgr: np.ndarray,
    in_frame_amodal_mask: np.ndarray,
    expansion_pixels: dict,
    subject_text: str,
    out_dir: Path,
    pad_color: int = 128,
) -> tuple:
    """Build a padded canvas, outpaint the off-frame margins with Flux-Fill
    so the subject's anatomy continues into them, then run SAM3 with a
    click on the in-frame subject centroid to segment the FULL silhouette
    (in-frame + outpainted) on the padded canvas.

    Returns (padded_full_mask, offframe_only_mask, padded_image, offsets).
      padded_full_mask   — H'×W' uint8 binary of the full subject
                           silhouette on the padded canvas (in-frame +
                           Flux-generated off-frame).
      offframe_only_mask — H'×W' uint8 binary of ONLY the off-frame portion
                           (= padded_full_mask zeroed inside the original
                           image rectangle).
      padded_image       — H'×W'×3 uint8 BGR of the OUTPAINTED padded canvas
                           (original in place + Flux-generated content in
                           the margins).
      offsets            — dict with {"top", "bottom", "left", "right",
                           "orig_h", "orig_w"} so the caller can map back to
                           original coordinates.

    Returns (None, None, None, None) on failure.
    """
    h, w = image_bgr.shape[:2]
    pad_t = max(0, int(expansion_pixels.get("top",    0)))
    pad_b = max(0, int(expansion_pixels.get("bottom", 0)))
    pad_l = max(0, int(expansion_pixels.get("left",   0)))
    pad_r = max(0, int(expansion_pixels.get("right",  0)))
    if pad_t + pad_b + pad_l + pad_r == 0:
        return None, None, None, None

    H, W = h + pad_t + pad_b, w + pad_l + pad_r

    # ── 1. Build the padded canvas (original in place, gray margins) ─────
    padded_bgr = np.full((H, W, 3), pad_color, dtype=np.uint8)
    padded_bgr[pad_t:pad_t + h, pad_l:pad_l + w] = image_bgr

    # ── 2. Padded in-frame amodal mask (same placement, just for viz) ────
    padded_in_mask = np.zeros((H, W), dtype=np.uint8)
    padded_in_mask[pad_t:pad_t + h, pad_l:pad_l + w] = (in_frame_amodal_mask > 0).astype(np.uint8)

    # ── 3. Build the inpaint mask = the off-frame margin region only ─────
    outpaint_mask = np.ones((H, W), dtype=np.uint8)
    outpaint_mask[pad_t:pad_t + h, pad_l:pad_l + w] = 0   # 0 = keep, 1 = inpaint

    # ── 3b. Narrow the inpaint mask to a dilated visible-bbox region ──────
    # Restricts Flux to only the gray padding near the subject. Outside the
    # extension bbox stays gray (Flux skips it), saving 2-4× Flux runtime
    # since those pixels would be cropped away by step 7 anyway.
    if getattr(config, "USE_EXTENSION_BBOX", False):
        ys, xs = np.where(padded_in_mask > 0)
        if len(ys) >= 16:
            vx1, vy1 = int(xs.min()), int(ys.min())
            vx2, vy2 = int(xs.max()), int(ys.max())
            vw, vh = vx2 - vx1 + 1, vy2 - vy1 + 1
            mult = float(getattr(config, "EXTENSION_MULTIPLIER", 2.0))
            min_ext = int(getattr(config, "MIN_EXTENSION_PX", 100))
            ext_w = max(int((vw * (mult - 1.0)) / 2), min_ext)
            ext_h = max(int((vh * (mult - 1.0)) / 2), min_ext)
            bx1 = max(0, vx1 - ext_w)
            by1 = max(0, vy1 - ext_h)
            bx2 = min(W, vx2 + ext_w + 1)
            by2 = min(H, vy2 + ext_h + 1)
            bbox_mask = np.zeros((H, W), dtype=np.uint8)
            bbox_mask[by1:by2, bx1:bx2] = 1
            before = int(outpaint_mask.sum())
            outpaint_mask = (outpaint_mask & bbox_mask).astype(np.uint8)
            after = int(outpaint_mask.sum())
            print(f"  [Amodal/offframe] narrowing mask to ext_bbox "
                  f"[{bx1}-{bx2} × {by1}-{by2}] (mult={mult}, ext=+{ext_w}×{ext_h}) "
                  f"→ {before} → {after} px ({100*after/max(before,1):.1f}% of full padding)")
            cv2.imwrite(str(out_dir / "amodal_offframe_ext_bbox_viz.png"),
                        (bbox_mask * 80 + outpaint_mask * 175).astype(np.uint8))

    # Save the input viz before generation so we can debug if Flux/SAM3 fails.
    cv2.imwrite(str(out_dir / "amodal_offframe_padded_input.png"), padded_bgr)
    cv2.imwrite(str(out_dir / "amodal_offframe_outpaint_mask.png"), outpaint_mask * 255)

    # ── 4. Run Flux-Fill on the padded canvas ────────────────────────────
    # Convert to RGB for Flux/diffusers; result is RGB too.
    padded_rgb_in = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)
    prompt = (
        f"Photorealistic complete {subject_text}, full anatomy visible "
        f"(legs, paws, tail, body), natural pose continuing from the "
        f"visible portion. EXTREMELY SHARP focus, crisp high-detail "
        f"textures (fur strands, individual claws, log bark grain), "
        f"matching exposure, lighting direction, shadows, and colour "
        f"temperature with the surrounding scene. Seamless continuation; "
        f"no seams, no duplicate body parts. 8K detail, DSLR photograph."
    )
    neg_extra = ("duplicate limbs, extra heads, second animal, frame, "
                 "border, blurry, soft, low-detail, painting, illustration, "
                 "smooth plastic, oversmoothed, low quality, distorted anatomy")

    try:
        flux_results = _run_flux_fill_inpaint(
            base_np=padded_rgb_in,
            inpaint_mask=outpaint_mask,
            amodal_rgb_256=np.full((256, 256, 3), 255, dtype=np.uint8),
            prompt=prompt,
            n_samples=1,
            out_dir=out_dir,
            prefix="flux_offframe",
            neg_extra=neg_extra,
            seed_offset=0,
        )
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [Amodal/offframe] Flux-Fill outpaint failed: {exc!r}")
        flux_results = []

    if not flux_results:
        print("  [Amodal/offframe] no Flux output — falling back to "
              "in-frame amodal mask placed on gray-padded canvas")
        padded_full   = padded_in_mask.copy()
        offframe_only = np.zeros_like(padded_full)
        offsets = {"top": pad_t, "bottom": pad_b, "left": pad_l, "right": pad_r,
                   "orig_h": h, "orig_w": w}
        return padded_full, offframe_only, padded_bgr, offsets

    # Flux returns a PIL image; convert and save the canonical outpainted canvas.
    flux_pil = flux_results[0]
    padded_rgb_out = np.array(flux_pil.convert("RGB"))
    if padded_rgb_out.shape[:2] != (H, W):
        padded_rgb_out = cv2.resize(padded_rgb_out, (W, H),
                                    interpolation=cv2.INTER_LANCZOS4)
    padded_bgr_out = cv2.cvtColor(padded_rgb_out, cv2.COLOR_RGB2BGR)
    # Hard-restore the original image inside the yellow rectangle (Flux can
    # subtly modify unmasked regions due to VAE round-trip).
    padded_bgr_out[pad_t:pad_t + h, pad_l:pad_l + w] = image_bgr
    cv2.imwrite(str(out_dir / "amodal_offframe_outpainted.png"), padded_bgr_out)

    # ── 5. Segment the FULL subject on the outpainted padded canvas ──────
    # Click prompt = centroid of the in-frame amodal mask, translated to
    # padded coordinates.  We also add the click coords of each non-empty
    # margin so SAM3 can also extend if the centroid alone misses.
    M = cv2.moments((in_frame_amodal_mask > 0).astype(np.uint8))
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"]) + pad_l
        cy = int(M["m01"] / M["m00"]) + pad_t
    else:
        cx, cy = pad_l + w // 2, pad_t + h // 2

    # Write the padded outpainted canvas to a temp PNG so _sam_segment_targeted
    # can re-read it (its API takes a path).
    flux_canvas_path = out_dir / "amodal_offframe_outpainted.png"
    sam_results = _sam_segment_targeted(
        str(flux_canvas_path),
        [{"label": "subject_padded", "x": cx, "y": cy}],
        out_dir,
    )

    padded_full = padded_in_mask.copy()
    if sam_results:
        sam_mask = sam_results[0]["mask"]
        if sam_mask.shape[:2] != (H, W):
            sam_mask = cv2.resize(sam_mask.astype(np.uint8), (W, H),
                                  interpolation=cv2.INTER_NEAREST)
        # Sanity check: SAM3 mask should overlap the in-frame amodal heavily.
        ovl = int(((sam_mask > 0) & (padded_in_mask > 0)).sum())
        ovl_ratio = ovl / max(int(padded_in_mask.sum()), 1)
        if ovl_ratio < 0.40:
            print(f"  [Amodal/offframe] SAM3 mask only covers {ovl_ratio:.2f} "
                  f"of in-frame amodal — rejecting, keeping in-frame mask")
        else:
            padded_full = np.clip((sam_mask > 0).astype(np.uint8) | padded_in_mask,
                                  0, 1).astype(np.uint8)
            print(f"  [Amodal/offframe] SAM3 segmented {int((sam_mask > 0).sum())} px "
                  f"on padded canvas (in-frame was {int(padded_in_mask.sum())} px) "
                  f"→ padded_full = {int(padded_full.sum())} px")
    else:
        print("  [Amodal/offframe] SAM3 returned no mask — keeping in-frame mask")

    # ── 6. Save off-frame-only mask + viz ────────────────────────────────
    offframe_only = padded_full.copy()
    offframe_only[pad_t:pad_t + h, pad_l:pad_l + w] = 0

    viz_out = padded_bgr_out.copy()
    ov = np.zeros_like(viz_out)
    ov[padded_full > 0]    = (255, 0, 255)   # magenta = full silhouette
    ov[padded_in_mask > 0] = (0, 255, 0)     # green   = in-frame portion
    viz_out = cv2.addWeighted(viz_out, 0.45, ov, 0.55, 0)
    cv2.rectangle(viz_out, (pad_l, pad_t), (pad_l + w - 1, pad_t + h - 1),
                  (0, 255, 255), 2)         # yellow = orig bounds
    cv2.circle(viz_out, (cx, cy), 6, (0, 255, 255), -1)  # click point
    cv2.imwrite(str(out_dir / "amodal_offframe_viz.png"), viz_out)

    offsets = {"top": pad_t, "bottom": pad_b, "left": pad_l, "right": pad_r,
               "orig_h": h, "orig_w": w}

    # ── 7. Post-hoc crop to subject's tight bbox + margin ────────────────
    # The padded canvas is intentionally oversized (auto-budget). After
    # segmentation we crop everything to the subject's tight bbox + a small
    # margin so downstream consumers get exactly the canvas they need.
    margin = int(getattr(config, "AUTO_CROP_MARGIN_PX", 30))
    ys, xs = np.where(padded_full > 0)
    if len(ys) >= 16:
        bx1 = max(0, int(xs.min()) - margin)
        by1 = max(0, int(ys.min()) - margin)
        bx2 = min(W, int(xs.max()) + margin + 1)
        by2 = min(H, int(ys.max()) + margin + 1)
        tight_w = bx2 - bx1
        tight_h = by2 - by1
        tight_canvas = padded_bgr_out[by1:by2, bx1:bx2].copy()
        tight_mask   = padded_full   [by1:by2, bx1:bx2].copy()
        tight_in_mask = padded_in_mask[by1:by2, bx1:bx2].copy()
        tight_offframe = offframe_only[by1:by2, bx1:bx2].copy()
        cv2.imwrite(str(out_dir / "subject_tight_canvas.png"),   tight_canvas)
        cv2.imwrite(str(out_dir / "subject_tight_mask.png"),     tight_mask * 255)
        cv2.imwrite(str(out_dir / "subject_tight_in_mask.png"),  tight_in_mask * 255)
        cv2.imwrite(str(out_dir / "subject_tight_offframe.png"), tight_offframe * 255)
        # Transparent RGBA cutout where alpha = subject mask
        rgba_tight = np.zeros((tight_h, tight_w, 4), dtype=np.uint8)
        rgba_tight[..., :3] = tight_canvas
        rgba_tight[..., 3]  = (tight_mask > 0).astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / "subject_tight_rgba.png"), rgba_tight)
        # Offsets relative to the padded canvas + back to original image:
        offsets["tight_bbox_in_padded"] = [bx1, by1, bx2, by2]
        offsets["tight_w"] = tight_w
        offsets["tight_h"] = tight_h
        # Original image coords relative to tight canvas: (pad_l - bx1, pad_t - by1)
        offsets["orig_in_tight"] = [pad_l - bx1, pad_t - by1,
                                    pad_l - bx1 + w, pad_t - by1 + h]
        print(f"  [Amodal/offframe] cropped to subject bbox "
              f"[{bx1}-{bx2} × {by1}-{by2}] → {tight_w}×{tight_h} "
              f"(was {W}×{H}) → subject_tight_*.png")
    else:
        print(f"  [Amodal/offframe] post-crop skipped (mask too small)")

    return padded_full, offframe_only, padded_bgr_out, offsets


def _gpt_amodal_subject_mask(
    image_bgr: np.ndarray,
    visible_mask: np.ndarray,
    occluder_mask: np.ndarray,
    subject_text: str,
    occluder_text: str,
    out_dir: Path,
) -> tuple:
    """Ask GPT-V to trace the FULL silhouette of the subject (visible +
    hidden behind the occluder) and return a binary mask of that silhouette.

    This is the user-requested replacement for pix2gestalt threshold output —
    pix2gestalt extends the silhouette in directions it learned during
    training (usually downward), which fails when the occluder is BESIDE the
    subject and the body should extend INTO the occluder's region.  GPT-V
    sees the whole scene and can reason about where the body goes.

    Returns (mask, polygon_pts) or (None, None) on failure.
    """
    h, w = image_bgr.shape[:2]

    # Build the input viz: original on left, original + green visible + red
    # occluder overlay on right.  GPT-V uses the overlay to know where the
    # subject's visible pixels and the occluder are; we put both panels
    # side-by-side so it can also see the un-overlaid image for context.
    overlay = image_bgr.copy()
    color_layer = np.zeros_like(image_bgr)
    color_layer[visible_mask > 0]  = (0, 255, 0)      # green = visible subject
    color_layer[occluder_mask > 0] = (0, 0, 255)      # red   = occluder
    both = (visible_mask > 0) & (occluder_mask > 0)
    color_layer[both]              = (0, 255, 255)    # yellow = both (shouldn't happen)
    overlay = cv2.addWeighted(overlay, 0.5, color_layer, 0.5, 0)
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(overlay, "GREEN=visible subject  RED=occluder",
                (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    viz_path = out_dir / "amodal_input_overlay.png"
    cv2.imwrite(str(viz_path), overlay)

    prompt = f"""You are a computer-vision assistant.  An image contains a SUBJECT
({subject_text}) that is partially occluded by an OCCLUDER ({occluder_text}).

Two images are provided:
  Image 1 — the original scene.
  Image 2 — the same scene with overlays:
              GREEN  = the visible pixels of the SUBJECT
              RED    = the OCCLUDER pixels (these hide part of the SUBJECT)

TASK: Return a polygon tracing the FULL ANATOMICAL SILHOUETTE of the
subject — its complete body outline as it would appear if you could
see the entire creature/object standing alone.  Include EVERY body part
of the subject's natural anatomy, even parts that are currently:
  (a) HIDDEN behind the red occluder, OR
  (b) HIDDEN below/beyond the photographer's framing (legs cut off at
      the bottom of the photo, tail cropped at the side, etc.), OR
  (c) HIDDEN by the photo composition (the body continues into a
      shadow / dirt / background area where it's hard to see).

Be ANATOMICALLY COMPLETE:
  • PERSON:     head + neck + torso + arms + hands + LEGS + feet
  • QUADRUPED:  head + body + 4 LEGS (or as many as anatomically present) + tail
  • BIRD:       head + neck + body + wings + LEGS + feet + tail
  • VEHICLE:    full chassis outline as if seen in profile
  • OBJECT:     the entire object boundary

⚠ IMPORTANT — be ANATOMICALLY TIGHT (not bloated):

  • Estimate body proportions from the visible parts.  For a bear with
    head + upper torso visible, the legs extend roughly 60-80% of the
    visible-body-height BELOW the body.  Use anatomy, not the occluder
    shape, to decide where the silhouette goes.
  • The polygon must follow the SUBJECT's actual body width at each
    level, NOT the occluder's width.  If a bear's leg is ~50 px wide
    but the log occluding it is 200 px wide, the polygon's leg portion
    must be ~50 px wide.
  • The polygon MAY extend beyond the occluder region AND beyond the
    visible region — that's expected when the subject's legs / tail /
    other parts continue past the occluder or are simply not visible
    in the photo.
  • The polygon MUST CONTAIN the visible (green) region.
  • The polygon's lower edge may extend down into background pixels
    (dirt, ground, grass) where the subject's legs/feet anatomically
    should land.  Do NOT include pixels that are clearly part of a
    separate background object (sky, distant trees).

Image is {w}×{h} pixels (width × height).  Coordinates: (x, y) where x is
horizontal (0=left, {w-1}=right), y is vertical (0=top, {h-1}=bottom).
Values can extend slightly outside the frame (e.g. y = {h} or y = {h+50})
if the subject is cropped at the edge — they'll be clipped to the canvas.

Worked example — bear with head + upper torso visible, log + dirt below:
  • Visible: bear's head + chest (green).
  • Occluder: horizontal log across belly (red).
  • Below the log: dirt/ground (no visible bear, but the bear's legs
    anatomically belong here).
  • Polygon: head + neck + full torso + 4 legs + paws, descending
    THROUGH the log and INTO the dirt area below.  Each leg ~40-60%
    of the head's width; legs extend ~70% of visible-body-height
    downward from the bottom of the torso.

Worked example — person partly behind a horse:
  • Visible: person's head + shoulders + one arm (green).
  • Occluder: horse to the side blocking the other shoulder + half torso.
  • Polygon: head + neck + torso + both arms (one visible, one behind
    horse) + both legs descending to feet on the ground (even if the
    legs are cropped at the bottom of the photo).

Worked example — bird with branch across body:
  • Visible: bird's head + wings + back (green).
  • Occluder: thin branch across belly (red).
  • Polygon: head + neck + body + wings + legs + feet + tail.  Even
    though legs and tail aren't visible at all, include them with
    typical-bird proportions.

Return JSON:
{{
  "full_subject_polygon": [[x, y], [x, y], ...],
  "rationale": "<one short sentence: which body parts are being added and where they extend>"
}}"""

    data = gpt_vision(
        [str(out_dir / "amodal_input_overlay.png")],
        prompt,
        schema=_GPT_AMODAL_SCHEMA,
        cache_key="amodal_completion_v1",
    )

    poly = data.get("full_subject_polygon", []) if isinstance(data, dict) else []
    if len(poly) < 3:
        print(f"  [GPT-V amodal] returned {len(poly)} points — not enough for a polygon")
        return None, None

    pts = np.array([[int(round(p[0])), int(round(p[1]))] for p in poly],
                   dtype=np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    completion = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(completion, [pts], 1)

    # Constraints on the GPT-drawn polygon:
    #   (a) completion = polygon − visible_mask (we paint only hidden parts;
    #       the visible part is already in the photo).
    #   (b) completion is clipped to an ANATOMICAL BBOX around the visible
    #       subject (not the occluder).  This lets the silhouette extend
    #       DOWN past the occluder into the legs/feet area while still
    #       rejecting wild over-extensions far from the visible body.
    before = int(completion.sum())
    completion = np.clip(completion.astype(np.int32)
                         - (visible_mask > 0).astype(np.int32), 0, 1).astype(np.uint8)
    after_subtract = int(completion.sum())
    if after_subtract < before:
        print(f"  [GPT-V amodal] removed {before - after_subtract} px overlapping visible_mask")

    # Anatomical bbox: expand the visible-body bbox by ~1.5× width on each
    # side and ~1.5× height downward (animals/people stand on their feet,
    # so most hidden anatomy is BELOW the visible body — legs).  Tail/wings
    # may add some upward/sideways extension too.  Don't clip to the
    # occluder — that was the bug; legs are NOT in the occluder.
    if int(visible_mask.sum()) > 0:
        ys, xs = np.where(visible_mask > 0)
        body_top    = int(ys.min())
        body_bottom = int(ys.max())
        body_left   = int(xs.min())
        body_right  = int(xs.max())
        body_h = body_bottom - body_top + 1
        body_w = body_right  - body_left  + 1
        # Anatomical band: extend mostly DOWN for legs (1.5× body height),
        # a little UP for head/ears (0.3× body height), and ~50% on each side.
        side_pad = int(0.50 * body_w)
        top_pad  = int(0.30 * body_h)
        bot_pad  = int(1.50 * body_h)
        ana_top    = max(0, body_top    - top_pad)
        ana_bottom = min(h - 1, body_bottom + bot_pad)
        ana_left   = max(0, body_left   - side_pad)
        ana_right  = min(w - 1, body_right + side_pad)
        ana_box = np.zeros_like(completion)
        ana_box[ana_top:ana_bottom + 1, ana_left:ana_right + 1] = 1
        before_box = int(completion.sum())
        completion = (completion.astype(bool) & ana_box.astype(bool)).astype(np.uint8)
        after_box = int(completion.sum())
        if after_box < before_box:
            print(f"  [GPT-V amodal] clipped {before_box - after_box} px "
                  f"outside anatomical bbox "
                  f"(x={ana_left}-{ana_right}, y={ana_top}-{ana_bottom})")

    # GEOMETRIC FALLBACK: if the clipped completion is empty (because GPT
    # placed the polygon entirely outside the occluder), derive the
    # completion from geometry.  The hidden body parts are wherever the
    # OCCLUDER is ADJACENT to the visible subject — that's the
    # transition zone where the visible body continues into hidden.
    #
    # Restricted to the SHOULDER Y-BAND of the visible body (the most
    # common occluded body part) so we don't include facial / leg edges.
    # Shoulder ≈ 15-50% from the top of the visible-body bbox.
    if int(completion.sum()) == 0 and int(occluder_mask.sum()) > 0 \
            and int(visible_mask.sum()) > 0:
        ys, xs = np.where(visible_mask > 0)
        body_top    = int(ys.min())
        body_bottom = int(ys.max())
        body_height = body_bottom - body_top
        body_width  = int(xs.max() - xs.min()) if xs.size > 0 else 50
        # Shoulder Y-band — upper-middle of the body
        shoulder_top    = body_top + int(0.15 * body_height)
        shoulder_bottom = body_top + int(0.55 * body_height)
        # Conservative reach: 25% of body width, capped 20–80 px
        reach = max(20, min(80, int(0.25 * body_width)))
        vis_dilated = cv2.dilate(
            (visible_mask > 0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=reach // 3,
        )
        # Build a Y-band mask
        band = np.zeros_like(visible_mask, dtype=np.uint8)
        band[shoulder_top:shoulder_bottom + 1, :] = 1
        geometric = ((vis_dilated > 0)
                     & (occluder_mask > 0)
                     & (band > 0)).astype(np.uint8)
        if int(geometric.sum()) > 0:
            print(f"  [GPT-V amodal] geometric fallback: completion = "
                  f"occluder ∩ dilated(visible, +{reach}px) ∩ "
                  f"shoulder_band(y={shoulder_top}-{shoulder_bottom}) = "
                  f"{int(geometric.sum())} px")
            completion = geometric

    # The full AMODAL silhouette = visible + completion.  We save both.
    amodal_full = np.clip((visible_mask > 0).astype(np.uint8) | completion, 0, 1).astype(np.uint8)

    # Save a viz: completion (yellow) + visible (green) + occluder (red).
    viz = image_bgr.copy()
    ov = np.zeros_like(image_bgr)
    ov[occluder_mask > 0] = (0, 0, 255)         # red    = occluder
    ov[visible_mask > 0]  = (0, 255, 0)         # green  = visible
    ov[completion > 0]    = (0, 255, 255)       # yellow = NEW completion region
    viz = cv2.addWeighted(viz, 0.45, ov, 0.55, 0)
    cv2.polylines(viz, [pts], isClosed=True, color=(255, 255, 255), thickness=2)
    cv2.imwrite(str(out_dir / "subject_full_amodal_mask_viz.png"), viz)

    if data.get("rationale"):
        print(f"  [GPT-V amodal] rationale: {data['rationale']}")
    print(f"  [GPT-V amodal] completion-only mask: {int(completion.sum())} px  "
          f"(amodal full = visible {int((visible_mask>0).sum())} + completion "
          f"= {int(amodal_full.sum())} px)")

    # ── Anatomical-bbox clip + occluder-body removal ─────────────────────
    # Two-step clip to prevent the amodal silhouette from drifting onto the
    # occluder's own body:
    #   1. Bbox clip: amodal must stay inside (visible_bbox + 30% padding).
    #   2. Occluder-body removal: remove "occluder-far-from-visible" pixels.
    #      The amodal CAN include occluder pixels that are within K px of
    #      visible_mask (= the hidden slice directly behind occluder), but
    #      NOT occluder pixels deep inside the occluder's own body.
    occ_b = (occluder_mask > 0).astype(np.uint8)
    ys, xs = np.where(visible_mask > 0)
    if len(ys) >= 16:
        vx1, vy1 = int(xs.min()), int(ys.min())
        vx2, vy2 = int(xs.max()), int(ys.max())
        bw, bh = vx2 - vx1, vy2 - vy1
        pad_x = int(0.30 * bw)
        pad_y = int(0.30 * bh)
        cx1 = max(0, vx1 - pad_x)
        cy1 = max(0, vy1 - pad_y)
        cx2 = min(w, vx2 + pad_x)
        cy2 = min(h, vy2 + pad_y)
        bbox_mask = np.zeros((h, w), dtype=np.uint8)
        bbox_mask[cy1:cy2, cx1:cx2] = 1
        before = int(amodal_full.sum())
        amodal_full = (amodal_full & bbox_mask).astype(np.uint8)
        # Step 2: compute the "occluder-far-from-visible" region.
        # Pixels inside occluder that are FAR from any visible_mask pixel
        # are the occluder's own body — disallow amodal there.
        # K = 10 px is tight enough to reject occluder-bodies that physically
        # touch the visible subject (e.g. horse's neck touching person's
        # shoulder); 30 px was too generous and let the horse-head survive.
        if int(occ_b.sum()) > 0 and int((visible_mask > 0).sum()) > 0:
            K = 10   # px — allowed reach of amodal INTO the occluder
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K * 2 + 1, K * 2 + 1))
            visible_neighbourhood = cv2.dilate(
                (visible_mask > 0).astype(np.uint8), kernel, iterations=1)
            occluder_far = ((occ_b == 1) & (visible_neighbourhood == 0)).astype(np.uint8)
            amodal_full = (amodal_full & ~occluder_far).astype(np.uint8)
        # Always keep visible_mask in the final amodal.
        amodal_full = np.clip(amodal_full | (visible_mask > 0).astype(np.uint8),
                              0, 1).astype(np.uint8)
        after = int(amodal_full.sum())
        print(f"  [GPT-V amodal] bbox clip [{cx1}-{cx2} × {cy1}-{cy2}] "
              f"+ occluder-far removal → {before} → {after} px")

    # Also persist the completion-only mask under its own filename for clarity.
    cv2.imwrite(str(out_dir / "subject_completion_mask.png"), completion * 255)

    # We return the FULL AMODAL silhouette (visible + completion) so the
    # caller can save it as `subject_full_amodal_mask.png` and Agent 2 can
    # use it as the ControlNet shape-prior.  The completion-only mask is
    # already saved alongside as `subject_completion_mask.png`.
    return amodal_full, pts


# ── Agent 1 — Occlusion Agent ─────────────────────────────────────────────────

def occlusion_agent(state: State) -> dict:
    print("\n─── Agent 1: Occlusion Agent ────────────────────────")
    img  = cv2.imread(state["image_path"])
    h, w = img.shape[:2]
    hint = (state.get("target") or "").strip()

    out_dir = BASE_DIR / "output" / Path(state["image_path"]).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    segments, sam3_viz_path = _sam_segment_all(state["image_path"], out_dir)
    if not segments:
        raise RuntimeError("[Occlusion Agent] SAM3 returned 0 segments — cannot proceed.")

    seg_summary = [
        {"id": s["id"], "bbox": s["bbox"], "area": s["area"], "polygon": s["polygon"][:12]}
        for s in segments
    ]

    target_instruction = (
        f'The user has indicated the occluded object is "{hint}". Use this as your guide.'
        if hint else
        "No target specified. Identify the most prominent occlusion — "
        "the object most clearly partially hidden behind another object."
    )

    prompt = f"""You are a world-class expert in computer vision, amodal completion, and animal/human anatomy.

Image size: {w}×{h} px (width × height).
{target_instruction}

Image 1: original photo.
Image 2: SAM3 visualisation — every segment is coloured and labelled with its numeric ID.

SAM3 segments (id, bbox [x1,y1,x2,y2], area px, polygon up to 12 [x,y] pts):
{json.dumps(seg_summary, indent=2)}

══════════════════════════════════════════════════════════════════════
STEP 1 — IDENTIFY THE SUBJECT
══════════════════════════════════════════════════════════════════════
Carefully examine the image and identify the primary subject (animal, person, object).
Fill `subject_description` with ALL of the following detail:
  • Species / type (e.g. "Silver Gull (Chroicocephalus novaehollandiae)")
  • Body coloring: head, back, wings, belly, beak, legs/feet (exact colors and textures)
  • Posture and orientation (e.g. "standing upright, facing left, wings folded")
  • Size relative to frame (e.g. "fills ~80% of image height")
  • Background (e.g. "bright blue sky, out-of-focus, stone/concrete post beneath")
  Example: "Silver Gull standing on a rounded stone post: white head and underparts,
   pale grey back and wings, bright red-orange beak and eye-ring, vivid red-orange
   legs and webbed feet, facing left, stone post is grey textured concrete."

Fill `visible_parts` with a precise inventory of what IS visible:
  • For birds: "head, neck, entire body/torso, both wings folded, both legs visible
    to ankle; feet and lower tarsus partially cut at bottom edge"
  • For people: "head, shoulders, torso, waist; legs entirely absent below frame"
  • Be specific about which body parts are partial vs. fully visible.

Fill `missing_parts` with what is CUT OFF or HIDDEN:
  • For birds: "feet and lower portion of tarsal legs — roughly lower 15% of bird height"
  • For people: "entire lower body from waist down"
  • Be specific about body parts and approximate size of the missing region.

══════════════════════════════════════════════════════════════════════
STEP 2 — DETERMINE THE MODE
══════════════════════════════════════════════════════════════════════

MODE A — Classic occlusion (frame_cropped = false):
  A foreground object WITHIN THE SCENE hides part of the subject INSIDE the frame.
  The missing part exists in the scene but is blocked by another object in front of it.
  Examples: pole covering a cat, fence in front of a deer, car blocking a person.

MODE B — Frame-crop completion (frame_cropped = true):
  The subject EXITS the image at a FRAME EDGE — meaning a portion of the body
  extends BEYOND the canvas edge and was simply never captured by the camera.
  The missing part does NOT exist anywhere in the current image.
  Examples:
    • Seagull standing on a post — feet and lower legs below the bottom frame edge
    • Person showing only torso — legs are outside the bottom of the frame
    • Bird in flight — one wing fully clipped by the right or left frame edge
    • Animal's body exits the frame on one side (head visible, hindquarters cut off)
    • Dog with only head and shoulders — body extends below the bottom edge

  ⚠ THRESHOLD: Only set frame_cropped=true when a MEANINGFUL portion is cut off.
    A tiny feather tip touching the edge is NOT frame_cropped.
    A body part that is >8% of the expected full body length/width IS frame_cropped.

══════════════════════════════════════════════════════════════════════
STEP 3 — EDGE-BY-EDGE ANALYSIS (for frame_cropped detection)
══════════════════════════════════════════════════════════════════════
For EACH of the four image edges, check:
  TOP (y=0): Does the subject's body reach and exit the top edge? What part?
  BOTTOM (y={h}): Does the subject's body reach and exit the bottom edge? What part?
  LEFT (x=0): Does the subject's body reach and exit the left edge? What part?
  RIGHT (x={w}): Does the subject's body reach and exit the right edge? What part?

For each edge where the subject exits, note the specific body part and approximate
percentage of the expected full body that is missing there.

Only include an edge in `expansion_directions` if the subject MEANINGFULLY exits there
(≥8% of the full body missing on that side).

══════════════════════════════════════════════════════════════════════
STEP 4 — FILL ALL FIELDS
══════════════════════════════════════════════════════════════════════

🔒 STRICT FORMAT RULES for the TEXT fields `occluded_object` and `occluder`:

The pipeline feeds these strings DIRECTLY to a referring-expression
segmenter (PSALM) and to CLIP scoring.  The exact phrasing determines
which pixels get segmented.  Follow these rules EXACTLY:

  1) FORMAT  :  `<position> <class>`  OR  `<class>` alone.
     `<position>` is OPTIONAL but REQUIRED when both objects are the
     same class (rule 3 below).  `<class>` is a single common noun
     for the object (animal/thing), NOT a body part or state.

  2) USE CONCRETE CLASS NOUNS:
     ✓ "zebra", "cat", "dog", "horse", "person", "car", "bicycle",
       "branch", "fence", "pumpkin", "log", "fence post", "tree trunk".
     ✗ NEVER use body parts as the class:
         BAD: "head neck", "torso", "'s belly", "shoulder", "rump"
     ✗ NEVER use relational/state words as the class:
         BAD: "nearer", "closer one", "overlapping belly", "in-front",
              "the partially obscured one", "foreground"
     ✗ NEVER use parentheticals or descriptions:
         BAD: "Plains Zebra (Equus quagga) — right/background individual"
         GOOD: "right zebra"

  3) SAME-CLASS DISAMBIGUATION (CRITICAL):
     When the occluder and subject are the SAME CLASS (e.g. zebra-on-
     zebra, cat-on-cat), you MUST prepend a POSITION word so the
     two phrases are distinguishable but USE THE SAME CLASS NOUN:
       GOOD pair:  occluded_object="right zebra"  occluder="left zebra"
       GOOD pair:  occluded_object="back cat"     occluder="front cat"
       BAD  pair:  occluded_object="zebra"        occluder="zebra"
                   (both texts identical — PSALM grabs both)
       BAD  pair:  occluded_object="Plains Zebra" occluder="left individual"
                   (no shared class noun — disambiguation fails)
     Valid position words: left, right, front, back, rear, foreground,
                           background, top, bottom, near, far.
     Pick the position that best describes the OBJECT'S LOCATION IN
     THE IMAGE (use left/right when objects are side-by-side, front/back
     when one is in front of the other).

  4) LENGTH  :  1–3 words.  No commas, no parens, no dashes.

  5) LOWERCASE preferred but accepted as-is.

Worked example for two zebras grazing, front (left) blocking rear (right):
  occluded_object = "right zebra"
  occluder        = "left zebra"

Worked example for a fence in front of a deer:
  occluded_object = "deer"
  occluder        = "fence"

Worked example for two cats, orange one in front of the gray one:
  occluded_object = "back cat"   (or "gray cat" if classes differ enough)
  occluder        = "front cat"  (or "orange cat")

For MODE A (frame_cropped = false):
  • `occluded_object`: see STRICT FORMAT RULES above.
    (1–3 words, class noun, position-prefixed when same-class.)
  • `occluder`: see STRICT FORMAT RULES above.
    (1–3 words, class noun, position-prefixed when same-class.)
  • `what_to_remove`: describe what to erase to reveal the hidden part
  • `selected_segment_ids`: SAM3 segment IDs whose union forms the OCCLUDER mask.
    Be CONSERVATIVE — only segments genuinely in front of the occluded object.
  • `visible_segment_ids`: SAM3 IDs of the VISIBLE (modal) portion of the subject.
    This mask is the critical input to pix2gestalt — make it as precise as possible.
  • `polygon_override`: [OCCLUDER ONLY] refined polygon if SAM3 missed occluder edges.
    Use this to refine the OCCLUDER mask only. Leave [] if SAM3 segments are sufficient.
  • `visible_polygon_override`: [VISIBLE SUBJECT ONLY] hand-drawn polygon tracing the
    VISIBLE portion of the subject when NO SAM3 segment cleanly covers it.
    ⚠ CRITICAL: If `visible_segment_ids` is empty (SAM3 missed the subject's visible region),
    you MUST draw a polygon here tracing the outline of the visible subject pixels.
    Example for a cat face peering through leaves: trace around the head, ears, and any
    visible fur. This polygon becomes the modal mask for pix2gestalt.
    Leave [] ONLY if `visible_segment_ids` already provides a good mask.
  • `hidden_region`: tight bounding box around the physically hidden (occluded) area.
  • `boundary_expansion`: dilation in pixels (15–40) for smooth blending at mask edges.
  • `expansion_directions`: [] — no canvas expansion needed.
  • `expansion_pixels`: {{"top":0,"bottom":0,"left":0,"right":0}}
  • `hidden_polygon`: precise polygon tracing the HIDDEN BODY SILHOUETTE — the
    shape of the SUBJECT's anatomy in the region behind the occluder.
    ⚠ ABSOLUTE RULE: this polygon traces the SUBJECT, NOT the OCCLUDER.
    If your polygon's outline matches the occluder's outline, you are WRONG.

    Anatomy mental check before submitting:
      - Cut the polygon out of the image. Does the cut-out look like a chunk
        of the subject (bird belly+legs, cat torso, human shoulder)?
        ✓ correct.
      - Does it look like a chunk of the occluder (branch, pumpkin rim,
        fence rail)? ✗ wrong — redraw it.

    Examples by occluder geometry:
      • Thin HORIZONTAL occluder (branch, fence rail, log) crossing an
        upright subject:
          → polygon must be a VERTICAL subject-body shape that extends
            DOWNWARD from the visible part, BELOW the occluder, covering
            the subject's lower body parts (belly, legs, feet, tail base).
          → polygon's height is taller than its width.
          → polygon EXTENDS PAST the occluder's outline on both sides
            (subject's feet stick out below the branch).
          → polygon is NOT a thin horizontal strip following the branch.
      • Thin VERTICAL occluder (post, tree trunk) crossing a horizontal
        subject (lying animal, person sideways):
          → polygon is a HORIZONTAL subject-body shape extending PAST the
            post on both sides.
      • Bowl/cavity occluder (pumpkin, basket, box) holding the subject:
          → polygon traces the subject's torso + hindquarters silhouette
            INSIDE the cavity, anatomically continuous with the visible
            head/shoulders. NOT the cavity rim.
      • Foreground subject occluding background subject (horse head in
        front of a person):
          → polygon traces the person's shoulder/torso silhouette behind
            the horse, NOT the horse's outline.

    Construction guidelines:
      - 8–20 points for a smooth anatomy-following outline.
      - Aspect ratio matches the subject's anatomy (tall for upright
        animals/humans; wide for lying ones).
      - The polygon must visually connect to `visible_polygon_override`
        (or the visible SAM3 segments) — imagine the subject with the
        occluder erased; the polygon completes its silhouette.
      - The polygon may extend OUTSIDE the occluder mask. That is correct
        when the hidden body is larger than the occluder.

    This polygon is used directly as the ControlNet inpaint mask. A
    misshapen polygon means the generator paints the subject in the wrong
    pixels — fatal. Be careful.
  • `occluder_click`: pixel (x, y) at the geometric CENTER of the occluder object.
    Used as a SAM3 point prompt to cleanly isolate the occluder.
  • `subject_click`: pixel (x, y) at the CENTER of the visible subject.
    Used as a SAM3 point prompt to cleanly isolate the subject from the occluder.

For MODE B (frame_cropped = true):
  • `occluded_object`: subject being cut off (e.g. "Silver Gull", "person", "German Shepherd")
  • `occluder`: "image frame boundary"
  • `what_to_remove`: "" (nothing to remove from the scene)
  • `selected_segment_ids`: [] (no in-scene occluder)
  • `visible_segment_ids`: SAM3 IDs of the ENTIRE VISIBLE subject body.
    Include all segments that are part of the subject — head, body, wings, visible legs, etc.
    This is the modal mask fed to pix2gestalt so it "sees" the full visible subject.
  • `polygon_override`: []
  • `hidden_region`: bounding box of the AREA NEAR THE FRAME EDGE where the subject exits.
    x1/y1/x2/y2 should match the edge pixel coordinate (e.g. if exiting bottom: y2={h}).
  • `boundary_expansion`: 15–25 px for the generation-to-original blending seam.
  • `hidden_polygon`: polygon tracing the STRIP of canvas that needs to be generated
    (the expansion area near the frame edge). Follow the frame edge on one side and
    the subject's body boundary on the other.
  • `occluder_click`: {{"x":0,"y":0}} (no in-scene occluder in frame-crop mode).
  • `subject_click`: pixel (x, y) at the CENTER of the visible subject body.
  • `expansion_directions`: list of edge names where subject exits with ≥8% body missing.
    Valid values: "top", "bottom", "left", "right"
  • `expansion_pixels`: pixels to ADD to canvas in each direction.
    Use this anatomy-based estimation:

    ┌─────────────────────────────────────────────────────────────┐
    │ ANATOMY PROPORTIONS (% of full body height or width)        │
    │                                                             │
    │ BIRD (standing, folded wings):                              │
    │   Head 15% · Neck 10% · Body/torso 40%                     │
    │   Legs (tarsus+toes) 20% · Tail feathers 15%               │
    │   If feet cut at bottom: expansion ≈ 25% × bird_height_px  │
    │   If tail cut at bottom: expansion ≈ 15% × bird_height_px  │
    │   If wing clipped at side: expansion ≈ 30-50% × bird_width │
    │                                                             │
    │ HUMAN (standing):                                           │
    │   Head 12% · Torso 38% · Upper leg 25% · Lower leg+foot 25%│
    │   If only torso visible: bottom ≈ 50% × visible_height     │
    │   If waist-down missing: bottom ≈ 50% × image_height       │
    │                                                             │
    │ QUADRUPED (dog/cat/horse standing):                         │
    │   Head+neck 25% · Body 40% · Legs 35%                      │
    │   If legs cut: bottom ≈ 40% × visible_body_height          │
    └─────────────────────────────────────────────────────────────┘

    Round to nearest 50 px. Minimum 150 px per active direction.
    Set inactive directions to 0.

══════════════════════════════════════════════════════════════════════
FINAL CHECK BEFORE YOU RESPOND
══════════════════════════════════════════════════════════════════════
Re-read your `occluded_object` and `occluder` strings:
  • Each is 1–3 words.
  • Each contains a CLASS NOUN (e.g. "zebra", "cat", "fence"),
    not a body part ("belly", "neck") or a state word ("nearer").
  • If the two objects are the same class, both strings MUST share
    that class noun and be disambiguated only by a position prefix
    (left/right/front/back/etc).
If your strings fail this check, fix them BEFORE responding.

Respond ONLY in JSON matching the schema."""

    # ── Single GPT-5 orchestrator call (no more geometry-self-check loop) ─────
    # We used to re-prompt GPT up to MAX_GPT_GEOMETRY_RETRIES times when its
    # click coords landed on the subject instead of the occluder.  That
    # was 1-2 extra GPT calls per image.  The same correction can be done
    # PROGRAMMATICALLY (and faster, and free of API flakiness):
    #   - run the single GPT call once
    #   - if click coords land inside the subject or outside the occluder
    #     region, snap them to the centroid of the largest occluder-class
    #     SAM3 segment (or to the GPT polygon_override's centroid if a
    #     polygon was supplied)
    # See block below for the programmatic fix-up.
    data = gpt_vision(
        [state["image_path"], str(sam3_viz_path)],
        prompt, schema=OCCLUSION_SCHEMA,
        cache_key="occlusion_analysis_v1",
    )

    sel_ids      = data.get("selected_segment_ids", [])
    vis_ids      = data.get("visible_segment_ids", [])
    poly_ovr     = data.get("polygon_override", [])
    region       = data.get("hidden_region", {})
    expansion    = int(data.get("boundary_expansion", config.MASK_EXPAND))
    frame_cropped = bool(data.get("frame_cropped", False))
    exp_dirs     = data.get("expansion_directions", [])
    exp_px       = data.get("expansion_pixels", {"top": 0, "bottom": 0, "left": 0, "right": 0})

    # ── Debug override: force ONLY the off-frame extension step ──────────
    # Keeps the rest of the pipeline (mode A vs B routing, mask fusion, Agent
    # 2 path) intact — only the padded-canvas / off-frame mask helper fires.
    # `force_offframe` is read further down where _gpt_extend_amodal_mask_offframe
    # is invoked.
    force_offframe = False
    if getattr(config, "FORCE_FRAME_CROPPED", False):
        force_offframe = True
        if getattr(config, "USE_AUTO_PADDING_BUDGET", False):
            # Auto-budget: pad each side by budget_px, capped so neither padded
            # dim exceeds MAX_PADDED_CANVAS_DIM. Symmetric on each axis.
            budget  = int(getattr(config, "FORCE_FRAME_CROPPED_BUDGET_PX", 400))
            max_dim = int(getattr(config, "MAX_PADDED_CANVAS_DIM", 1280))
            pad_w   = min(budget, max(0, (max_dim - w) // 2))
            pad_h   = min(budget, max(0, (max_dim - h) // 2))
            force_exp_px = {"top": pad_h, "bottom": pad_h,
                            "left": pad_w, "right": pad_w}
            print(f"  [FORCE] auto-budget padding (budget={budget}, max_dim={max_dim}) "
                  f"→ {force_exp_px}  padded={w + 2*pad_w}×{h + 2*pad_h}")
        else:
            force_exp_px = dict(getattr(config, "FORCE_EXPANSION_PIXELS",
                                        {"top": 0, "bottom": 120, "left": 0, "right": 0}))
            print(f"  [FORCE] fixed-padding mode "
                  f"px={force_exp_px}")
    bbox         = [region.get("x1", 0), region.get("y1", 0),
                    region.get("x2", w), region.get("y2", h)]
    # Fix 2: new fields
    hidden_poly  = data.get("hidden_polygon", [])
    occ_click    = data.get("occluder_click", {})
    sub_click    = data.get("subject_click",  {})

    print(f"  Occluded object  : {data.get('occluded_object', '')}")
    print(f"  Occluder         : {data.get('occluder', '')}")
    print(f"  Frame-cropped    : {frame_cropped}")
    if frame_cropped:
        print(f"  Expand dirs      : {exp_dirs}")
        print(f"  Expand pixels    : {exp_px}")
    else:
        print(f"  Occluder seg IDs : {sel_ids}")
    print(f"  Visible seg IDs  : {vis_ids}")
    print(f"  Hidden region    : {region}")
    print(f"  Expansion        : {expansion} px")
    if hidden_poly:
        print(f"  Hidden polygon   : {len(hidden_poly)} pts")
    if occ_click.get("x") or occ_click.get("y"):
        print(f"  Occluder click   : ({occ_click['x']}, {occ_click['y']})")
    if sub_click.get("x") or sub_click.get("y"):
        print(f"  Subject click    : ({sub_click['x']}, {sub_click['y']})")

    (out_dir / "occlusion.json").write_text(json.dumps(data, indent=2))

    seg_by_id = {s["id"]: s for s in segments}

    # ── CLIP-grounded segment labeling (CVPR'25 amodal-style) ────────────────
    # We classify every SAM3 segment against {target, occluder, background, other}
    # using CLIP, then:
    #   - visible_mask candidate = ⋃ segments labeled as target
    #   - occluder_mask candidate = ⋃ segments labeled as occluder
    #     filtered by adjacency (must be within CLIP_ADJACENCY_PX of visible)
    # When both candidates pass quality checks, they replace the GPT-polygon
    # mask. Otherwise we fall back to the existing polygon path.  This avoids
    # GPT coordinate hallucinations (the bear-face / pumpkin-rim bug).
    target_class   = (data.get("occluded_object", "") or "").strip()
    occluder_class = (data.get("occluder", "") or "").strip()
    target_short   = _extract_short_label(target_class)
    occluder_short = _extract_short_label(occluder_class)
    clip_visible   = np.zeros((h, w), dtype=np.uint8)
    clip_occluder  = np.zeros((h, w), dtype=np.uint8)
    clip_grounded  = False

    if (
        getattr(config, "USE_CLIP_GROUNDING", False)
        and not frame_cropped
        and target_class and occluder_class
        and len(segments) > 0
    ):
        # Short labels work better for CLIP than the verbose GPT descriptions.
        # Track both "verbose" and "short" forms for clearer logs.
        target_label   = target_short or target_class
        occluder_label = occluder_short or occluder_class
        print(f"  [CLIP] Labels: target='{target_label}'  occluder='{occluder_label}'")
        text_labels = [target_label, occluder_label, "background", "other object"]
        seg_labels  = _clip_label_segments(img_rgb := cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                            segments, text_labels)
        _save_clip_label_viz(img, segments, seg_by_id, seg_labels,
                             target_label, occluder_label,
                             out_dir / "clip_labels.png")
        # Build CLIP candidate masks (compare against the short labels we sent)
        for sl in seg_labels:
            sid = sl["id"]
            if sid not in seg_by_id:
                continue
            if sl["label"] is None or sl["score"] < config.CLIP_MIN_SCORE:
                continue
            if sl["label"] == target_label:
                clip_visible = np.clip(clip_visible | seg_by_id[sid]["mask"], 0, 1)
            elif sl["label"] == occluder_label:
                clip_occluder = np.clip(clip_occluder | seg_by_id[sid]["mask"], 0, 1)

        # Adjacency filter: only keep occluder segments touching/near the visible
        if clip_visible.sum() > 0 and clip_occluder.sum() > 0:
            adj_k = max(3, config.CLIP_ADJACENCY_PX)
            adj_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (adj_k * 2 + 1, adj_k * 2 + 1))
            vis_dilated = cv2.dilate(clip_visible, adj_kernel, iterations=1)
            before = int(clip_occluder.sum())
            clip_occluder = np.clip(clip_occluder & vis_dilated, 0, 1).astype(np.uint8)
            after  = int(clip_occluder.sum())
            print(f"  [CLIP] candidates: visible={int(clip_visible.sum())} px, "
                  f"occluder={before} px → {after} px (after adjacency filter)")

        # Decide whether to USE the CLIP-grounded masks
        if (
            clip_visible.sum() > 0
            and clip_occluder.sum() >= config.CLIP_OCCLUDER_MIN_AREA
        ):
            # Refuse to replace if the CLIP "occluder" mostly overlaps the visible
            # subject — would mean CLIP got confused by similar textures.
            ovl = int((clip_occluder.astype(bool) & clip_visible.astype(bool)).sum())
            ovl_ratio = ovl / max(int(clip_occluder.sum()), 1)
            if ovl_ratio < 0.35:
                clip_grounded = True
                print(f"  [CLIP-grounded] ✓ Using CLIP segment masks "
                      f"(visible={int(clip_visible.sum())} px, "
                      f"occluder={int(clip_occluder.sum())} px, ovl={ovl_ratio:.2f}) → "
                      f"clip_labels.png")
            else:
                print(f"  [CLIP-grounded] ✗ occluder/visible overlap {ovl_ratio:.2f} too high — "
                      f"falling back to GPT polygon path")
        else:
            print(f"  [CLIP-grounded] ✗ no confident occluder segments found "
                  f"(visible={int(clip_visible.sum())} px, "
                  f"occluder={int(clip_occluder.sum())} px) — "
                  f"trying Grounding DINO + CLIP-grid fallbacks")

    # ── Fallback chain when primary CLIP-on-segments missed the occluder ─────
    # D: Grounding DINO → bbox(es) for the short occluder noun → SAM3 point-
    #    prompt at each bbox center → union the masks.
    # C: CLIP-on-grid → top-K patch centers as SAM3 point prompts.
    # Either path produces a `clip_occluder` mask we then run through the same
    # adjacency filter and acceptance gate as the primary path.
    if (
        not clip_grounded
        and not frame_cropped
        and target_class and occluder_class
        and len(segments) > 0
    ):
        fallback_occ = np.zeros((h, w), dtype=np.uint8)
        fallback_src = None

        # ---- D: Grounding DINO + SAM3 BBOX prompt ---------------------------
        # Pass bboxes directly to SAM3 (not just centers) so SAM3 segments the
        # whole occluder inside the bbox, not a small region around a click.
        # This is the critical fix for log-shaped/elongated occluders that a
        # point-prompt would only partially cover.
        if getattr(config, "USE_GROUNDING_DINO", False):
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            detections = _grounding_dino_detect(pil_img, occluder_short or occluder_class)
            boxes = []
            for det in detections[:5]:  # take up to 5 best detections
                bx1, by1, bx2, by2 = det["box"]
                bx1 = max(0, min(w - 1, int(bx1)))
                by1 = max(0, min(h - 1, int(by1)))
                bx2 = max(0, min(w - 1, int(bx2)))
                by2 = max(0, min(h - 1, int(by2)))
                if bx2 - bx1 < 8 or by2 - by1 < 8:
                    continue
                boxes.append({"label": f"gdino_{len(boxes)}",
                              "box": [bx1, by1, bx2, by2]})
            if boxes:
                targeted = _sam_segment_with_boxes(state["image_path"], boxes, out_dir)
                img_area = h * w
                for t in targeted:
                    tarea = int(t["mask"].sum())
                    if 0 < tarea < 0.55 * img_area:   # allow a slightly larger occluder
                        fallback_occ = np.clip(fallback_occ | t["mask"], 0, 1).astype(np.uint8)
                if fallback_occ.sum() > 0:
                    fallback_src = "GroundingDINO+SAM3-bbox"

        # ---- C: CLIP-on-grid → SAM3 point-prompt -----------------------------
        if fallback_occ.sum() == 0 and getattr(config, "USE_CLIP_GRID_FALLBACK", False):
            print("  [Fallback] Grounding DINO returned nothing — trying CLIP-grid")
            heatmap, hits = _clip_grid_locate(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                occluder_short or occluder_class,
                grid_n=config.CLIP_GRID_N,
            )
            _save_clip_grid_viz(img, heatmap, out_dir / "clip_grid_heatmap.png")
            anchors = []
            for r, c, score, cx, cy in hits[: config.CLIP_GRID_TOP_K]:
                if score < config.CLIP_GRID_MIN_SCORE:
                    break
                anchors.append({"label": f"grid_{r}_{c}", "x": int(cx), "y": int(cy)})
            if anchors:
                targeted = _sam_segment_targeted(state["image_path"], anchors, out_dir)
                img_area = h * w
                for t in targeted:
                    tarea = int(t["mask"].sum())
                    if 0 < tarea < 0.50 * img_area:
                        fallback_occ = np.clip(fallback_occ | t["mask"], 0, 1).astype(np.uint8)
                if fallback_occ.sum() > 0:
                    fallback_src = "CLIP-grid+SAM3"

        # ---- Adjacency filter + acceptance gate (same as primary path) -------
        if fallback_occ.sum() > 0:
            # Need a visible mask to anchor adjacency. Prefer CLIP visible if we
            # have it; else fall back to the visible_polygon_override.
            adj_anchor = clip_visible.copy()
            if adj_anchor.sum() == 0:
                vis_poly_ovr_local = data.get("visible_polygon_override", [])
                if len(vis_poly_ovr_local) >= 3:
                    pts_v = np.array([[int(p[0]), int(p[1])] for p in vis_poly_ovr_local], dtype=np.int32)
                    cv2.fillPoly(adj_anchor, [pts_v], 1)
            if adj_anchor.sum() > 0:
                adj_k = max(3, config.CLIP_ADJACENCY_PX)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (adj_k * 2 + 1, adj_k * 2 + 1))
                vis_dil = cv2.dilate(adj_anchor, k, iterations=1)
                before = int(fallback_occ.sum())
                fallback_occ = np.clip(fallback_occ & vis_dil, 0, 1).astype(np.uint8)
                print(f"  [{fallback_src}] adjacency filter: {before} → {int(fallback_occ.sum())} px")

            ovl = int((fallback_occ.astype(bool) & adj_anchor.astype(bool)).sum())
            ovl_ratio = ovl / max(int(fallback_occ.sum()), 1)
            if (
                fallback_occ.sum() >= config.CLIP_OCCLUDER_MIN_AREA
                and ovl_ratio < 0.50
            ):
                clip_occluder  = fallback_occ
                clip_visible   = adj_anchor if adj_anchor.sum() > clip_visible.sum() else clip_visible
                clip_grounded  = True
                cv2.imwrite(str(out_dir / f"fallback_occluder_{fallback_src}.png"),
                            fallback_occ * 255)
                print(f"  [Fallback] ✓ {fallback_src} produced usable occluder "
                      f"({int(fallback_occ.sum())} px, ovl={ovl_ratio:.2f})")
            else:
                print(f"  [Fallback] ✗ {fallback_src} mask rejected "
                      f"(area={int(fallback_occ.sum())} px, ovl={ovl_ratio:.2f}) — "
                      f"falling through to GPT polygon path")
        else:
            print("  [Fallback] No fallback path produced a non-empty occluder")

    # ── Occluder mask candidates ──────────────────────────────────────────────
    # Each source produces a binary candidate; we fuse them according to
    # config.MASK_FUSION_MODE ("majority" / "union" / "intersection" / "priority").
    # This replaces the old priority-only chain — every source is now a vote.
    mask_candidates: list = []   # [(name: str, mask: np.uint8 0/1)]
    poly_used = None

    # A) Vision-grounded path (CLIP-on-SAM3 + GroundingDINO + CLIP-grid all
    #    funnel into clip_occluder; we treat them as a single vote here).
    if clip_grounded and clip_occluder.sum() > 0:
        mask_candidates.append(("clip_segments", clip_occluder.astype(np.uint8).copy()))

    # B) PSALM referring-expression segmentation (own venv subprocess).
    psalm_text = (occluder_short or (data.get("occluder", "") or "")).strip()
    psalm_mask = _psalm_referring_seg(state["image_path"], psalm_text, out_dir)
    if psalm_mask is not None and psalm_mask.sum() > 0:
        if psalm_mask.shape[:2] != (h, w):
            psalm_mask = cv2.resize(psalm_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_candidates.append(("psalm", psalm_mask.astype(np.uint8)))

    # C) GPT polygon — REMOVED.  GPT polygons + GPT segment IDs are not
    # independent of the GPT-supplied occluder description that drives PSALM
    # and CLIP; including them as separate "votes" creates correlated noise
    # (they all swing the same way when GPT's narrative is wrong).  We keep
    # the polygon coordinates around for the geometry self-check only.
    if len(poly_ovr) >= 3:
        pts = np.array([[int(p[0]), int(p[1])] for p in poly_ovr], dtype=np.int32)
        poly_used = pts

    # D) GPT segments — REMOVED for the same reason as C.

    # E) InstaOrder learned pairwise occlusion-order.
    # Resolves same-class occlusion (zebra/zebra, cat/cat) that CLIP-text and
    # GPT polygons cannot — we ask: of all SAM3 segments, which are predicted
    # to be IN FRONT of the visible target?  Their union is the occluder.
    if getattr(config, "USE_INSTAORDER", False):
        # Build the target (visible) mask the same way the downstream
        # visible_mask pass does, but limited to signals we have RIGHT NOW
        # (clip_visible + GPT vis_ids).  This is a pre-fusion seed.
        io_target = np.zeros((h, w), dtype=np.uint8)
        if clip_grounded and clip_visible.sum() > 0:
            io_target = np.clip(io_target | clip_visible.astype(np.uint8), 0, 1)
        for sid in (vis_ids or []):
            if sid in seg_by_id:
                io_target = np.clip(io_target | seg_by_id[sid]["mask"], 0, 1)

        if io_target.sum() > 0 and seg_by_id:
            try:
                from instaorder_helper import (
                    get_instaorder_model, occluders_above,
                )
                io_model = get_instaorder_model(
                    repo_dir=config.INSTAORDER_REPO_DIR,
                    ckpt_path=config.INSTAORDER_CKPT,
                )
                io_cands = [(f"sam3_{sid}", seg_by_id[sid]["mask"])
                            for sid in seg_by_id.keys()]
                # img is BGR from cv2.imread at the top of occlusion_agent.
                io_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                io_mask = occluders_above(
                    model           = io_model,
                    image_rgb       = io_image_rgb,
                    target_mask     = io_target,
                    candidate_masks = io_cands,
                    input_size      = getattr(config, "INSTAORDER_INPUT_SIZE", 384),
                    occ_prob_thresh = getattr(config, "INSTAORDER_OCC_THRESH", 0.5),
                )
                if io_mask is not None and io_mask.sum() > 0:
                    mask_candidates.append(("instaorder", io_mask.astype(np.uint8)))
            except Exception as exc:                          # noqa: BLE001
                print(f"  [InstaOrder] skipped: {exc!r}")
        else:
            print("  [InstaOrder] no target seed yet — skipping")

    # F) PSALM-class + InstaOrder pairwise split (same-class occlusion).
    # When the occluder and the subject are instances of the SAME class
    # (zebra-on-zebra, cat-on-cat), CLIP text scoring labels both segments
    # identically and PSALM with the class noun returns BOTH instances.
    # This block:
    #   1. Calls PSALM with the shared class noun → unified instance map
    #   2. Splits the result into connected components
    #   3. Uses InstaOrder pairwise to rank each component by "frontness"
    #   4. Top component → occluder candidate (instaorder_pair_occluder)
    #      All others   → subject visible candidate (instaorder_pair_subject,
    #      stored in `instaorder_pair_visible` for use during visible_mask build)
    instaorder_pair_visible: Optional[np.ndarray] = None

    def _core_class_noun(text: str) -> str:
        """Pull the class noun out of a GPT description.

        Strips parenthesised species names, em/en/hyphen-dash side-clauses
        (e.g. '— right/background individual'), and a small dictionary of
        position/role words (left/right/front/back/foreground/background/
        individual/specimen/etc) so the remaining last word is the actual
        class noun ('zebra', 'cat', 'log').  Returns "" if it can't find one.
        """
        if not text:
            return ""
        s = re.sub(r"\([^)]*\)", "", text)
        # Strip everything after the first em/en/hyphen dash (used as a
        # parenthetical role descriptor).
        s = re.split(r"[—–]|\s-\s", s, maxsplit=1)[0]
        role_words = {
            "left", "right", "rear", "front", "back", "foreground", "background",
            "individual", "specimen", "instance", "object", "side", "main",
            "primary", "the", "a", "an", "of", "in", "with", "from",
            "occluding", "occluded", "subject", "occluder",
        }
        words = [w for w in re.findall(r"[A-Za-z]+", s)
                 if w.lower() not in role_words
                 and w.lower() not in _QUALIFIER_WORDS]
        return words[-1].lower() if words else ""

    try:
        tgt_short = _core_class_noun(target_class) or _core_class_noun(target_short)
        occ_short = _core_class_noun(occluder_class) or _core_class_noun(occluder_short)
        # Exact match: tgt_short and occ_short agree on the core class noun.
        same_class = bool(tgt_short) and tgt_short == occ_short

        # Containment match: when extraction picked different last words
        # ("zebra" vs "neck"), check if EITHER description contains the OTHER
        # core noun anywhere as a whole-word substring.  Covers cases like
        # target='Plains zebra' / occluder='foreground zebra head-neck region'
        # where both texts reference 'zebra' but only one ends in 'zebra'.
        if not same_class and tgt_short and occluder_class:
            same_class = bool(re.search(rf"\b{re.escape(tgt_short)}\b",
                                        occluder_class, re.I))
        if not same_class and occ_short and target_class:
            same_class = bool(re.search(rf"\b{re.escape(occ_short)}\b",
                                        target_class, re.I))
        print(f"  [InstaOrder-pair-debug] tgt='{tgt_short}' occ='{occ_short}' "
              f"same_class={same_class}")
    except Exception:
        same_class = False

    if same_class and getattr(config, "USE_INSTAORDER", False):
        print(f"  [InstaOrder-pair] same-class occlusion detected: '{tgt_short}'")
        # Use the broader class noun (the SHORT label) so PSALM matches both.
        psalm_class_text = tgt_short
        psalm_class_mask = _psalm_referring_seg(
            state["image_path"], psalm_class_text, out_dir,
            out_filename="psalm_class_mask.png",
        )
        if psalm_class_mask is not None and psalm_class_mask.sum() > 0:
            if psalm_class_mask.shape[:2] != (h, w):
                psalm_class_mask = cv2.resize(
                    psalm_class_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Connected components — each is a candidate instance.
            # When two instances touch in pixel-space (e.g. shoulder against
            # shoulder), PSALM returns them merged into one blob.  Iterative
            # erosion breaks the thin joining bridge so CC can split them;
            # we then DILATE each component back into the original PSALM
            # mask using a watershed-like expansion (per-component dilate
            # restricted by the original mask).
            min_area = max(200, int(0.005 * h * w))

            def _split_via_cc(binary_mask: np.ndarray, k_erode: int) -> list:
                eroded = (binary_mask.copy()
                          if k_erode <= 0
                          else cv2.erode(
                              binary_mask,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                         (3, 3)),
                              iterations=k_erode))
                n_cc_, lbl_map_ = cv2.connectedComponents(eroded.astype(np.uint8))
                out = []
                for k in range(1, n_cc_):
                    m = (lbl_map_ == k).astype(np.uint8)
                    if int(m.sum()) >= min_area:
                        out.append((k, int(m.sum()), m))
                return out

            cc_masks = _split_via_cc(psalm_class_mask, 0)

            # If we got only 1 blob, try erosions of 2 / 4 / 6 to break a
            # touching bridge.  Stop as soon as we get ≥2 components.
            erode_used = 0
            if len(cc_masks) < 2:
                for k_erode in (2, 4, 6, 8):
                    cand = _split_via_cc(psalm_class_mask, k_erode)
                    if len(cand) >= 2:
                        # Re-grow each component back inside the original mask
                        # using a few dilations clipped by psalm_class_mask.
                        kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (3, 3))
                        regrown = []
                        for cid, _, m in cand:
                            grown = cv2.dilate(m, kernel,
                                               iterations=k_erode + 1)
                            grown = (grown.astype(bool)
                                     & psalm_class_mask.astype(bool)
                                    ).astype(np.uint8)
                            regrown.append((cid, int(grown.sum()), grown))
                        # Resolve any overlap by giving each pixel to the
                        # nearest seed component (one-pass argmax of dilation
                        # count).  Approximation: process in size order, later
                        # components only claim pixels not already claimed.
                        regrown.sort(key=lambda t: t[1], reverse=True)
                        used = np.zeros_like(psalm_class_mask, dtype=bool)
                        final_ccs: list = []
                        for cid, _, g in regrown:
                            unique = (g.astype(bool) & ~used).astype(np.uint8)
                            if int(unique.sum()) >= min_area:
                                final_ccs.append((cid, int(unique.sum()), unique))
                                used |= unique.astype(bool)
                        cc_masks = final_ccs
                        erode_used = k_erode
                        break

            cc_masks.sort(key=lambda t: t[1], reverse=True)
            print(f"  [InstaOrder-pair] PSALM('{psalm_class_text}') → "
                  f"{int(psalm_class_mask.sum())} px, {len(cc_masks)} "
                  f"component(s) ≥ {min_area} px "
                  f"(erode_k={erode_used})")

            # Last-resort split: ask SAM3 to do INSTANCE segmentation directly,
            # using the two GPT-supplied click points (occluder_click +
            # subject_click).  When erosion can't break the bridge between
            # touching instances, SAM3 with TWO positive prompts returns
            # ONE clean mask PER POINT — which is exactly what we need to
            # feed InstaOrder pairwise.  We intersect each click-mask with
            # the original PSALM-class blob so we never wander off-subject.
            if len(cc_masks) < 2:
                ox = int(occ_click.get("x", 0))
                oy = int(occ_click.get("y", 0))
                sx2 = int(sub_click.get("x", 0))
                sy2 = int(sub_click.get("y", 0))
                if (0 < ox < w and 0 < oy < h
                        and 0 < sx2 < w and 0 < sy2 < h):
                    print("  [InstaOrder-pair] erosion couldn't split — "
                          "falling back to SAM3 dual-click (occluder + subject)")
                    # SAM3's transformers pipeline only emits ONE mask per
                    # batched call regardless of how many prompt-sets we pass,
                    # so call it SEPARATELY for each click to get 2 distinct
                    # per-instance masks.
                    dual_occ = _sam_segment_targeted(
                        state["image_path"],
                        [{"label": "pair_occ", "x": ox, "y": oy}],
                        out_dir,
                    )
                    dual_sub = _sam_segment_targeted(
                        state["image_path"],
                        [{"label": "pair_sub", "x": sx2, "y": sy2}],
                        out_dir,
                    )
                    if dual_occ and dual_sub:
                        occ_blob = dual_occ[0]["mask"]
                        sub_blob = dual_sub[0]["mask"]
                        if occ_blob.shape[:2] != (h, w):
                            occ_blob = cv2.resize(
                                occ_blob, (w, h), interpolation=cv2.INTER_NEAREST)
                        if sub_blob.shape[:2] != (h, w):
                            sub_blob = cv2.resize(
                                sub_blob, (w, h), interpolation=cv2.INTER_NEAREST)
                        # Restrict each click-mask to the PSALM-class blob
                        # (anything outside it isn't even of the right class).
                        occ_blob = (occ_blob.astype(bool)
                                    & psalm_class_mask.astype(bool)
                                   ).astype(np.uint8)
                        sub_blob = (sub_blob.astype(bool)
                                    & psalm_class_mask.astype(bool)
                                   ).astype(np.uint8)
                        # Resolve any pixel claimed by both: give it to
                        # whichever click-mask is closer to its own click.
                        both = occ_blob.astype(bool) & sub_blob.astype(bool)
                        if both.any():
                            # Coarse: assign each disputed pixel to whichever
                            # click is geometrically nearer.
                            ys, xs = np.where(both)
                            d_occ = (ys - oy) ** 2 + (xs - ox) ** 2
                            d_sub = (ys - sy2) ** 2 + (xs - sx2) ** 2
                            sub_blob[ys[d_occ < d_sub], xs[d_occ < d_sub]] = 0
                            occ_blob[ys[d_sub <= d_occ], xs[d_sub <= d_occ]] = 0
                        # Replace cc_masks with the two clean per-instance blobs.
                        a_occ = int(occ_blob.sum())
                        a_sub = int(sub_blob.sum())
                        print(f"  [InstaOrder-pair] SAM3-dual: "
                              f"occluder={a_occ} px, subject={a_sub} px")
                        if a_occ >= min_area and a_sub >= min_area:
                            cc_masks = [
                                (1, a_occ, occ_blob),
                                (2, a_sub, sub_blob),
                            ]
                            erode_used = -1  # signal "via SAM3 dual-click"
                            cc_masks.sort(key=lambda t: t[1], reverse=True)

            if len(cc_masks) >= 2:
                try:
                    from instaorder_helper import (
                        get_instaorder_model, rank_by_frontness,
                    )
                    io_model = get_instaorder_model(
                        repo_dir=config.INSTAORDER_REPO_DIR,
                        ckpt_path=config.INSTAORDER_CKPT,
                    )
                    cc_only_masks = [c[2] for c in cc_masks]
                    instaorder_ranking = None
                    if io_model is not None:
                        io_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        instaorder_ranking = rank_by_frontness(
                            io_model, io_image_rgb, cc_only_masks,
                            input_size=getattr(config, "INSTAORDER_INPUT_SIZE", 384),
                        )

                    # Independent vision-only ranker: depth.  Higher
                    # predicted_depth = closer to camera = front.
                    depth_ranking = None
                    if getattr(config, "USE_DEPTH_RANK", True):
                        depth_ranking = _depth_rank_masks(
                            state["image_path"], cc_only_masks)
                        if depth_ranking is not None:
                            print("  [Depth-rank] " + " ".join(
                                f"cc{r['index']}:{r['median_depth']:.2f}"
                                for r in depth_ranking))

                    # Combine InstaOrder + Depth.  If they agree, high
                    # confidence.  If they disagree, prefer depth (it's a
                    # pure-vision signal that doesn't depend on any model
                    # learned on a specific dataset).
                    if instaorder_ranking is not None and depth_ranking is not None:
                        io_top    = instaorder_ranking[0]["index"]
                        depth_top = depth_ranking[0]["index"]
                        if io_top == depth_top:
                            print(f"  [InstaOrder-pair] AGREE → front=cc{io_top}")
                            ranking = instaorder_ranking
                        else:
                            print(f"  [InstaOrder-pair] DISAGREE — "
                                  f"InstaOrder→cc{io_top}, Depth→cc{depth_top} "
                                  f"; preferring Depth")
                            ranking = depth_ranking
                    elif depth_ranking is not None:
                        print("  [InstaOrder-pair] using Depth-only ranking "
                              "(InstaOrder model unavailable)")
                        ranking = depth_ranking
                    elif instaorder_ranking is not None:
                        print("  [InstaOrder-pair] using InstaOrder-only ranking "
                              "(Depth unavailable)")
                        ranking = instaorder_ranking
                    else:
                        print("  [InstaOrder-pair] both rankers unavailable — "
                              "skipping pair split")
                        ranking = None

                    if ranking is not None:
                        # Most-front component = occluder.
                        front_idx = ranking[0]["index"]
                        front_mask = cc_masks[front_idx][2]
                        # Everyone else = visible subject parts.
                        back_mask = np.zeros_like(front_mask)
                        for idx, _ in enumerate(cc_masks):
                            if idx != front_idx:
                                back_mask = np.clip(
                                    back_mask | cc_masks[idx][2], 0, 1
                                ).astype(np.uint8)

                        # Tie-breaker: if subject_click landed on `front_idx`,
                        # the ranking is reversed — swap.
                        sx = int(sub_click.get("x", 0))
                        sy = int(sub_click.get("y", 0))
                        if 0 < sx < w and 0 < sy < h and front_mask[sy, sx] > 0:
                            print("  [InstaOrder-pair] subject_click lands on "
                                  "front candidate → swapping (treating it as "
                                  "back instead)")
                            front_mask, back_mask = back_mask, front_mask

                        # Save + register as candidates.
                        cv2.imwrite(str(out_dir / "cand_instaorder_pair_occluder.png"),
                                    front_mask * 255)
                        cv2.imwrite(str(out_dir / "cand_instaorder_pair_subject.png"),
                                    back_mask * 255)
                        front_px = int(front_mask.sum())
                        back_px  = int(back_mask.sum())
                        # Score key depends on ranker; both work for printing.
                        score_key = ("front_score" if "front_score" in ranking[0]
                                     else "median_depth")
                        print(f"  [InstaOrder-pair] front (occluder) = {front_px} px, "
                              f"back (subject) = {back_px} px  "
                              f"{score_key}s="
                              f"{[round(r[score_key], 2) for r in ranking]}")
                        if front_px > 0:
                            mask_candidates.append(
                                ("instaorder_pair", front_mask))
                        if back_px > 0:
                            instaorder_pair_visible = back_mask
                except Exception as exc:                       # noqa: BLE001
                    print(f"  [InstaOrder-pair] skipped: {exc!r}")
            else:
                print(f"  [InstaOrder-pair] only {len(cc_masks)} component(s) — "
                      "can't pair-rank, skipping")

    # Persist each candidate for inspection.
    for name, m in mask_candidates:
        cv2.imwrite(str(out_dir / f"cand_{name}.png"), (m.astype(np.uint8) * 255))
    print(f"  Mask candidates  : "
          f"{[(n, int(m.sum())) for n, m in mask_candidates] or 'none'}")

    # ── Fuse candidates → final occluder mask ────────────────────────────────
    fusion_mode  = getattr(config, "MASK_FUSION_MODE", "priority")
    fusion_min   = int(getattr(config, "MASK_FUSION_MIN_AGREE", 2))
    fusion_order = getattr(config, "MASK_FUSION_PRIORITY", None)

    if mask_candidates:
        occluder_mask, used_sources = _fuse_masks(
            mask_candidates, mode=fusion_mode,
            min_agree=fusion_min, priority=fusion_order,
        )
        extra = f" min_agree={fusion_min}" if fusion_mode == "majority" else ""
        print(f"  [Fusion] mode={fusion_mode}{extra}  "
              f"used={used_sources}  final={int(occluder_mask.sum())} px")

        # Majority/intersection can be empty when sources disagree strongly.
        # Retry as union before falling through to the bbox fallback so we at
        # least mask SOMETHING that the downstream pipeline can work with.
        if occluder_mask.sum() == 0 and fusion_mode != "union":
            print("  [Fusion] empty result — retrying as union")
            occluder_mask, used_sources = _fuse_masks(mask_candidates, mode="union")
            print(f"  [Fusion] union → {int(occluder_mask.sum())} px "
                  f"(sources: {used_sources})")
    elif frame_cropped:
        occluder_mask = np.zeros((h, w), dtype=np.uint8)
        print("  Frame-crop mode: no in-scene occluder mask (canvas will be expanded at inpainting step)")
    else:
        print("  WARNING: no occluder candidates — falling back to hidden_region bbox")
        occluder_mask = np.zeros((h, w), dtype=np.uint8)
        occluder_mask[
            max(0, int(bbox[1])):min(h, int(bbox[3])),
            max(0, int(bbox[0])):min(w, int(bbox[2])),
        ] = 1

    # Hidden object mask: the missing part to create. Keep this separate from the
    # occluder mask; mixing them makes background fill and object synthesis fight.
    hidden_mask = np.zeros((h, w), dtype=np.uint8)
    hidden_poly_data = data.get("hidden_polygon", [])
    if len(hidden_poly_data) >= 3:
        pts_h = np.array([[int(p[0]), int(p[1])] for p in hidden_poly_data], dtype=np.int32)
        cv2.fillPoly(hidden_mask, [pts_h], 1)
        print(f"  Hidden object mask: {hidden_mask.sum()} px")

    mask_save = out_dir / "occluder_mask.png"
    cv2.imwrite(str(mask_save), occluder_mask * 255)
    print(f"  Occluder mask    : {occluder_mask.sum()} px → {mask_save}")

    hidden_mask_save = out_dir / "hidden_object_mask.png"
    cv2.imwrite(str(hidden_mask_save), hidden_mask * 255)
    print(f"  Hidden mask      : {hidden_mask.sum()} px → {hidden_mask_save}")

    # ── Visible object mask (modal mask for pix2gestalt) ──────────────────────
    visible_mask = np.zeros((h, w), dtype=np.uint8)

    # Highest-priority seed: PSALM-class + InstaOrder pair-rank "back" side.
    # When same-class occlusion is detected, this is the most reliable signal —
    # it's the actual subject instance silhouette, not a text-derived guess.
    if instaorder_pair_visible is not None and instaorder_pair_visible.sum() > 0:
        visible_mask = instaorder_pair_visible.copy()
        print(f"  InstaOrder-pair visible seed: {int(visible_mask.sum())} px")

    # Seed from CLIP-labeled "target" segments when available
    if clip_grounded and clip_visible.sum() > 0:
        visible_mask = np.clip(visible_mask | clip_visible, 0, 1).astype(np.uint8)
        print(f"  CLIP-grounded visible seed: {int(visible_mask.sum())} px")

    if vis_ids:
        for sid in vis_ids:
            if sid in seg_by_id:
                visible_mask = np.clip(visible_mask | seg_by_id[sid]["mask"], 0, 1)
                print(f"  Merged visible  seg {sid:03d} : area={seg_by_id[sid]['area']} px")
            else:
                print(f"  WARNING: visible segment {sid} not found — skipped")

    # Apply visible_polygon_override — used when SAM3 missed the visible subject
    vis_poly_ovr = data.get("visible_polygon_override", [])
    vis_poly_used = None
    if len(vis_poly_ovr) >= 3:
        pts_v = np.array([[int(p[0]), int(p[1])] for p in vis_poly_ovr], dtype=np.int32)
        vis_poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(vis_poly_mask, [pts_v], 1)
        visible_mask = np.clip(visible_mask | vis_poly_mask, 0, 1)
        vis_poly_used = pts_v
        print(f"  Visible polygon override applied ({len(vis_poly_ovr)} pts, area={vis_poly_mask.sum()} px)")

    # ── Expand visible_mask via a SECOND PSALM call on the SUBJECT class ─────
    # The first PSALM call segmented the occluder.  Now query PSALM for the
    # subject so we capture ALL visible parts of it, not just the head/sliver
    # GPT's polygon happens to include.
    #
    # PSALM with a class noun (e.g. "Plains Zebra") will return EVERY instance
    # of that class — including the occluder if it's the same class.  After
    # subtracting the occluder mask, we restrict to the CONNECTED COMPONENT
    # that contains `subject_click` (a pixel GPT identified as on the
    # subject), so we keep only the rear-zebra blob and drop any leftover
    # front-zebra pixels caused by an under-sized occluder mask.
    subject_text = (data.get("occluded_object", "") or "").strip()
    # Drop trailing parentheticals like "(Equus quagga)" which confuse PSALM.
    if "(" in subject_text:
        subject_text = subject_text.split("(")[0].strip()
    if subject_text:
        psalm_subj = _psalm_referring_seg(
            state["image_path"], subject_text, out_dir,
            out_filename="psalm_subject_mask.png",
        )
        if psalm_subj is not None and psalm_subj.sum() > 0:
            if psalm_subj.shape[:2] != (h, w):
                psalm_subj = cv2.resize(psalm_subj, (w, h),
                                        interpolation=cv2.INTER_NEAREST)
            # Subject can't include occluder pixels — subtract.
            psalm_subj_clean = np.clip(psalm_subj.astype(np.int32)
                                       - occluder_mask.astype(np.int32),
                                       0, 1).astype(np.uint8)

            # Connected-component filter: keep only the blob(s) connected to
            # subject_click.  If no click is available or it doesn't land on
            # any blob, fall back to the largest connected component.
            n_lbl, lbl_map = cv2.connectedComponents(psalm_subj_clean)
            sx = int(sub_click.get("x", 0))
            sy = int(sub_click.get("y", 0))
            target_lbl = 0
            if 0 < sx < w and 0 < sy < h and lbl_map[sy, sx] != 0:
                target_lbl = int(lbl_map[sy, sx])
            else:
                # No usable click — pick the largest non-background component.
                sizes = [int((lbl_map == k).sum()) for k in range(1, n_lbl)]
                if sizes:
                    target_lbl = 1 + sizes.index(max(sizes))
            psalm_subj_picked = ((lbl_map == target_lbl).astype(np.uint8)
                                 if target_lbl > 0
                                 else np.zeros_like(psalm_subj_clean))

            # Tighten via SAM3 point-prompt at subject_click — but ONLY when
            # SAM3 returns a mask comparable in size to PSALM-subject.  A
            # tiny SAM3 click-mask (e.g. 2k px when PSALM-subject is 20k+)
            # almost always means SAM3 latched onto a sub-part (head only,
            # a leg only).  Intersecting with it would WIPE the larger
            # correct PSALM mask.
            if 0 < sx < w and 0 < sy < h and int(psalm_subj_picked.sum()) > 0:
                targeted_subj = _sam_segment_targeted(
                    state["image_path"],
                    [{"label": "subject_refine", "x": sx, "y": sy}],
                    out_dir,
                )
                if targeted_subj and int(targeted_subj[0]["mask"].sum()) > 0:
                    sam_subj = targeted_subj[0]["mask"]
                    if sam_subj.shape[:2] != (h, w):
                        sam_subj = cv2.resize(sam_subj, (w, h),
                                              interpolation=cv2.INTER_NEAREST)
                    psalm_px = int(psalm_subj_picked.sum())
                    sam_px   = int(sam_subj.sum())
                    # Guard: only intersect when SAM3 covers ≥ 40% of PSALM
                    # AND SAM3 ≥ 25% of psalm_subj_picked.  Otherwise SAM3 is
                    # too narrow and the intersection would destroy signal.
                    if sam_px >= 0.40 * psalm_px:
                        sam_dilated = cv2.dilate(
                            sam_subj.astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                            iterations=1,
                        )
                        intersected = (psalm_subj_picked.astype(bool)
                                       & sam_dilated.astype(bool)
                                      ).astype(np.uint8)
                        intersected = np.clip(
                            intersected.astype(np.int32)
                            - occluder_mask.astype(np.int32),
                            0, 1).astype(np.uint8)
                        if int(intersected.sum()) >= 0.25 * psalm_px:
                            psalm_subj_picked = intersected
                            print(f"  PSALM-subject ∩ SAM3-point-prompt: "
                                  f"{psalm_px} → {int(psalm_subj_picked.sum())} "
                                  f"px (sam3={sam_px}, dilated="
                                  f"{int(sam_dilated.sum())})")
                        else:
                            print(f"  [SAM3-intersect] skipped — intersection "
                                  f"would drop to {int(intersected.sum())} px "
                                  f"(< 25% of psalm {psalm_px})")
                    else:
                        print(f"  [SAM3-intersect] skipped — sam3 mask "
                              f"{sam_px} px is < 40% of psalm {psalm_px} "
                              f"(too narrow, would over-trim)")

            before = int(visible_mask.sum())
            kept_px = int(psalm_subj_picked.sum())

            # PSALM-subject is now the SINGLE source of truth for the
            # visible mask.  The CLIP-grounded seed + GPT
            # visible_polygon_override added earlier are noisier polygons
            # that inflate the mask — we drop them entirely in favour of
            # PSALM's clean silhouette.  PSALM falls back to retaining the
            # prior seed ONLY if it returned nothing usable.
            if kept_px > 0:
                visible_mask = psalm_subj_picked.copy()
                after = int(visible_mask.sum())
                print(f"  PSALM-subject REPLACED visible_mask: "
                      f"{before} → {after} px  "
                      f"(psalm={int(psalm_subj.sum())} - occluder = "
                      f"{int(psalm_subj_clean.sum())}, kept-cc={kept_px}, "
                      f"#cc={n_lbl - 1}, click=({sx},{sy})→lbl{target_lbl})")
            else:
                # PSALM produced nothing — keep whatever the prior seed
                # contained so the pipeline doesn't crash with an empty
                # visible mask.
                after = before
                print(f"  PSALM-subject empty — keeping prior seed "
                      f"({before} px)")

    # Fallback: when visible_mask is absent or too small, estimate visible region
    # from the area ABOVE the hidden_region (the subject's head is above the hidden body).
    # Threshold of 5000 px rejects "tiny dot" masks that confuse pix2gestalt.
    _MIN_VIS_AREA = max(200, int(0.005 * h * w))
    if visible_mask.sum() < _MIN_VIS_AREA:
        if visible_mask.sum() > 0:
            print(f"  WARNING: visible_mask too small ({visible_mask.sum()} px < {_MIN_VIS_AREA}) — estimating from hidden_region")
        else:
            print("  WARNING: visible_mask empty — estimating from hidden_region top edge")
        hry1 = max(0, int(region.get("y1", h // 2)))
        hrx1 = max(0, int(region.get("x1", 0)))
        hrx2 = min(w, int(region.get("x2", w)))
        # Use full height from image top to pumpkin rim (vy1=0 keeps aspect ratio square-ish)
        vy1 = 0
        vy2 = hry1  # top of hidden region = bottom of visible head
        # Narrow x-range to ~50% of hidden region width centred on pumpkin opening
        # so the crop aspect ratio stays near 1:1 for pix2gestalt
        center_x   = (hrx1 + hrx2) // 2
        half_w     = min((hrx2 - hrx1) // 3, 130)
        fb_x1 = max(0, center_x - half_w)
        fb_x2 = min(w, center_x + half_w)
        # Exclude the occluder itself from the fallback visible mask
        fallback = np.zeros((h, w), dtype=np.uint8)
        fallback[vy1:vy2, fb_x1:fb_x2] = 1
        fallback = np.clip(fallback & ~occluder_mask, 0, 1)
        if fallback.sum() > visible_mask.sum():
            visible_mask = fallback
            print(f"  Visible mask fallback → {fallback.sum()} px (y={vy1}–{vy2}, x={fb_x1}–{fb_x2})")

    # ── Programmatic click snap (replaces the old GPT geometry retry loop) ───
    # If GPT's click coords land in the wrong region (occluder_click on the
    # subject mask, or subject_click on the occluder mask, or either outside
    # both), snap them to the centroid of the largest connected component of
    # the correct fused mask. This is faster + free of API flakiness than
    # re-prompting GPT.
    occ_x, occ_y = int(occ_click.get("x", 0)), int(occ_click.get("y", 0))
    sub_x, sub_y = int(sub_click.get("x", 0)), int(sub_click.get("y", 0))

    def _centroid_of_largest_cc(mask: np.ndarray) -> tuple[int, int] | None:
        if mask is None or int(mask.sum()) == 0:
            return None
        n, lbl, stats, cents = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        if n <= 1:
            return None
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cx, cy = cents[biggest]
        # Pull the centroid back onto the mask if it landed on a hole.
        if not bool(mask[int(cy), int(cx)]):
            ys, xs = np.where(lbl == biggest)
            d2 = (ys - cy) ** 2 + (xs - cx) ** 2
            k  = int(np.argmin(d2))
            cx, cy = xs[k], ys[k]
        return int(cx), int(cy)

    def _click_inside(mask: np.ndarray, x: int, y: int) -> bool:
        if mask is None or not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
            return False
        return bool(mask[y, x])

    if not frame_cropped:
        occ_inside_occ = _click_inside(occluder_mask, occ_x, occ_y) if (occ_x or occ_y) else False
        occ_inside_vis = _click_inside(visible_mask,  occ_x, occ_y) if (occ_x or occ_y) else False
        if (occ_x or occ_y) and (not occ_inside_occ or occ_inside_vis):
            new_occ = _centroid_of_largest_cc(occluder_mask)
            if new_occ is not None:
                old = (occ_x, occ_y)
                occ_x, occ_y = new_occ
                occ_click["x"], occ_click["y"] = occ_x, occ_y
                print(f"  [click-snap] occluder click {old} → ({occ_x}, {occ_y}) "
                      f"(was {'on subject' if occ_inside_vis else 'off-mask'})")
        sub_inside_vis = _click_inside(visible_mask,  sub_x, sub_y) if (sub_x or sub_y) else False
        sub_inside_occ = _click_inside(occluder_mask, sub_x, sub_y) if (sub_x or sub_y) else False
        if (sub_x or sub_y) and (not sub_inside_vis or sub_inside_occ):
            new_sub = _centroid_of_largest_cc(visible_mask)
            if new_sub is not None:
                old = (sub_x, sub_y)
                sub_x, sub_y = new_sub
                sub_click["x"], sub_click["y"] = sub_x, sub_y
                print(f"  [click-snap] subject click {old} → ({sub_x}, {sub_y}) "
                      f"(was {'on occluder' if sub_inside_occ else 'off-mask'})")

    # ── Fix 1: SAM3 point-prompt refinement (MODE A only) ────────────────────────
    # Auto-segment may merge touching objects (e.g. horse head + woman body) into
    # a single blob.  Use the click coordinates GPT returned to re-run SAM3 with
    # targeted point prompts and get a clean mask per object.
    # Skip when CLIP already grounded the masks — the CLIP-labeled segments are
    # higher confidence than a single GPT click point.
    if not frame_cropped and not clip_grounded and (occ_x or occ_y) and (sub_x or sub_y):
        targeted = _sam_segment_targeted(
            state["image_path"],
            [
                {"label": "occluder", "x": occ_x, "y": occ_y},
                {"label": "subject",  "x": sub_x, "y": sub_y},
            ],
            out_dir,
        )
        img_area = h * w
        for t in targeted:
            tarea = int(t["mask"].sum())
            tmask = t["mask"].astype(bool)
            if t["label"] == "occluder":
                cur     = int(occluder_mask.sum())
                cur_b   = occluder_mask.astype(bool)
                vis_b   = visible_mask.astype(bool)
                overlap_cur = int((tmask & cur_b).sum())
                overlap_vis = int((tmask & vis_b).sum())
                overlap_ratio_cur = overlap_cur / max(cur, 1)
                overlap_ratio_vis = overlap_vis / max(int(vis_b.sum()), 1)
                contains_click = bool(tmask[occ_y, occ_x]) if 0 <= occ_y < h and 0 <= occ_x < w else False
                # Area band: within 0.5×–2× of existing estimate (skip when no existing estimate).
                area_in_band = (cur == 0) or (0.5 * cur <= tarea <= 2.0 * cur)
                # Acceptance criteria, ALL must hold:
                # (a) contains the click point (the point-prompt actually segmented at the prompt)
                # (b) area not pathological (< 40 % of frame)
                # (c) area within band of existing estimate when one exists (avoids replace-with-bg)
                # (d) does not heavily overlap visible_mask (< 25 % of visible)
                # (e) overlaps existing estimate when one exists (≥ 35 %)
                accept = (
                    tarea > 0
                    and contains_click
                    and tarea < 0.40 * img_area
                    and area_in_band
                    and overlap_ratio_vis < 0.25
                    and (cur == 0 or overlap_ratio_cur >= 0.35)
                )
                if accept:
                    occluder_mask = t["mask"].copy()
                    print(f"  [Fix1] Point-prompt occluder mask → {tarea} px "
                          f"(click✓ band✓ vis-ovl={overlap_ratio_vis:.2f} cur-ovl={overlap_ratio_cur:.2f})")
                    mask_save = out_dir / "occluder_mask.png"
                    cv2.imwrite(str(mask_save), occluder_mask * 255)
                else:
                    reason = []
                    if not contains_click: reason.append("click∉mask")
                    if tarea >= 0.40 * img_area: reason.append(f"area={100*tarea/img_area:.0f}%img")
                    if not area_in_band: reason.append(f"area-band(cur={cur})")
                    if overlap_ratio_vis >= 0.25: reason.append(f"vis-ovl={overlap_ratio_vis:.2f}")
                    if cur > 0 and overlap_ratio_cur < 0.35: reason.append(f"cur-ovl={overlap_ratio_cur:.2f}")
                    print(f"  [Fix1] Occluder point-prompt rejected "
                          f"(area={tarea}px; {', '.join(reason) or 'unknown'}) — keeping auto-segment mask")
            if t["label"] == "subject":
                cur     = int(visible_mask.sum())
                cur_b   = visible_mask.astype(bool)
                occ_b   = occluder_mask.astype(bool)
                overlap_cur = int((tmask & cur_b).sum())
                overlap_occ = int((tmask & occ_b).sum())
                overlap_ratio_cur = overlap_cur / max(cur, 1)
                overlap_ratio_occ = overlap_occ / max(tarea, 1)
                contains_click = bool(tmask[sub_y, sub_x]) if 0 <= sub_y < h and 0 <= sub_x < w else False
                # Subject must contain the click, not be mostly inside the occluder, and
                # be ≥ existing visible estimate (avoids locking onto a sub-feature).
                accept = (
                    tarea > 200
                    and contains_click
                    and overlap_ratio_occ < 0.5
                    and tarea >= cur
                )
                if accept:
                    visible_mask = t["mask"].copy()
                    print(f"  [Fix1] Point-prompt visible mask → {tarea} px  (was {cur} px) "
                          f"click✓ occ-ovl={overlap_ratio_occ:.2f}")
                    visible_mask_save = out_dir / "visible_mask.png"
                    cv2.imwrite(str(visible_mask_save), visible_mask * 255)
                else:
                    reason = []
                    if tarea <= 200: reason.append(f"area={tarea}px")
                    if not contains_click: reason.append("click∉mask")
                    if overlap_ratio_occ >= 0.5: reason.append(f"occ-ovl={overlap_ratio_occ:.2f}")
                    if tarea < cur: reason.append(f"smaller than cur={cur}")
                    print(f"  [Fix1] Subject point-prompt rejected "
                          f"(area={tarea}px; {', '.join(reason) or 'unknown'}) — keeping polygon/segment mask")

    visible_mask_save = out_dir / "visible_mask.png"
    cv2.imwrite(str(visible_mask_save), visible_mask * 255)
    print(f"  Visible mask     : {visible_mask.sum()} px → {visible_mask_save}")

    # ── Geometric hidden_object_mask reconstruction ──────────────────────────
    # The original `hidden_mask` came from GPT's hidden_polygon, which is often
    # placed in the wrong region (e.g. on the bear's face instead of the chest).
    # We rebuild it geometrically:
    #   geometric_hidden = (occluder_mask − visible_mask)
    # Then UNION with the GPT polygon (which may contribute the truly out-of-
    # occluder area like below-the-frame body parts), and subtract visible_mask
    # so the modal subject is never overwritten.
    occ_b = occluder_mask.astype(bool)
    vis_b = visible_mask.astype(bool)
    geometric_hidden = (occ_b & ~vis_b).astype(np.uint8)
    polygon_hidden   = hidden_mask.copy()        # what GPT gave us
    combined_hidden  = np.clip(geometric_hidden | polygon_hidden, 0, 1).astype(np.uint8)
    # Always remove any pixels that overlap the visible subject — these would
    # repaint the bear's face / cat's head etc.
    combined_hidden  = np.clip(combined_hidden & ~visible_mask, 0, 1).astype(np.uint8)

    g_area  = int(geometric_hidden.sum())
    p_area  = int(polygon_hidden.sum())
    c_area  = int(combined_hidden.sum())
    print(f"  Hidden mask reconstruction: geometric={g_area} px, "
          f"polygon={p_area} px → combined={c_area} px")
    hidden_mask = combined_hidden
    cv2.imwrite(str(hidden_mask_save), hidden_mask * 255)

    # ── Five canonical Agent-1 mask outputs (user-requested taxonomy) ─────────
    # 1. query_mask         = modal / visible part of the target object
    # 2. occluder_mask      = object(s) in front of the target  (already in mem)
    # 3. outpaint_mask      = extra-canvas / out-of-frame region the
    #                          subject is expected to extend INTO
    # 4. inpainting_mask    = occluder ∪ outpaint  (everything Agent 2 fills)
    # 5. final_amodal_mask  = query ∪ inpainting (PLANNED full subject silhouette)
    #
    # All five are saved in TWO subdirectories of out_dir:
    #   masks_binary/<name>.png       — white-on-black uint8 (255 = mask)
    #   masks_transparent/<name>.png  — RGBA, mask pixels opaque-coloured,
    #                                   non-mask pixels fully transparent

    binary_dir = out_dir / "masks_binary"
    transp_dir = out_dir / "masks_transparent"
    binary_dir.mkdir(parents=True, exist_ok=True)
    transp_dir.mkdir(parents=True, exist_ok=True)

    # Build the 5 masks first, then save both forms in a loop.
    # Hard invariant: query (visible part of subject) MUST NOT overlap the
    # occluder.  PSALM-class can leak a few px over the boundary on either
    # side; enforce ∅ intersection here so downstream consumers see clean,
    # mutually-exclusive masks.
    query_mask      = np.clip(visible_mask.astype(np.int32)
                              - occluder_mask.astype(np.int32),
                              0, 1).astype(np.uint8)
    outpaint_mask   = np.clip(hidden_mask.astype(np.int32)
                              - occluder_mask.astype(np.int32),
                              0, 1).astype(np.uint8)
    inpainting_mask = np.clip(occluder_mask | outpaint_mask, 0, 1).astype(np.uint8)
    final_amodal    = np.clip(query_mask | inpainting_mask, 0, 1).astype(np.uint8)

    # img is the source BGR image loaded at the top of occlusion_agent.
    masks_to_save: list[tuple[str, np.ndarray]] = [
        ("query_mask",        query_mask),
        ("occluder_mask",     occluder_mask),
        ("outpaint_mask",     outpaint_mask),
        ("inpainting_mask",   inpainting_mask),
        ("final_amodal_mask", final_amodal),
    ]
    for name, m in masks_to_save:
        # binary form: white-on-black, single channel
        cv2.imwrite(str(binary_dir / f"{name}.png"), m * 255)
        # transparent form: ORIGINAL image pixels inside the mask, fully
        # transparent everywhere else.  This is a true RGBA "cutout" — drop
        # the file on any background and only the masked region shows.
        rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)
        # OpenCV reads BGR; PNG with alpha in cv2.imwrite expects BGRA, so
        # we can copy the BGR planes directly.
        rgba[..., :3] = img
        rgba[..., 3]  = (m > 0).astype(np.uint8) * 255
        cv2.imwrite(str(transp_dir / f"{name}.png"), rgba)
        print(f"  {name:<18}: {int(m.sum()):>6} px → "
              f"masks_binary/ + masks_transparent/")

    # ── Comparison grid: original + all 5 masks overlaid (BGR colours) ───────
    def _panel(title: str, mask: np.ndarray, color: tuple,
               raw: bool = False) -> np.ndarray:
        if raw:
            out = img.copy()
        else:
            ov = np.zeros_like(img)
            ov[mask > 0] = color
            out = cv2.addWeighted(img, 0.55, ov, 0.45, 0)
        # title banner
        cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
        if raw:
            label = title
        else:
            label = f"{title}  {int((mask > 0).sum())}px"
        cv2.putText(out, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    panels = [
        _panel("1. ORIGINAL",       np.zeros_like(occluder_mask), (0, 0, 0), raw=True),
        _panel("2. query (modal)",  query_mask,        (0, 255, 0)),       # green
        _panel("3. occluder",       occluder_mask,     (0, 0, 255)),       # red
        _panel("4. outpaint",       outpaint_mask,     (255, 0, 0)),       # blue
        _panel("5. inpainting",     inpainting_mask,   (0, 255, 255)),     # yellow
        _panel("6. final amodal",   final_amodal,      (255, 0, 255)),     # magenta
    ]
    row1 = np.hstack(panels[:3])
    row2 = np.hstack(panels[3:])
    grid = np.vstack([row1, row2])
    grid_save = out_dir / "masks_comparison.png"
    cv2.imwrite(str(grid_save), grid)
    print(f"  comparison grid  : → {grid_save.name}")

    # ── Amodal subject mask: pix2gestalt → review  OR  direct GPT-V draw ─────
    # Three paths:
    #   • USE_AMODAL_COMPLETION=False → use visible_mask AS-IS (no completion)
    #   • USE_PIX2GESTALT_AMODAL=True → pix2gestalt seed + GPT-V review
    #   • USE_PIX2GESTALT_AMODAL=False → direct GPT-V silhouette call
    try:
        if not getattr(config, "USE_AMODAL_COMPLETION", True):
            # No amodal extension — use visible_mask as the amodal mask.
            # PSALM's visible segmentation is treated as authoritative; no
            # GPT-V silhouette generation (which often drifts onto the occluder).
            reviewed_mask = (visible_mask > 0).astype(np.uint8)
            print(f"  [Amodal/skip] no completion — using visible_mask "
                  f"as amodal ({int(reviewed_mask.sum())} px)")
        elif getattr(config, "USE_PIX2GESTALT_AMODAL", True):
            # --- step 1: pix2gestalt amodal completion ----------------------
            amodal_dir = out_dir / "amodal_review"
            amodal_dir.mkdir(parents=True, exist_ok=True)
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            aligned_shape_p2g, _amodal_rgb_256 = _run_shape_prior(
                image_rgb, visible_mask, (h, w), amodal_dir,
                hidden_polygon=None,
            )
            print(f"  [Amodal/pix2gestalt] mask area: {int(aligned_shape_p2g.sum())} px")

            # --- step 2: GPT-V review ---------------------------------------
            reviewed_mask = _gpt_review_amodal_mask(
                image_bgr=img,
                visible_mask=visible_mask,
                occluder_mask=occluder_mask,
                pix2gestalt_mask=aligned_shape_p2g,
                subject_text=target_class or target_short or "the subject",
                occluder_text=occluder_class or occluder_short or "the occluder",
                out_dir=out_dir,
            )
        else:
            # Minimal-pipeline path: skip pix2gestalt; ask GPT-V to trace
            # the full silhouette from scratch in a single vision call.
            print("  [Amodal/minimal] skipping pix2gestalt + review — using "
                  "_gpt_amodal_subject_mask (one GPT-V call)")
            gpt_mask, _gpt_polys = _gpt_amodal_subject_mask(
                image_bgr=img,
                visible_mask=visible_mask,
                occluder_mask=occluder_mask,
                subject_text=target_class or target_short or "the subject",
                occluder_text=occluder_class or occluder_short or "the occluder",
                out_dir=out_dir,
            )
            if gpt_mask is None:
                # Fallback: use visible_mask alone (no amodal completion).
                print("  [Amodal/minimal] GPT-V returned no polygon — using "
                      "visible_mask as amodal (no completion)")
                reviewed_mask = (visible_mask > 0).astype(np.uint8)
            else:
                # Always include visible_mask in the final amodal.
                reviewed_mask = np.clip(
                    gpt_mask.astype(np.uint8) |
                    (visible_mask > 0).astype(np.uint8),
                    0, 1).astype(np.uint8)
                print(f"  [Amodal/minimal] GPT-V silhouette: {int(reviewed_mask.sum())} px")

        # Persist final reviewed mask
        cv2.imwrite(str(out_dir / "subject_full_amodal_mask.png"),
                    reviewed_mask * 255)
        transp_dir = out_dir / "masks_transparent"
        transp_dir.mkdir(parents=True, exist_ok=True)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = img
        rgba[..., 3]  = (reviewed_mask > 0).astype(np.uint8) * 255
        cv2.imwrite(str(transp_dir / "subject_full_amodal_mask.png"), rgba)
        # And the completion-only (= reviewed − visible) for clarity.
        completion = np.clip(reviewed_mask.astype(np.int32)
                             - (visible_mask > 0).astype(np.int32),
                             0, 1).astype(np.uint8)
        cv2.imwrite(str(out_dir / "subject_completion_mask.png"), completion * 255)
        print(f"  [Amodal/reviewed] final mask: {int(reviewed_mask.sum())} px "
              f"(completion-only = {int(completion.sum())} px) → "
              f"subject_full_amodal_mask.png")

        # ── step 3: off-frame extension via GPT-V on padded canvas ─────────
        # Fires when GPT flagged frame_cropped=True with non-zero
        # expansion_pixels, OR when config.FORCE_FRAME_CROPPED is set (debug).
        offframe_px = (force_exp_px if force_offframe else exp_px) \
            if (force_offframe or frame_cropped) else None
        if (
            offframe_px is not None
            and isinstance(offframe_px, dict)
            and (offframe_px.get("top", 0) or offframe_px.get("bottom", 0)
                 or offframe_px.get("left", 0) or offframe_px.get("right", 0))
        ):
            padded_full, offframe_only, padded_img, offsets = \
                _flux_extend_amodal_mask_offframe(
                    image_bgr=img,
                    in_frame_amodal_mask=reviewed_mask,
                    expansion_pixels=offframe_px,
                    subject_text=target_class or target_short or "the subject",
                    out_dir=out_dir,
                )
            if padded_full is not None:
                # Persist padded outputs.
                cv2.imwrite(str(out_dir / "padded_canvas.png"), padded_img)
                cv2.imwrite(str(out_dir / "subject_full_amodal_padded.png"),
                            padded_full * 255)
                cv2.imwrite(str(out_dir / "subject_offframe_only_mask.png"),
                            offframe_only * 255)
                # Transparent versions for visual inspection.
                Hp, Wp = padded_full.shape[:2]
                rgba_pad = np.zeros((Hp, Wp, 4), dtype=np.uint8)
                rgba_pad[..., :3] = padded_img
                rgba_pad[..., 3]  = (padded_full > 0).astype(np.uint8) * 255
                cv2.imwrite(str(transp_dir / "subject_full_amodal_padded.png"),
                            rgba_pad)
                # Save offsets json so downstream can map padded↔original.
                (out_dir / "padded_offsets.json").write_text(
                    json.dumps(offsets, indent=2))
                print(f"  [Amodal/offframe] padded mask {int(padded_full.sum())} px, "
                      f"off-frame only {int(offframe_only.sum())} px → "
                      f"subject_full_amodal_padded.png "
                      f"+ subject_offframe_only_mask.png")
    except Exception as exc:                                      # noqa: BLE001
        traceback.print_exc()
        print(f"  [Amodal/reviewed] skipped: {exc!r}")

    # ── Mask consistency hard-gate ───────────────────────────────────────────
    # If after all of the above the masks are still inconsistent, log it.
    # The reviewer will route MASK_INACCURATE on the next pass.
    occ_area = int(occluder_mask.sum())
    vis_area = int(visible_mask.sum())
    hid_area = int(hidden_mask.sum())
    img_area = h * w
    occ_in_vis = int((occ_b & vis_b).sum())
    occ_vs_vis_ratio = occ_in_vis / max(occ_area, 1)

    # Cover-fraction: how much of GPT's hidden_region bbox is inside (occluder ∪ hidden)
    cover_mask = np.clip(occluder_mask | hidden_mask, 0, 1).astype(np.uint8)
    bbox_y1 = max(0, int(bbox[1])); bbox_y2 = min(h, int(bbox[3]))
    bbox_x1 = max(0, int(bbox[0])); bbox_x2 = min(w, int(bbox[2]))
    bbox_area = max((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), 1)
    bbox_cover = int(cover_mask[bbox_y1:bbox_y2, bbox_x1:bbox_x2].sum()) / bbox_area

    issues = []
    if occ_area < 0.005 * img_area:
        issues.append(f"occluder_mask very small ({occ_area} px = {100*occ_area/img_area:.2f}% of image)")
    if vis_area < 0.005 * img_area:
        issues.append(f"visible_mask very small ({vis_area} px = {100*vis_area/img_area:.2f}% of image)")
    if occ_vs_vis_ratio > 0.40:
        issues.append(f"occluder overlaps visible by {100*occ_vs_vis_ratio:.0f}% — masks contradict each other")
    if bbox_cover < 0.30 and bbox_area > 500:
        issues.append(f"occluder∪hidden covers only {100*bbox_cover:.0f}% of GPT hidden_region bbox")

    if issues:
        print("  [Mask-consistency] WARNING:")
        for iss in issues:
            print(f"    • {iss}")
    else:
        print(f"  [Mask-consistency] ✓ occ={occ_area} vis={vis_area} hidden={hid_area} "
              f"bbox-cover={100*bbox_cover:.0f}%")

    # ── Visualisation: red = occluder, green = visible object, cyan = hidden ────
    viz_out = img.copy()
    overlay = np.zeros_like(viz_out)
    overlay[occluder_mask == 1] = (0, 0, 255)   # red
    overlay[visible_mask  == 1] = (0, 255, 0)   # green
    viz_out = cv2.addWeighted(viz_out, 0.55, overlay, 0.45, 0)

    contours, _ = cv2.findContours(occluder_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(viz_out, contours, -1, (0, 0, 200), 2)
    if poly_used is not None:
        cv2.polylines(viz_out, [poly_used], True, (0, 200, 0), 2)
    if vis_poly_used is not None:
        cv2.polylines(viz_out, [vis_poly_used], True, (0, 255, 120), 2)
    # Fix 2: draw hidden_polygon in cyan
    if len(hidden_poly) >= 3:
        pts_h = np.array([[int(p[0]), int(p[1])] for p in hidden_poly], dtype=np.int32)
        cv2.polylines(viz_out, [pts_h], True, (255, 255, 0), 2)   # cyan
        print(f"  Hidden polygon   : {len(hidden_poly)} pts drawn on viz")

    occ_viz_path = out_dir / "occluder_viz.png"
    cv2.imwrite(str(occ_viz_path), viz_out)

    removed = img.copy()
    removed[occluder_mask == 1] = 0
    removed_path = out_dir / "occluder_removed.png"
    cv2.imwrite(str(removed_path), removed)

    print(f"  Viz saved        : {occ_viz_path}")

    print(f"  Subject desc     : {data.get('subject_description', '')[:120]}")
    print(f"  Visible parts    : {data.get('visible_parts', '')}")
    print(f"  Missing parts    : {data.get('missing_parts', '')}")

    return {
        **state,
        "occluded_object":       data.get("occluded_object", hint or "unknown"),
        "occluder":              data.get("occluder", ""),
        "what_to_remove":        data.get("what_to_remove", ""),
        "subject_description":   data.get("subject_description", ""),
        "visible_parts":         data.get("visible_parts", ""),
        "missing_parts":         data.get("missing_parts", ""),
        "bbox":                  bbox,
        "boundary_expansion":    expansion,
        "region_desc":           region.get("description", ""),
        "frame_cropped":         frame_cropped,
        "expansion_directions":  exp_dirs,
        "expansion_pixels":      exp_px,
        "mask_path":             str(mask_save),
        "visible_mask_path":     str(visible_mask_save),
        "hidden_mask_path":      str(hidden_mask_save),
        "occluder_removed_path": str(removed_path),
        "occluder_viz_path":     str(occ_viz_path),
        "hidden_polygon":        hidden_poly if len(hidden_poly) >= 3 else None,
        "pix2gestalt_dir":       None,   # invalidate cached samples on mask re-run
        "mask_retry_count":      state.get("mask_retry_count", 0) + 1,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_binary_mask(path: str, size: tuple) -> np.ndarray:
    """Load a mask PNG and resize to (w, h). Returns uint8 array with values 0/1."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def _build_subject_mask(out_dir: Path, seed_mask: np.ndarray, result_hw: tuple,
                        overlap_thresh: float = 0.05) -> np.ndarray:
    """Union of all saved SAM3 segments that overlap >= overlap_thresh of their area
    with seed_mask (the known subject region).  Falls back to seed_mask if no masks found."""
    h, w = result_hw
    seed_rs = cv2.resize(seed_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    seed_rs = (seed_rs > 0)
    subject = seed_rs.copy()
    masks_dir = out_dir / "sam3_masks"
    if not masks_dir.is_dir():
        return seed_rs.astype(np.uint8)
    for p in sorted(masks_dir.glob("seg_*.png")):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m_bin = (m > 127)
        overlap = (m_bin & seed_rs).sum()
        if m_bin.sum() > 0 and overlap / m_bin.sum() >= overlap_thresh:
            subject |= m_bin
    return subject.astype(np.uint8)


def _poisson_blend(src_bgr: np.ndarray, dst_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Poisson seamless blending of src into dst at mask region.
    Falls back to a feathered alpha blend when the mask's bbox can't fit OpenCV's
    internal ROI (which it computes as a square centered at the centroid that spans
    the larger bbox dim — too big for masks covering most of the image)."""
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return src_bgr
    H, W = dst_bgr.shape[:2]

    # seamlessClone's internal ROI is centered at (cx, cy) with side = max(bbox_w,
    # bbox_h). If that ROI doesn't fit inside the image, OpenCV asserts. Detect
    # and skip straight to the feathered fallback in that case.
    ys, xs = np.where(mask > 0)
    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_h = int(ys.max() - ys.min() + 1)
    side = max(bbox_w, bbox_h) + 2
    cx = int(np.clip(M["m10"] / M["m00"], side // 2, W - side // 2 - 1))
    cy = int(np.clip(M["m01"] / M["m00"], side // 2, H - side // 2 - 1))
    fits = side <= min(H, W) - 2

    if fits:
        try:
            return cv2.seamlessClone(
                src_bgr, dst_bgr,
                (mask * 255).astype(np.uint8), (cx, cy), cv2.MIXED_CLONE,
            )
        except cv2.error as e:
            print(f"  [Poisson] seamlessClone failed ({e}), using feathered fallback")
    else:
        print(f"  [Poisson] mask bbox {bbox_w}x{bbox_h} too large for image {W}x{H} — using feathered blend")

    # Feathered alpha blend: Gaussian-blur the binary mask edge so the composite
    # transitions smoothly from src to dst rather than hard-cutting (which leaves
    # a visible seam and is what the original fallback produced).
    alpha = (mask.astype(np.float32) * 255.0)
    feather = max(3, min(H, W) // 100)
    if feather % 2 == 0:
        feather += 1
    alpha = cv2.GaussianBlur(alpha, (feather, feather), 0) / 255.0
    alpha = alpha[:, :, None]
    return (src_bgr.astype(np.float32) * alpha
            + dst_bgr.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


# ── Subject crop helper ───────────────────────────────────────────────────────

def _crop_to_subject(
    image_np: np.ndarray,
    visible_mask: np.ndarray,
    padding_frac: float = 0.3,
) -> tuple:
    """Crop image tightly around visible_mask bounding box with padding.
    Returns (img_crop, msk_crop, (cx1, cy1, cx2, cy2)).
    Falls back to the full image when the mask is empty.
    """
    ys, xs = np.where(visible_mask > 0)
    h, w = image_np.shape[:2]
    if len(ys) == 0:
        return image_np, visible_mask, (0, 0, w, h)

    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    pad_x = max(int((x2 - x1) * padding_frac), 20)
    pad_y = max(int((y2 - y1) * padding_frac), 20)

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    return image_np[cy1:cy2, cx1:cx2], visible_mask[cy1:cy2, cx1:cx2], (cx1, cy1, cx2, cy2)


# ── Shape prior (pix2gestalt → binary silhouette) ─────────────────────────────

def _run_shape_prior(
    image_np: np.ndarray,       # H×W×3 uint8 RGB, original image
    visible_mask: np.ndarray,   # H×W uint8 binary
    orig_hw: tuple,             # (H, W) of original image
    out_dir: Path,
    n_samples: int = 1,
    hidden_polygon: list = None,  # If provided, hint is stretched to cover the full hidden area
) -> tuple:
    """
    Run pix2gestalt on a subject-centred crop to predict the amodal shape.
    Returns (aligned_shape, amodal_hint):
      aligned_shape : H×W uint8 binary mask in original image coordinates
      amodal_hint   : H×W×3 uint8 RGB — pix2gestalt output placed back on a white
                      full-size canvas (used as ControlNet colour/texture hint)
    """
    SZ = 256
    orig_h, orig_w = orig_hw
    feat_path       = out_dir.parent / "sam3_features.pt"
    aisformer_cache = out_dir / "aisformer_shape.png"
    mask_cache      = out_dir / "shape_prior_mask.png"   # full-res binary, persisted

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if (out_dir / "shape_prior_0.png").exists() and mask_cache.exists():
        amodal_hint = np.array(Image.open(str(out_dir / "shape_prior_0.png")).convert("RGB"))
        if aisformer_cache.exists():
            raw = cv2.imread(str(aisformer_cache), cv2.IMREAD_GRAYSCALE)
            aligned = (cv2.resize(raw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
            print("  [AISFormer] Using cached shape prior")
        else:
            raw = cv2.imread(str(mask_cache), cv2.IMREAD_GRAYSCALE)
            aligned = (cv2.resize(raw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
        return aligned, amodal_hint

    # ── Crop around visible subject ───────────────────────────────────────────
    img_crop, msk_crop, (cx1, cy1, cx2, cy2) = _crop_to_subject(image_np, visible_mask)
    crop_h, crop_w = cy2 - cy1, cx2 - cx1
    print(f"  [pix2gestalt] Subject crop: ({cx1},{cy1})→({cx2},{cy2})  {crop_w}×{crop_h} px")

    img_256 = cv2.resize(img_crop, (SZ, SZ), interpolation=cv2.INTER_AREA)
    msk_256 = cv2.resize(msk_crop * 255, (SZ, SZ), interpolation=cv2.INTER_NEAREST)
    msk_256 = np.clip(msk_256, 0, 255).astype(np.uint8)
    rgb_mask = np.stack([msk_256, msk_256, msk_256], axis=2)

    model = _get_pix2gestalt()
    repo = str(config.PIX2GESTALT_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from inference import run_pix2gestalt

    print(f"  [pix2gestalt] Generating shape prior (steps={config.PIX2GESTALT_STEPS})…")
    try:
        results = run_pix2gestalt(
            model, DEVICE,
            img_256, rgb_mask,
            scale=config.PIX2GESTALT_SCALE,
            n_samples=n_samples,
            ddim_steps=config.PIX2GESTALT_STEPS,
            ddim_eta=config.PIX2GESTALT_ETA,
        )
    finally:
        _free_pix2gestalt()

    amodal_crop = results[0]   # 256×256 RGB in crop space

    # ── Save crop-space output for inspection ─────────────────────────────────
    Image.fromarray(amodal_crop).save(str(out_dir / "shape_prior_crop_0.png"))
    print(f"  [pix2gestalt] Crop output saved → shape_prior_crop_0.png")

    r_min, r_max = int(amodal_crop.min()), int(amodal_crop.max())
    object_px = int((amodal_crop.min(axis=2) < config.SHAPE_PRIOR_THRESH).sum())
    total_px  = SZ * SZ
    print(f"  [pix2gestalt] pixel stats  min={r_min}  max={r_max}  mean={amodal_crop.mean():.1f}")
    print(f"  [pix2gestalt] object pixels (any channel < {config.SHAPE_PRIOR_THRESH}): "
          f"{object_px} / {total_px}  ({100 * object_px / total_px:.1f}%)")
    if object_px == 0:
        print("  [pix2gestalt] WARNING: output is entirely white — visible_mask may be empty")
    elif object_px == total_px:
        print("  [pix2gestalt] WARNING: output is entirely dark — SHAPE_PRIOR_THRESH may be too high")

    # ── Build full-size hint: white canvas with crop result placed back ────────
    # Stretch hint to cover full body extent (hidden_polygon bottom) so ControlNet
    # sees cat body content in the lower hidden area rather than white background.
    if hidden_polygon and len(hidden_polygon) >= 3:
        hp_pts = np.array([[int(p[0]), int(p[1])] for p in hidden_polygon], np.int32)
        body_bottom = min(orig_h, int(hp_pts[:, 1].max()) + 20)
        hint_h = max(crop_h, body_bottom - cy1)
        hint_x1 = min(cx1, int(hp_pts[:, 0].min()))
        hint_x2 = max(cx2, int(hp_pts[:, 0].max()))
        hint_w = hint_x2 - hint_x1
    else:
        hint_h, hint_w, hint_x1, hint_x2 = crop_h, crop_w, cx1, cx2

    crop_result = np.array(Image.fromarray(amodal_crop).resize((hint_w, hint_h), Image.LANCZOS))
    amodal_hint = np.full((orig_h, orig_w, 3), 255, dtype=np.uint8)
    paste_y2 = min(orig_h, cy1 + hint_h)
    paste_x2 = min(orig_w, hint_x1 + hint_w)
    amodal_hint[cy1:paste_y2, hint_x1:paste_x2] = crop_result[:paste_y2 - cy1, :paste_x2 - hint_x1]
    Image.fromarray(amodal_hint).save(str(out_dir / "shape_prior_0.png"))
    print(f"  [pix2gestalt] Full-size hint saved → shape_prior_0.png  ({hint_w}×{hint_h} at ({hint_x1},{cy1}))")

    # ── Map binary shape from crop 256×256 → full image coords ────────────────
    shape_crop_256  = _threshold_shape_prior(amodal_crop)
    cv2.imwrite(str(out_dir / "shape_prior_threshold.png"), shape_crop_256 * 255)
    print(f"  [pix2gestalt] Threshold mask (256×256) → shape_prior_threshold.png  ({shape_crop_256.sum()} px)")

    shape_in_crop   = cv2.resize(shape_crop_256, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
    aligned_pix2g   = np.zeros((orig_h, orig_w), dtype=np.uint8)
    aligned_pix2g[cy1:cy2, cx1:cx2] = shape_in_crop
    print(f"  [pix2gestalt] Shape prior in scene: {aligned_pix2g.sum()} px")
    cv2.imwrite(str(mask_cache), aligned_pix2g * 255)

    # ── AISFormer override (when checkpoint is configured) ────────────────────
    aisformer_mask = _try_run_aisformer(feat_path, visible_mask, (SZ, SZ))
    if aisformer_mask is not None:
        aligned = cv2.resize(aisformer_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(aisformer_cache), aligned * 255)
        print(f"  [AISFormer] Shape prior saved → {aisformer_cache.name}  ({aligned.sum()} px)")
    else:
        aligned = aligned_pix2g

    return aligned.astype(np.uint8), amodal_hint


def _threshold_shape_prior(amodal_rgb: np.ndarray) -> np.ndarray:
    """Binary object mask: pixels where any channel < SHAPE_PRIOR_THRESH."""
    object_px = (amodal_rgb.min(axis=2) < config.SHAPE_PRIOR_THRESH).astype(np.uint8)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    object_px = cv2.morphologyEx(object_px, cv2.MORPH_OPEN,  kernel)
    object_px = cv2.morphologyEx(object_px, cv2.MORPH_CLOSE, kernel)
    return object_px


# ── LaMa background inpainting ────────────────────────────────────────────────

def _run_lama_inpaint(image_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    """
    Fill mask_np == 1 region in image_np using LaMa.
    image_np: H×W×3 uint8 RGB
    mask_np : H×W uint8 binary (1 = fill)
    Returns H×W×3 uint8 RGB.
    """
    h, w = image_np.shape[:2]
    lama = _get_lama()
    pil_img  = Image.fromarray(image_np)
    pil_mask = Image.fromarray((mask_np * 255).astype(np.uint8))
    result   = lama(pil_img, pil_mask)
    out = np.array(result.convert("RGB"))
    # SimpleLama may pad/resize internally; restore original dimensions.
    if out.shape[:2] != (h, w):
        out = np.array(Image.fromarray(out).resize((w, h), Image.LANCZOS))
    return out


# ── Flux-Fill inpainter (alternative backend) ─────────────────────────────────

def _run_flux_fill_inpaint(
    base_np: np.ndarray,
    inpaint_mask: np.ndarray,
    amodal_rgb_256: np.ndarray,
    prompt: str,
    n_samples: int,
    out_dir: Path,
    prefix: str = "result",
    neg_extra: str = "",
    seed_offset: int = 0,
    strength: float = 1.0,    # 1.0 = full re-inpaint; 0.4-0.7 = refinement pass
) -> list:
    """Run FLUX.1-Fill-dev on the inpaint region.

    Flux-Fill takes (image, mask, prompt) and produces an inpainted image
    in one shot — no ControlNet, no shape-prior conditioning image, no
    UNet hooks.  Anatomy quality comes from the model's native priors
    (DiT, 12B params) rather than from external scaffolding.

    Falls back to ControlNet-Inpaint if the pipeline fails to load.
    """
    from PIL import Image as PILImage

    h, w = base_np.shape[:2]
    pipe = _get_flux_fill_pipe()
    if pipe is None:
        # Backend unavailable — recursively call the ControlNet path
        # by temporarily flipping the config.
        print("  [Flux-Fill] backend unavailable — using ControlNet fallback")
        old_backend = config.INPAINT_BACKEND
        config.INPAINT_BACKEND = "controlnet_sd15"
        try:
            return _run_controlnet_inpaint(
                base_np=base_np, inpaint_mask=inpaint_mask,
                amodal_rgb_256=amodal_rgb_256, prompt=prompt,
                n_samples=n_samples, out_dir=out_dir, prefix=prefix,
                neg_extra=neg_extra, seed_offset=seed_offset,
            )
        finally:
            config.INPAINT_BACKEND = old_backend

    pil_base = PILImage.fromarray(base_np)
    # Flux wants the mask as a single-channel PIL with white=inpaint.
    pil_mask = PILImage.fromarray((inpaint_mask * 255).astype(np.uint8)).convert("L")

    # Flux works best at multiples of 16; cap at 1024 for memory.
    flux_w = min((round(w / 16) * 16), 1024)
    flux_h = min((round(h / 16) * 16), 1024)

    # Construct prompt — Flux doesn't take negative prompts in the same
    # way SD does; merge neg_extra into the positive prompt as an
    # exclusion clause.
    flux_prompt = prompt
    if neg_extra:
        flux_prompt = f"{prompt}.  Avoid: {neg_extra}."

    print(f"  [Flux-Fill] Generating {n_samples} samples "
          f"(steps={config.FLUX_FILL_STEPS}, guidance={config.FLUX_FILL_GUIDANCE_SCALE}, "
          f"strength={strength}, {flux_w}×{flux_h}, seed_offset={seed_offset})…")
    print(f"  Prompt: {flux_prompt[:140]}")

    results = []
    try:
        for i in range(n_samples):
            seed = 42 + seed_offset * 1000 + i
            generator = torch.Generator(device="cpu").manual_seed(seed)
            call_kwargs = dict(
                prompt=flux_prompt,
                image=pil_base,
                mask_image=pil_mask,
                height=flux_h,
                width=flux_w,
                num_inference_steps=config.FLUX_FILL_STEPS,
                guidance_scale=config.FLUX_FILL_GUIDANCE_SCALE,
                max_sequence_length=config.FLUX_FILL_MAX_SEQUENCE_LEN,
                generator=generator,
            )
            # FluxFillPipeline supports `strength` (img2img-like partial denoise)
            # in diffusers ≥ 0.32. Older versions only do full denoise.
            if strength < 1.0:
                call_kwargs["strength"] = strength
            try:
                out = pipe(**call_kwargs).images[0]
            except TypeError as exc:
                if "strength" in str(exc):
                    call_kwargs.pop("strength", None)
                    print(f"  [Flux-Fill] pipeline doesn't accept 'strength' — "
                          f"falling back to full denoise")
                    out = pipe(**call_kwargs).images[0]
                else:
                    raise
            if out.size != (w, h):
                out = out.resize((w, h), PILImage.LANCZOS)
            save_path = out_dir / f"{prefix}_{i}.png"
            out.save(str(save_path))
            results.append(out)
    finally:
        _free_flux_fill()
    return results


# ── ControlNet-Inpaint with pix2gestalt shape prior ───────────────────────────

def _prep_control_image(
    base_np: np.ndarray,        # H×W×3 uint8 RGB (LaMa result or expanded canvas)
    inpaint_mask: np.ndarray,   # H×W uint8 binary (kept for API compatibility)
    amodal_rgb_256: np.ndarray, # pix2gestalt full-scene hint (white background outside silhouette)
    orig_hw: tuple,
) -> "Image.Image":
    """
    Prepare ControlNet control image for control_v11p_sd15_inpaint.
    Inside the inpaint region, paste pix2gestalt's hint ONLY where the silhouette
    is non-white (any channel < SHAPE_PRIOR_THRESH). This keeps the LaMa-filled
    scene visible around the silhouette and avoids feeding ControlNet large
    rectangles of white pixels — which previously caused white blobs in the output.
    """
    h, w = base_np.shape[:2]
    hint = amodal_rgb_256
    if hint.shape[:2] != (h, w):
        hint = np.array(Image.fromarray(hint).resize((w, h), Image.LANCZOS))

    control = base_np.copy()
    inpaint_b   = inpaint_mask.astype(bool)
    silhouette  = (hint.min(axis=2) < config.SHAPE_PRIOR_THRESH)
    paste_where = inpaint_b & silhouette
    control[paste_where] = hint[paste_where]
    return Image.fromarray(control)


def _run_controlnet_inpaint(
    base_np: np.ndarray,         # H×W×3 uint8 RGB (LaMa-filled scene)
    inpaint_mask: np.ndarray,    # H×W uint8 binary
    amodal_rgb_256: np.ndarray,  # 256×256×3 shape prior hint
    prompt: str,
    n_samples: int,
    out_dir: Path,
    prefix: str = "result",
    neg_extra: str = "",         # additional terms for negative prompt (e.g. occluder nouns)
    original_image_np: Optional[np.ndarray] = None,  # H×W×3 uint8 unmodified RGB (for MCDS)
    visible_mask: Optional[np.ndarray] = None,        # H×W uint8 binary visible-object mask (for MCDS)
    seed_offset: int = 0,        # per-attempt offset so retries produce different samples
    frame_cropped: bool = False, # MCDS only fires when this is True (canvas-extension mode)
) -> list:
    """
    Run inpaint with the configured backend (ControlNet-Inpaint or Flux-Fill).
    Returns list of PIL.Image RGB at base_np's resolution.
    Saves <prefix>_N.png files to out_dir.

    When config.USE_MIXED_CONTEXT is True and both `original_image_np` and
    `visible_mask` are provided, applies Mixed Context Diffusion Sampling
    (port of Xu et al. CVPR 2024) to suppress co-occurrence bias:
      - Scene outside (visible ∪ inpaint) is greyed out in the SD input.
      - The latents outside the visible-object mask are swapped for noisy
        LaMa-clean-bg latents during the first MIXED_CONTEXT_TIMESTEP_FRAC
        of the denoise schedule.

    Dispatch on config.INPAINT_BACKEND:
      "controlnet_sd15" (default) → SD-1.5 + ControlNet-Inpaint with MCDS
      "flux_fill"                  → FLUX.1-Fill-dev (no MCDS, stronger anatomy)
    """
    backend = getattr(config, "INPAINT_BACKEND", "controlnet_sd15")
    if backend == "flux_fill":
        return _run_flux_fill_inpaint(
            base_np         = base_np,
            inpaint_mask    = inpaint_mask,
            amodal_rgb_256  = amodal_rgb_256,
            prompt          = prompt,
            n_samples       = n_samples,
            out_dir         = out_dir,
            prefix          = prefix,
            neg_extra       = neg_extra,
            seed_offset     = seed_offset,
        )
    # else fall through to the original ControlNet-Inpaint path below.
    from PIL import Image as PILImage

    h, w      = base_np.shape[:2]
    orig_hw   = (h, w)
    pipe      = _get_controlnet_pipe()

    # MCDS gating — fire whenever we're in a "BORDER" case where the
    # subject's silhouette extends into UNSEEN TERRITORY (areas with no
    # nearby anchor pixels of the subject).  Three triggers:
    #   • Mode B  : GPT marked frame_cropped=True (legs/tail off the frame)
    #   • Mode A+ : significant amodal extension OUTSIDE the occluder
    #               (bear's legs descending into dirt below the log).  These
    #               pixels have no scene anchor anywhere near them, so SD
    #               needs MCDS's clean-bg-latent swap to free the
    #               structural decisions from the dominant local texture
    #               (dirt/grass/sky pulling the body toward background).
    #   • Pure Mode A (occluder fully contains the completion): skip MCDS;
    #     the surrounding visible context is correct guidance.
    # Border-extension detector: read the occluder mask from disk (we
    # don't get it as a parameter here, but Agent 1 always saves it) and
    # measure how much of the inpaint mask lies OUTSIDE that occluder.
    # "Outside the occluder" pixels are unseen territory — there's no
    # nearby anchor of the subject's appearance, just background scene
    # (dirt, sky, grass).  MCDS context-suppression is exactly what
    # those pixels need.
    mcds_extension_px = 0
    if inpaint_mask is not None and inpaint_mask.sum() > 0:
        try:
            occ_path = out_dir.parent / "occluder_mask.png"
            if occ_path.exists():
                occ_raw = cv2.imread(str(occ_path), cv2.IMREAD_GRAYSCALE)
                if occ_raw is not None:
                    if occ_raw.shape != inpaint_mask.shape:
                        occ_raw = cv2.resize(
                            occ_raw,
                            (inpaint_mask.shape[1], inpaint_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    occ_b = occ_raw > 0
                    # Dilate occluder slightly so "right next to occluder"
                    # pixels still count as Mode A, not border.
                    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    occ_dil = cv2.dilate(occ_b.astype(np.uint8), kern, iterations=2)
                    outside_occluder = (inpaint_mask > 0) & (occ_dil == 0)
                    mcds_extension_px = int(outside_occluder.sum())
        except Exception:
            mcds_extension_px = 0
    mcds_extension_threshold = 500   # >500 px of inpaint outside occluder
    border_extension = mcds_extension_px >= mcds_extension_threshold

    use_mc = (
        getattr(config, "USE_MIXED_CONTEXT", False)
        and (frame_cropped or border_extension)
        and original_image_np is not None
        and visible_mask is not None
        and visible_mask.sum() > 0
    )
    if getattr(config, "USE_MIXED_CONTEXT", False):
        if use_mc:
            trigger = "frame_cropped" if frame_cropped else (
                f"border-extension ({mcds_extension_px} px far from visible)")
            print(f"  [MCDS] enabled — trigger: {trigger}")
        else:
            print(f"  [MCDS] skipped — in-scene occlusion fully near visible "
                  f"(extension={mcds_extension_px} px < "
                  f"{mcds_extension_threshold} threshold)")

    if use_mc:
        print(f"  [MCDS] Mixed Context Diffusion Sampling enabled (frac={config.MIXED_CONTEXT_TIMESTEP_FRAC})")
        # CRITICAL: pass the LaMa-filled image (base_np) as the source, NOT
        # the original image_np.  base_np has the occluder REMOVED (filled
        # with background); the original still has horse/branch/leaf pixels
        # which SD-inpaint would PRESERVE outside the small inpaint_mask.
        # Without this, the occluder leaks through into the final result.
        sd_input_np, object_removed_bg_np = build_mixed_context_inputs(
            image_np      = base_np,             # ← was original_image_np
            visible_mask  = visible_mask,
            inpaint_mask  = inpaint_mask,
            lama_inpaint_fn = _run_lama_inpaint,
            gray_value    = config.MIXED_CONTEXT_GRAY,
        )
        # _run_lama_inpaint internally _get_lama()s; pair with _free_lama
        # so we hand VRAM back to ControlNet/the rest of the pipeline.
        _free_lama()
    else:
        sd_input_np = base_np
        object_removed_bg_np = None

    pil_base  = Image.fromarray(sd_input_np)
    pil_mask  = Image.fromarray((inpaint_mask * 255).astype(np.uint8)).convert("L")
    ctrl_img  = _prep_control_image(sd_input_np, inpaint_mask, amodal_rgb_256, orig_hw)

    # SD works best at multiples of 8; keep within a reasonable budget
    sd_w = min(round(w / 8) * 8, 768)
    sd_h = min(round(h / 8) * 8, 768)

    neg_prompt = (
        "blurry, low quality, artefacts, duplicate, extra limbs, "
        "deformed anatomy, watermark, text"
    )
    if neg_extra:
        neg_prompt = neg_extra + ", " + neg_prompt

    mc_callback = None
    callback_inputs = None
    mc_hook_handles: list = []
    mc_state = None
    if use_mc:
        # (C) + (D): install UNet pre-hook (9-channel masked-image latent swap)
        # and up_blocks[2] forward hook (KMeans-based query-mask refinement).
        # These must be registered fresh per call because they reference the
        # current sample's inpaint mask + object-removed bg.
        try:
            from mixed_context import register_mcds_unet_hook, reset_mcds_sample_state
            mc_hook_handles, mc_state = register_mcds_unet_hook(
                pipe                  = pipe,
                object_removed_bg_np  = object_removed_bg_np,
                visible_mask_np       = visible_mask,
                inpaint_mask_np       = inpaint_mask,
                total_steps           = config.SD_INPAINT_STEPS,
                sd_h                  = sd_h,
                sd_w                  = sd_w,
                mc_step_frac          = config.MIXED_CONTEXT_TIMESTEP_FRAC,
                use_up_ft_kmeans      = getattr(config, "MCDS_USE_UP_FT_KMEANS", True),
                num_clusters          = getattr(config, "MCDS_NUM_CLUSTERS", 8),
                up_block_idx          = getattr(config, "MCDS_UP_BLOCK_IDX", 2),
                intersect_thresh      = getattr(config, "MCDS_INTERSECT_THRESH", 0.2),
            )
        except Exception as exc:                              # noqa: BLE001
            print(f"  [MCDS] UNet hook install failed: {exc!r} — falling back to latent-only swap")

        # (E): latent-space callback, now state-aware so it picks up the
        # KMeans-refined visible mask if the up_ft hook ran.
        mc_callback = make_mc_latent_callback(
            pipe                 = pipe,
            object_removed_bg_np = object_removed_bg_np,
            visible_mask_np      = visible_mask,
            total_steps          = config.SD_INPAINT_STEPS,
            sd_h                 = sd_h,
            sd_w                 = sd_w,
            mc_step_frac         = config.MIXED_CONTEXT_TIMESTEP_FRAC,
            state                = mc_state,
        )
        callback_inputs = ["latents"]

    results = []
    try:
        for i in range(n_samples):
            # Reset MCDS state so each new sample's KMeans refinement starts
            # from the *original* visible mask, not the previous sample's
            # already-refined one.
            if mc_state is not None:
                reset_mcds_sample_state(mc_state)
            pipe_kwargs = dict(
                prompt=prompt,
                negative_prompt=neg_prompt,
                image=pil_base,
                mask_image=pil_mask,
                control_image=ctrl_img,
                height=sd_h,
                width=sd_w,
                num_inference_steps=config.SD_INPAINT_STEPS,
                guidance_scale=config.SD_INPAINT_GUIDANCE_SCALE,
                controlnet_conditioning_scale=config.CONTROLNET_CONDITIONING_SCALE,
                strength=config.SD_INPAINT_STRENGTH,
                generator=torch.manual_seed(42 + seed_offset * 1000 + i),
            )
            if mc_callback is not None:
                pipe_kwargs["callback_on_step_end"] = mc_callback
                pipe_kwargs["callback_on_step_end_tensor_inputs"] = callback_inputs
            out = pipe(**pipe_kwargs).images[0]
            # Resize back to original resolution if SD resized internally
            if out.size != (w, h):
                out = out.resize((w, h), Image.LANCZOS)
            save_path = out_dir / f"{prefix}_{i}.png"
            out.save(str(save_path))
            results.append(out)
    finally:
        for h in mc_hook_handles:
            try: h.remove()
            except Exception: pass
        _free_controlnet()
    return results


def _build_completion_prompt(state: "State") -> str:
    """Build a Goal-B SD inpainting prompt: reconstruct the subject's hidden
    anatomy inside the masked region, matching the visible subject's texture
    and lighting. No occluder references — those are handled by the negative
    prompt and the masks."""
    obj      = state.get("occluded_object", "animal")
    missing  = state.get("missing_parts", "")
    occluder = state.get("occluder", "")

    # Strip occluder noun(s) from missing_parts so SD doesn't re-generate them
    clean_missing = missing
    if occluder:
        for word in occluder.lower().split():
            if len(word) > 3:
                clean_missing = clean_missing.replace(word, "").replace(word.capitalize(), "")

    # Anatomical body-part vocabulary (mammal + bird + human)
    _BODY_TOKENS = [
        # mammals / quadrupeds
        "torso", "abdomen", "lower chest", "chest", "hindquarters", "hind leg", "hind legs",
        "front leg", "front legs", "paw", "paws", "leg", "legs",
        "back", "belly", "flank", "shoulder", "rump", "hips", "knee", "ankle",
        # birds
        "wing", "wings", "wing base", "tail base", "tail", "feet", "foot",
        "vent", "lower belly", "rump", "tarsus",
        # humans
        "torso", "shoulder", "arm", "arms", "hand", "hands", "thigh", "thighs",
        "waist", "hip", "hips",
    ]
    found = []
    for tok in _BODY_TOKENS:
        if tok in clean_missing.lower() and tok not in found:
            found.append(tok)

    parts = [
        f"Photorealistic complete {obj}, full anatomy visible, natural pose.",
        "Sharp focus, matching texture and lighting, anatomically correct, seamless integration.",
    ]
    if found:
        parts.append(f"Show entire {', '.join(found[:6])}, fully visible, anatomically correct, no occlusion.")
    return " ".join(parts)[:300]


# ── Agent 2 — Inpainting Agent (LaMa + ControlNet-Inpaint + shape prior) ──────

def inpainting_agent(state: State) -> dict:
    attempt = state["attempt"] + 1
    print(f"\n─── Agent 2: Inpainting Agent  (attempt {attempt}) ───")

    img_path = Path(state["image_path"])
    out_dir  = BASE_DIR / "output" / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    image_np = np.array(Image.open(img_path).convert("RGB"))
    orig_h, orig_w = image_np.shape[:2]

    occluder_mask = (
        _load_binary_mask(state["mask_path"], (orig_w, orig_h))
        if state.get("mask_path") and Path(state["mask_path"]).exists()
        else np.zeros((orig_h, orig_w), dtype=np.uint8)
    )
    visible_mask = (
        _load_binary_mask(state["visible_mask_path"], (orig_w, orig_h))
        if state.get("visible_mask_path") and Path(state["visible_mask_path"]).exists()
        else np.ones((orig_h, orig_w), dtype=np.uint8)
    )
    hidden_mask = (
        _load_binary_mask(state["hidden_mask_path"], (orig_w, orig_h))
        if state.get("hidden_mask_path") and Path(state["hidden_mask_path"]).exists()
        else np.zeros((orig_h, orig_w), dtype=np.uint8)
    )

    # ── Pre-inpaint mask sanity checks ─────────────────────────────────────────
    # Detect the failure mode where GPT drew `hidden_polygon` along the
    # occluder's outline instead of the subject's body silhouette. When that
    # happens the occluder mask sits inside the hidden mask (or vice-versa),
    # so the generator paints the subject in the occluder's shape.
    _occ_b = occluder_mask.astype(bool)
    _hid_b = hidden_mask.astype(bool)
    _occ_area = int(_occ_b.sum())
    _hid_area = int(_hid_b.sum())
    _inter    = int((_occ_b & _hid_b).sum())
    occ_inside_hidden  = _inter / max(_occ_area, 1)   # fraction of occluder swallowed by hidden
    hidden_is_occluder = _inter / max(_hid_area, 1)   # fraction of hidden that equals occluder
    print(f"  [Sanity] occluder={_occ_area}px hidden={_hid_area}px "
          f"occ⊂hidden={occ_inside_hidden:.2f} hidden⊂occ={hidden_is_occluder:.2f}")
    if _occ_area > 0 and _hid_area > 0:
        if occ_inside_hidden > 0.50:
            print(f"  [Sanity] ⚠ hidden_object_mask CONTAINS the occluder "
                  f"({100*occ_inside_hidden:.0f}%) — hidden polygon was likely drawn "
                  f"as the occluder shape instead of the subject's body silhouette.")
        if hidden_is_occluder > 0.50:
            print(f"  [Sanity] ⚠ hidden_object_mask IS occluder-shaped "
                  f"({100*hidden_is_occluder:.0f}%) — ControlNet will paint the "
                  f"subject inside the occluder's outline, not the subject's anatomy.")

    # ── Frame-crop canvas expansion ────────────────────────────────────────────
    pad_top = pad_bottom = pad_left = pad_right = 0
    if state.get("frame_cropped"):
        exp_px     = state.get("expansion_pixels") or {}
        pad_top    = max(0, int(exp_px.get("top",    0)))
        pad_bottom = max(0, int(exp_px.get("bottom", 0)))
        pad_left   = max(0, int(exp_px.get("left",   0)))
        pad_right  = max(0, int(exp_px.get("right",  0)))

        if pad_top + pad_bottom + pad_left + pad_right > 0:
            image_np = cv2.copyMakeBorder(
                image_np, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REPLICATE,
            )
            orig_h, orig_w = image_np.shape[:2]

            vis_exp = cv2.copyMakeBorder(
                visible_mask * 255, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0,
            )
            visible_mask = (vis_exp > 127).astype(np.uint8)

            hid_exp = cv2.copyMakeBorder(
                hidden_mask * 255, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0,
            )
            hidden_mask = (hid_exp > 127).astype(np.uint8)

            occluder_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            if pad_top:    occluder_mask[:pad_top, :]              = 1
            if pad_bottom: occluder_mask[orig_h - pad_bottom:, :]  = 1
            if pad_left:   occluder_mask[:, :pad_left]              = 1
            if pad_right:  occluder_mask[:, orig_w - pad_right:]    = 1

            # In frame-crop mode the padded canvas is the missing-object region.
            hidden_mask = np.clip(hidden_mask | occluder_mask, 0, 1).astype(np.uint8)

            print(f"  [Frame-crop] Canvas expanded → {orig_w}×{orig_h} "
                  f"(T:{pad_top} B:{pad_bottom} L:{pad_left} R:{pad_right})")

    exp     = state["boundary_expansion"]
    k_exp   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (exp * 2 + 1, exp * 2 + 1))
    occ_dil = cv2.dilate(occluder_mask, k_exp, iterations=1)

    # ── Cache dir: one per mask-retry so shape priors regenerate on mask retry ──
    mask_gen = state.get("mask_retry_count", 1)
    comp_dir = out_dir / f"completions_mask{mask_gen}"
    comp_dir.mkdir(exist_ok=True)

    # ── Step 1: pix2gestalt → binary shape prior ───────────────────────────────
    aligned_shape, amodal_rgb_256 = _run_shape_prior(
        image_np, visible_mask, (orig_h, orig_w), comp_dir,
        hidden_polygon=None if state.get("frame_cropped") else state.get("hidden_polygon"),
    )
    print(f"  [Shape prior] object px in scene: {aligned_shape.sum()}")

    # ── Step 1b: GPT-V full-amodal mask override ──────────────────────────────
    # If Agent 1 produced subject_full_amodal_mask.png (GPT-V's polygon trace
    # of the COMPLETE subject silhouette), prefer it over pix2gestalt's
    # threshold output.  GPT-V's mask covers the correct region even when
    # the occluder is BESIDE the subject — pix2gestalt only extends in
    # trained directions and misses side-occluder cases (e.g. horse next
    # to person).  We also synthesise an amodal_rgb_256 hint from this
    # mask so the ControlNet shape-prior conditioning matches.
    gpt_amodal_path = out_dir / "subject_full_amodal_mask.png"
    if gpt_amodal_path.exists():
        gpt_amodal = cv2.imread(str(gpt_amodal_path), cv2.IMREAD_GRAYSCALE)
        if gpt_amodal is not None and gpt_amodal.shape != (orig_h, orig_w):
            gpt_amodal = cv2.resize(gpt_amodal, (orig_w, orig_h),
                                    interpolation=cv2.INTER_NEAREST)
        if gpt_amodal is not None and int((gpt_amodal > 0).sum()) > 0:
            aligned_shape = (gpt_amodal > 0).astype(np.uint8)
            print(f"  [GPT-V amodal] OVERRIDING pix2gestalt shape prior with "
                  f"GPT-V silhouette ({int(aligned_shape.sum())} px)")
            # Build a 256x256 hint that mirrors pix2gestalt's format:
            # subject pixels = sampled from the visible_mask region of the
            # original image (matching texture) on white background.
            hint = np.ones((256, 256, 3), dtype=np.uint8) * 255
            small_mask = cv2.resize(aligned_shape * 255, (256, 256),
                                    interpolation=cv2.INTER_NEAREST)
            # Use the original-image colours within the amodal silhouette so
            # ControlNet has both a shape AND a colour cue.
            small_img = cv2.resize(image_np, (256, 256),
                                   interpolation=cv2.INTER_AREA)
            sub_b = small_mask > 0
            hint[sub_b] = small_img[sub_b]
            amodal_rgb_256 = hint
            cv2.imwrite(str(comp_dir / "shape_prior_from_gptv.png"), amodal_rgb_256)

            # CRITICAL: hidden_mask = AMODAL SILHOUETTE − VISIBLE.
            # The completion region is EVERYTHING in the full subject
            # silhouette that is NOT already in the visible mask — that
            # includes parts behind the occluder AND parts that extend
            # beyond the occluder (e.g. legs descending into the dirt
            # below a bear's log, or a person's legs cropped at the bottom
            # of the photo).  Painting only the occluder ∩ amodal slice
            # leaves the legs un-inpainted, so the cutout shows dirt
            # pixels where bear legs should be.
            new_hidden = np.clip(
                (aligned_shape > 0).astype(np.int32)
                - (visible_mask > 0).astype(np.int32),
                0, 1,
            ).astype(np.uint8)
            # Light dilation (≈4 px) for boundary smoothness so SD doesn't
            # produce hard seams; cap inside the amodal silhouette so we
            # don't bleed into the surrounding scene.
            dilated = cv2.dilate(
                new_hidden,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=2,         # ≈ 4 px outward
            )
            new_hidden = (dilated.astype(bool)
                          & (aligned_shape > 0)).astype(np.uint8)
            print(f"  [GPT-V amodal] hidden_mask refined: "
                  f"{int(hidden_mask.sum())} → {int(new_hidden.sum())} px "
                  f"(amodal − visible, +4 px dilation inside amodal)")
            hidden_mask = new_hidden
            # Flag so downstream sanity / auto-rescue logic knows the
            # masks are already internally consistent and shouldn't be
            # second-guessed.
            gpt_v_amodal_override = True
        else:
            gpt_v_amodal_override = False
    else:
        gpt_v_amodal_override = False

    # ── Sanity: does the shape prior actually overlap the hidden region? ──────
    _shape_b = aligned_shape.astype(bool)
    hidden_shape_iou = 1.0
    if _hid_b.sum() > 0 and _shape_b.sum() > 0:
        _union = int((_hid_b | _shape_b).sum())
        _ix    = int((_hid_b & _shape_b).sum())
        hidden_shape_iou = _ix / max(_union, 1)
        print(f"  [Sanity] hidden ∩ shape_prior IoU = {hidden_shape_iou:.2f}")
        if hidden_shape_iou < 0.10:
            print("  [Sanity] ⚠ Shape prior does not overlap the hidden region "
                  "— ControlNet has no useful subject silhouette to follow inside "
                  "the inpaint mask.")

    # ── Auto-rescue: GPT polygon is occluder-shaped → rebuild silhouette ──────
    # Trigger when sanity flagged the polygon as occluder-shaped or the shape
    # prior doesn't overlap. Build a subject-body silhouette from three
    # independent signals (more reliable than any single one):
    #   • shape_prior  − visible: pix2gestalt's predicted body
    #   • hidden_bbox  − visible: GPT's hidden_region bounding box (when it's
    #     a sensible non-occluder rectangle, e.g. a tall body extension below
    #     the visible region)
    #   • occluder     − visible: guarantees pixels under the local occluder
    #     get inpainted as subject, not preserved as background
    # Subtracting visible everywhere keeps the modal subject untouched.
    rescue_triggered = False
    # Skip the auto-rescue when the GPT-V amodal override has been applied:
    # in that case hidden_mask is already a tight subset of aligned_shape
    # (occluder ∩ amodal) and the IoU metric is intentionally low because
    # the completion is much smaller than the full silhouette.  Rescuing
    # would re-inflate hidden_mask back to the entire occluder.
    if (hidden_is_occluder > 0.50 or hidden_shape_iou < 0.10) and not gpt_v_amodal_override:
        vis_b = visible_mask.astype(bool)
        components = []
        if _shape_b.sum() > 0:
            shape_minus_vis = _shape_b & ~vis_b
            if shape_minus_vis.sum() > 100:
                components.append(("shape_prior", shape_minus_vis))
        # GPT hidden_region bbox as a body-silhouette hint
        bbox_arr = state.get("bbox")
        if bbox_arr and len(bbox_arr) == 4:
            bx1 = max(0, int(bbox_arr[0])); by1 = max(0, int(bbox_arr[1]))
            bx2 = min(orig_w, int(bbox_arr[2])); by2 = min(orig_h, int(bbox_arr[3]))
            if bx2 > bx1 and by2 > by1:
                bbox_mask = np.zeros((orig_h, orig_w), dtype=bool)
                bbox_mask[by1:by2, bx1:bx2] = True
                bbox_minus_vis = bbox_mask & ~vis_b
                if bbox_minus_vis.sum() > 100:
                    components.append(("hidden_bbox", bbox_minus_vis))
        if _occ_b.sum() > 0:
            occ_minus_vis = _occ_b & ~vis_b
            if occ_minus_vis.sum() > 100:
                components.append(("occluder", occ_minus_vis))

        if components:
            rescued = np.zeros((orig_h, orig_w), dtype=bool)
            comp_names = []
            for name, m in components:
                rescued |= m
                comp_names.append(f"{name}({int(m.sum())}px)")
            rescued_u8 = rescued.astype(np.uint8)
            if rescued_u8.sum() > 100:
                print(f"  [Rescue] Replacing GPT hidden_mask ({_hid_area}px, occluder-shaped) "
                      f"with union of: {', '.join(comp_names)} − visible "
                      f"= {int(rescued_u8.sum())}px")
                hidden_mask = rescued_u8
                _hid_b = hidden_mask.astype(bool)
                _hid_area = int(_hid_b.sum())
                rescue_triggered = True
                cv2.imwrite(
                    str(out_dir / "hidden_object_mask_rescued.png"),
                    hidden_mask * 255,
                )

    # ── Step 2: split removable background from missing object ────────────────
    # Three paths, in order of preference:
    #   (a) Both hidden_mask AND a usable shape prior → REFINE hidden_mask by
    #       intersecting with the prior so pumpkin-rim / non-object pixels in the
    #       user polygon get dropped.  Only refine when the prior overlaps the
    #       hidden_mask substantially; otherwise the prior is misaligned and we
    #       trust the hidden polygon as-is.
    #   (b) Only hidden_mask → use it directly.
    #   (c) Only shape prior → fall back to occ_dil ∩ shape prior.
    if hidden_mask.sum() > 0 and aligned_shape.sum() > 0:
        intersect = (hidden_mask.astype(bool) & aligned_shape.astype(bool))
        hm_b      = hidden_mask.astype(bool)
        coverage  = int(intersect.sum()) / max(int(hm_b.sum()), 1)
        # Be conservative about trimming hidden_mask with the shape prior:
        # pix2gestalt's silhouette is often misaligned vertically (esp. when
        # most of the body is hidden, so the visible-mask crop is small).
        # Trimming away the lower-body pixels would ask the generator to
        # complete only the upper body — exactly the bear-feet bug.
        # Require both: (a) high coverage AND (b) the intersection is at
        # least 60 % of hidden_mask before we accept the trim. Otherwise keep
        # the full hidden_mask.
        intersect_ratio = int(intersect.sum()) / max(int(hm_b.sum()), 1)
        if coverage >= 0.60 and intersect_ratio >= 0.60:
            obj_inpaint = intersect.astype(np.uint8)
            print(f"  [Masks] hidden_object ∩ shape_prior → obj_inpaint {obj_inpaint.sum()} px  "
                  f"(coverage={coverage:.2f}, kept after strict trim)")
        else:
            obj_inpaint = hidden_mask.astype(np.uint8)
            print(f"  [Masks] hidden_object kept whole (coverage={coverage:.2f}, "
                  f"intersect_ratio={intersect_ratio:.2f}) — "
                  f"shape prior would over-trim lower body")
        bg_inpaint  = np.clip(occ_dil & ~obj_inpaint, 0, 1).astype(np.uint8)
    elif hidden_mask.sum() > 0:
        obj_inpaint = hidden_mask.astype(np.uint8)
        bg_inpaint  = np.clip(occ_dil & ~obj_inpaint, 0, 1).astype(np.uint8)
        print(f"  [Masks] hidden_object → obj_inpaint {obj_inpaint.sum()} px  bg_inpaint {bg_inpaint.sum()} px")
    else:
        obj_inpaint = np.clip(occ_dil & aligned_shape,  0, 1).astype(np.uint8)
        bg_inpaint  = np.clip(occ_dil & ~aligned_shape, 0, 1).astype(np.uint8)
        print(f"  [Masks] shape-prior fallback → obj_inpaint {obj_inpaint.sum()} px  bg_inpaint {bg_inpaint.sum()} px")
    if obj_inpaint.sum() == 0:
        print("  [Shape prior] WARNING: empty inpaint region — treating full occluder as object region")
        obj_inpaint = occ_dil.copy()
        bg_inpaint  = np.zeros_like(occ_dil)

    # Persist the final (post-refinement) hidden_object_mask for inspection
    cv2.imwrite(str(out_dir / f"hidden_object_mask_refined_attempt{attempt}.png"),
                obj_inpaint * 255)

    # allow_region covers BOTH the dilated occluder and the hidden polygon so that
    # ControlNet-generated content is not overwritten outside the occluder mask.
    allow_region = np.clip(occ_dil.astype(np.int32) + obj_inpaint.astype(np.int32), 0, 1).astype(np.uint8)

    # ── Step 3: LaMa fills the ENTIRE occluder region ─────────────────────────
    # IMPORTANT: we LaMa-fill the WHOLE occluder (occ_dil), not just
    # bg_inpaint.  Earlier the LaMa pass only erased the slivers of the
    # occluder OUTSIDE the subject silhouette (bg_inpaint = occ_dil & ~
    # shape_prior), leaving the bulk of the occluder pixels INTACT in the
    # base image that SD inpaint sees.  Those leftover occluder pixels
    # encoded into masked-image-latents and biased UNet's early denoising
    # steps to RE-SYNTHESIZE the occluder shape — that's why the horse /
    # branch / leaf stayed visible in the ControlNet output even with
    # MCDS active.  Filling the whole occluder with LaMa gives SD a
    # genuinely occluder-free starting context.
    full_occluder_inpaint = occ_dil.astype(np.uint8)
    if full_occluder_inpaint.sum() > 0:
        print(f"  [LaMa] Filling FULL occluder region "
              f"({int(full_occluder_inpaint.sum())} px = obj {int(obj_inpaint.sum())} "
              f"+ bg {int(bg_inpaint.sum())})…")
        try:
            lama_result = _run_lama_inpaint(image_np, full_occluder_inpaint)
        finally:
            _free_lama()
    else:
        lama_result = image_np.copy()

    lama_result[allow_region == 0] = image_np[allow_region == 0]

    # ── Step 4: ControlNet-Inpaint fills object region ────────────────────────
    # If the rescue rebuilt the hidden mask, drop any cached results from the
    # bad mask so ControlNet regenerates against the corrected silhouette.
    if rescue_triggered:
        for old in comp_dir.glob("result_*.png"):
            old.unlink()
        print(f"  [Rescue] Invalidated stale ControlNet cache in {comp_dir.name}")
    cached_results = sorted(comp_dir.glob("result_*.png"))
    # On a retry attempt, if we have already cycled through every cached sample,
    # regenerate a fresh batch with a different seed series instead of looping
    # back to index 0 — that loop was producing byte-identical attempt_*_rgb.png
    # outputs on consecutive retries.
    if cached_results and attempt > 1 and attempt > len(cached_results):
        print(f"  [Retry] Exhausted {len(cached_results)} cached samples on "
              f"attempt {attempt} — wiping cache to force fresh seeds")
        for old in cached_results:
            old.unlink()
        cached_results = []
    if not cached_results:
        prompt = _build_completion_prompt(state)
        # Build negative terms from occluder so SD doesn't re-generate it
        occluder_str = state.get("occluder", "")
        neg_extra = ", ".join(
            w for w in occluder_str.lower().split()
            if len(w) > 3 and w not in {"with", "that", "from", "into", "over"}
        ) if occluder_str else ""
        print(f"  [ControlNet] Generating {config.N_SAMPLES} samples (seed_offset={attempt - 1})…")
        print(f"  Prompt: {prompt[:120]}")
        if neg_extra:
            print(f"  Neg-extra: {neg_extra}")
        _run_controlnet_inpaint(
            base_np           = lama_result,
            inpaint_mask      = obj_inpaint,
            amodal_rgb_256    = amodal_rgb_256,
            prompt            = prompt,
            n_samples         = config.N_SAMPLES,
            out_dir           = comp_dir,
            neg_extra         = neg_extra,
            original_image_np = image_np,
            visible_mask      = visible_mask,
            seed_offset       = attempt - 1,
            frame_cropped     = bool(state.get("frame_cropped", False)),
        )
        cached_results = sorted(comp_dir.glob("result_*.png"))
        print(f"  [ControlNet] {len(cached_results)} results cached → {comp_dir}")
    else:
        print(f"  [ControlNet] {len(cached_results)} cached results in {comp_dir}")

    sample_idx  = (attempt - 1) % len(cached_results)
    sample_path = cached_results[sample_idx]
    result_np   = np.array(Image.open(str(sample_path)).convert("RGB"))
    print(f"  Using sample {sample_idx} ({sample_path.name})")

    # ── Restore original pixels outside the occluded region ───────────────────
    result_np[allow_region == 0] = image_np[allow_region == 0]
    if pad_top + pad_bottom + pad_left + pad_right > 0:
        orig_file_np = np.array(Image.open(img_path).convert("RGB"))
        ofh, ofw     = orig_file_np.shape[:2]
        result_np[pad_top:pad_top + ofh, pad_left:pad_left + ofw] = orig_file_np

    blend_dst = image_np.copy()
    blend_dst[bg_inpaint == 1] = lama_result[bg_inpaint == 1]
    if pad_top + pad_bottom + pad_left + pad_right > 0:
        orig_file_np = np.array(Image.open(img_path).convert("RGB"))
        ofh, ofw     = orig_file_np.shape[:2]
        blend_dst[pad_top:pad_top + ofh, pad_left:pad_left + ofw] = orig_file_np

    result_np = _poisson_blend(result_np, blend_dst, allow_region.astype(np.uint8))

    # ── Flux refinement cascade (gated by config.USE_FLUX_REFINEMENT) ────────
    # Take the ControlNet result and run Flux-Fill over the inpaint region at
    # partial denoising strength. ControlNet provided shape compliance via the
    # pix2gestalt prior; Flux now polishes anatomy + texture quality.
    if getattr(config, "USE_FLUX_REFINEMENT", False):
        try:
            print(f"  [Flux-Refine] Cascading ControlNet → Flux-Fill "
                  f"(strength={config.FLUX_REFINEMENT_STRENGTH})…")
            refine_prompt = (
                f"Photorealistic complete {state.get('occluded_object', 'subject')}, "
                f"full anatomy visible, natural pose. Sharp focus, matching texture "
                f"and lighting, anatomically accurate. Avoid: cartoon, distorted, "
                f"duplicate parts, color bleed."
            )
            # Save the ControlNet result as the cascade input for debugging
            cv2.imwrite(str(out_dir / f"_cascade_input_attempt_{attempt}.png"),
                        cv2.cvtColor(result_np.astype(np.uint8), cv2.COLOR_RGB2BGR))
            flux_refined = _run_flux_fill_inpaint(
                base_np=result_np.astype(np.uint8),
                inpaint_mask=allow_region.astype(np.uint8),
                amodal_rgb_256=np.full((256, 256, 3), 255, dtype=np.uint8),
                prompt=refine_prompt,
                n_samples=1,
                out_dir=out_dir,
                prefix=f"flux_refine_attempt_{attempt}",
                neg_extra="",
                seed_offset=attempt,
                strength=float(config.FLUX_REFINEMENT_STRENGTH),
            )
            if flux_refined:
                refined_np = np.array(flux_refined[0].convert("RGB"))
                if refined_np.shape[:2] != result_np.shape[:2]:
                    refined_np = cv2.resize(
                        refined_np, (result_np.shape[1], result_np.shape[0]),
                        interpolation=cv2.INTER_LANCZOS4)
                # Restore pixels OUTSIDE the inpaint region from the ControlNet
                # blended result (Flux preserves these but VAE round-trip can drift).
                refined_np[allow_region == 0] = result_np[allow_region == 0]
                result_np = refined_np
                print(f"  [Flux-Refine] ✓ refined output saved")
            else:
                print(f"  [Flux-Refine] ✗ Flux returned nothing — keeping ControlNet result")
        except Exception as exc:                                  # noqa: BLE001
            print(f"  [Flux-Refine] failed: {exc!r} — keeping ControlNet result")

    # ── Save outputs ──────────────────────────────────────────────────────────
    rgb_path  = out_dir / f"attempt_{attempt}_rgb.png"
    rgba_path = out_dir / f"attempt_{attempt}_rgba.png"
    comp_path = out_dir / f"attempt_{attempt}_comparison.png"

    result_pil = Image.fromarray(result_np.astype(np.uint8))
    result_pil.save(str(rgb_path))

    rgba_arr = np.dstack([result_np, occ_dil * 255])
    Image.fromarray(rgba_arr.astype(np.uint8), mode="RGBA").save(str(rgba_path))

    # ── Post-generation SAM3 segmentation of the subject in the result ───────
    # The pre-generation pix2gestalt silhouette is computed from the VISIBLE
    # crop only — when the bird's lower body / legs / feet are entirely hidden
    # in the input, pix2gestalt has no way to extend its prediction down to
    # those parts. As a result `aligned_shape ∩ obj_inpaint` excludes them
    # from the cutout even when ControlNet did generate plausible content.
    #
    # We re-segment the generated result with SAM3 at the visible bird's
    # centroid. SAM3 sees the FULL bird in the result (visible + generated)
    # and gives us a clean silhouette that includes the inpainted legs/feet.
    post_gen_mask = None
    try:
        ys, xs = np.where(visible_mask > 0)
        if len(xs) > 0:
            cx = int(np.median(xs))
            cy = int(np.median(ys))
            tmp_result_path = out_dir / f"_tmp_result_for_segment_{attempt}.png"
            Image.fromarray(result_np.astype(np.uint8)).save(str(tmp_result_path))
            print(f"  [PostSeg] Re-segmenting bird in result at ({cx},{cy})…")
            post_results = _sam_segment_targeted(
                str(tmp_result_path),
                [{"label": f"post_gen_subject_{attempt}", "x": cx, "y": cy}],
                out_dir,
            )
            try:
                tmp_result_path.unlink()
            except OSError:
                pass
            if post_results:
                m = post_results[0]["mask"]
                m_area = int(m.sum())
                # Sanity: must be at least as large as visible_mask, not absurdly large
                if m_area >= int(visible_mask.sum()) * 0.8 and m_area < 0.7 * (orig_h * orig_w):
                    post_gen_mask = m
                    print(f"  [PostSeg] ✓ post-gen mask {m_area} px (was visible={int(visible_mask.sum())})")
                else:
                    print(f"  [PostSeg] ✗ rejected (area={m_area}, visible={int(visible_mask.sum())})")
    except Exception as exc:
        print(f"  [PostSeg] failed (non-fatal): {exc}")

    # ── Full-subject cutout on white background ──────────────────────────────
    # Composite the entire subject (visible parts + generated parts) on a
    # clean white canvas. This is the CVPR'25-style "amodal completion" output:
    # just the bird/cat/etc with its previously-occluded body now generated.
    #
    # The subject silhouette is NOT obj_inpaint — that has the OCCLUDER's shape
    # (e.g. a horizontal branch that extends far beyond the bird's body, which
    # would put bird+branch in the cutout). Instead we use:
    #   subject = visible_mask  ∪  (aligned_shape ∩ obj_inpaint)
    # i.e. all visible bird pixels + only the bird-shaped part of the inpaint
    # region (pix2gestalt predicts where the body should be).
    # Priority order for the cutout silhouette:
    #   1. post_gen_mask  — SAM3 on the generated result (ground truth of
    #                        what's actually bird-shaped in the output)
    #   2. visible_mask ∪ (aligned_shape ∩ obj_inpaint)
    #                     — pre-gen prediction (may miss legs/feet)
    #   3. visible_mask ∪ obj_inpaint
    #                     — fallback (may include occluder tails)
    bound = np.clip(visible_mask.astype(np.int32) + obj_inpaint.astype(np.int32), 0, 1).astype(np.uint8)

    # NEW priority: the GPT-V amodal silhouette saved by Agent 1 is the
    # canonical full-subject shape (visible + completion).  Use it directly
    # so the cutout always shows the COMPLETE subject body including
    # behind-the-occluder parts — that's what the user wants for the final
    # transparent PNG.  SAM3 post-gen segmentation is a poor fallback
    # because it only sees what the inpaint generated, which often misses
    # the completion region.
    gpt_amodal_path = out_dir / "subject_full_amodal_mask.png"
    gpt_amodal_silhouette = None
    # Derive the target H,W from the visible_mask (or the bound mask
    # already built above) — they all share the inpainting_agent's
    # image resolution.
    _tgt_h, _tgt_w = visible_mask.shape[:2]
    if gpt_amodal_path.exists():
        _raw = cv2.imread(str(gpt_amodal_path), cv2.IMREAD_GRAYSCALE)
        if _raw is not None:
            if _raw.shape != (_tgt_h, _tgt_w):
                _raw = cv2.resize(_raw, (_tgt_w, _tgt_h), interpolation=cv2.INTER_NEAREST)
            gpt_amodal_silhouette = (_raw > 127).astype(np.uint8)

    if gpt_amodal_silhouette is not None and int(gpt_amodal_silhouette.sum()) > 0:
        full_subject = gpt_amodal_silhouette
        print(f"  [WhiteBG] GPT-V amodal silhouette → "
              f"{int(full_subject.sum())} px (canonical)")
    elif post_gen_mask is not None:
        # Use SAM3's post-generation segmentation as the canonical bird shape.
        # Combine with visible_mask to be safe (SAM3 might miss a sliver).
        full_subject = np.clip(
            post_gen_mask.astype(np.int32) + visible_mask.astype(np.int32),
            0, 1,
        ).astype(np.uint8)
        print(f"  [WhiteBG] post-gen SAM3 mask → {int(full_subject.sum())} px (canonical)")
    elif aligned_shape is not None and int(aligned_shape.sum()) > 0:
        # subject = visible_mask  ∪  (aligned_shape ∩ obj_inpaint)
        hidden_subject = (aligned_shape.astype(bool) & obj_inpaint.astype(bool)).astype(np.uint8)
        full_subject   = np.clip(
            visible_mask.astype(np.int32) + hidden_subject.astype(np.int32),
            0, 1,
        ).astype(np.uint8)
        full_subject = cv2.dilate(full_subject,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                  iterations=1)
        full_subject = (full_subject & bound).astype(np.uint8)
        print(f"  [WhiteBG] amodal ∩ inpaint fallback → {int(full_subject.sum())} px")
    else:
        full_subject = bound.copy()
        print(f"  [WhiteBG] last-resort visible ∪ obj_inpaint → {int(full_subject.sum())} px")

    # Soft alpha: feather the edges by 2 px so the cutout doesn't look hard-edged
    alpha = (full_subject * 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

    white_bg = np.full_like(result_np, 255)
    white_path  = out_dir / f"attempt_{attempt}_white_bg.png"
    cutout_path = out_dir / f"attempt_{attempt}_cutout_rgba.png"
    if full_subject.sum() > 0:
        # White-background composite (alpha blend)
        a_f = (alpha[..., None].astype(np.float32) / 255.0)
        white_comp = (a_f * result_np.astype(np.float32)
                      + (1.0 - a_f) * white_bg.astype(np.float32)).astype(np.uint8)
        Image.fromarray(white_comp).save(str(white_path))
        # RGBA cutout (transparent background) — also useful
        cutout_rgba = np.dstack([result_np, alpha])
        Image.fromarray(cutout_rgba.astype(np.uint8), mode="RGBA").save(str(cutout_path))
        print(f"  WhiteBG → {white_path}  ({int(full_subject.sum())} px subject)")
        print(f"  Cutout  → {cutout_path}")
    else:
        # Fallback: dump the full result on white if we couldn't compute a subject mask
        Image.fromarray(result_np).save(str(white_path))
        print(f"  WhiteBG → {white_path}  (no subject mask — saved full result)")

    src_pil      = Image.open(img_path).convert("RGB")
    src_w, src_h = src_pil.size
    res_w, res_h = result_pil.size
    comp_h       = max(src_h, res_h)
    if src_h != comp_h:
        src_pil = src_pil.resize((int(src_w * comp_h / src_h), comp_h), Image.LANCZOS)
        src_w   = src_pil.width
    res_comp = result_pil if res_h == comp_h else result_pil.resize(
        (int(res_w * comp_h / res_h), comp_h), Image.LANCZOS
    )
    res_w = res_comp.width
    pd, lh = 8, 28
    comparison = Image.new("RGB", (src_w + res_w + pd * 3, comp_h + pd * 2 + lh), (30, 30, 30))
    comparison.paste(src_pil,  (pd, pd + lh))
    comparison.paste(res_comp, (src_w + pd * 2, pd + lh))
    cn = np.array(comparison)
    cv2.putText(cn, "ORIGINAL",
                (pd + 6, pd + lh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(cn, f"LaMa+ControlNet+ShapePrior  attempt {attempt}  sample {sample_idx}",
                (src_w + pd * 2 + 6, pd + lh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(cn, (src_w + pd + pd // 2, 0),
             (src_w + pd + pd // 2, comp_h + pd * 2 + lh), (80, 80, 80), pd)
    cv2.imwrite(str(comp_path), cn[:, :, ::-1])

    print(f"  RGB   → {rgb_path}")
    print(f"  RGBA  → {rgba_path}")
    print(f"  Comp  → {comp_path}")

    return {
        **state,
        "output_path":      str(rgb_path),
        "output_rgba_path": str(rgba_path),
        "pix2gestalt_dir":  str(comp_dir),
        "attempt":          attempt,
    }


# ── Reviewer ──────────────────────────────────────────────────────────────────

def reviewer(state: State) -> dict:
    print(f"\n─── Reviewer  (attempt {state['attempt']}) ──────────────────")

    target    = state["occluded_object"]
    all_codes = sorted(config.ALL_FAILURE_CODES)

    frame_cropped      = state.get("frame_cropped", False)
    subject_desc       = state.get("subject_description", "")
    visible_parts_desc = state.get("visible_parts", "")
    missing_parts_desc = state.get("missing_parts", "")

    if frame_cropped:
        mode_context = f"""MODE: Frame-crop completion
  - "{target}" was cut off at the image frame boundary.
  - Expansion direction(s): {state.get("expansion_directions", [])}
  - Subject description: {subject_desc}
  - Parts that WERE visible in original: {visible_parts_desc}
  - Parts that WERE MISSING and should now appear: {missing_parts_desc}
  - Image 2 is LARGER than Image 1 — the added region contains the generated missing parts."""
        job_line = f'verify that "{target}" was successfully completed beyond the frame boundary — the missing body parts should now appear naturally in the expanded region'
    else:
        occluder_label = state.get("occluder", "occluder")
        mode_context = f"""MODE: In-scene occlusion removal
  - "{target}" was partially hidden behind "{occluder_label}" in the original scene.
  - Image 2 is the MODIFIED SCENE: the occluder has been removed and the hidden body parts have been inpainted.
  - Background: original scene background — NOT a white background (the scene background is correct and expected).
  - Subject description: {subject_desc}
  - Parts that WERE visible in original: {visible_parts_desc}
  - Parts that WERE HIDDEN and must now appear: {missing_parts_desc}
  - The occluder ("{occluder_label}") should no longer dominate the image.
  - DO NOT penalise for the dark/original background — only evaluate the completeness and anatomy of the revealed subject."""
        job_line = (
            f'verify that (1) the occluder ("{occluder_label}") is significantly reduced or removed, '
            f'and (2) "{target}" now has all the body parts listed in missing_parts — '
            f'every part must be anatomically correct and consistent with the visible parts'
        )

    prompt = f"""You are a meticulous image-completion quality reviewer with deep knowledge of animal and human anatomy.

YOUR ONLY JOB: {job_line}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{mode_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO INSPECT IN IMAGE 2 (the result)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ANATOMY CHECK — penalise heavily for any of these:
  ✗ Duplicated or mirrored body parts (e.g. four legs on a bird, two bodies, ghost copy)
  ✗ Wrong body-part COUNT for the species:
      Birds: exactly 2 legs, 2 wings, 1 beak, 1 tail
      Humans: 2 arms, 2 legs, 5 fingers per hand
      Quadrupeds: 4 legs, 1 tail
  ✗ Wrong proportions (legs too short/long, head too large, etc.)
  ✗ Missing body parts that should have been generated
  ✗ Deformed, melted, or surreal anatomy
  ✗ Background objects appearing inside the subject's body

COLOUR & TEXTURE CHECK — penalise for:
  ✗ Colour mismatch between the generated region and the original visible subject
      (e.g. legs are different shade, feet different colour than beak/eye-ring)
  ✗ Texture inconsistency (feathers/fur/skin grain doesn't match visible part)
  ✗ Lighting direction mismatch (shadows on wrong side)

SEAM CHECK — penalise for:
  ✗ Visible horizontal or vertical boundary line between original and generated area
  ✗ White halo, dark halo, or fringe at the join
  ✗ Colour banding or sudden colour shift at the boundary
  ✗ Blurry or low-resolution generated area vs. sharp original

OVERALL COMPLETION CHECK:
  ✓ The specific missing parts listed above are NOW PRESENT and look correct
  ✓ The subject looks like a single, complete, naturally-photographed specimen
  ✓ A casual viewer would not notice any editing

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Score 1–10 as the AVERAGE of:
  (A) COMPLETENESS: Are ALL missing parts now present and anatomically correct?
  (B) SEAMLESSNESS: Does the result look like a single unedited photograph?

  9–10 : Both criteria excellent — indistinguishable from a real photo
  7–8  : Mostly good — very minor issue (slight softness, tiny colour shift)
  5–6  : Partial success — one criterion clearly fails
  3–4  : Clear failure — duplicated parts, wrong anatomy, or obvious seam
  1–2  : Complete failure — subject still truncated or looks impossibly wrong

Retry threshold: score < {config.SCORE_THRESHOLD}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FAILURE CODES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return EXACTLY ONE code from: {all_codes}
  ACCEPTED               — score >= {config.SCORE_THRESHOLD}, result is convincing
  MASK_INACCURATE        — generated region is in the wrong location (mask wrong → re-run masks)
  OCCLUDER_REMNANT_SOLID — the occluder is still PRESENT and OPAQUE — you can clearly
                           see the occluding object as a solid shape, with its own
                           texture/colours, fully covering the subject. Indicates the
                           mask did not cover the whole occluder (mask wrong → re-run masks).
  OCCLUDER_GHOST         — the occluder has been REMOVED but the area where it was now
                           contains TRANSLUCENT GHOST / HALO / BLEED-THROUGH artifacts
                           (faint duplicated stripes, semi-transparent shapes, smeared
                           textures). The original occluder is gone but the fill is
                           messy.  This is a FILL-QUALITY problem, not a mask problem —
                           the mask was correct, the SD inpaint just didn't synthesize
                           clean replacement pixels.  (cycle SD sample, do NOT shrink mask)
  SHAPE_PRIOR_BAD        — generated body silhouette is grossly wrong (e.g. random shape, no
                           clear animal outline, second-body ghost) → regenerate shape prior
  ANATOMY_WRONG    — duplicated parts, wrong count, deformed, or wrong species features
  COLOR_MISMATCH   — colours/texture of generated area don't match the visible subject
  SEAM_VISIBLE     — clear boundary line, halo, or colour banding at the join
  PROMPT_WEAK      — generic filler, textureless blob, or unrecognisable result
  BLURRY_OUTPUT    — generated area is notably blurrier or lower-detail than original

Respond ONLY in JSON:
{{
  "score": <float 1-10>,
  "feedback": "<2–3 sentences: what specific anatomy/colour/seam issues were found, and what exactly is wrong or right>",
  "failure_code": "<one of: {', '.join(all_codes)}>",
  "improved_prompt": "",
  "improved_negative_prompt": ""
}}"""

    data         = gpt_vision([state["image_path"], state["output_path"]], prompt, schema=REVIEWER_SCHEMA, cache_key="reviewer_v1")
    score        = float(data.get("score", 5.0))
    feedback     = data.get("feedback", "")
    failure_code = data.get("failure_code", "PROMPT_WEAK")

    print(f"  Score        : {score:.1f}/10")
    print(f"  Failure code : {failure_code}")
    print(f"  Feedback     : {feedback}")

    best_score   = state.get("best_score", 0.0)
    best_attempt = state.get("best_attempt", state["attempt"])
    if score > best_score:
        best_score   = score
        best_attempt = state["attempt"]
        print(f"  ★ New best   : attempt {best_attempt}  score {best_score:.1f}")

    log_file = BASE_DIR / "output" / Path(state["image_path"]).stem / "review_log.json"
    log = json.loads(log_file.read_text()) if log_file.exists() else []
    log.append({
        "attempt":      state["attempt"],
        "score":        score,
        "feedback":     feedback,
        "failure_code": failure_code,
    })
    log_file.write_text(json.dumps(log, indent=2))

    return {
        **state,
        "review_score":    score,
        "review_feedback": feedback,
        "failure_code":    failure_code,
        "best_score":      best_score,
        "best_attempt":    best_attempt,
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def route(state: State) -> str:
    score        = state["review_score"]
    failure_code = state.get("failure_code", "")
    attempt      = state["attempt"]
    mask_retries = state.get("mask_retry_count", 0)

    if score >= config.SCORE_THRESHOLD:
        print(f"\n✓ Accepted — score {score:.1f}")
        return "end"

    if attempt >= config.MAX_RETRIES:
        best = state.get("best_attempt", attempt)
        print(f"\n✗ Max retries ({config.MAX_RETRIES}) reached. Best: attempt {best} "
              f"(score {state.get('best_score', score):.1f})")
        return "end"

    out_dir = BASE_DIR / "output" / Path(state["image_path"]).stem

    # Best-attempt revert: if a prior attempt already scored higher than the
    # current one, the current path is regressing.  Don't let a low-scoring
    # MASK_* failure re-segment further — we'd just be walking deeper into a
    # bad region of mask-space.  Fall back to cycling SD samples on the prior
    # (better) mask instead.  This is exactly the failure mode we observed on
    # the zebra image: attempt 1 (broad mask, front zebra removed) was
    # rejected as OCCLUDER_REMNANT, mask was shrunk for attempt 2/3, and
    # we lost the removal entirely.
    best_score = float(state.get("best_score", 0.0))
    if (failure_code in config.MASK_FAILURE_CODES
            and score < best_score
            and attempt >= 2):
        print(f"\n↻ Mask failure ({failure_code}) but score {score:.1f} < best "
              f"{best_score:.1f} — REVERTING to retry_sample on prior mask "
              f"(would otherwise regress mask further)")
        return "retry_sample"

    if failure_code in config.MASK_FAILURE_CODES and mask_retries <= config.MAX_MASK_RETRIES:
        # Wipe stale ControlNet caches so the next inpainting pass regenerates with the
        # new masks instead of cycling through old samples produced from the bad masks.
        for d in out_dir.glob("completions_mask*"):
            for f in d.glob("result_*.png"):
                try: f.unlink()
                except OSError: pass
        print(f"\n↻ Mask failure ({failure_code}) — re-analysing occlusion "
              f"(mask retry {mask_retries}/{config.MAX_MASK_RETRIES}); cleared cached samples")
        return "retry_mask"

    if failure_code in config.SHAPE_FAILURE_CODES:
        # Wipe shape prior + ControlNet caches so pix2gestalt re-runs with fresh seed.
        mask_gen = state.get("mask_retry_count", 1)
        comp_dir = out_dir / f"completions_mask{mask_gen}"
        for f in list(comp_dir.glob("shape_prior*.png")) + list(comp_dir.glob("result_*.png")):
            try: f.unlink()
            except OSError: pass
        print(f"\n↻ Shape-prior failure ({failure_code}) — regenerating pix2gestalt + ControlNet samples")
        return "retry_sample"

    print(f"\n↻ Trying next pix2gestalt sample — score {score:.1f} < {config.SCORE_THRESHOLD}  [{failure_code}]")
    return "retry_sample"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(State)

    g.add_node("occlusion_agent",  occlusion_agent)
    g.add_node("inpainting_agent", inpainting_agent)
    g.add_node("reviewer",         reviewer)

    g.set_entry_point("occlusion_agent")
    g.add_edge("occlusion_agent",  "inpainting_agent")
    g.add_edge("inpainting_agent", "reviewer")
    g.add_conditional_edges(
        "reviewer",
        route,
        {
            "retry_sample": "inpainting_agent",
            "retry_mask":   "occlusion_agent",
            "end":          END,
        },
    )

    return g.compile()


# ── Entry ─────────────────────────────────────────────────────────────────────

def run():
    img_path = Path(config.IMAGE_PATH)
    if not img_path.exists():
        print(f"Error: image not found: {img_path}")
        sys.exit(1)

    initial: State = {
        "image_path":         str(img_path),
        "target":             config.TARGET,
        "occluded_object":    "",
        "occluder":           "",
        "what_to_remove":     "",
        "bbox":               None,
        "boundary_expansion": config.MASK_EXPAND,
        "region_desc":        "",
        "subject_description":   "",
        "visible_parts":         "",
        "missing_parts":         "",
        "frame_cropped":         False,
        "expansion_directions":  [],
        "expansion_pixels":      None,
        "mask_path":              None,
        "visible_mask_path":      None,
        "occluder_removed_path":  None,
        "occluder_viz_path":      None,
        "hidden_polygon":         None,
        "hidden_mask_path":       None,
        "pix2gestalt_dir":        None,
        "output_path":        "",
        "output_rgba_path":   "",
        "review_score":       0.0,
        "review_feedback":    "",
        "failure_code":       "",
        "attempt":            0,
        "mask_retry_count":   0,
        "best_attempt":       0,
        "best_score":         0.0,
    }

    pipeline = build_graph()
    final    = pipeline.invoke(initial)

    stem         = img_path.stem
    best_attempt = final.get("best_attempt", final["attempt"])
    best_score   = final.get("best_score",   final["review_score"])

    print(f"\n{'='*52}")
    print(f"  Best output  : output/{stem}/attempt_{best_attempt}_rgb.png  (score {best_score:.1f}/10)")
    print(f"  RGBA output  : output/{stem}/attempt_{best_attempt}_rgba.png")
    print(f"  White BG     : output/{stem}/attempt_{best_attempt}_white_bg.png  (full subject, no scene)")
    print(f"  Cutout RGBA  : output/{stem}/attempt_{best_attempt}_cutout_rgba.png  (transparent bg)")
    print(f"  Last score   : {final['review_score']:.1f}/10  (attempt {final['attempt']})")
    print(f"  Review log   : output/{stem}/review_log.json")
    print(f"  pix2gestalt  : output/{stem}/pix2gestalt_mask*/")


if __name__ == "__main__":
    run()
