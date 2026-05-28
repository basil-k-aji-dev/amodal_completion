"""
models/flux.py — FLUX.1-Fill inpainting (local + remote).

Local Flux-Fill pipeline (lazy singleton, offload mode driven by runtime's
VRAM auto-detect) plus a remote-server client (_run_flux_remote) selected when
config.INPAINT_BACKEND == "remote_flux".
"""

from __future__ import annotations

import time

import numpy as np
import torch

import config
from runtime import DEVICE, SERVER_MODE, _release_ram, free_all_except, register_free_fn

# ── Lazy singleton ──────────────────────────────────────────────────────────
_FLUX_FILL_PIPE = None


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
        free_all_except("flux")
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
        pipe = FluxFillPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        if bool(getattr(config, "FLUX_FILL_SEQUENTIAL_OFFLOAD", False)):
            pipe.enable_sequential_cpu_offload()
            mode = "sequential_cpu_offload"
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
        print(f"  [Flux-Fill] load failed: {exc!r}")
        _FLUX_FILL_PIPE = None
    return _FLUX_FILL_PIPE


def _free_flux_fill():
    """Unload Flux from GPU. No-op in SERVER_MODE — stays resident."""
    if SERVER_MODE:
        return
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



def _run_flux_remote(
    base_np: np.ndarray,
    inpaint_mask: np.ndarray,
    prompt: str,
    n_samples: int,
    neg_extra: str = "",
    seed_offset: int = 0,
) -> list:
    """Call a remote Flux server (server.py /inpaint, e.g. on a Lightning
    Studio or Colab). Returns a list of PIL Images matching the local
    backend's interface."""
    import requests, base64, io
    from PIL import Image as PILImage

    url = getattr(config, "REMOTE_FLUX_URL", "").rstrip("/")
    if not url:
        raise RuntimeError("INPAINT_BACKEND='remote_flux' but REMOTE_FLUX_URL is empty")
    timeout = float(getattr(config, "REMOTE_FLUX_TIMEOUT", 900))

    # Encode inputs as raw PNG bytes (smaller than data URIs).
    base_buf = io.BytesIO(); PILImage.fromarray(base_np).save(base_buf, "PNG")
    mask_pil = PILImage.fromarray((inpaint_mask * 255).astype(np.uint8)).convert("L")
    mask_buf = io.BytesIO(); mask_pil.save(mask_buf, "PNG")

    flux_prompt = prompt
    if neg_extra:
        flux_prompt = f"{prompt}.  Avoid: {neg_extra}."

    print(f"  [Flux-remote] POST {url}/inpaint  ({base_np.shape[1]}×{base_np.shape[0]}, "
          f"steps={config.FLUX_FILL_STEPS}, guidance={config.FLUX_FILL_GUIDANCE_SCALE})")
    t0 = time.time()
    try:
        resp = requests.post(
            f"{url}/inpaint",
            files={
                "image": ("image.png", base_buf.getvalue(), "image/png"),
                "mask":  ("mask.png",  mask_buf.getvalue(), "image/png"),
            },
            data={
                "prompt":   flux_prompt,
                "steps":    int(config.FLUX_FILL_STEPS),
                "guidance": float(config.FLUX_FILL_GUIDANCE_SCALE),
                "samples":  int(n_samples),
                "seed":     int(seed_offset),
            },
            timeout=timeout,
        )
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [Flux-remote] request failed: {exc!r}")
        raise

    if resp.status_code != 200:
        raise RuntimeError(f"[Flux-remote] server returned HTTP {resp.status_code}: "
                           f"{resp.text[:300]}")
    data = resp.json()
    imgs = []
    for b64 in data.get("images_b64", []):
        imgs.append(PILImage.open(io.BytesIO(base64.b64decode(b64))))
    print(f"  [Flux-remote] received {len(imgs)} image(s) in {time.time() - t0:.1f}s")
    return imgs


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

    Raises if the Flux backend is unavailable. This pipeline intentionally
    does not fall back to ControlNet.
    """
    from PIL import Image as PILImage

    # Remote backend short-circuit — Lightning Studio / self-hosted Flux.
    if getattr(config, "INPAINT_BACKEND", "flux_fill") == "remote_flux":
        imgs = _run_flux_remote(
            base_np=base_np, inpaint_mask=inpaint_mask, prompt=prompt,
            n_samples=n_samples, neg_extra=neg_extra, seed_offset=seed_offset,
        )
        # Save to out_dir for parity with local backend.
        for i, im in enumerate(imgs):
            im.save(out_dir / f"{prefix}_{i}.png")
        return imgs

    h, w = base_np.shape[:2]
    pipe = _get_flux_fill_pipe()
    if pipe is None:
        raise RuntimeError(
            "Flux-Fill backend unavailable. ControlNet fallback is disabled; "
            "install/use Flux or set INPAINT_BACKEND='remote_flux'."
        )

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



register_free_fn("flux", _free_flux_fill)
