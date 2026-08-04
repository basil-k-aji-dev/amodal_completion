"""Agent 2 — FLUX.1-Fill-dev diffusion inpainting."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config
from runtime import DEVICE, _release_ram

_FLUX_FILL_PIPE = None

def _get_flux_fill_pipe():
    """Lazy-load FluxFillPipeline.

    Flux-Fill is a 12B-param DiT-based inpainter with strong anatomy priors.
    Drawback: ~16-24 GB VRAM.

    Returns None when the diffusers version is too old to support
    `FluxFillPipeline` or the model checkpoint can't be loaded.
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
        # Hard barrier — free every other GPU-resident singleton (SAM3,
        # SAM3-text, etc.) so Flux's 12B-param weight load doesn't fight for
        # the last MB. Without this, `cpu_offload` mode (~10 GB peak) OOMs
        # on 12 GB cards at the SAM3→Flux handoff.
        from model_lifecycle import _free_all_gpu_models_except  # local import breaks the flux_fill<->model_lifecycle cycle
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
        #   model      → whole pipeline on device during inference (~24 GB
        #                for Flux — OOMs on 12 GB cards, only for SD-1.5).
        #   none       → fully on device, requires 24 GB+ headroom (A100).
        use_mmgp  = bool(getattr(config, "USE_MMGP_OFFLOAD", False))
        use_group = bool(getattr(config, "FLUX_FILL_GROUP_OFFLOAD", False))
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
                print(f"  [Flux-Fill] mmgp setup failed: {exc!r} — falling back to group/model offload")
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
        elif getattr(config, "FLUX_FILL_CPU_OFFLOAD", False):
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
        print(f"  [Flux-Fill] load failed: {exc!r}")
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
    keep_loaded: bool = False,  # skip the post-call GPU release (see below)
) -> list:
    """Run FLUX.1-Fill-dev on the inpaint region.

    Flux-Fill takes (image, mask, prompt) and produces an inpainted image
    in one shot — no shape-prior conditioning image, no UNet hooks. Anatomy
    quality comes from the model's native priors (DiT, 12B params) rather
    than from external scaffolding.

    `keep_loaded=True` skips freeing the pipeline from GPU after this call.
    Loading/releasing a 12B-param model is the dominant cost of a call (the
    30-step generation itself takes seconds; the load+release teardown
    around it takes ~100s+) — that overhead is only necessary right before
    something else (SAM3, InstaFormer) needs the GPU next. A caller making
    several back-to-back Flux calls with nothing else touching the GPU in
    between (e.g. the reviewer retry loop) should pass `keep_loaded=True`
    on every call and free the pipeline itself once, via `_free_flux_fill`,
    when it's actually done — see pipeline.py's retry loop.
    """
    from PIL import Image as PILImage

    h, w = base_np.shape[:2]
    pipe = _get_flux_fill_pipe()
    if pipe is None:
        raise RuntimeError(
            "Flux-Fill pipeline failed to load — see the "
            "'[Flux-Fill] load failed' log line above for the cause."
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

    # FluxFillPipeline has two text encoders: CLIP (pooled embedding, hard
    # 77-token cap) and T5 (sequence embedding, up to max_sequence_length).
    # If `prompt_2` isn't given it defaults to `prompt`, so CLIP silently
    # truncates whatever we pass as `prompt` — observed in practice eating
    # the tail of the merged prompt+"Avoid:" clause (and any long
    # reviewer-improved retry prompt) right where the corrective
    # instructions live. Keep CLIP's copy short and front-loaded with the
    # core instruction; let T5's `prompt_2` carry the full text.
    clip_prompt = " ".join(flux_prompt.split()[:45])

    # T5's max_sequence_length (config.FLUX_FILL_MAX_SEQUENCE_LEN, e.g. 512)
    # is a real ceiling, but every padding token past the prompt's actual
    # length still costs attention compute on every diffusion step.
    # Hardcoding one smaller fixed value would risk silently truncating a
    # long reviewer-improved retry prompt — the same failure mode just
    # fixed for CLIP above, at a higher token budget. Instead, measure
    # THIS prompt's real token count with T5's own tokenizer and pick the
    # smallest safe bucket that fits it; a long prompt automatically gets
    # bumped up to whatever it actually needs, up to the configured
    # ceiling — it never gets truncated.
    configured_max = int(getattr(config, "FLUX_FILL_MAX_SEQUENCE_LEN", 512))
    try:
        n_tokens = len(pipe.tokenizer_2(flux_prompt, truncation=False)["input_ids"])
    except Exception:
        n_tokens = len(flux_prompt.split()) * 2   # generous fallback if tokenizer_2 is unavailable
    max_seq_len = configured_max
    for bucket in (64, 128, 192, 256, 384):
        if bucket <= configured_max and n_tokens + 8 <= bucket:
            max_seq_len = bucket
            break

    print(f"  [Flux-Fill] Generating {n_samples} samples "
          f"(steps={config.FLUX_FILL_STEPS}, guidance={config.FLUX_FILL_GUIDANCE_SCALE}, "
          f"strength={strength}, {flux_w}×{flux_h}, seed_offset={seed_offset}, "
          f"max_seq_len={max_seq_len}/{configured_max} [{n_tokens} tokens])…")
    print(f"  Prompt: {flux_prompt[:140]}")

    results = []
    try:
        for i in range(n_samples):
            seed = 42 + seed_offset * 1000 + i
            generator = torch.Generator(device="cpu").manual_seed(seed)
            call_kwargs = dict(
                prompt=clip_prompt,
                prompt_2=flux_prompt,
                image=pil_base,
                mask_image=pil_mask,
                height=flux_h,
                width=flux_w,
                num_inference_steps=config.FLUX_FILL_STEPS,
                guidance_scale=config.FLUX_FILL_GUIDANCE_SCALE,
                max_sequence_length=max_seq_len,
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
        if not keep_loaded:
            _free_flux_fill()
    return results


