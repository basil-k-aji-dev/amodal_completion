"""Mixed Context Diffusion Sampling for amodal_test3.

Port of Xu et al., "Amodal Completion via Progressive Mixed Context Diffusion"
(CVPR 2024, see amodal_k8xu/) onto the diffusers ControlNet-Inpaint pipeline.

The reference pipeline (amodal_k8xu/diffusers/.../
pipeline_stable_diffusion_inpaint_mixed_context.py:1362-1515) does three
things that this module reproduces by registering hooks on a stock
diffusers pipeline:

  (A) Background swap — replace pixels outside (visible ∪ SD-inpaint) with
      neutral gray BEFORE the pipeline ever sees the image, so the scene
      context never enters the masked-image latent or ControlNet conditioning.
      Implemented in `build_mixed_context_inputs`.

  (B) Object-removed background — LaMa-inpaint (visible ∪ SD-inpaint) to
      supply a clean-scene reference image. Returned alongside (A).

  (C) 9-channel masked-image latent swap — SD-inpaint's UNet takes 9 input
      channels: 4 noisy-latent + 1 mask + 4 masked-image latents. The
      reference swaps channels 5-8 to the *object-removed* masked-image
      latents during the late denoising window. Implemented as a UNet
      `forward_pre_hook` in `register_mcds_unet_hook`.

  (D) up_blocks[2] KMeans mask refinement — at the transition step, the
      reference clusters UNet decoder features (up_ft[2]) with KMeans and
      grows the query-mask to include clusters that overlap it; this
      captures the synthesized-object pixels that lie outside the original
      mask. Implemented as a UNet up_blocks[2] `forward_hook` paired with
      a state dict held in closure (`register_mcds_unet_hook`).

  (E) Latent-space swap — for the early portion of denoising, replace
      latents outside the (now refined) query mask with noisy clean-bg
      latents. Plugged in via `callback_on_step_end` in
      `make_mc_latent_callback`. (Unchanged from the original test3
      implementation.)

Together (A)+(B)+(C)+(D)+(E) reproduce the k8xu mixed-context behaviour
without forking the diffusers pipeline.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image


# ── (A) + (B) Input-side preprocessing ────────────────────────────────────────


def build_mixed_context_inputs(
    image_np: np.ndarray,
    visible_mask: np.ndarray,
    inpaint_mask: np.ndarray,
    lama_inpaint_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    gray_value: int = 127,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (swapped_input_image, object_removed_background).

    swapped_input_image  — H×W×3 uint8 RGB; pixels outside (visible ∪ inpaint)
                           replaced with `gray_value`. Goes to SD as the
                           inpaint input.
    object_removed_bg    — H×W×3 uint8 RGB; LaMa-filled (visible ∪ inpaint)
                           erased. Source of clean-bg latents.
    """
    keep = visible_mask.astype(bool) | inpaint_mask.astype(bool)

    swapped = image_np.copy()
    swapped[~keep] = gray_value

    object_remove_mask = keep.astype(np.uint8)
    object_removed_bg = lama_inpaint_fn(image_np, object_remove_mask)

    return swapped, object_removed_bg


# ── Latent encoding helper ────────────────────────────────────────────────────


def _encode_to_latents(pipe, image_np: np.ndarray, dtype, device) -> torch.Tensor:
    """Encode an H×W×3 uint8 RGB array to scaled VAE latents."""
    img = torch.from_numpy(image_np).to(device=device, dtype=torch.float32) / 127.5 - 1.0
    img = img.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)
    with torch.no_grad():
        latent_dist = pipe.vae.encode(img).latent_dist
        latent = latent_dist.mean * pipe.vae.config.scaling_factor
    return latent


# ── (C) + (D) UNet input swap + up_ft[2] KMeans refinement ────────────────────


def register_mcds_unet_hook(
    pipe,
    object_removed_bg_np: np.ndarray,
    visible_mask_np: np.ndarray,
    inpaint_mask_np: np.ndarray,
    total_steps: int,
    sd_h: int,
    sd_w: int,
    mc_step_frac: float = 0.7,
    use_up_ft_kmeans: bool = True,
    num_clusters: int = 8,
    up_block_idx: int = 2,
    intersect_thresh: float = 0.2,
) -> tuple[list, dict]:
    """Install the 9-channel masked-image latent swap + KMeans mask refinement.

    Returns (handles, state):
      handles : list of torch hook handles — caller must `.remove()` each
                after pipeline generation finishes.
      state   : mutable dict shared with the latent-swap callback; carries
                the (possibly refined) visible-object mask at SD-latent
                resolution under key 'vis_mask_latent_t' (torch tensor).
    """
    device = next(pipe.unet.parameters()).device
    dtype  = pipe.unet.dtype

    bg_pil   = Image.fromarray(object_removed_bg_np).resize((sd_w, sd_h), Image.LANCZOS)
    bg_np_sd = np.array(bg_pil)
    bg_image_t = (torch.from_numpy(bg_np_sd).to(device=device, dtype=torch.float32)
                  / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)

    with torch.no_grad():
        bg_latent_dist = pipe.vae.encode(bg_image_t).latent_dist
        bg_latent_full = bg_latent_dist.mean * pipe.vae.config.scaling_factor  # [1, 4, h_lat, w_lat]

    h_lat, w_lat = bg_latent_full.shape[-2:]

    # Visible-mask resampled to latent resolution; held in `state` so the
    # KMeans hook can REPLACE it with a refined mask and the latent callback
    # picks up the update.
    import cv2  # local import keeps top of file dep-light
    vis_lat_np = cv2.resize((visible_mask_np > 0).astype(np.uint8),
                            (w_lat, h_lat), interpolation=cv2.INTER_NEAREST)
    inp_lat_np = cv2.resize((inpaint_mask_np > 0).astype(np.uint8),
                            (w_lat, h_lat), interpolation=cv2.INTER_NEAREST)
    _orig_vis_t = (torch.from_numpy(vis_lat_np.astype(np.float32))
                        .to(device=device, dtype=dtype)[None, None]).clone()
    state: dict = {
        "vis_mask_latent_np":  vis_lat_np.copy(),
        "inp_mask_latent_np":  inp_lat_np,
        "vis_mask_latent_t":   _orig_vis_t.clone(),
        # Pristine copies so reset_mcds_sample_state can restore them between
        # samples — without this, sample N+1's KMeans refinement compounds on
        # sample N's already-grown mask (observed: 123 → 262 → 282 → 314 → 576).
        "_orig_vis_mask_latent_np": vis_lat_np.copy(),
        "_orig_vis_mask_latent_t":  _orig_vis_t,
        "captured_up_ft":      None,
        "step":                {"i": -1, "last_t": None},
        "cutoff_step":         int(round(total_steps * mc_step_frac)),
        "kmeans_done":         False,
    }

    # ── Pre-hook on UNet: swap channels 5-8 of input sample ───────────────────
    # SD-inpaint UNet has 9 input channels: [noisy_latents(4), mask(1),
    # masked_image_latents(4)]. We replace channels 5-8 with bg latents in the
    # late-denoising window (matches k8xu's `t < timesteps[mc_timestep]`).
    def _pre_hook(module, args, kwargs):
        if not args:
            return None
        sample = args[0]
        if not isinstance(sample, torch.Tensor) or sample.ndim != 4 or sample.shape[1] != 9:
            return None  # ControlNet or non-inpaint call; pass through

        # Advance step counter once per unique timestep (UNet may be called
        # twice per step under CFG but with identical t).
        t_arg = args[1] if len(args) > 1 else kwargs.get("timestep")
        if isinstance(t_arg, torch.Tensor):
            t_val = float(t_arg.flatten()[0].item())
        else:
            t_val = float(t_arg)
        step = state["step"]
        if step["last_t"] != t_val:
            step["i"] += 1
            step["last_t"] = t_val

        if step["i"] < state["cutoff_step"]:
            # Early denoising — leave original masked-image latents.
            return None

        bg_l = bg_latent_full.to(dtype=sample.dtype, device=sample.device)
        if bg_l.shape[0] != sample.shape[0]:
            bg_l = bg_l.expand(sample.shape[0], -1, -1, -1)

        new_sample = sample.clone()
        new_sample[:, 5:9, :, :] = bg_l
        new_args = (new_sample,) + tuple(args[1:])
        return new_args, kwargs

    unet_pre = pipe.unet.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    handles = [unet_pre]

    # ── Forward-hook on up_blocks[N]: capture features for KMeans ─────────────
    if use_up_ft_kmeans and hasattr(pipe.unet, "up_blocks"):
        if up_block_idx >= len(pipe.unet.up_blocks):
            up_block_idx = len(pipe.unet.up_blocks) - 1

        def _up_hook(module, args, output):
            # Capture only at the transition step, and only once.
            if state["kmeans_done"]:
                return
            if state["step"]["i"] != state["cutoff_step"]:
                return
            # `output` is the up_block return — either a tensor or a tuple.
            feat = output if isinstance(output, torch.Tensor) else output[0]
            # Under CFG (batch=2 with uncond+cond), index 1 is the text-cond
            # branch (matches k8xu: `unet_ft = unet_ft[1]`). With batch=1, use 0.
            idx = 1 if feat.shape[0] >= 2 else 0
            state["captured_up_ft"] = feat[idx].detach().to(torch.float32).cpu().numpy()
            state["kmeans_done"] = True
            # Run KMeans + mask refinement immediately so the next pre-hook /
            # latent callback sees the updated visible mask.
            _refine_visible_mask_from_up_ft(
                state, num_clusters, intersect_thresh,
                device=device, dtype=dtype,
            )

        up_handle = pipe.unet.up_blocks[up_block_idx].register_forward_hook(_up_hook)
        handles.append(up_handle)

    return handles, state


def reset_mcds_sample_state(state: dict) -> None:
    """Restore the MCDS state to its as-registered values for a new sample.

    Resets the step counter, the captured up_ft buffer, the kmeans-done flag,
    AND the visible-mask snapshots — so each new sample's KMeans refinement
    starts from the *original* visible mask, not the previous sample's
    already-refined one.
    """
    if state is None:
        return
    state["step"]["i"] = -1
    state["step"]["last_t"] = None
    state["kmeans_done"] = False
    state["captured_up_ft"] = None
    orig_np = state.get("_orig_vis_mask_latent_np")
    orig_t  = state.get("_orig_vis_mask_latent_t")
    if orig_np is not None:
        state["vis_mask_latent_np"] = orig_np.copy()
    if orig_t is not None:
        state["vis_mask_latent_t"] = orig_t.clone()


def _refine_visible_mask_from_up_ft(
    state: dict, num_clusters: int, intersect_thresh: float,
    device, dtype,
) -> None:
    """Cluster captured up_ft with KMeans and grow the visible mask.

    Mirrors the `get_interm_query_mask` routine in
    amodal_k8xu/diffusers/.../pipeline_stable_diffusion_inpaint_mixed_context.py
    """
    feat = state.get("captured_up_ft")
    if feat is None:
        return
    # feat: (C, h, w) — flatten spatial → (h*w, C) for clustering
    C, h, w = feat.shape
    flat = feat.reshape(C, -1).T  # (N, C)

    try:
        from scipy.cluster.vq import kmeans2
        centroids, labels = kmeans2(flat, num_clusters, minit="++", seed=0)
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [MCDS] KMeans skipped ({exc!r}); leaving visible mask un-refined")
        return

    cluster_map = labels.reshape(h, w).astype(np.int32)

    # Upsample cluster map to latent resolution
    import cv2
    vis_lat = state["vis_mask_latent_np"]
    inp_lat = state["inp_mask_latent_np"]
    H_lat, W_lat = vis_lat.shape
    cluster_lat = cv2.resize(cluster_map, (W_lat, H_lat),
                             interpolation=cv2.INTER_NEAREST)

    query_inpaint_overlap = ((vis_lat | inp_lat) > 0).astype(np.uint8)
    refined = vis_lat.copy().astype(np.int32)
    thresh_px = max(1, int(intersect_thresh * max(int(vis_lat.sum()), 1)))

    for cid in np.unique(cluster_lat):
        cluster_mask = (cluster_lat == cid)
        inter = int((cluster_mask & (vis_lat > 0)).sum())
        if inter > thresh_px:
            refined[cluster_mask] = 1

    refined = (refined > 0).astype(np.uint8)
    refined = (refined & query_inpaint_overlap).astype(np.uint8)

    n_before = int(vis_lat.sum())
    n_after  = int(refined.sum())
    print(f"  [MCDS] up_ft KMeans refined visible mask: {n_before} → {n_after} latent px")

    state["vis_mask_latent_np"] = refined
    state["vis_mask_latent_t"]  = (torch.from_numpy(refined.astype(np.float32))
                                       .to(device=device, dtype=dtype)[None, None])


# ── (E) Latent-space callback (now reads refined visible mask from state) ─────


def make_mc_latent_callback(
    pipe,
    object_removed_bg_np: np.ndarray,
    visible_mask_np: np.ndarray,
    total_steps: int,
    sd_h: int,
    sd_w: int,
    mc_step_frac: float = 0.7,
    state: Optional[dict] = None,
):
    """Build a `callback_on_step_end` that swaps background latents.

    If `state` is provided (from `register_mcds_unet_hook`), the callback
    reads the (possibly KMeans-refined) visible mask from it on every step;
    otherwise it uses the resampled `visible_mask_np` directly.

    Existing test3 semantics (swap during EARLY denoising, step_index <
    cutoff) are preserved.
    """
    device = pipe._execution_device if hasattr(pipe, "_execution_device") else next(pipe.unet.parameters()).device
    dtype  = pipe.unet.dtype

    bg_pil = Image.fromarray(object_removed_bg_np).resize((sd_w, sd_h), Image.LANCZOS)
    bg_np_sd = np.array(bg_pil)
    bg_latent = _encode_to_latents(pipe, bg_np_sd, dtype=dtype, device=device)

    H_lat, W_lat = bg_latent.shape[2], bg_latent.shape[3]
    vis_pil = Image.fromarray((visible_mask_np * 255).astype(np.uint8))
    vis_resized = np.array(vis_pil.resize((W_lat, H_lat), Image.NEAREST),
                           dtype=np.float32) / 255.0
    fallback_vis_t = torch.from_numpy(vis_resized).to(device=device, dtype=dtype)[None, None, :, :]

    cutoff_step = int(round(total_steps * mc_step_frac))

    def callback(pipe_, step_index: int, t, callback_kwargs):
        if step_index >= cutoff_step:
            return callback_kwargs
        latents = callback_kwargs.get("latents")
        if latents is None:
            return callback_kwargs

        bg_l = bg_latent.to(dtype=latents.dtype, device=latents.device)
        if bg_l.shape[0] != latents.shape[0]:
            bg_l = bg_l.expand(latents.shape[0], -1, -1, -1)

        noise = torch.randn(bg_l.shape, device=latents.device, dtype=latents.dtype, generator=None)
        t_tensor = t if isinstance(t, torch.Tensor) else torch.tensor([t], device=latents.device)
        noisy_bg = pipe_.scheduler.add_noise(bg_l, noise, t_tensor)

        if state is not None and isinstance(state.get("vis_mask_latent_t"), torch.Tensor):
            mask = state["vis_mask_latent_t"].to(dtype=latents.dtype, device=latents.device)
        else:
            mask = fallback_vis_t.to(dtype=latents.dtype, device=latents.device)
        callback_kwargs["latents"] = latents * mask + noisy_bg * (1.0 - mask)
        return callback_kwargs

    return callback
