"""Iterative off-frame canvas-extension (Jiang Ao CVPR'25 style): pads the
canvas and outpaints when the subject's amodal silhouette touches a frame
edge. Kept in the codebase but switched off via config.USE_OFFFRAME_EXTENSION
— the current pipeline scope is the in-frame hidden region only.
"""
from pathlib import Path

import cv2
import numpy as np

import config
from models.flux_fill import _run_flux_fill_inpaint
from models.sam3 import _sam_segment_targeted

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


