"""
pipeline/nodes.py — LangGraph nodes for the amodal completion pipeline.

  occlusion_agent  — Agent 1: SAM3 auto-seg + GPT-V + CLIP grounding produce
                     the visible mask, occluder mask, and amodal target region.
  completion_agent — Agent 2: builds the subject cutout, runs Flux-Fill on the
                     hidden slice, re-segments with SAM3, then iteratively
                     extends off-frame (Jiang Ao CVPR'25). Writes the RGB /
                     RGBA / white-bg outputs + comparison + metrics.
  reviewer         — Agent 3: GPT-V scores the result and emits a failure code.

Plus the GPT/SAM3/CLIP/Flux support helpers those nodes call.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config
from runtime import BASE_DIR
from pipeline.state import State
from pipeline.schemas import (
    OCCLUSION_SCHEMA,
    REVIEWER_SCHEMA,
    _MASK_REVIEW_SCHEMA,
    _GPT_AMODAL_SCHEMA,
    _GPT_REVIEW_SCHEMA,
)
from models.gpt import gpt_vision
from models.sam3 import (
    _sam_segment_all,
    _sam_segment_targeted,
    _sam_segment_text_prompt,
    _mask_from_text_segments,
)
from models.clip import (
    _extract_short_label,
    _clip_label_segments,
    _clip_grid_locate,
    _save_clip_grid_viz,
    _save_clip_label_viz,
    _clip_verify_mask,
    _clip_verified_strict,
)
from models.flux import _run_flux_fill_inpaint

# Neutral gray fill outside the amodal silhouette (completion_agent).
PAD_COLOR          = 128
# Iterative off-frame extension (Jiang Ao CVPR'25).
MAX_OFFFRAME_ITERS = 3
PAD_PER_ITER       = 150
BOUNDARY_GAP_PX    = 10


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




def _gpt_verify_visible_mask(
    image_bgr: np.ndarray,
    visible_mask: np.ndarray,
    target_class: str,
    out_dir: Path,
) -> dict:
    """Build green-mask overlay viz, ask GPT-V to verify it captures
    ``target_class`` correctly. Returns
    {'ok', 'rationale', 'corrective_click_x', 'corrective_click_y'}.
    """
    h, w = image_bgr.shape[:2]
    overlay = image_bgr.copy()
    layer = np.zeros_like(image_bgr)
    layer[visible_mask > 0] = (0, 255, 0)
    overlay = cv2.addWeighted(overlay, 0.55, layer, 0.45, 0)
    cv2.putText(overlay,
                f"green = visible mask for '{target_class}'",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    viz_path = out_dir / "_mask_verify_input.png"
    cv2.imwrite(str(viz_path), overlay)

    prompt = (
        f"You are reviewing a segmentation mask.\n"
        f"The GREEN overlay shows what the pipeline selected as the visible "
        f"part of the '{target_class}' in the image. Look carefully:\n"
        f"\n"
        f"  • Does the green region clearly cover the visible portion of the "
        f"correct '{target_class}'?\n"
        f"  • Or does it cover the wrong subject (a different instance, an "
        f"occluder, just a small body part, or empty background)?\n"
        f"\n"
        f"If the mask is correct, return ok=true and set corrective_click "
        f"coords to 0,0 (they will be ignored).\n"
        f"If the mask is wrong, return ok=false and provide a corrective "
        f"click at a clearly-visible pixel of the correct '{target_class}' "
        f"body (use the centroid of the largest visible region of the right "
        f"target). Pixel coordinates: x=column (0={w-1} right), "
        f"y=row (0=top, {h-1}=bottom)."
    )

    return gpt_vision([str(viz_path)], prompt,
                      schema=_MASK_REVIEW_SCHEMA,
                      cache_key="mask_review_v1")




def _maybe_review_and_correct_visible(
    image_bgr: np.ndarray,
    visible_mask: np.ndarray,
    occluder_mask: np.ndarray,
    target_class: str,
    out_dir: Path,
    *,
    max_retries: int = 1,
    min_area_frac: float = 0.02,
) -> np.ndarray:
    """Optionally re-segment the visible mask via GPT-V review + corrective
    click. Returns the (possibly updated) visible mask. Always called from
    occlusion_agent when ``USE_MASK_REVIEW_FIRST`` is True.

    Flow per attempt:
      1. CLIP gate (if USE_CLIP_VERIFY): if score ≥ threshold, ACCEPT.
         Skips the GPT-V call entirely on confident masks.
      2. Otherwise call _gpt_verify_visible_mask to get ok / corrective click.
      3. If GPT says wrong, re-segment SAM3 with the click and loop.

    Loop terminates when CLIP+GPT both ok or after ``max_retries``.
    """
    h, w = image_bgr.shape[:2]
    img_area = h * w
    cur_mask = visible_mask.copy()
    occ_b = (occluder_mask > 0).astype(np.uint8)

    for attempt in range(max_retries + 1):
        cur_area = int((cur_mask > 0).sum())
        force_review = cur_area < int(min_area_frac * img_area)

        # ── CLIP gate: cheap accept on confident masks ─────────────────
        clip_score, clip_passed = _clip_verify_mask(image_bgr, cur_mask, target_class)
        if clip_passed and not force_review:
            # CLIP is confident — skip the GPT-V call entirely.
            print(f"  [MaskReview] CLIP gate accepted mask (score={clip_score:.3f}) — skipping GPT-V")
            return cur_mask
        if not clip_passed:
            print(f"  [MaskReview] CLIP gate FAILED (score={clip_score:.3f}) — calling GPT-V for correction")

        try:
            review = _gpt_verify_visible_mask(image_bgr, cur_mask, target_class, out_dir)
        except Exception as exc:                              # noqa: BLE001
            print(f"  [MaskReview] GPT-V verify call failed: {exc!r} — keeping current mask")
            return cur_mask

        ok = bool(review.get("ok", False)) and not force_review
        rationale = review.get("rationale", "")
        cx        = int(review.get("corrective_click_x", 0))
        cy        = int(review.get("corrective_click_y", 0))

        if ok:
            print(f"  [MaskReview] visible mask accepted — {rationale[:120]}")
            return cur_mask

        if force_review and bool(review.get("ok", False)):
            print(f"  [MaskReview] force-review (area {cur_area} px < "
                  f"{min_area_frac:.0%} of image) — GPT said ok but re-checking")

        if attempt >= max_retries:
            print(f"  [MaskReview] exhausted retries — keeping current mask")
            return cur_mask

        if not (0 <= cx < w and 0 <= cy < h):
            print(f"  [MaskReview] GPT returned out-of-bounds click ({cx},{cy}) — keeping current")
            return cur_mask

        # If GPT's click landed inside the occluder, the suggested correction
        # is bogus; refuse to apply.
        if occ_b[cy, cx] == 1:
            print(f"  [MaskReview] corrective click ({cx},{cy}) landed inside occluder — keeping current")
            return cur_mask

        print(f"  [MaskReview] mask rejected: {rationale[:120]}")
        print(f"  [MaskReview] re-segmenting with corrective click ({cx},{cy})")

        # Save image to disk for the SAM3 helper (it reads from path).
        src_path = out_dir / "_mask_verify_source.png"
        cv2.imwrite(str(src_path), image_bgr)
        sam_res = _sam_segment_targeted(
            str(src_path),
            [{"label": "mask_review_corrective", "x": cx, "y": cy}],
            out_dir,
        )
        # Pick the best mask: largest area that contains the click, not
        # >85% of image, and not overlapping occluder more than 10%.
        best_mask = None
        best_score = 0.0
        for r in sam_res:
            m = r.get("mask")
            if m is None:
                continue
            if m.shape[:2] != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            mb = (m > 0).astype(np.uint8)
            if not mb[cy, cx]:
                continue
            area = int(mb.sum())
            if area > 0.85 * img_area:
                continue
            occ_overlap = int((mb & occ_b).sum()) / max(area, 1)
            if occ_overlap > 0.10:
                continue
            score = area * (1.0 - occ_overlap)
            if score > best_score:
                best_score = score
                best_mask = mb

        if best_mask is None:
            print(f"  [MaskReview] no SAM3 candidate from corrective click passed filters — keeping current")
            return cur_mask

        new_area = int(best_mask.sum())
        print(f"  [MaskReview] corrective SAM3 mask: {new_area} px "
              f"(was {cur_area} px) — applying")
        cur_mask = best_mask
        # Save the new mask alongside the originals for traceability.
        cv2.imwrite(str(out_dir / "visible_mask_after_review.png"),
                    cur_mask * 255)

    return cur_mask


# ── SAM3 point-prompt helper (Fix 1) ─────────────────────────────────────────



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

    target_instruction = (
        f'The user has indicated the occluded object is "{hint}". Use this as your guide.'
        if hint else
        "No target specification. Identify the most prominent occlusion — "
        "the object most clearly partially hidden behind another object."
    )

    # ── PRIMARY PATH: SAM3 point-prompt from text-derived click ──────────
    # First, call GPT with just the raw image to get occluder_click coordinates.
    # NO auto-segment yet — that's only for fallback.
    prompt = f"""You are a world-class expert in computer vision, amodal completion, and animal/human anatomy.

Image size: {w}×{h} px (width × height).
{target_instruction}

Image: the original photo.

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
  • `selected_segment_ids`: [] on the first raw-image pass. If a second image
    with numbered SAM3 auto-segments is provided, fill this with the segment IDs
    whose union forms the OCCLUDER mask. Be conservative — only segments genuinely
    in front of the occluded object.
  • `visible_segment_ids`: [] on the first raw-image pass. If a second image
    with numbered SAM3 auto-segments is provided, fill this with the segment IDs
    of the VISIBLE (modal) portion of the subject.
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
  • `visible_segment_ids`: [] on the first raw-image pass. If a second image
    with numbered SAM3 auto-segments is provided, include SAM3 IDs of the ENTIRE
    VISIBLE subject body — head, body, wings, visible legs, etc.
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
        [state["image_path"]],
        prompt, schema=OCCLUSION_SCHEMA,
        cache_key="occlusion_analysis_text_first_v1",
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

    target_class   = (data.get("occluded_object", "") or "").strip()
    occluder_class = (data.get("occluder", "") or "").strip()
    target_short   = _extract_short_label(target_class)
    occluder_short = _extract_short_label(occluder_class)

    (out_dir / "occlusion_text_first.json").write_text(json.dumps(data, indent=2))

    # ── Text-first SAM3 flow ────────────────────────────────────────────────
    # Do not auto-segment up front. First ask SAM3 for the requested text
    # prompts, then let CLIP verify those masks. Only if that fails do we run
    # SAM3 automatic segmentation and ask GPT to pick numbered candidates.
    segments: list = []
    sam3_viz_path: Optional[Path] = None
    text_occluder_mask = np.zeros((h, w), dtype=np.uint8)
    text_visible_mask  = np.zeros((h, w), dtype=np.uint8)
    sam3_text_occluder_good = False
    sam3_text_visible_good  = False

    text_target_prompt = target_class or target_short or hint
    text_occluder_prompt = occluder_class or occluder_short

    if frame_cropped:
        print("  [Text-first] frame-crop mode: segment visible subject by text; no occluder")
    if text_target_prompt:
        target_text_segments = _sam_segment_text_prompt(
            state["image_path"], text_target_prompt, out_dir, "target")
        text_visible_mask = _mask_from_text_segments(
            target_text_segments, (h, w), sub_click, "target")
        if text_visible_mask.sum() > 0:
            tv_score, tv_passed = _clip_verified_strict(img, text_visible_mask, text_target_prompt)
            sam3_text_visible_good = bool(tv_passed)
            print(f"  [Text-first] target SAM3-text + CLIP "
                  f"{'PASS' if tv_passed else 'FAILED'} "
                  f"(score={tv_score:.3f}) → {int(text_visible_mask.sum())} px")

    if not frame_cropped and text_occluder_prompt:
        occluder_text_segments = _sam_segment_text_prompt(
            state["image_path"], text_occluder_prompt, out_dir, "occluder")
        text_occluder_mask = _mask_from_text_segments(
            occluder_text_segments, (h, w), occ_click, "occluder")
        if text_occluder_mask.sum() > 0:
            to_score, to_passed = _clip_verified_strict(img, text_occluder_mask, text_occluder_prompt)
            sam3_text_occluder_good = bool(to_passed)
            print(f"  [Text-first] occluder SAM3-text + CLIP "
                  f"{'PASS' if to_passed else 'FAILED'} "
                  f"(score={to_score:.3f}) → {int(text_occluder_mask.sum())} px")

    need_auto_fallback = (
        (not frame_cropped and not sam3_text_occluder_good)
        or not sam3_text_visible_good
    )
    if need_auto_fallback:
        print("  [Text-first] CLIP rejected text-prompt mask(s) — running SAM3 auto-seg fallback")
        segments, sam3_viz_path = _sam_segment_all(state["image_path"], out_dir)
        fallback_prompt = prompt + f"""

══════════════════════════════════════════════════════════════════════
FALLBACK PASS — NUMBERED SAM3 CANDIDATES
══════════════════════════════════════════════════════════════════════
At least one SAM3 text-prompt mask was rejected by CLIP. Image 1 is the original
photo. Image 2 is a numbered SAM3 automatic-segmentation overlay.

Accepted text-prompt masks:
  • visible target accepted: {sam3_text_visible_good}
  • occluder accepted: {sam3_text_occluder_good}

Now fill `selected_segment_ids` and `visible_segment_ids` using the numbered
overlay for any rejected mask. Keep all other fields consistent with the
original image.
"""
        data = gpt_vision(
            [state["image_path"], str(sam3_viz_path)],
            fallback_prompt, schema=OCCLUSION_SCHEMA,
            cache_key="occlusion_analysis_auto_fallback_v1",
        )
        sel_ids      = data.get("selected_segment_ids", [])
        vis_ids      = data.get("visible_segment_ids", [])
        poly_ovr     = data.get("polygon_override", [])
        region       = data.get("hidden_region", {})
        expansion    = int(data.get("boundary_expansion", config.MASK_EXPAND))
        frame_cropped = bool(data.get("frame_cropped", False))
        exp_dirs     = data.get("expansion_directions", [])
        exp_px       = data.get("expansion_pixels", {"top": 0, "bottom": 0, "left": 0, "right": 0})
        bbox         = [region.get("x1", 0), region.get("y1", 0),
                        region.get("x2", w), region.get("y2", h)]
        hidden_poly  = data.get("hidden_polygon", [])
        occ_click    = data.get("occluder_click", {})
        sub_click    = data.get("subject_click",  {})
        target_class   = (data.get("occluded_object", "") or "").strip()
        occluder_class = (data.get("occluder", "") or "").strip()
        target_short   = _extract_short_label(target_class)
        occluder_short = _extract_short_label(occluder_class)
        print(f"  [Fallback/GPT] occluder seg IDs={sel_ids} visible seg IDs={vis_ids}")
        (out_dir / "occlusion.json").write_text(json.dumps(data, indent=2))
    else:
        print("  [Text-first] CLIP accepted SAM3 text-prompt masks — skipping auto-seg fallback")
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

        # ---- CLIP-on-grid → SAM3 point-prompt -----------------------------
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

    # ── PRIMARY PATH: SAM3 point-prompt from text-derived click ──────────────
    # Use GPT's text-derived occluder_click to run SAM3 point-prompt directly.
    # Then CLIP-verify the result. If CLIP passes, use it. If fails, fall
    # through to the auto-segment + GPT numbered-candidate selection.
    mask_candidates: list = []   # [(name: str, mask: np.uint8 0/1)]
    poly_used = None
    sam3_click_occluder_good = False

    if sam3_text_occluder_good and text_occluder_mask.sum() > 0:
        mask_candidates.append(("sam3_text", text_occluder_mask.astype(np.uint8).copy()))
        print(f"  [Occluder/SAM3-text] using accepted text-prompt mask "
              f"({int(text_occluder_mask.sum())} px)")

    ox = int(occ_click.get("x", 0))
    oy = int(occ_click.get("y", 0))
    occ_cls = occluder_class or occluder_short or (data.get("occluder", "") or "").strip()

    if sam3_text_occluder_good:
        print("  [Occluder/SAM3-click] skipped — SAM3 text-prompt mask already passed CLIP")
    elif 0 < ox < w and 0 < oy < h and occ_cls:
        print(f"  [Occluder/SAM3-click] SAM3 point-prompt at occluder_click=({ox},{oy}) "
              f"for '{occ_cls}'…")
        targeted = _sam_segment_targeted(
            state["image_path"],
            [{"label": "occluder_point_prompt", "x": ox, "y": oy}],
            out_dir,
        )
        if targeted and int(targeted[0]["mask"].sum()) > 200:
            sam_occ = targeted[0]["mask"].astype(np.uint8)
            if sam_occ.shape[:2] != (h, w):
                sam_occ = cv2.resize(sam_occ, (w, h),
                                     interpolation=cv2.INTER_NEAREST)
            # CLIP-verify the SAM3 click mask against the occluder class.
            sc_score, sc_passed = _clip_verify_mask(img, sam_occ, occ_cls)
            if sc_passed:
                occluder_mask = sam_occ.copy()
                sam3_click_occluder_good = True
                mask_candidates.append(("sam3_click", occluder_mask))
                print(f"  [Occluder/SAM3-click] SAM3 point-prompt + CLIP PASS "
                      f"(score={sc_score:.3f}) → {int(occluder_mask.sum())} px")
            else:
                print(f"  [Occluder/SAM3-click] SAM3 click mask CLIP FAILED "
                      f"(score={sc_score:.3f}) — falling through to numbered "
                      f"auto-segment candidates")
        else:
            print(f"  [Occluder/SAM3-click] SAM3 click returned empty — "
                  f"falling through to numbered auto-segment candidates")
    else:
        print(f"  [Occluder/SAM3-click] no valid occluder_click or class — "
              f"falling through to numbered auto-segment candidates")

    if not sam3_click_occluder_good:
        # ── FALLBACK PATH: auto-segment + GPT numbered candidates ─────────
        # SAM3 point-prompt from text-derived click failed or was rejected by
        # CLIP. Fall back to auto-segmented SAM3 candidates with GPT's
        # numbered segment selection and InstaOrder ordering.
        # GPT already selected `selected_segment_ids` from the SAM3 auto-seg
        # viz during the GPT call.  Treat these as numbered candidates.
        # After fusion, CLIP-verify the result.

        # A) Vision-grounded path (CLIP-on-SAM3 + GroundingDINO + CLIP-grid all
        #    funnel into clip_occluder; we treat them as a single vote here).
        if clip_grounded and clip_occluder.sum() > 0:
            mask_candidates.append(("clip_segments", clip_occluder.astype(np.uint8).copy()))

    # C) GPT polygon — kept as fallback reference.
    if len(poly_ovr) >= 3:
        pts = np.array([[int(p[0]), int(p[1])] for p in poly_ovr], dtype=np.int32)
        poly_used = pts

    # D) GPT segment IDs as numbered candidates (auto-segment fallback).
    # GPT selects these from the SAM3 auto-seg viz (the numbered overlay).
    if sel_ids:
        sel_mask = np.zeros((h, w), dtype=np.uint8)
        for sid in sel_ids:
            if sid in seg_by_id:
                sel_mask = np.clip(sel_mask | seg_by_id[sid]["mask"], 0, 1).astype(np.uint8)
        if sel_mask.sum() > 0:
            mask_candidates.append(("gpt_segments", sel_mask))

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

    if sam3_text_visible_good and text_visible_mask.sum() > 0:
        visible_mask = np.clip(visible_mask | text_visible_mask, 0, 1).astype(np.uint8)
        print(f"  SAM3-text visible seed: {int(text_visible_mask.sum())} px")

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
    _allow_gpt_click = bool(getattr(config, "USE_GPT_CLICK", True))
    if not _allow_gpt_click:
        print(f"  [Fix1] skipped — USE_GPT_CLICK=False (no SAM3 point-prompt rescue)")
    if not frame_cropped and not clip_grounded and _allow_gpt_click and (occ_x or occ_y) and (sub_x or sub_y):
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

    # ── Mask-first review (USE_MASK_REVIEW_FIRST) ────────────────────────
    # GPT-V verifies the visible mask captures the right target. If wrong
    # (e.g. picked the back-bear in a same-class scene), GPT issues a
    # corrective click and we re-segment.
    if getattr(config, "USE_MASK_REVIEW_FIRST", False):
        try:
            img_for_review = cv2.imread(state["image_path"])
            target_text = (state.get("target") or state.get("occluded_object")
                           or "subject").strip()
            # If GPT clicks are disabled, run review for the verdict but
            # force max_retries=0 so no corrective click fires.
            _max_retries = (
                int(getattr(config, "MASK_REVIEW_MAX_RETRIES", 1))
                if bool(getattr(config, "USE_GPT_CLICK", True))
                else 0
            )
            updated = _maybe_review_and_correct_visible(
                img_for_review,
                visible_mask,
                occluder_mask,
                target_text,
                out_dir,
                max_retries=_max_retries,
                min_area_frac=float(getattr(config, "MASK_REVIEW_MIN_AREA_FRAC", 0.02)),
            )
            if updated is not None and not np.array_equal(updated, visible_mask):
                visible_mask = updated.astype(np.uint8)
                cv2.imwrite(str(visible_mask_save), visible_mask * 255)
                print(f"  Visible mask     : {int(visible_mask.sum())} px → {visible_mask_save} (after review)")
        except Exception as exc:                              # noqa: BLE001
            print(f"  [MaskReview] review step failed: {exc!r} — keeping pre-review mask")

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



# ── Completion-agent image helpers (Jiang Ao CVPR'25 off-frame blending) ──────

def _check_touch_boundary(mask: np.ndarray, gap: int = BOUNDARY_GAP_PX) -> set:
    """Sides {top,bottom,left,right} where the mask reaches within `gap` of
    the canvas edge. Mirrors amodal/main.py:check_touch_boundary (Jiang Ao).
    """
    H, W = mask.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return set()
    sides = set()
    if int(ys.min()) <= gap:               sides.add("top")
    if int(ys.max()) >= H - 1 - gap:       sides.add("bottom")
    if int(xs.min()) <= gap:               sides.add("left")
    if int(xs.max()) >= W - 1 - gap:       sides.add("right")
    return sides


# ── Edge feathering + alpha blending (ported from amodal/main.py:1099,1159) ──

def _shrink_edges_to_transparent(rgba: np.ndarray, shrink_amount: int = 10) -> np.ndarray:
    """Shrink the visible alpha region by ``shrink_amount`` px.

    Equivalent to dilating the *transparent* region into the silhouette.
    Used before alpha-blending so the blend transition band lives entirely
    inside the original silhouette.
    """
    if rgba.shape[2] != 4:
        raise ValueError("RGBA expected")
    alpha = rgba[..., 3]
    transparent = (alpha == 0).astype(np.uint8)
    k = shrink_amount * 2 + 1
    dilated = cv2.dilate(transparent, np.ones((k, k), np.uint8), iterations=1)
    out = rgba.copy()
    out[dilated > 0, 3] = 0
    return out


def _alpha_blending(src_rgba: np.ndarray, dst_rgba: np.ndarray,
                    transi_wid: int = 5) -> np.ndarray:
    """Feathered RGBA blend: keep src pixels exact inside its silhouette,
    fade smoothly to dst over a ``transi_wid``-px band at the boundary.
    Ported from amodal/main.py:alpha_blending.
    """
    if src_rgba.shape[2] != 4 or dst_rgba.shape[2] != 4:
        raise ValueError("Both must be RGBA")
    if src_rgba.shape[:2] != dst_rgba.shape[:2]:
        raise ValueError("Shape mismatch")

    src_mask = (src_rgba[..., 3] > 0).astype(np.uint8)
    kernel = np.ones((transi_wid, transi_wid), np.uint8)
    src_interior = cv2.erode(src_mask, kernel, iterations=1)
    transition_region = src_mask - src_interior

    dist = cv2.distanceTransform((1 - src_mask).astype(np.uint8),
                                 cv2.DIST_L2, 5)
    dist = np.clip(dist / max(transi_wid, 1), 0, 1)
    w_src = np.where(transition_region > 0, dist, 1.0)
    w_dst = 1.0 - w_src

    out = dst_rgba.copy()
    out[src_mask > 0] = src_rgba[src_mask > 0]

    trans_idx = transition_region > 0
    dst_alpha_t = dst_rgba[trans_idx, 3]
    blend = dst_alpha_t > 0
    out[trans_idx, :3][blend] = (
        w_dst[trans_idx, np.newaxis][blend] * dst_rgba[trans_idx, :3][blend] +
        w_src[trans_idx, np.newaxis][blend] * src_rgba[trans_idx, :3][blend]
    )
    out[trans_idx, :3][~blend] = src_rgba[trans_idx, :3][~blend]
    out[..., 3] = np.maximum(src_rgba[..., 3], dst_rgba[..., 3])
    return out


def _make_comparison(panels: list, labels: list, out_path: Path) -> None:
    h = max(p.shape[0] for p in panels)
    resized = []
    for p in panels:
        if p.ndim == 2:
            p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        if p.shape[0] != h:
            scale = h / p.shape[0]
            p = cv2.resize(p, (int(p.shape[1] * scale), h),
                           interpolation=cv2.INTER_LINEAR)
        resized.append(p)
    grid = np.hstack(resized)
    x = 0
    for p, lbl in zip(resized, labels):
        cv2.rectangle(grid, (x, 0), (x + p.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(grid, lbl, (x + 6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        x += p.shape[1]
    cv2.imwrite(str(out_path), grid)


# ── Agent 2 — Completion (cutout → Flux-Fill → SAM3 re-seg → off-frame) ────────

def completion_agent(state: State) -> dict:
    """Build the subject cutout, Flux-Fill the hidden slice, re-segment with
    SAM3, then iteratively extend off-frame (Jiang Ao CVPR'25). Writes RGB /
    RGBA / white-bg outputs + comparison + metrics; sets state['output_path']
    and state['output_rgba_path'] for the reviewer."""
    img_path = Path(state["image_path"])
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return {**state, "output_path": "", "output_rgba_path": ""}

    stem = img_path.stem
    out_dir = BASE_DIR / "output" / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dir = out_dir / "_flux_cutout_person"
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"Out dir: {test_dir}")

    # Masks come from occlusion_agent (Agent 1), which runs before this node.
    # If they're missing (standalone invocation), run Agent 1 now.
    vis_path    = out_dir / "visible_mask.png"
    amodal_path = out_dir / "subject_full_amodal_mask.png"
    if not (vis_path.exists() and amodal_path.exists()):
        print("Masks missing — running Agent 1 (occlusion_agent)…")
        config.IMAGE_PATH = str(img_path)
        t0 = time.time()
        occlusion_agent(state)
        print(f"Agent 1 done in {time.time() - t0:.1f}s")
    else:
        print(f"Reusing cached masks from {out_dir}/")
    # ── Step 2: load image + masks ───────────────────────────────────────
    img_bgr = cv2.imread(str(img_path))
    h, w = img_bgr.shape[:2]
    vis_mask = cv2.imread(str(vis_path), cv2.IMREAD_GRAYSCALE)
    am_mask  = cv2.imread(str(amodal_path), cv2.IMREAD_GRAYSCALE)
    if vis_mask.shape != (h, w):
        vis_mask = cv2.resize(vis_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if am_mask.shape != (h, w):
        am_mask  = cv2.resize(am_mask,  (w, h), interpolation=cv2.INTER_NEAREST)
    vis_b = (vis_mask > 127).astype(np.uint8)
    am_b  = (am_mask  > 127).astype(np.uint8)
    print(f"  visible mask: {int(vis_b.sum())} px")
    print(f"  amodal  mask: {int(am_b.sum())} px")

    # ── Step 3: Flux inpaints the dilated-occluder region (Jiang Ao-style) ─
    # No predicted amodal silhouette. The inpaint region is exactly what
    # Jiang Ao uses for in-frame iter 0: dilate(occluder_mask) − visible.
    # Their kernel is 5×5 with 3 iters (~15 px). Flux fills "complete <subject>"
    # into the dilated occluder; SAM3 post-seg of Flux's output then
    # recovers the actual amodal silhouette (including thin legs that no
    # 60-vertex polygon could trace).
    occluder_path = out_dir / "occluder_mask.png"
    if occluder_path.exists():
        occ_mask = cv2.imread(str(occluder_path), cv2.IMREAD_GRAYSCALE)
        if occ_mask.shape != (h, w):
            occ_mask = cv2.resize(occ_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        occ_b = (occ_mask > 127).astype(np.uint8)
    else:
        occ_b = np.zeros((h, w), dtype=np.uint8)
    # Match amodal/main.py:725-726 for iter 0
    JA_KERNEL = np.ones((5, 5), np.uint8)
    JA_ITERS  = 3
    occ_dilated = cv2.dilate(occ_b, JA_KERNEL, iterations=JA_ITERS).astype(np.uint8)
    hidden = (occ_dilated & (1 - vis_b)).astype(np.uint8)
    am_b   = (vis_b | hidden).astype(np.uint8)   # used by off-frame boundary check below
    print(f"  visible: {int(vis_b.sum())} px   occluder: {int(occ_b.sum())} px   "
          f"occ_dilated: {int(occ_dilated.sum())} px")
    print(f"  hidden = dilate(occluder, 5x5, 3) \\ visible: {int(hidden.sum())} px  "
          f"(Jiang Ao iter-0 mask)")

    # CUTOUT: keep ONLY (eroded) visible person pixels.  Hidden slice +
    # everything else is neutral gray.  The 5×5 erosion (Jiang Ao
    # amodal/main.py:693-697) prevents Flux from preserving any
    # edge-bleed pixels at the visible/occluder boundary — those
    # 3-5 px get repainted by Flux instead.
    vis_b_eroded = cv2.erode(vis_b, np.ones((5, 5), np.uint8),
                             iterations=1).astype(np.uint8)
    cutout = np.full_like(img_bgr, 255)   # white bg matches Jiang Ao amodal/main.py:697
    cutout[vis_b_eroded == 1] = img_bgr[vis_b_eroded == 1]
    cv2.imwrite(str(test_dir / "person_cutout.png"), cutout)
    cv2.imwrite(str(test_dir / "hidden_inpaint_mask.png"), hidden * 255)

    # ── Step 5: run Flux-Fill (skip if hidden region is empty) ───────────
    if int(hidden.sum()) < 50:
        print(f"\nHidden region is empty / tiny ({int(hidden.sum())} px) — "
              f"skipping Flux. Cutout is the final output.")
        flux_bgr = cutout.copy()
        # Save the same outputs as the Flux path for downstream consistency.
        cv2.imwrite(str(test_dir / "flux_completed_restored.png"), flux_bgr)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = flux_bgr
        rgba[..., 3]  = (am_b * 255).astype(np.uint8)
        cv2.imwrite(str(test_dir / "flux_completed_rgba.png"), rgba)

        overlay_in = img_bgr.copy()
        ov = np.zeros_like(overlay_in)
        ov[hidden == 1] = (255, 0, 255)
        ov[vis_b == 1]  = (0, 255, 0)
        overlay_in = cv2.addWeighted(overlay_in, 0.55, ov, 0.45, 0)
        _make_comparison(
            panels=[img_bgr, overlay_in, cutout,
                    cv2.cvtColor(hidden * 255, cv2.COLOR_GRAY2BGR), flux_bgr],
            labels=["A: original", "B: green=visible (no hidden region)",
                    "C: person cutout (final)", "D: hidden mask (empty)",
                    "E: same as cutout (Flux skipped)"],
            out_path=test_dir / "comparison.png",
        )
        print(f"\nOutputs → {test_dir}/")
        return {
            **state,
            "attempt":          state.get("attempt", 0) + 1,
            "output_path":      str(test_dir / "flux_completed_restored.png"),
            "output_rgba_path": str(test_dir / "flux_completed_rgba.png"),
        }

    cutout_rgb = cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB)
    subject_text = (getattr(config, "TARGET", "") or "subject").strip()
    prompt = (
        f"Photorealistic complete {subject_text}, full body continuing "
        f"seamlessly from the visible portion behind the occluder. Matching "
        f"lighting, texture, anatomy and color. Sharp focus, high detail. "
        f"Plain neutral background."
    )
    neg_extra = "distorted anatomy, duplicate parts, blurry, low detail"

    print(f"\nRunning Flux-Fill on the cutout…")
    t0 = time.time()
    flux_results = _run_flux_fill_inpaint(
        base_np=cutout_rgb,
        inpaint_mask=hidden,
        amodal_rgb_256=np.full((256, 256, 3), 255, dtype=np.uint8),
        prompt=prompt,
        n_samples=1,
        out_dir=test_dir,
        prefix="flux_completed",
        neg_extra=neg_extra,
        seed_offset=0,
        strength=1.0,
    )
    dt = time.time() - t0
    print(f"  Flux done in {dt:.1f}s")

    if not flux_results:
        print("Flux returned nothing — aborting")
        return {
            **state,
            "attempt":          state.get("attempt", 0) + 1,
            "output_path":      "",
            "output_rgba_path": "",
        }

    flux_rgb = np.array(flux_results[0].convert("RGB"))
    if flux_rgb.shape[:2] != (h, w):
        flux_rgb = cv2.resize(flux_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
    flux_bgr = cv2.cvtColor(flux_rgb, cv2.COLOR_RGB2BGR)
    # Restore visible-person pixels (Flux's VAE drift can soften them)
    flux_bgr[vis_b == 1] = img_bgr[vis_b == 1]
    # ── Re-segment Flux's output to capture the FULL generated silhouette ─
    # Jiang Ao approach (amodal/main.py:517 filter_out_amodal_segmentation):
    # take ALL SAM3 mask candidates and pick the one with the highest
    # IoU vs the visible mask. No click-inside requirement (which used
    # to fail when GPT's click landed a pixel off the bird).
    final_mask = am_b.copy()
    try:
        ys, xs = np.where(vis_b > 0)
        if len(ys) >= 16:
            cy = int(ys.mean())
            cx = int(xs.mean())
            flux_raw_path = test_dir / "_flux_raw_for_segment.png"
            cv2.imwrite(str(flux_raw_path),
                        cv2.cvtColor(flux_rgb, cv2.COLOR_RGB2BGR))
            sam_res = _sam_segment_targeted(
                str(flux_raw_path),
                [{"label": "final_subject", "x": cx, "y": cy}],
                test_dir,
            )
            best_mask = None
            best_iou  = 0.0
            vis_sum   = int(vis_b.sum())
            candidates = []
            for r in sam_res:
                m = r["mask"]
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST)
                candidates.append((m > 0).astype(np.uint8))
            # First pass: direct masks.
            for m_b in candidates:
                area = int(m_b.sum())
                if area > 0.85 * h * w:
                    continue
                inter = int(((m_b > 0) & (vis_b > 0)).sum())
                union = area + vis_sum - inter
                iou   = inter / max(union, 1)
                if iou > best_iou:
                    best_iou  = iou
                    best_mask = m_b
            # Second pass: complements — SAM3 on gray-background cutouts often
            # inverts foreground/background, returning the gray region instead
            # of the textured subject.  Try flipping each mask.
            if best_iou < 0.20:
                for m_b in candidates:
                    m_comp = (1 - m_b).astype(np.uint8)
                    area = int(m_comp.sum())
                    if area > 0.85 * h * w or area == 0:
                        continue
                    # Complement must still contain the vis_b centroid.
                    if m_comp[cy, cx] == 0:
                        continue
                    inter = int(((m_comp > 0) & (vis_b > 0)).sum())
                    union = area + vis_sum - inter
                    iou   = inter / max(union, 1)
                    if iou > best_iou:
                        best_iou  = iou
                        best_mask = m_comp
                if best_iou >= 0.20:
                    print(f"  [PostSeg] complement flip applied (IoU={best_iou:.2f})")
            if best_mask is not None and best_iou >= 0.20:
                final_mask = (best_mask | vis_b).astype(np.uint8)
                print(f"  [PostSeg] argmax-IoU mask  IoU={best_iou:.2f}  "
                      f"area={int(final_mask.sum())} px "
                      f"(was am_b={int(am_b.sum())} px)")
            else:
                print(f"  [PostSeg] no SAM3 mask passed (best IoU "
                      f"{best_iou:.2f}) — keeping am_b")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [PostSeg] failed: {exc!r}")

    flux_bgr[final_mask == 0]  = PAD_COLOR
    cv2.imwrite(str(test_dir / "flux_completed_restored.png"), flux_bgr)

    # ── Feathered RGBA via Jiang Ao alpha_blending ───────────────────────
    # src = original visible bird pixels on transparent (sharp inside,
    #       transparent outside its silhouette)
    # dst = Flux-painted full silhouette on transparent
    # Blend: keep src pixel-exact inside its mask; smoothly fade to dst
    # over a 5-px transition band at the visible-mask edge.
    src_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    src_rgba[..., :3] = img_bgr
    src_rgba[..., 3]  = (vis_b * 255).astype(np.uint8)
    dst_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    dst_rgba[..., :3] = flux_bgr
    dst_rgba[..., 3]  = (final_mask * 255).astype(np.uint8)
    rgba = _alpha_blending(_shrink_edges_to_transparent(src_rgba, 5),
                           dst_rgba, transi_wid=5)
    cv2.imwrite(str(test_dir / "flux_completed_rgba.png"), rgba)

    # White-background version composited from the feathered RGBA
    white_bg = np.full((h, w, 3), 255, dtype=np.uint8)
    alpha_f = rgba[..., 3:4].astype(np.float32) / 255.0
    white_bg = (rgba[..., :3].astype(np.float32) * alpha_f +
                white_bg.astype(np.float32) * (1.0 - alpha_f)).astype(np.uint8)
    cv2.imwrite(str(test_dir / "flux_completed_white_bg.png"), white_bg)

    # ── Step 5b: iterative off-frame extension (Jiang Ao CVPR'25) ────────
    # If the in-frame silhouette still reaches the canvas edge, pad those
    # sides with gray and rerun Flux into the new strip. Loop up to
    # MAX_OFFFRAME_ITERS, stopping early when no edge is touched or the
    # post-Flux SAM3 re-segmentation fails to extend the silhouette.
    current_bgr   = flux_bgr.copy()
    current_mask  = final_mask.copy()
    sides_touched = _check_touch_boundary(current_mask)
    print(f"\n[OffFrame] sides touched after in-frame Flux: {sides_touched or 'none'}")

    iter_canvases   = []   # (label, bgr) tuples for comparison
    iter_canvases.append(("OF0: in-frame Flux", current_bgr.copy()))

    for it in range(1, MAX_OFFFRAME_ITERS + 1):
        if not sides_touched:
            break
        pt = PAD_PER_ITER if "top"    in sides_touched else 0
        pb = PAD_PER_ITER if "bottom" in sides_touched else 0
        pl = PAD_PER_ITER if "left"   in sides_touched else 0
        pr = PAD_PER_ITER if "right"  in sides_touched else 0
        H0, W0 = current_bgr.shape[:2]
        H_new, W_new = H0 + pt + pb, W0 + pl + pr
        print(f"[OffFrame iter {it}] padding "
              f"top={pt} bottom={pb} left={pl} right={pr} → "
              f"canvas {W0}×{H0} → {W_new}×{H_new}")

        padded_bgr = np.full((H_new, W_new, 3), PAD_COLOR, dtype=np.uint8)
        padded_bgr[pt:pt + H0, pl:pl + W0] = current_bgr

        padded_mask = np.zeros((H_new, W_new), dtype=np.uint8)
        padded_mask[pt:pt + H0, pl:pl + W0] = current_mask

        # Outpaint mask = ONLY the new gray strip. 1 = inpaint, 0 = keep.
        outpaint_mask = np.ones((H_new, W_new), dtype=np.uint8)
        outpaint_mask[pt:pt + H0, pl:pl + W0] = 0

        cv2.imwrite(str(test_dir / f"offframe_iter{it}_input.png"),    padded_bgr)
        cv2.imwrite(str(test_dir / f"offframe_iter{it}_mask.png"),     outpaint_mask * 255)

        # Build a per-side directional prompt suffix (subject-agnostic).
        dir_phrases = []
        if "bottom" in sides_touched: dir_phrases.append("body extending downward")
        if "top"    in sides_touched: dir_phrases.append("body extending upward")
        if "left"   in sides_touched: dir_phrases.append("body extending to the left")
        if "right"  in sides_touched: dir_phrases.append("body extending to the right")
        dir_clause = "; ".join(dir_phrases) or "body continuing into the surrounding area"
        iter_prompt = (
            f"Photorealistic complete {subject_text}, {dir_clause}. "
            f"Natural anatomy and form continuing seamlessly from the "
            f"visible body, with matching lighting, texture and color. "
            f"Sharp focus, high detail. Plain neutral background."
        )

        t1 = time.time()
        iter_results = _run_flux_fill_inpaint(
            base_np=cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB),
            inpaint_mask=outpaint_mask,
            amodal_rgb_256=np.full((256, 256, 3), 255, dtype=np.uint8),
            prompt=iter_prompt,
            n_samples=1,
            out_dir=test_dir,
            prefix=f"offframe_iter{it}_flux",
            neg_extra=neg_extra,
            seed_offset=it * 7,
            strength=1.0,
        )
        print(f"  [OffFrame iter {it}] Flux done in {time.time() - t1:.1f}s")
        if not iter_results:
            print(f"  [OffFrame iter {it}] Flux returned nothing — stopping")
            break

        iter_rgb = np.array(iter_results[0].convert("RGB"))
        if iter_rgb.shape[:2] != (H_new, W_new):
            iter_rgb = cv2.resize(iter_rgb, (W_new, H_new), interpolation=cv2.INTER_LANCZOS4)
        iter_bgr = cv2.cvtColor(iter_rgb, cv2.COLOR_RGB2BGR)
        # Preserve the kept (in-frame) region pixel-exact (Flux VAE drift).
        iter_bgr[pt:pt + H0, pl:pl + W0] = current_bgr

        # SAM3 click on the existing silhouette centroid → full new silhouette
        canvas_path = test_dir / f"offframe_iter{it}_canvas.png"
        cv2.imwrite(str(canvas_path), iter_bgr)
        ys_in, xs_in = np.where(padded_mask > 0)
        if len(ys_in) < 16:
            print(f"  [OffFrame iter {it}] padded_mask too small — stopping")
            break
        cy = int(ys_in.mean()); cx = int(xs_in.mean())
        try:
            sam_res = _sam_segment_targeted(
                str(canvas_path),
                [{"label": f"offframe_iter{it}_subject", "x": cx, "y": cy}],
                test_dir,
            )
        except Exception as exc:                              # noqa: BLE001
            print(f"  [OffFrame iter {it}] SAM3 failed: {exc!r} — stopping")
            break

        best_mask = None
        best_overlap = 0.0
        _of_candidates = []
        for r in sam_res:
            m = r["mask"]
            if m.shape[:2] != (H_new, W_new):
                m = cv2.resize(m.astype(np.uint8), (W_new, H_new),
                               interpolation=cv2.INTER_NEAREST)
            _of_candidates.append((m > 0).astype(np.uint8))
        # First pass: direct masks.
        for mb in _of_candidates:
            if not (0 <= cy < H_new and 0 <= cx < W_new and mb[cy, cx] == 1):
                continue
            if mb.sum() > 0.85 * H_new * W_new:
                continue
            ovl = int(((mb > 0) & (padded_mask > 0)).sum())
            ovl_ratio = ovl / max(int(padded_mask.sum()), 1)
            if ovl_ratio > best_overlap:
                best_overlap = ovl_ratio
                best_mask = mb
        # Second pass: complements — SAM3 on gray-background cutouts often
        # inverts foreground/background.  Try flipping each rejected mask.
        if best_overlap < 0.40:
            for mb in _of_candidates:
                mb_comp = (1 - mb).astype(np.uint8)
                area_comp = int(mb_comp.sum())
                if area_comp > 0.85 * H_new * W_new or area_comp == 0:
                    continue
                if not (0 <= cy < H_new and 0 <= cx < W_new and mb_comp[cy, cx] == 1):
                    continue
                ovl = int(((mb_comp > 0) & (padded_mask > 0)).sum())
                ovl_ratio = ovl / max(int(padded_mask.sum()), 1)
                if ovl_ratio > best_overlap:
                    best_overlap = ovl_ratio
                    best_mask = mb_comp
            if best_mask is not None and best_overlap >= 0.40:
                print(f"  [OffFrame iter {it}] SAM3 complement flip applied "
                      f"(overlap={best_overlap:.2f})")
        if best_mask is None or best_overlap < 0.40:
            print(f"  [OffFrame iter {it}] no SAM3 mask passed "
                  f"(best overlap {best_overlap:.2f}) — stopping")
            break

        new_mask = (best_mask | padded_mask).astype(np.uint8)
        delta = int(new_mask.sum()) - int(padded_mask.sum())
        # Gray out everything outside the new silhouette so VAE drift can't
        # leak into the background.
        iter_bgr[new_mask == 0] = PAD_COLOR
        cv2.imwrite(str(test_dir / f"offframe_iter{it}_restored.png"), iter_bgr)
        cv2.imwrite(str(test_dir / f"offframe_iter{it}_silhouette.png"),
                    new_mask * 255)

        current_bgr  = iter_bgr
        current_mask = new_mask
        sides_touched = _check_touch_boundary(current_mask)
        print(f"  [OffFrame iter {it}] silhouette {int(new_mask.sum())} px "
              f"(+{delta} px); sides still touched: {sides_touched or 'none'}")
        iter_canvases.append((f"OF{it}: +{','.join(sorted({s for s in 'tblr' if (s == 't' and pt) or (s == 'b' and pb) or (s == 'l' and pl) or (s == 'r' and pr)}))}",
                              current_bgr.copy()))

    # Final padded outputs (= in-frame canvas if loop didn't fire).
    final_canvas_bgr  = current_bgr.copy()
    final_canvas_mask = current_mask.copy()
    final_canvas_bgr[final_canvas_mask == 0] = PAD_COLOR
    cv2.imwrite(str(test_dir / "offframe_final_canvas.png"), final_canvas_bgr)
    cv2.imwrite(str(test_dir / "offframe_final_mask.png"),
                final_canvas_mask * 255)

    rgba_final = np.zeros((*final_canvas_mask.shape, 4), dtype=np.uint8)
    rgba_final[..., :3] = current_bgr
    rgba_final[..., 3]  = (final_canvas_mask * 255).astype(np.uint8)
    cv2.imwrite(str(test_dir / "offframe_final_rgba.png"), rgba_final)

    # White-bg version on the padded canvas
    white_final = np.full_like(final_canvas_bgr, 255)
    white_final[final_canvas_mask == 1] = current_bgr[final_canvas_mask == 1]
    cv2.imwrite(str(test_dir / "offframe_final_white_bg.png"), white_final)

    # Per-iter side-by-side
    if len(iter_canvases) >= 2:
        _make_comparison(
            panels=[bgr for _, bgr in iter_canvases],
            labels=[lbl for lbl, _ in iter_canvases],
            out_path=test_dir / "offframe_iters_comparison.png",
        )
        print(f"[OffFrame] saved offframe_iters_comparison.png "
              f"({len(iter_canvases)} stages)")

    # ── Step 6: side-by-side comparison ──────────────────────────────────
    overlay_in = img_bgr.copy()
    ov = np.zeros_like(overlay_in)
    ov[hidden == 1] = (255, 0, 255)
    ov[vis_b == 1]  = (0, 255, 0)
    overlay_in = cv2.addWeighted(overlay_in, 0.55, ov, 0.45, 0)

    _make_comparison(
        panels=[
            img_bgr,
            overlay_in,
            cutout,
            cv2.cvtColor(hidden * 255, cv2.COLOR_GRAY2BGR),
            flux_bgr,
        ],
        labels=[
            "A: original",
            "B: green=visible, magenta=hidden",
            "C: person cutout (Flux input)",
            "D: hidden inpaint mask",
            "E: Flux-completed person",
        ],
        out_path=test_dir / "comparison.png",
    )

    # ── Per-run metrics (IoU, Boundary F1, LPIPS) ───────────────────────
    try:
        from metrics import compute_run_metrics                # noqa: WPS433
        compute_run_metrics(
            out_dir          = test_dir,
            original_image   = img_bgr,
            visible_mask     = vis_b,
            predicted_amodal = am_b,
            flux_output      = flux_bgr,
            final_silhouette = final_mask,
        )
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [metrics] skipped: {exc!r}")

    print(f"\nOutputs → {test_dir}/")

    # Prefer the off-frame-extended canvas when it was produced; else the
    # in-frame white-bg completion. RGBA mirrors the same choice.
    offframe_white = test_dir / "offframe_final_white_bg.png"
    offframe_rgba  = test_dir / "offframe_final_rgba.png"
    if offframe_white.exists():
        output_path = offframe_white
        rgba_path   = offframe_rgba if offframe_rgba.exists() else (test_dir / "flux_completed_rgba.png")
    else:
        output_path = test_dir / "flux_completed_white_bg.png"
        rgba_path   = test_dir / "flux_completed_rgba.png"
    return {
        **state,
        "attempt":          state.get("attempt", 0) + 1,
        "output_path":      str(output_path),
        "output_rgba_path": str(rgba_path),
    }
