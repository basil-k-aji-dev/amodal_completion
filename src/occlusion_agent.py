"""Agent 1 — Occlusion Agent.

Runs SAM3 automatic mask generation to segment every object in the image,
and sends the segments + image to GPT for occluder/subject identification.
Occluder-mask candidates are fused (see masks._fuse_masks /
config.MASK_FUSION_*) from InstaFormer's holistic occlusion+depth-order
prediction (instaformer_helper.py, run as a subprocess) and SAM3's own
text-prompted (Promptable Concept Segmentation) fallback.
Produces: occluder mask, visible-object modal mask, and (via
`_gpt_amodal_subject_mask`) the full amodal subject silhouette.
Can be re-invoked on mask failures (MASK_INACCURATE / OCCLUDER_REMNANT).
"""
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config
from runtime import BASE_DIR, gpt_vision
from schemas import State, OCCLUSION_SCHEMA, _GPT_AMODAL_SCHEMA
from masks import _fuse_masks, _extract_short_label
from offframe import _flux_extend_amodal_mask_offframe
from models.sam3 import (
    _sam3_text_segment,
    _sam_segment_all,
    _sam_segment_targeted,
)

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

    # Short class nouns (still used below by the SAM3-text fallback prompts,
    # even though the CLIP/GroundingDINO grounding paths that used to consume
    # them have been removed — both were permanently disabled in favour of
    # InstaFormer + SAM3-text).
    target_class   = (data.get("occluded_object", "") or "").strip()
    occluder_class = (data.get("occluder", "") or "").strip()
    target_short   = _extract_short_label(target_class)
    occluder_short = _extract_short_label(occluder_class)

    # ── Occluder mask candidates ──────────────────────────────────────────────
    # Each source produces a binary candidate; we fuse them according to
    # config.MASK_FUSION_MODE ("majority" / "union" / "intersection" / "priority").
    # This replaces the old priority-only chain — every source is now a vote.
    mask_candidates: list = []   # [(name: str, mask: np.uint8 0/1)]

    # A) CLIP-on-SAM3 / GroundingDINO / CLIP-grid — REMOVED.  All three were
    # permanently disabled (USE_CLIP_GROUNDING / USE_CLIP_GRID_FALLBACK /
    # USE_GROUNDING_DINO = False) in favour of InstaFormer + SAM3-text.

    # B) PSALM referring-expression segmentation — REMOVED (USE_PSALM=False;
    # was over-segmenting nature scenes). Falls back to InstaFormer + SAM3-text.

    # C) GPT polygon — REMOVED.  GPT polygons + GPT segment IDs are not
    # independent of the GPT-supplied occluder description that drives
    # InstaFormer's target match; including them as separate "votes" creates
    # correlated noise (they all swing the same way when GPT's narrative is
    # wrong).

    # D) GPT segments — REMOVED for the same reason as C.

    # E+F) InstaFormer holistic occlusion+depth-order predictor.
    # Single forward pass over the whole scene: its own panoptic
    # segmentation + an occlusion matrix over every instance. Replaces the
    # old pairwise InstaOrder calls entirely — same-class occlusion
    # (zebra/zebra, cat/cat) falls out naturally since InstaFormer assigns
    # each instance its own segment id and ranks them directly, no
    # PSALM-class-split / connected-components / SAM3-dual-click needed.
    instaformer_result = None
    instaformer_visible_seed: Optional[np.ndarray] = None
    if getattr(config, "USE_INSTAFORMER", False):
        try:
            from instaformer_helper import (
                run_instaformer, occluders_above, same_class_split,
                best_matching_segment,
            )
            instaformer_result = run_instaformer(
                image_path  = state["image_path"],
                out_dir     = out_dir,
                venv_python = config.INSTAFORMER_VENV_PYTHON,
                repo_dir    = config.INSTAFORMER_REPO_DIR,
                config_file = config.INSTAFORMER_CONFIG,
                ckpt_path   = config.INSTAFORMER_CKPT,
            )
        except Exception as exc:                              # noqa: BLE001
            print(f"  [InstaFormer] skipped: {exc!r}")
            instaformer_result = None

        if instaformer_result is not None:
            sx = int(sub_click.get("x", 0))
            sy = int(sub_click.get("y", 0))
            click_xy = (sx, sy) if (sx or sy) else None
            target_idx = (best_matching_segment(instaformer_result, click_xy=click_xy)
                          if click_xy is not None else None)

            if target_idx is not None:
                io_mask = occluders_above(instaformer_result, click_xy=click_xy)
                if io_mask is not None and io_mask.sum() > 0:
                    mask_candidates.append(("instaformer", io_mask.astype(np.uint8)))
                    print(f"  [InstaFormer] occluder candidate: {int(io_mask.sum())} px")

                pair = same_class_split(instaformer_result, target_idx)
                if pair is not None:
                    front_idx, back_idx = pair
                    if front_idx != target_idx:
                        front_mask = instaformer_result["segments"][front_idx]["mask"]
                        back_mask  = instaformer_result["segments"][back_idx]["mask"]
                        if int(front_mask.sum()) > 0:
                            mask_candidates.append(
                                ("instaformer_pair", front_mask.astype(np.uint8)))
                            instaformer_visible_seed = back_mask.astype(np.uint8)
                            print(f"  [InstaFormer-pair] same-class occlusion resolved: "
                                  f"front={int(front_mask.sum())} px, "
                                  f"back={int(back_mask.sum())} px")
                else:
                    # No same-class sibling (the common case) — InstaFormer's
                    # own panoptic segment for the target IS the visible
                    # subject silhouette, and it's a real segmentation (not a
                    # guess), so use it directly as the visible_mask seed.
                    target_seg_mask = instaformer_result["segments"][target_idx]["mask"]
                    if int(target_seg_mask.sum()) > 0:
                        instaformer_visible_seed = target_seg_mask.astype(np.uint8)
                        print(f"  [InstaFormer] target segment as visible seed: "
                              f"{int(instaformer_visible_seed.sum())} px")
            else:
                print("  [InstaFormer] no subject_click to match a target segment "
                      "— skipping")


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
        # No occluder candidate from any segmentation source — usually
        # because the occluder isn't a COCO category InstaFormer knows
        # (leaves, branch, log, fountain, snowbank, ...). Try SAM3's own
        # open-vocabulary text-prompted segmentation (Promptable Concept
        # Segmentation) before falling back to a crude geometric guess —
        # it can find ANY described object by name, no fixed category list.
        occ_text = (occluder_short or occluder_class).strip()
        sam3_text_hits = _sam3_text_segment(state["image_path"], occ_text) if occ_text else []
        if sam3_text_hits and sam3_text_hits[0]["mask"].sum() > 0:
            raw_mask = sam3_text_hits[0]["mask"].astype(np.uint8)
            if raw_mask.shape[:2] != (h, w):
                raw_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            raw_area = int(raw_mask.sum())

            # Scattered-instance textures (leaves, grass, foliage, ...) match
            # EVERY instance across the whole scene, not just the bit near
            # the subject — unlike a single discrete object (knife, cup,
            # snowbank). Restrict to only the connected component(s) that
            # actually overlap the GPT hidden_region bbox (dilated), so a
            # leaf touching the subject is kept but scattered background
            # leaves elsewhere in the frame are dropped.
            bbox_mask = np.zeros((h, w), dtype=np.uint8)
            bx1, by1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
            bx2, by2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
            bbox_mask[by1:by2, bx1:bx2] = 1
            anchor_k = max(3, int(0.05 * max(h, w)))
            anchor_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (anchor_k * 2 + 1, anchor_k * 2 + 1))
            anchor_dilated = cv2.dilate(bbox_mask, anchor_kernel, iterations=1)

            n_cc, cc_labels = cv2.connectedComponents(raw_mask)
            kept = np.zeros((h, w), dtype=np.uint8)
            for cc_id in range(1, n_cc):
                comp = (cc_labels == cc_id).astype(np.uint8)
                if int((comp & anchor_dilated).sum()) > 0:
                    kept = np.clip(kept | comp, 0, 1).astype(np.uint8)

            if int(kept.sum()) > 0:
                occluder_mask = kept
                print(f"  [SAM3-text] occluder fallback: '{occ_text}' → "
                      f"{raw_area} px raw, {int(occluder_mask.sum())} px "
                      f"after adjacency filter (score={sam3_text_hits[0]['score']:.2f})")
            else:
                # Nothing survived adjacency filtering (e.g. the match was
                # entirely elsewhere in the frame) — use the raw mask
                # rather than silently producing an empty occluder.
                occluder_mask = raw_mask
                print(f"  [SAM3-text] occluder fallback: '{occ_text}' → "
                      f"{raw_area} px (adjacency filter kept nothing — "
                      f"using raw mask, score={sam3_text_hits[0]['score']:.2f})")
        else:
            print("  WARNING: no occluder candidates (incl. SAM3-text) — "
                  "falling back to hidden_region bbox")
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

    # Highest-priority seed: InstaFormer same-class pair-rank "back" side.
    # When same-class occlusion is detected, this is the most reliable signal —
    # it's the actual subject instance silhouette, not a text-derived guess.
    if instaformer_visible_seed is not None and instaformer_visible_seed.sum() > 0:
        visible_mask = instaformer_visible_seed.copy()
        print(f"  InstaFormer-pair visible seed: {int(visible_mask.sum())} px")

    if vis_ids:
        for sid in vis_ids:
            if sid in seg_by_id:
                visible_mask = np.clip(visible_mask | seg_by_id[sid]["mask"], 0, 1)
                print(f"  Merged visible  seg {sid:03d} : area={seg_by_id[sid]['area']} px")
            else:
                print(f"  WARNING: visible segment {sid} not found — skipped")

    _MIN_VIS_AREA = max(200, int(0.005 * h * w))
    if visible_mask.sum() < _MIN_VIS_AREA:
        print(f"  WARNING: visible_mask too small ({int(visible_mask.sum())} px "
              f"< {_MIN_VIS_AREA}) — trying SAM3 text-prompt fallback")
        # Same rationale as the occluder fallback above: the subject may
        # not be a COCO category InstaFormer/vis_ids could match (bread,
        # tamarin, butterfly, ...). SAM3's open-vocabulary text prompt
        # can find it directly by name instead of leaving this empty.
        subj_text = (target_short or target_class).strip()
        sam3_text_hits = _sam3_text_segment(state["image_path"], subj_text) if subj_text else []
        if sam3_text_hits and sam3_text_hits[0]["mask"].sum() > _MIN_VIS_AREA:
            candidate = sam3_text_hits[0]["mask"].astype(np.uint8)
            if candidate.shape[:2] != (h, w):
                candidate = cv2.resize(candidate, (w, h), interpolation=cv2.INTER_NEAREST)
            # Exclude any occluder overlap — the subject's visible region
            # can't include pixels already claimed by the occluder.
            candidate = np.clip(candidate.astype(np.int32) - occluder_mask.astype(np.int32),
                                0, 1).astype(np.uint8)
            if int(candidate.sum()) > _MIN_VIS_AREA:
                visible_mask = candidate
                print(f"  [SAM3-text] visible_mask fallback: '{subj_text}' → "
                      f"{int(visible_mask.sum())} px "
                      f"(score={sam3_text_hits[0]['score']:.2f})")
        if visible_mask.sum() < _MIN_VIS_AREA:
            print(f"  WARNING: visible_mask still too small "
                  f"({int(visible_mask.sum())} px) after SAM3-text fallback")

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
    if not frame_cropped and (occ_x or occ_y) and (sub_x or sub_y):
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

    # ── Amodal subject mask: visible_mask AS-IS  OR  direct GPT-V draw ───────
    # Two paths (pix2gestalt seed + GPT-V review was removed — permanently
    # disabled via USE_PIX2GESTALT_AMODAL=False, superseded by direct Flux
    # inpainting):
    #   • USE_AMODAL_COMPLETION=False → use visible_mask AS-IS (no completion)
    #   • USE_AMODAL_COMPLETION=True  → direct GPT-V silhouette call
    try:
        if not getattr(config, "USE_AMODAL_COMPLETION", True):
            # No amodal extension — use visible_mask as the amodal mask.
            # PSALM's visible segmentation is treated as authoritative; no
            # GPT-V silhouette generation (which often drifts onto the occluder).
            reviewed_mask = (visible_mask > 0).astype(np.uint8)
            print(f"  [Amodal/skip] no completion — using visible_mask "
                  f"as amodal ({int(reviewed_mask.sum())} px)")
        else:
            # Minimal-pipeline path: ask GPT-V to trace the full silhouette
            # from scratch in a single vision call.
            print("  [Amodal/minimal] using "
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

# ── Flux-Fill inpainter (alternative backend) ─────────────────────────────────

