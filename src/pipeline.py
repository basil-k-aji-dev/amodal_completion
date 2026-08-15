"""Pipeline entry point: runs Agent 1 (occlusion_agent) if masks aren't
cached, builds the Jiang-Ao-style hidden-region cutout, then runs Agent 2
(FLUX.1-Fill-dev inpainting) through Agent 3 (a GPT-vision reviewer/retry
loop that scores each completion and retries with a corrected prompt on a
low score), and finally the iterative off-frame extension stage (kept in
the codebase but disabled by default via config.USE_OFFFRAME_EXTENSION).

Usage:
    .venv/bin/python src/pipeline.py [image_path] [input_prompt]

`input_prompt`, if given, overrides config.INPUT_PROMPT for this run only
(lets batch runners hint Agent 1 per-image without editing config.py).
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from runtime import BASE_DIR, gpt_vision
from schemas import REVIEWER_SCHEMA
from occlusion_agent import occlusion_agent
from models.flux_fill import _run_flux_fill_inpaint, _free_flux_fill
from models.sam3 import _sam_segment_targeted


DEFAULT_IMAGE = ("/home/basil-k-aji/Desktop/Workspace/RD/website/"
                 "bird-8296358_640_pigeon.jpg")
PAD_COLOR     = 128   # neutral gray fill outside the amodal silhouette

# Iterative off-frame extension (Jiang Ao CVPR'25, pad_pixels=150).
# After the in-frame Flux pass, if the final silhouette still touches the
# canvas edge, pad those sides with gray and re-run Flux on the new strip.
# Loop until no boundary is touched or MAX_OFFFRAME_ITERS is hit.
MAX_OFFFRAME_ITERS = 3
PAD_PER_ITER       = 150   # px per touched side per iter (Ao default)
BOUNDARY_GAP_PX    = 10    # how close to the edge counts as "touching"


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


def _publish_final(test_dir: Path, rgba_src: Path, subject_slug: str) -> None:
    """Copy the final RGBA cutout + comparison grid into test_dir/final/,
    then optionally kick off Hunyuan3D-2.1 3D generation on it (see
    config.RUN_3D_GENERATION_HUNYUAN3D — off by default)."""
    final_dir = test_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_rgba = final_dir / f"{subject_slug}_final.png"
    if rgba_src.exists():
        shutil.copy2(rgba_src, final_rgba)
    comparison_src = test_dir / "comparison.png"
    if comparison_src.exists():
        shutil.copy2(comparison_src, final_dir / "comparison.png")

    if getattr(config, "RUN_3D_GENERATION_HUNYUAN3D", False) and final_rgba.exists():
        _run_hunyuan3d_generation(final_rgba, final_dir, subject_slug)


def _run_hunyuan3d_generation(image_path: Path, out_dir: Path, subject_slug: str) -> None:
    """Run Hunyuan3D-2.1 shape+paint generation on the finished 2D image.

    Hunyuan3D-2.1 needs its own conda env (different torch/CUDA build than
    this project's .venv), so it's invoked as a subprocess via
    `conda run`, not imported in-process. Failure is non-fatal — the 2D
    result is already published; a 3D miss shouldn't fail the whole run.
    """
    import subprocess

    repo_dir = getattr(config, "HUNYUAN3D_REPO_DIR", None)
    conda_env = getattr(config, "HUNYUAN3D_CONDA_ENV", "hunyuan3d")
    if not repo_dir:
        print("  [3D] RUN_3D_GENERATION_HUNYUAN3D is set but HUNYUAN3D_REPO_DIR "
              "isn't configured — skipping")
        return

    torch_lib = (f"/home/ubuntu/miniconda3/envs/{conda_env}/lib/python3.10/"
                 f"site-packages/torch/lib")
    cudart_lib = (f"/home/ubuntu/miniconda3/envs/{conda_env}/lib/python3.10/"
                  f"site-packages/nvidia/cuda_runtime/lib")
    cmd = (
        f'cd {repo_dir} && '
        f'source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && '
        f'conda activate {conda_env} && '
        f'export LD_LIBRARY_PATH="{torch_lib}:{cudart_lib}:${{LD_LIBRARY_PATH:-}}" && '
        f'python generate_3d.py --image {image_path} --out-dir {out_dir} '
        f'--prefix {subject_slug}_hunyuan3d'
    )
    print(f"  [3D] Running Hunyuan3D-2.1 generation on {image_path.name}…")
    t0 = time.time()
    try:
        result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            print(f"  [3D] Hunyuan3D-2.1 done in {time.time() - t0:.1f}s -> "
                  f"{out_dir}/{subject_slug}_hunyuan3d.glb")
        else:
            print(f"  [3D] Hunyuan3D-2.1 failed (exit {result.returncode}): "
                  f"{result.stderr[-500:] if result.stderr else '(no stderr)'}")
    except subprocess.TimeoutExpired:
        print("  [3D] Hunyuan3D-2.1 generation timed out after 30 min — skipping")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [3D] Hunyuan3D-2.1 generation errored: {exc!r}")


def _apply_occlusion_info(info: dict, subject_text: str, input_prompt: str,
                           source: str) -> tuple:
    """Extract occluder_text (and subject_text, when no explicit input
    prompt was given) from an Agent-1 result — either freshly returned by
    occlusion_agent() or loaded from a cached occlusion.json written by an
    earlier run. Both share the same 'occluder'/'occluded_object' keys, so
    a cache hit gets the exact same occluder-exclusion prompt clause a
    fresh Agent 1 run would produce, instead of silently losing it.
    """
    occluder_text = (info.get("occluder", "") or "").strip()
    if occluder_text:
        print(f"  Occluder identified as '{occluder_text}' ({source}) — will "
              f"be explicitly excluded from the Flux prompt")
    detected = (info.get("occluded_object", "") or "").strip()
    if detected:
        # Prefer Agent 1's own detected identity over a generic CLI hint
        # whenever it's a refinement of that hint (e.g. hint="horse",
        # detected="left horse") rather than a wholesale override — this is
        # exactly the same-class disambiguation Agent 1's prompt is designed
        # to produce (see the "two zebras" worked example in
        # occlusion_agent.py), and without it the prompt falls back to the
        # bare class noun, which is ambiguous between two same-class
        # instances and leaves Flux no way to tell which one to continue.
        if not input_prompt or input_prompt.lower() in detected.lower():
            if subject_text != detected:
                print(f"  Using {source} detection for the completion prompt "
                      f"(more specific than the input hint): '{detected}'")
            subject_text = detected
    return subject_text, occluder_text


def _mask_covers_majority_of_frame(mask_path: Path, h: int, w: int,
                                    max_fraction: float = 0.55) -> bool:
    """True if `mask_path` is missing/unreadable, or covers more than
    `max_fraction` of the frame. A single subject's visible-mask should
    never be the majority of the frame — seen in practice, a stale/corrupt
    Agent 1 cache can save visible_mask.png with foreground and background
    swapped (e.g. the wall gets marked "visible", not the animal), which
    then silently poisons every downstream mask (hidden region, PostSeg
    argmax-IoU, the final gray-fill). Used both to invalidate a cached
    mask before trusting it and, as a last resort, to auto-flip one that's
    still implausible after a fresh Agent 1 run.
    """
    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return True
    if raw.shape != (h, w):
        raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    return int((raw > 127).sum()) > max_fraction * h * w


def main() -> int:
    img_path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE)
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return 1
    if len(sys.argv) > 2 and sys.argv[2].strip():
        config.INPUT_PROMPT = sys.argv[2].strip()

    stem = img_path.stem
    input_prompt = (getattr(config, "INPUT_PROMPT", "") or "").strip()
    subject_text = input_prompt or "subject"
    subject_slug = re.sub(r"[^a-z0-9]+", "_", subject_text.lower()).strip("_") or "subject"

    out_dir = BASE_DIR / "output" / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dir = out_dir / f"_flux_cutout_{subject_slug}"
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"Out dir: {test_dir}")

    img_bgr = cv2.imread(str(img_path))
    h, w = img_bgr.shape[:2]

    # ── Step 1: ensure we have visible + amodal masks ────────────────────
    vis_path     = out_dir / "visible_mask.png"
    amodal_path  = out_dir / "subject_full_amodal_mask.png"
    need_agent1 = not (vis_path.exists() and amodal_path.exists())
    if not need_agent1 and _mask_covers_majority_of_frame(vis_path, h, w):
        print("  [WARN] cached visible_mask.png covers most of the frame — "
              "looks inverted/stale; forcing Agent 1 to re-run")
        need_agent1 = True
    if need_agent1:
        print(f"Masks missing — running Agent 1 (occlusion_agent)…")
        config.IMAGE_PATH = str(img_path)
        state = {
            "image_path":            str(img_path),
            "target":                input_prompt,
            "occluded_object":       "",
            "occluder":              "",
            "what_to_remove":        "",
            "bbox":                  None,
            "boundary_expansion":    config.MASK_EXPAND,
            "region_desc":           "",
            "subject_description":   "",
            "visible_parts":         "",
            "missing_parts":         "",
            "frame_cropped":         False,
            "expansion_directions":  [],
            "expansion_pixels":      None,
            "mask_path":             None,
            "visible_mask_path":     None,
            "occluder_removed_path": None,
            "occluder_viz_path":     None,
            "hidden_polygon":        None,
            "hidden_mask_path":      None,
            "pix2gestalt_dir":       None,
            "output_path":           "",
            "output_rgba_path":      "",
            "review_score":          0.0,
            "review_feedback":       "",
            "failure_code":          "",
            "attempt":               0,
            "mask_retry_count":      0,
            "best_attempt":          0,
            "best_score":            0.0,
        }
        t0 = time.time()
        agent1_result = occlusion_agent(state)
        print(f"Agent 1 done in {time.time() - t0:.1f}s")
        subject_text, occluder_text = _apply_occlusion_info(
            agent1_result or {}, subject_text, input_prompt, "Agent 1's own")
    else:
        print(f"Reusing cached masks from {out_dir}/")
        cached_occlusion = {}
        occlusion_json_path = out_dir / "occlusion.json"
        if occlusion_json_path.exists():
            try:
                cached_occlusion = json.loads(occlusion_json_path.read_text())
            except Exception as exc:
                print(f"  [WARN] failed to read cached occlusion.json: {exc!r}")
        subject_text, occluder_text = _apply_occlusion_info(
            cached_occlusion, subject_text, input_prompt, "cached occlusion.json")

    # `test_dir` was named from the CLI hint (or the generic "subject"
    # placeholder) BEFORE Agent 1 ran, since its real detected subject
    # wasn't known yet. When no hint was given, every no-hint run ended up
    # with an identically-named "_flux_cutout_subject" (or, before the
    # config.INPUT_PROMPT default was fixed, "_flux_cutout_horse") folder
    # regardless of the image's actual content. Nothing is written into
    # test_dir until after this point, so it's safe to rename it now that
    # subject_text reflects Agent 1's real answer.
    if not input_prompt:
        new_slug = re.sub(r"[^a-z0-9]+", "_", subject_text.lower()).strip("_") or subject_slug
        if new_slug != subject_slug:
            new_test_dir = out_dir / f"_flux_cutout_{new_slug}"
            if test_dir.exists() and not any(test_dir.iterdir()):
                test_dir.rmdir()
            test_dir = new_test_dir
            test_dir.mkdir(parents=True, exist_ok=True)
            subject_slug = new_slug
            print(f"  Renamed output folder to reflect detected subject → {test_dir}")

    # ── Step 2: load masks ────────────────────────────────────────────────
    vis_mask = cv2.imread(str(vis_path), cv2.IMREAD_GRAYSCALE)
    am_mask  = cv2.imread(str(amodal_path), cv2.IMREAD_GRAYSCALE)
    if vis_mask.shape != (h, w):
        vis_mask = cv2.resize(vis_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if am_mask.shape != (h, w):
        am_mask  = cv2.resize(am_mask,  (w, h), interpolation=cv2.INTER_NEAREST)
    vis_b = (vis_mask > 127).astype(np.uint8)
    am_b  = (am_mask  > 127).astype(np.uint8)
    if int(vis_b.sum()) > 0.55 * h * w:
        print("  [WARN] visible_mask still covers most of the frame after "
              "Agent 1 — flipping foreground/background polarity")
        vis_b = 1 - vis_b
        # Persist the correction. Without this, every future run re-reads
        # the same still-inverted file on disk, re-trips this exact check,
        # and re-triggers a full Agent 1 re-run every single time — the
        # cache never sticks for this image.
        cv2.imwrite(str(vis_path), vis_b * 255)
        cv2.imwrite(str(amodal_path), vis_b * 255)
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
    # ── Area-ratio sanity check: trim a disproportionately large occluder ──
    # A correctly segmented occluder is normally comparable to (or smaller
    # than) the subject it's blocking. A wildly oversized occluder mask
    # (e.g. a "teacup" segmentation bleeding into the whole tabletop) means
    # dilate(occluder)\visible below would inpaint a huge, mostly-irrelevant
    # region — this is the vase failure mode's actual root cause. Bbox
    # clipping further down only bounds the OUTER extent; it doesn't check
    # whether the occluder mask itself is a sane size. If the ratio is too
    # high, keep only the occluder pixels physically near the visible
    # subject — a real occluder must touch/border what it's hiding.
    vis_area  = int(vis_b.sum())
    occ_area  = int(occ_b.sum())
    max_ratio = float(getattr(config, "OCCLUDER_MAX_AREA_RATIO", 2.5))
    if vis_area > 0 and occ_area > max_ratio * vis_area:
        prox_px = int(getattr(config, "OCCLUDER_PROXIMITY_PX", 60))
        prox_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (prox_px * 2 + 1, prox_px * 2 + 1))
        vis_dilated = cv2.dilate(vis_b, prox_kernel, iterations=1)
        trimmed = (occ_b & vis_dilated).astype(np.uint8)
        print(f"  [OcclSanity] occluder {occ_area}px is {occ_area / vis_area:.1f}x the "
              f"visible subject ({vis_area}px, max {max_ratio}x) — trimming to within "
              f"{prox_px}px of the subject: {occ_area} → {int(trimmed.sum())} px")
        if int(trimmed.sum()) > 0:
            occ_b = trimmed
        else:
            print("  [OcclSanity] trim left nothing — keeping untrimmed occluder mask")

    # Match amodal/main.py:725-726 for iter 0
    JA_KERNEL = np.ones((5, 5), np.uint8)
    JA_ITERS  = 3
    occ_dilated = cv2.dilate(occ_b, JA_KERNEL, iterations=JA_ITERS).astype(np.uint8)
    hidden = (occ_dilated & (1 - vis_b)).astype(np.uint8)
    print(f"  visible: {int(vis_b.sum())} px   occluder: {int(occ_b.sum())} px   "
          f"occ_dilated: {int(occ_dilated.sum())} px")
    print(f"  hidden = dilate(occluder, 5x5, 3) \\ visible: {int(hidden.sum())} px  "
          f"(Jiang Ao iter-0 mask)")

    # Bound the fill region to the subject's plausible extent. Unbounded,
    # `hidden` is dilate(occluder) minus visible — fine for a COMPACT
    # occluder (~subject-sized, e.g. a person standing in front of a
    # horse), but for a large occluder (e.g. a snowbank a rabbit sits
    # behind) that's the entire occluder: observed in practice at 6-7x the
    # visible subject's area, up to 4 body-widths away from the subject.
    # Flux then has no reason to paint one plausible continuation rather
    # than several/oversized subjects tiling the space — which is exactly
    # the DUPLICATE_SUBJECT failures the reviewer kept flagging. Clipping
    # to an expanded visible-bbox is adaptive: a compact occluder is
    # barely affected (it's already close to bbox-sized); a huge one gets
    # clipped down to near the subject.
    if getattr(config, "BOUND_HIDDEN_TO_SUBJECT_BBOX", True):
        ys, xs = np.where(vis_b > 0)
        if len(ys) > 0:
            vx1, vy1, vx2, vy2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            vw, vh = vx2 - vx1, vy2 - vy1
            mult    = float(getattr(config, "HIDDEN_REGION_BBOX_MULTIPLIER", 2.0))
            min_ext = int(getattr(config, "HIDDEN_REGION_MIN_EXPANSION_PX", 100))
            ex = max(int(vw * (mult - 1) / 2), min_ext)
            ey = max(int(vh * (mult - 1) / 2), min_ext)
            bbox_mask = np.zeros((h, w), dtype=np.uint8)
            bbox_mask[max(0, vy1 - ey):min(h, vy2 + ey),
                      max(0, vx1 - ex):min(w, vx2 + ex)] = 1
            hidden_before = int(hidden.sum())
            bounded = (hidden & bbox_mask).astype(np.uint8)
            # If the expanded bbox doesn't overlap the occluder-derived
            # hidden region AT ALL, bounding would zero out the fill region
            # entirely and Flux would never run — strictly worse than the
            # unbounded (if oversized) region this was meant to shrink.
            # Most likely cause: Agent 1's occluder detection landed on the
            # wrong/misaligned region this run. Keep the unbounded region
            # rather than silently skipping inpainting.
            if int(bounded.sum()) == 0 and hidden_before > 0:
                print(f"  [BoundHidden] expanded bbox has ZERO overlap with the "
                      f"{hidden_before}px hidden region (likely a mismatched "
                      f"occluder detection this run) — keeping it unbounded "
                      f"rather than filling nothing")
            else:
                hidden = bounded
                if int(hidden.sum()) < hidden_before:
                    print(f"  [BoundHidden] expanded visible-bbox ({mult}x, min {min_ext}px) "
                          f"clipped hidden region {hidden_before} → {int(hidden.sum())} px")

    am_b = (vis_b | hidden).astype(np.uint8)   # used by off-frame boundary check below

    # CUTOUT: keep ONLY (eroded) visible person pixels.  Hidden slice +
    # everything else is neutral gray.  The 5×5 erosion (Jiang Ao
    # amodal/main.py:693-697) prevents Flux from preserving any
    # edge-bleed pixels at the visible/occluder boundary — those
    # 3-5 px get repainted by Flux instead.
    vis_b_eroded = cv2.erode(vis_b, np.ones((5, 5), np.uint8),
                             iterations=1).astype(np.uint8)
    cutout = np.full_like(img_bgr, PAD_COLOR)
    cutout[vis_b_eroded == 1] = img_bgr[vis_b_eroded == 1]
    cv2.imwrite(str(test_dir / f"{subject_slug}_cutout.png"), cutout)
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
                    f"C: {subject_text} cutout (final)", "D: hidden mask (empty)",
                    "E: same as cutout (Flux skipped)"],
            out_path=test_dir / "comparison.png",
        )
        _publish_final(test_dir, test_dir / "flux_completed_rgba.png", subject_slug)
        print(f"\nOutputs → {test_dir}/")
        return 0

    cutout_rgb = cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB)
    # NOTE on wording: this is a strict region-only inpaint, not a "generate
    # a subject" prompt. Two failure modes showed up repeatedly at review
    # time with an earlier, vaguer version of this prompt ("...full body...
    # Plain neutral background."): the model would paint a second, separate
    # animal (DUPLICATE_SUBJECT) or replace the surrounding scene wholesale
    # (SEAM_VISIBLE / background regenerated) instead of a small, local
    # continuation of the one existing subject. Every clause below exists
    # to rule out one of those specific failures — keep them explicit
    # rather than trimming back to something vaguer.
    # Deliberately do NOT name the occluder noun here (e.g. "bowl",
    # "sticker") in a "draw no X" instruction — diffusion models handle
    # negation poorly (text encoders have no true logical NOT), so naming
    # the forbidden object still primes cross-attention toward painting
    # it. Observed in practice: "draw no bowl pixels" -> Flux paints a
    # bowl anyway (OCCLUDER_REGENERATED). Use only generic, positive
    # phrasing instead — describe what TO paint, not what to avoid by name.
    occluder_clause = (
        f" This area was covered by something in front of the "
        f"{subject_text} that has since been digitally removed — paint "
        f"only {subject_text} body continuing naturally here, with "
        f"nothing else in this region."
    ) if occluder_text else ""
    prompt = (
        f"Inpaint only this masked region. Continue the SAME single "
        f"{subject_text} already visible in the photo — exactly one "
        f"subject, no duplicate, no second animal, no new subject anywhere. "
        f"Match its exact color, texture, pose and lighting. Keep every "
        f"unmasked pixel unchanged; do not invent, replace, or regenerate "
        f"any background or scenery.{occluder_clause}"
    )
    neg_extra = (
        "distorted anatomy, duplicate subject, second animal, extra animal, "
        "collage, multiple subjects, sprite sheet, blurry, low detail, "
        "background replaced, new scenery, studio backdrop, plain backdrop"
    )
    if occluder_text:
        neg_extra += f", {occluder_text}"

    # ── Agent 2 (Flux-Fill) + Agent 3 (reviewer/retry loop) ──────────────
    use_reviewer = getattr(config, "USE_REVIEWER", False)
    score_thresh = getattr(config, "REVIEWER_SCORE_THRESHOLD", 7.0)
    max_retries  = getattr(config, "REVIEWER_MAX_RETRIES", 0) if use_reviewer else 0

    current_prompt    = prompt
    current_neg_extra = neg_extra
    review_log        = []
    flux_bgr = flux_rgb = None
    best_score = -1.0
    best_cand  = None   # (attempt, cand_bgr, cand_rgb) — highest-scoring attempt seen

    fill_base_rgb = cutout_rgb
    fill_strength = 1.0
    if getattr(config, "USE_DEPTH_GUIDED_FILL", False):
        from models.flux_depth import build_depth_guided_base
        depth_guided_bgr = build_depth_guided_base(
            cutout_bgr=cutout, hidden_mask=hidden, prompt=prompt,
        )
        if depth_guided_bgr is not cutout:   # only switch strength if the guide stage actually ran
            fill_base_rgb = cv2.cvtColor(depth_guided_bgr, cv2.COLOR_BGR2RGB)
            fill_strength = float(getattr(config, "DEPTH_GUIDED_FILL_STRENGTH", 0.65))
            cv2.imwrite(str(test_dir / "depth_guided_base.png"), depth_guided_bgr)

    for attempt in range(1, max_retries + 2):
        print(f"\nRunning Flux-Fill on the cutout (attempt {attempt}/{max_retries + 1})…")
        t0 = time.time()
        flux_results = _run_flux_fill_inpaint(
            base_np=fill_base_rgb,
            inpaint_mask=hidden,
            amodal_rgb_256=np.full((256, 256, 3), 255, dtype=np.uint8),
            prompt=current_prompt,
            n_samples=1,
            out_dir=test_dir,
            prefix=f"flux_completed_attempt{attempt}",
            neg_extra=current_neg_extra,
            seed_offset=(attempt - 1) * 7,
            strength=fill_strength,
            keep_loaded=True,   # nothing else needs the GPU between retries — freed once, below
        )
        dt = time.time() - t0
        print(f"  Flux done in {dt:.1f}s")

        if not flux_results:
            print("Flux returned nothing — aborting")
            return 1

        cand_rgb = np.array(flux_results[0].convert("RGB"))
        if cand_rgb.shape[:2] != (h, w):
            cand_rgb = cv2.resize(cand_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
        cand_bgr = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2BGR)
        # Restore visible-person pixels (Flux's VAE drift can soften them)
        cand_bgr[vis_b == 1] = img_bgr[vis_b == 1]

        if not use_reviewer:
            flux_bgr, flux_rgb = cand_bgr, cand_rgb
            break

        cand_path = test_dir / f"flux_completed_attempt{attempt}_review.png"
        cv2.imwrite(str(cand_path), cand_bgr)

        review_prompt = (
            f"You are reviewing an AI-completed image. The original photo (first image) "
            f"shows a '{subject_text}' partly hidden behind "
            f"{('a ' + occluder_text) if occluder_text else 'another object'}. The second "
            f"image is the completed result, where the hidden region has been filled in by "
            f"a diffusion model.\n\n"
            f"IMPORTANT CONTEXT — read before scoring: this pipeline's deliverable is an "
            f"ISOLATED CUTOUT of the {subject_text}, not a scene-preserving edit. The second "
            f"image is deliberately composited on a flat neutral-gray background, with the "
            f"rest of the original scene (walls, floor, foliage, snow, sky, etc.) "
            f"intentionally removed — that gray background, and the hard boundary between "
            f"the subject's silhouette and that gray field, is the CORRECT, EXPECTED output "
            f"format, not a defect. Do NOT score down for the background being replaced by "
            f"gray, for the original scenery/context being absent, or for a clean edge "
            f"between the subject and the gray field — none of that was ever part of the "
            f"task. Judge ONLY the {subject_text} itself: does the newly-painted region of "
            f"its OWN body look right, and does it read as the same continuous subject as "
            f"the visible portion (matching fur/texture/color/pose/lighting ON the subject "
            f"itself)? A perfect score is possible even though the background is flat gray.\n\n"
            f"Score the result 1-10 on: (a) whether the completed region shows the "
            f"{subject_text}'s OWN body continuing naturally — NOT "
            f"{occluder_text or 'the occluder'} regenerated in that space, and NOT a "
            f"duplicate {subject_text}; (b) anatomical correctness; (c) seamless lighting/"
            f"texture/color match with the visible portion; (d) absence of blur or "
            f"visible seams.\n"
            f"failure_code must be one of: ACCEPTED, OCCLUDER_REGENERATED, "
            f"DUPLICATE_SUBJECT, ANATOMY_WRONG, BLURRY_OUTPUT, SEAM_VISIBLE.\n"
            f"If score < {score_thresh}, also fill improved_prompt and "
            f"improved_negative_prompt with a corrected version of this prompt: "
            f"{current_prompt!r} (negative terms: {current_neg_extra!r}) that fixes the "
            f"specific problem you found. IMPORTANT constraint on improved_prompt: never "
            f"name {occluder_text or 'the occluder'} inside a 'no X' / 'remove X' / "
            f"'without X' instruction — diffusion models handle negation poorly, so "
            f"writing e.g. 'no {occluder_text or 'occluder'}' or 'remove the "
            f"{occluder_text or 'occluder'}' primes the model to paint it anyway (this is "
            f"the exact OCCLUDER_REGENERATED failure). Put occluder terms ONLY in "
            f"improved_negative_prompt (which uses real classifier-free guidance, not "
            f"text negation) — improved_prompt itself should describe only what TO paint "
            f"(the subject's own body), using generic phrasing like 'nothing else in this "
            f"region' instead of naming the object."
        )
        try:
            review = gpt_vision(
                images=[str(img_path), str(cand_path)],
                prompt=review_prompt,
                schema=REVIEWER_SCHEMA,
                cache_key="agent3_reviewer",
            )
        except Exception as exc:
            print(f"  [Agent 3: Reviewer] call failed ({exc!r}) — accepting attempt as-is")
            flux_bgr, flux_rgb = cand_bgr, cand_rgb
            break

        score = review.get("score", 0)
        failure_code = review.get("failure_code", "")
        print(f"  [Agent 3: Reviewer] score={score}/10  failure_code={failure_code}")
        print(f"  Feedback: {review.get('feedback', '')[:200]}")
        review_log.append({"attempt": attempt, **review})

        if score > best_score:
            best_score = score
            best_cand  = (attempt, cand_bgr, cand_rgb)

        if score >= score_thresh:
            flux_bgr, flux_rgb = cand_bgr, cand_rgb
            break
        if attempt >= max_retries + 1:
            # No attempt reached score_thresh. Keep the best-scoring attempt
            # seen across the whole run, NOT just this last one — Flux/the
            # reviewer's own retries are not monotonically improving, so the
            # last attempt tried is not necessarily the best one produced.
            best_attempt, flux_bgr, flux_rgb = best_cand
            print(f"  [Agent 3: Reviewer] no attempt reached {score_thresh}/10 after "
                  f"{attempt} attempts — keeping best-scoring attempt "
                  f"{best_attempt} (score={best_score})")
            break

        print(f"  [Agent 3: Reviewer] score below threshold ({score_thresh}) — retrying")
        if review.get("improved_prompt"):
            current_prompt = review["improved_prompt"]
        if review.get("improved_negative_prompt"):
            current_neg_extra = review["improved_negative_prompt"]

    if review_log:
        with open(test_dir / "review_log.json", "w") as f:
            json.dump(review_log, f, indent=2)
    # Retry loop kept Flux resident across attempts (keep_loaded=True) —
    # release it now, once, before SAM3 needs the GPU below.
    _free_flux_fill()
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
            for r in sam_res:
                m = r["mask"]
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST)
                m_b = (m > 0).astype(np.uint8)
                area = int(m_b.sum())
                # Drop blobs covering most of the image (background segment).
                if area > 0.85 * h * w:
                    continue
                inter = int(((m_b > 0) & (vis_b > 0)).sum())
                union = area + vis_sum - inter
                iou   = inter / max(union, 1)
                if iou > best_iou:
                    best_iou  = iou
                    best_mask = m_b
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
    if sides_touched and not getattr(config, "USE_OFFFRAME_EXTENSION", False):
        print("[OffFrame] disabled via config.USE_OFFFRAME_EXTENSION — skipping")
        sides_touched = set()

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
            f"Sharp focus, high detail. Plain neutral background.{occluder_clause}"
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
        for r in sam_res:
            m = r["mask"]
            if m.shape[:2] != (H_new, W_new):
                m = cv2.resize(m.astype(np.uint8), (W_new, H_new),
                               interpolation=cv2.INTER_NEAREST)
            mb = (m > 0).astype(np.uint8)
            if not (0 <= cy < H_new and 0 <= cx < W_new and mb[cy, cx] == 1):
                continue
            if mb.sum() > 0.85 * H_new * W_new:
                continue
            ovl = int(((mb > 0) & (padded_mask > 0)).sum())
            ovl_ratio = ovl / max(int(padded_mask.sum()), 1)
            if ovl_ratio > best_overlap:
                best_overlap = ovl_ratio
                best_mask = mb
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
            f"C: {subject_text} cutout (Flux input)",
            "D: hidden inpaint mask",
            f"E: Flux-completed {subject_text}",
        ],
        out_path=test_dir / "comparison.png",
    )

    _publish_final(test_dir, test_dir / "offframe_final_rgba.png", subject_slug)
    print(f"\nOutputs → {test_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
