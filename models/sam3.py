"""
models/sam3.py — SAM3 segmentation.

Auto mask generation, text-prompt segmentation, and point/box-prompt
segmentation, plus the segment->dict helpers and visualisations. Models are
lazy-loaded singletons cached for the process lifetime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

import config
from runtime import DEVICE, _release_ram, register_free_fn

# ── Lazy singletons ─────────────────────────────────────────────────────────
_SAM3_PIPE           = None
_SAM3_TEXT_MODEL     = None
_SAM3_TEXT_PROCESSOR = None


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
    global _SAM3_PIPE, _SAM3_TEXT_MODEL, _SAM3_TEXT_PROCESSOR
    if _SAM3_PIPE is None and _SAM3_TEXT_MODEL is None and _SAM3_TEXT_PROCESSOR is None:
        return
    # Walk all pipeline attributes and move any nn.Module to CPU.
    # Using isinstance(_, torch.nn.Module) is more reliable than duck-typing.
    if _SAM3_PIPE is not None:
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
    if isinstance(_SAM3_TEXT_MODEL, torch.nn.Module):
        try:
            _SAM3_TEXT_MODEL.cpu()
        except Exception:
            pass
    if _SAM3_PIPE is not None:
        del _SAM3_PIPE
    if _SAM3_TEXT_MODEL is not None:
        del _SAM3_TEXT_MODEL
    if _SAM3_TEXT_PROCESSOR is not None:
        del _SAM3_TEXT_PROCESSOR
    _SAM3_PIPE = None
    _SAM3_TEXT_MODEL = None
    _SAM3_TEXT_PROCESSOR = None
    _release_ram()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  [SAM3] Released from GPU")




def _get_sam3_text():
    """Lazy-load SAM3's native text-prompt model/processor path."""
    global _SAM3_TEXT_MODEL, _SAM3_TEXT_PROCESSOR
    if _SAM3_TEXT_MODEL is None or _SAM3_TEXT_PROCESSOR is None:
        try:
            from transformers.models.sam3 import Sam3Model, Sam3Processor
        except ImportError:
            raise ImportError("transformers is not installed. Run: pip install transformers")
        print(f"  [SAM3/Text] Loading model ({config.SAM3_MODEL_ID})…")
        _release_ram()
        _SAM3_TEXT_PROCESSOR = Sam3Processor.from_pretrained(config.SAM3_MODEL_ID)
        _SAM3_TEXT_MODEL = Sam3Model.from_pretrained(
            config.SAM3_MODEL_ID,
            low_cpu_mem_usage=True,
        )
        _SAM3_TEXT_MODEL.to(DEVICE)
        _SAM3_TEXT_MODEL.eval()
    return _SAM3_TEXT_MODEL, _SAM3_TEXT_PROCESSOR


def _segments_from_binary_masks(
    masks,
    scores,
    score_thresh: float,
    min_area: int,
) -> list:
    """Convert model masks/scores into the segment dict shape used downstream."""
    segments: list = []
    for mask_obj, score_obj in zip(masks, scores):
        mask = (np.asarray(mask_obj) > 0).astype(np.uint8)
        area = int(mask.sum())
        score = float(score_obj)
        if area < min_area or score < score_thresh:
            continue

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon = []
        if contours:
            largest = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            polygon = approx.reshape(-1, 2).tolist()

        segments.append({
            "id": len(segments),
            "mask": mask,
            "bbox": [x1, y1, x2, y2],
            "polygon": polygon,
            "area": area,
            "iou": score,
        })
    segments.sort(key=lambda s: s["area"], reverse=True)
    for i, seg in enumerate(segments):
        seg["id"] = i
    return segments


def _save_segments_viz(img_bgr: np.ndarray, segments: list, out_path: Path, prefix: str = "") -> None:
    """Save the numbered overlay used by GPT / debugging."""
    viz = img_bgr.copy()
    rng = np.random.default_rng(42)
    for seg in segments:
        colour = rng.integers(60, 210, 3).tolist()
        overlay = np.zeros_like(viz)
        overlay[seg["mask"] == 1] = colour
        viz = cv2.addWeighted(viz, 0.65, overlay, 0.35, 0)
        M = cv2.moments(seg["mask"])
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            label = f"{prefix}{seg['id']}"
            cv2.putText(viz, label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(viz, label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), viz)


def _sam_segment_text_prompt(image_path: str, text_prompt: str, out_dir: Path, tag: str) -> list:
    """Run SAM3's native text-prompted instance segmentation."""
    if not text_prompt:
        return []

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    pil_img = Image.fromarray(img_rgb)

    try:
        model, processor = _get_sam3_text()
        print(f"  [SAM3/Text] Segmenting '{text_prompt}' ({tag})…")
        inputs = processor(images=pil_img, text=text_prompt, return_tensors="pt")
        inputs = {
            k: (v.to(DEVICE) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=float(getattr(config, "SAM3_TEXT_SCORE_THRESH", config.SAM3_SCORE_THRESH)),
            mask_threshold=float(getattr(config, "SAM3_TEXT_MASK_THRESH", 0.5)),
            target_sizes=[(h, w)],
        )[0]
        masks = results.get("masks", [])
        scores = results.get("scores", [1.0] * len(masks))
        masks = [m.detach().to("cpu").numpy() if hasattr(m, "detach") else m for m in masks]
        scores = [s.detach().to("cpu").item() if hasattr(s, "detach") else s for s in scores]
        segments = _segments_from_binary_masks(
            masks,
            scores,
            score_thresh=float(getattr(config, "SAM3_TEXT_SCORE_THRESH", config.SAM3_SCORE_THRESH)),
            min_area=int(getattr(config, "SAM3_TEXT_MIN_AREA", config.SAM3_MIN_AREA)),
        )

        masks_dir = out_dir / f"sam3_text_{tag}_masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        for seg in segments:
            cv2.imwrite(str(masks_dir / f"seg_{seg['id']:03d}.png"), seg["mask"] * 255)
        viz_path = out_dir / f"sam3_text_{tag}_viz.png"
        _save_segments_viz(img_bgr, segments, viz_path)
        print(f"  [SAM3/Text] {tag}: {len(segments)} segment(s) → {viz_path}")
        return segments
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [SAM3/Text] failed for '{text_prompt}' ({tag}): {exc!r}")
        return []
    finally:
        _free_sam3()


def _mask_from_text_segments(
    segments: list,
    shape: tuple[int, int],
    click: Optional[dict],
    tag: str,
) -> np.ndarray:
    """Choose the text-prompt segment instance nearest to GPT's click."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)
    if not segments:
        return out

    cx = int((click or {}).get("x", 0))
    cy = int((click or {}).get("y", 0))
    if 0 < cx < w and 0 < cy < h:
        containing = [s for s in segments if s["mask"].shape[:2] == (h, w) and s["mask"][cy, cx] > 0]
        if containing:
            best = max(containing, key=lambda s: int(s["mask"].sum()))
            print(f"  [SAM3/Text] {tag}: selected segment {best['id']} containing click "
                  f"({cx},{cy}) area={int(best['mask'].sum())} px")
            return best["mask"].astype(np.uint8).copy()

        def _centroid_dist(seg: dict) -> float:
            m = seg["mask"]
            M = cv2.moments(m.astype(np.uint8))
            if M["m00"] <= 0:
                return float("inf")
            mx, my = M["m10"] / M["m00"], M["m01"] / M["m00"]
            return float((mx - cx) ** 2 + (my - cy) ** 2)

        best = min(segments, key=_centroid_dist)
        print(f"  [SAM3/Text] {tag}: click ({cx},{cy}) was outside text masks; "
              f"selected nearest segment {best['id']} area={int(best['mask'].sum())} px")
        return best["mask"].astype(np.uint8).copy()

    for seg in segments:
        out = np.clip(out | seg["mask"].astype(np.uint8), 0, 1).astype(np.uint8)
    print(f"  [SAM3/Text] {tag}: no valid click; unioned {len(segments)} segment(s) "
          f"area={int(out.sum())} px")
    return out




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


# ── Mask-review-first verification (USE_MASK_REVIEW_FIRST flow) ──────────────
#
# After Agent 1 finalizes the visible mask, this helper builds a green-overlay
# viz and asks GPT-V to verify whether the mask correctly captures the target
# subject. If GPT rejects it (mask is too small / on wrong subject / picked
# the occluder by accident), GPT returns a corrective click and we re-run
# SAM3 point-prompt with that click. Bounded by MASK_REVIEW_MAX_RETRIES.



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



register_free_fn("sam3", _free_sam3)
