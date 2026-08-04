"""SAM3 segmentation: the generic auto/point-prompt pipeline, the direct
Sam3Model/Sam3Processor path for open-vocabulary text-prompted segmentation
(Promptable Concept Segmentation), and the segment-everything / targeted
click-prompt helpers used by Agent 1.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import config
from runtime import DEVICE, _release_ram

_SAM3_PIPE = None


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

_SAM3_TEXT_MODEL = None
_SAM3_TEXT_PROCESSOR = None


def _get_sam3_text():
    """Lazy-load Sam3Model/Sam3Processor directly for open-vocabulary
    text-prompted segmentation (SAM3's "Promptable Concept Segmentation").

    The generic `pipeline("mask-generation", ...)` wrapper used by
    `_get_sam3()` only exposes auto-segment/point/box prompts — text
    prompts require calling Sam3Model/Sam3Processor directly.
    """
    global _SAM3_TEXT_MODEL, _SAM3_TEXT_PROCESSOR
    if _SAM3_TEXT_MODEL is None:
        from transformers import Sam3Model, Sam3Processor
        print(f"  [SAM3-text] Loading model ({config.SAM3_MODEL_ID})…")
        _release_ram()
        _SAM3_TEXT_MODEL = Sam3Model.from_pretrained(
            config.SAM3_MODEL_ID, torch_dtype=torch.float32
        ).to(DEVICE)
        _SAM3_TEXT_PROCESSOR = Sam3Processor.from_pretrained(config.SAM3_MODEL_ID)
    return _SAM3_TEXT_MODEL, _SAM3_TEXT_PROCESSOR


def _free_sam3_text():
    global _SAM3_TEXT_MODEL, _SAM3_TEXT_PROCESSOR
    if _SAM3_TEXT_MODEL is None:
        return
    try:
        _SAM3_TEXT_MODEL.cpu()
    except Exception:
        pass
    del _SAM3_TEXT_MODEL
    _SAM3_TEXT_MODEL = None
    _SAM3_TEXT_PROCESSOR = None
    _release_ram()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  [SAM3-text] Released from GPU")


def _sam3_text_segment(
    image_path: str,
    text: str,
    threshold: float = 0.3,
    mask_threshold: float = 0.5,
) -> list:
    """Open-vocabulary text-prompted segmentation via SAM3's Promptable
    Concept Segmentation — finds all instances matching a free-text noun
    phrase (e.g. "bread", "rabbit", "fountain"), no fixed category list
    and no click-point needed. Unlike InstaFormer (COCO-only categories)
    or the old geometric-guess fallback, this can find ANY described
    subject or occluder directly.

    Returns a list of {'score': float, 'box': [x1,y1,x2,y2],
    'mask': np.uint8 0/1} sorted by score descending, or [] on no match
    or failure.
    """
    try:
        model, processor = _get_sam3_text()
        pil = Image.open(image_path).convert("RGB")
        inputs = processor(images=pil, text=text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=mask_threshold,
            target_sizes=[pil.size[::-1]],
        )[0]
        out = []
        for score, box, mask in zip(results["scores"], results["boxes"], results["masks"]):
            out.append({
                "score": float(score),
                "box": [float(x) for x in box],
                "mask": mask.cpu().numpy().astype(np.uint8),
            })
        out.sort(key=lambda d: d["score"], reverse=True)
        if out:
            print(f"  [SAM3-text] '{text}': {len(out)} instance(s), "
                  f"top score={out[0]['score']:.2f}, "
                  f"area={int(out[0]['mask'].sum())} px")
        else:
            print(f"  [SAM3-text] '{text}': no match")
        return out
    except Exception as exc:                                      # noqa: BLE001
        print(f"  [SAM3-text] segmentation failed: {exc!r}")
        return []

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


