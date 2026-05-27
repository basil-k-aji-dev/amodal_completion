"""
metrics.py — per-run quality metrics.

Computes and saves four standard amodal-completion benchmark numbers
into ``output/<stem>/_flux_cutout_person/metrics.json``:

  1. IoU                  — silhouette overlap (vs ground truth if given,
                            else self-consistency vs SAM3 PostSeg)
  2. Boundary F1          — edge agreement, tolerance ≈ 3 px
  3. LPIPS                — perceptual similarity (visible-region drift +
                            full-image)
  4. FID                  — distribution-level; **not applicable per image**,
                            requires a batch. We emit a note + reserve the
                            field for a separate batch-eval script.

Self-consistency vs ground truth:
  • Without ground truth (our typical case on the ``website/`` set), the
    "IoU" and "Boundary F1" numbers measure how well the predicted amodal
    mask matches the silhouette SAM3 actually traced in Flux's output.
    High self-consistency = Flux drew what Agent 1 predicted; low = the
    polygon and the painted result disagree.
  • With ground truth (pass ``gt_mask_path`` to ``compute_run_metrics``),
    the same two metrics are computed against the GT and reported under
    ``iou_vs_gt`` / ``boundary_f1_vs_gt`` instead.

LPIPS requires ``pip install lpips`` — falls back to MSE if unavailable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# Cache the LPIPS model so we don't reload per call.
_LPIPS_MODEL = None

# Cache the CLIP model for clip_score_region — shared across calls.
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None


def _get_clip_for_scoring(model_name: str = "ViT-B/32"):
    """Lazy-load OpenAI CLIP for region scoring. Returns (model, preprocess,
    device) or (None, None, None) if the clip package isn't importable."""
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    try:
        import clip as openai_clip                              # noqa: WPS433
        import torch                                            # noqa: WPS433
    except ImportError:
        return None, None, None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m, p = openai_clip.load(model_name, device=device)
        m.eval()
        _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE = m, p, device
        return m, p, device
    except Exception:                                           # noqa: BLE001
        return None, None, None


def clip_score_region(image_bgr: np.ndarray,
                      mask: np.ndarray,
                      text: str,
                      *,
                      pad_px: int = 8,
                      model_name: str = "ViT-B/32") -> float:
    """Cosine similarity between the masked region of ``image_bgr`` and
    ``text`` ("a photo of a {target}"), in [-1, 1] (typically 0.0-0.4).

    The masked region is cropped to its bounding box (with ``pad_px``
    padding), pixels outside the mask are zeroed, and the result is
    encoded by CLIP. Returns ``float('nan')`` if CLIP isn't available
    or the mask is empty.
    """
    if image_bgr is None or mask is None:
        return float("nan")
    m = _ensure_binary(mask)
    if m.sum() == 0:
        return float("nan")

    ys, xs = np.where(m > 0)
    y0, y1 = max(0, int(ys.min()) - pad_px), min(image_bgr.shape[0], int(ys.max()) + pad_px + 1)
    x0, x1 = max(0, int(xs.min()) - pad_px), min(image_bgr.shape[1], int(xs.max()) + pad_px + 1)
    crop = image_bgr[y0:y1, x0:x1].copy()
    crop_m = m[y0:y1, x0:x1]
    crop[crop_m == 0] = 0  # zero pixels outside the mask

    model, preprocess, device = _get_clip_for_scoring(model_name)
    if model is None:
        return float("nan")

    try:
        import clip as openai_clip                              # noqa: WPS433
        import torch                                            # noqa: WPS433
        from PIL import Image                                   # noqa: WPS433
    except ImportError:
        return float("nan")

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    image_input = preprocess(pil).unsqueeze(0).to(device)
    text_input = openai_clip.tokenize([text]).to(device)
    with torch.no_grad():
        img_f = model.encode_image(image_input)
        txt_f = model.encode_text(text_input)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        sim = float((img_f @ txt_f.T).item())
    return sim


def _ensure_binary(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 127).astype(np.uint8) if mask.dtype == np.uint8 and mask.max() > 1 \
           else (mask > 0).astype(np.uint8)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over Union of two binary masks (same shape)."""
    if a is None or b is None:
        return float("nan")
    a_b = _ensure_binary(a).astype(bool)
    b_b = _ensure_binary(b).astype(bool)
    if a_b.shape != b_b.shape:
        h = min(a_b.shape[0], b_b.shape[0])
        w = min(a_b.shape[1], b_b.shape[1])
        a_b = a_b[:h, :w]
        b_b = b_b[:h, :w]
    inter = int((a_b & b_b).sum())
    union = int((a_b | b_b).sum())
    if union == 0:
        return 1.0  # both empty → trivially identical
    return inter / union


def boundary_f1(a: np.ndarray, b: np.ndarray, tolerance: int = 3) -> float:
    """F1 score on mask boundaries with ``tolerance`` px slack.

    Returns harmonic mean of precision (fraction of A-boundary pixels
    within tolerance of B's boundary) and recall (B-boundary within
    tolerance of A's). 1.0 = identical edges, 0.0 = no overlap.
    """
    if a is None or b is None:
        return float("nan")
    a_b = _ensure_binary(a)
    b_b = _ensure_binary(b)
    if a_b.shape != b_b.shape:
        h = min(a_b.shape[0], b_b.shape[0])
        w = min(a_b.shape[1], b_b.shape[1])
        a_b = a_b[:h, :w]
        b_b = b_b[:h, :w]

    k_erode = np.ones((3, 3), np.uint8)
    a_boundary = a_b - cv2.erode(a_b, k_erode)
    b_boundary = b_b - cv2.erode(b_b, k_erode)
    if a_boundary.sum() == 0 or b_boundary.sum() == 0:
        return float("nan")

    k_tol = np.ones((tolerance * 2 + 1, tolerance * 2 + 1), np.uint8)
    a_dil = cv2.dilate(a_boundary, k_tol)
    b_dil = cv2.dilate(b_boundary, k_tol)

    precision = (a_boundary * b_dil).sum() / max(int(a_boundary.sum()), 1)
    recall    = (b_boundary * a_dil).sum() / max(int(b_boundary.sum()), 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _get_lpips_model():
    """Lazy-load LPIPS (AlexNet backbone, smaller than VGG). Returns None
    if the lpips package isn't installed; callers fall back to MSE."""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL
    try:
        import lpips  # noqa: WPS433
        import torch  # noqa: WPS433
        m = lpips.LPIPS(net="alex", verbose=False)
        if torch.cuda.is_available():
            m = m.cuda()
        m.eval()
        _LPIPS_MODEL = m
        return m
    except Exception:                                         # noqa: BLE001
        return None


def lpips_score(img_a: np.ndarray, img_b: np.ndarray,
                mask: Optional[np.ndarray] = None) -> dict:
    """Perceptual similarity between two BGR uint8 images of the same size.
    Returns ``{lpips, mse_fallback}`` — if lpips lib not installed,
    ``lpips`` is None and MSE is reported instead.

    If ``mask`` is provided (binary, same HxW), only the masked region is
    compared (pixels outside the mask are set to 0 in both).
    """
    if img_a is None or img_b is None:
        return {"lpips": float("nan"), "mse": float("nan")}
    if img_a.shape != img_b.shape:
        h = min(img_a.shape[0], img_b.shape[0])
        w = min(img_a.shape[1], img_b.shape[1])
        img_a = img_a[:h, :w]
        img_b = img_b[:h, :w]
    if mask is not None:
        m = _ensure_binary(mask)
        if m.shape != img_a.shape[:2]:
            m = cv2.resize(m, (img_a.shape[1], img_a.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        img_a = img_a.copy(); img_b = img_b.copy()
        img_a[m == 0] = 0
        img_b[m == 0] = 0

    # MSE — always cheap to compute as a fallback / sanity number.
    mse = float(((img_a.astype(np.float32) - img_b.astype(np.float32)) ** 2).mean())

    model = _get_lpips_model()
    if model is None:
        return {"lpips": None, "mse": mse}

    import torch                                              # noqa: WPS433
    def _t(img):
        # BGR uint8 → RGB tensor in [-1, 1], NCHW
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
        t = (t / 127.5) - 1.0
        if torch.cuda.is_available():
            t = t.cuda()
        return t
    with torch.no_grad():
        d = float(model(_t(img_a), _t(img_b)).item())
    return {"lpips": d, "mse": mse}


def compute_run_metrics(
    *,
    out_dir: Path,
    original_image: np.ndarray,
    visible_mask: np.ndarray,
    predicted_amodal: np.ndarray,
    flux_output: Optional[np.ndarray] = None,
    final_silhouette: Optional[np.ndarray] = None,
    gt_mask_path: Optional[Path] = None,
) -> dict:
    """Compute the four standard metrics + auxiliary stats and write
    ``metrics.json`` into ``out_dir``.

    Parameters
    ----------
    out_dir          where to write metrics.json
    original_image   input image (BGR uint8, HxWx3)
    visible_mask     binary HxW — visible portion of subject
    predicted_amodal binary HxW — pipeline's predicted amodal silhouette
    flux_output      Flux-completed BGR image (HxWx3) or None
    final_silhouette SAM3 PostSeg result on Flux output (HxW) or None
    gt_mask_path     optional path to ground-truth amodal mask PNG
    """
    t0 = time.time()
    metrics: dict = {
        "version":             "1",
        "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%S"),
        "visible_mask_px":     int((_ensure_binary(visible_mask) > 0).sum()),
        "predicted_amodal_px": int((_ensure_binary(predicted_amodal) > 0).sum()),
    }

    # Completion ratio: how much new silhouette was added beyond visible.
    vis_n = metrics["visible_mask_px"]
    amo_n = metrics["predicted_amodal_px"]
    metrics["completion_ratio"] = (amo_n - vis_n) / max(amo_n, 1)

    # ── 1 & 2. IoU + Boundary F1 ─────────────────────────────────────
    # Compare predicted amodal against ground truth (preferred) OR
    # against the SAM3 PostSeg result (self-consistency proxy).
    gt = None
    if gt_mask_path is not None and Path(gt_mask_path).exists():
        gt = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        if gt is not None and gt.shape != predicted_amodal.shape[:2]:
            gt = cv2.resize(gt, (predicted_amodal.shape[1], predicted_amodal.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    if gt is not None:
        metrics["iou_vs_gt"]         = round(iou(predicted_amodal, gt), 4)
        metrics["boundary_f1_vs_gt"] = round(boundary_f1(predicted_amodal, gt, 3), 4)
    if final_silhouette is not None:
        metrics["iou_self_consistency"]         = round(iou(predicted_amodal, final_silhouette), 4)
        metrics["boundary_f1_self_consistency"] = round(boundary_f1(predicted_amodal, final_silhouette, 3), 4)

    # ── 3. LPIPS (perceptual) ────────────────────────────────────────
    if flux_output is not None:
        # 3a. Visible-region drift — pixel-copied, so should be ~0.
        #     If it's high, the visible mask boundary is broken.
        vis = lpips_score(original_image, flux_output, mask=visible_mask)
        metrics["lpips_visible_drift"] = round(vis["lpips"], 4) if vis["lpips"] is not None else None
        metrics["mse_visible_drift"]   = round(vis["mse"], 2)

        # 3b. Full-image — measures how much the Flux pass changed the scene.
        full = lpips_score(original_image, flux_output, mask=None)
        metrics["lpips_full"] = round(full["lpips"], 4) if full["lpips"] is not None else None
        metrics["mse_full"]   = round(full["mse"], 2)

    # ── 4. FID ──────────────────────────────────────────────────────
    # FID is a distribution-level metric — it needs N≥1 000 samples and
    # a reference distribution. Per-image FID isn't meaningful.
    metrics["fid"] = None
    metrics["fid_note"] = ("FID requires a batch of generated + reference images. "
                          "Run a separate batch_eval script over multiple runs.")

    metrics["compute_time_s"] = round(time.time() - t0, 2)

    out_path = Path(out_dir) / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [metrics] wrote {out_path.name}  iou_sc={metrics.get('iou_self_consistency','—')}  "
          f"f1_sc={metrics.get('boundary_f1_self_consistency','—')}  "
          f"lpips_full={metrics.get('lpips_full','—')}")
    return metrics
