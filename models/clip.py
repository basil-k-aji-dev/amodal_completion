"""
models/clip.py — OpenAI CLIP grounding helpers.

Lazy CLIP loader plus the segment-labelling, grid-localisation, mask
verification, and short-label extraction helpers used to ground occluder /
subject masks against text.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import config
from runtime import DEVICE, _release_ram, register_free_fn

# ── Lazy singletons ─────────────────────────────────────────────────────────
_CLIP_MODEL      = None
_CLIP_PREPROCESS = None


_QUALIFIER_WORDS = {
    "large", "small", "big", "huge", "tiny", "short", "long", "tall",
    "thin", "thick", "wide", "narrow", "round", "rounded", "square",
    "pale", "dark", "light", "bright", "dull", "weathered", "old", "new",
    "clean", "dirty", "rough", "smooth", "shiny", "matte",
    "horizontal", "vertical", "front", "back", "side", "main", "primary",
    "foreground", "background", "central", "centre", "center",
    # colours
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "black", "white", "grey", "gray", "brown", "tan", "beige", "cream",
    "golden", "silver", "amber", "ivory",
    # texture words
    "wooden", "metallic", "plastic", "glass", "leather", "fabric",
    "soft", "hard", "fluffy", "shaggy",
    # filler
    "the", "a", "an", "of", "in", "on", "with", "and", "or",
    "very", "quite", "rather", "somewhat",
}

_PHRASE_BREAKERS = re.compile(
    r"\b(?:"
    r"in front of|behind|covered by|partially|"
    r"running (?:across|along|through)|"
    r"lying (?:on|under|across)|"
    r"draped (?:over|across)|"
    r"resting (?:on|against)|"
    r"protruding|extending|sticking|standing|"
    r"covering|blocking|hiding|"
    r"across the|along the|near the"
    r")\b",
    re.I,
)


def _extract_short_label(phrase: str) -> str:
    """Reduce a long descriptive phrase to its head noun(s).

    Examples:
      "large pale weathered wooden log/stump in front of the bear" → "log"
      "horizontal driftwood log running across the bottom"        → "driftwood log"
      "brown bear (Ursus arctos)"                                  → "brown bear"
    """
    if not phrase:
        return ""
    # Drop parenthesised qualifiers like "(Ursus arctos)"
    s = re.sub(r"\([^)]*\)", "", phrase).strip()
    # Cut at relational phrase breakers ("in front of …" etc.)
    s = _PHRASE_BREAKERS.split(s)[0].strip().rstrip(",")
    # Slash-separated alternatives → take the first option ("log/stump" → "log")
    s = s.split("/")[0]
    # Tokenise + drop qualifiers
    tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z\-]*", s)
              if t.lower() not in _QUALIFIER_WORDS]
    if not tokens:
        # Fallback: keep last 1-2 words of the original phrase
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]*", phrase)[-2:]
    # Keep at most the last 2 head words ("driftwood log" or just "log")
    return " ".join(tokens[-2:]).lower()


# ── CLIP — vision-grounded segment labeling (CVPR'25 amodal-style) ────────────

def _get_clip():
    """Lazy-load OpenAI CLIP. Falls back to (None, None) if the package isn't
    importable so the rest of the pipeline can still run."""
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS
    try:
        import clip as openai_clip
    except ImportError:
        print("  [CLIP] openai/CLIP not installed — skipping CLIP segment labeling")
        return None, None
    print(f"  [CLIP] Loading {config.CLIP_MODEL_NAME}…")
    _release_ram()
    try:
        _CLIP_MODEL, _CLIP_PREPROCESS = openai_clip.load(
            config.CLIP_MODEL_NAME, device=DEVICE,
        )
        _CLIP_MODEL.eval()
    except Exception as exc:
        print(f"  [CLIP] Load failed (non-fatal): {exc}")
        _CLIP_MODEL = None
        _CLIP_PREPROCESS = None
    return _CLIP_MODEL, _CLIP_PREPROCESS


def _free_clip():
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if _CLIP_MODEL is not None:
        try:
            _CLIP_MODEL.cpu()
        except Exception:
            pass
        _CLIP_MODEL = None
        _CLIP_PREPROCESS = None
        _release_ram()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("  [CLIP] Released")


def _clip_label_segments(
    image_rgb: np.ndarray,
    segments: list,
    text_labels: list,
) -> list:
    """For each SAM3 segment: crop its bbox (with small padding), grey out the
    non-segment pixels, run CLIP image-text similarity against text_labels, and
    return the best-matching label per segment.

    Returns list of {"id": int, "label": str | None, "score": float, "scores": dict}.
    """
    try:
        import clip as openai_clip
    except ImportError:
        return [{"id": s["id"], "label": None, "score": 0.0, "scores": {}} for s in segments]

    model, preprocess = _get_clip()
    if model is None:
        return [{"id": s["id"], "label": None, "score": 0.0, "scores": {}} for s in segments]

    # Encode the text labels once
    prompts = [f"a photo of {t}" if t else "a photo" for t in text_labels]
    text_tokens = openai_clip.tokenize(prompts).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    h, w = image_rgb.shape[:2]
    out = []
    for seg in segments:
        x1, y1, x2, y2 = seg["bbox"]
        pad = max(4, int(0.05 * max(x2 - x1, y2 - y1)))
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad + 1), min(h, y2 + pad + 1)
        crop  = image_rgb[cy1:cy2, cx1:cx2].copy()
        smask = seg["mask"][cy1:cy2, cx1:cx2]
        # Grey out non-segment pixels so CLIP focuses on the segment content
        crop[smask == 0] = 128

        try:
            pil = Image.fromarray(crop)
            inp = preprocess(pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feats = model.encode_image(inp)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                sim   = (100.0 * feats @ text_features.T).softmax(dim=-1)[0]
            scores  = {text_labels[i]: float(sim[i].item()) for i in range(len(text_labels))}
            best_i  = int(sim.argmax().item())
            out.append({
                "id":     seg["id"],
                "label":  text_labels[best_i],
                "score":  float(sim[best_i].item()),
                "scores": scores,
            })
        except Exception as exc:
            print(f"  [CLIP] Segment {seg['id']} scoring failed: {exc}")
            out.append({"id": seg["id"], "label": None, "score": 0.0, "scores": {}})

    _free_clip()
    return out


def _clip_grid_locate(
    image_rgb: np.ndarray,
    text_label: str,
    grid_n: int = 16,
    patch_overlap: float = 0.25,
) -> "tuple[np.ndarray, list]":
    """
    Score every grid cell of the image with CLIP against `text_label` and the
    standard distractor classes.  Returns:
      heatmap : (grid_n, grid_n) float — softmax probability of `text_label`
      hits    : list of (row, col, score, cx, cy) sorted by score desc

    Used as a fallback when SAM3 auto-segment missed the occluder entirely:
    the highest-scoring grid cell becomes a SAM3 point-prompt anchor that
    forces SAM3 to produce a mask at that location.
    """
    try:
        import clip as openai_clip
    except ImportError:
        return np.zeros((grid_n, grid_n), dtype=np.float32), []

    model, preprocess = _get_clip()
    if model is None:
        return np.zeros((grid_n, grid_n), dtype=np.float32), []

    h, w = image_rgb.shape[:2]
    cell_h = h / grid_n
    cell_w = w / grid_n
    overlap_h = int(cell_h * patch_overlap)
    overlap_w = int(cell_w * patch_overlap)

    distractors = ["background", "wall", "sky", "ground", "animal fur"]
    prompts = [f"a photo of {text_label}"] + [f"a photo of {d}" for d in distractors]
    text_tokens = openai_clip.tokenize(prompts).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    heatmap = np.zeros((grid_n, grid_n), dtype=np.float32)
    hits    = []

    for r in range(grid_n):
        for c in range(grid_n):
            y1 = max(0, int(r * cell_h) - overlap_h)
            y2 = min(h, int((r + 1) * cell_h) + overlap_h)
            x1 = max(0, int(c * cell_w) - overlap_w)
            x2 = min(w, int((c + 1) * cell_w) + overlap_w)
            if y2 - y1 < 16 or x2 - x1 < 16:
                continue
            patch = image_rgb[y1:y2, x1:x2]
            try:
                pil = Image.fromarray(patch)
                inp = preprocess(pil).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    feats = model.encode_image(inp)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    sim   = (100.0 * feats @ text_features.T).softmax(dim=-1)[0]
                score = float(sim[0].item())
                heatmap[r, c] = score
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                hits.append((r, c, score, cx, cy))
            except Exception:
                continue

    hits.sort(key=lambda t: t[2], reverse=True)
    _free_clip()
    return heatmap, hits


def _save_clip_grid_viz(
    image_bgr: np.ndarray,
    heatmap: np.ndarray,
    out_path: Path,
) -> None:
    """Overlay a CLIP-on-grid heatmap on the image and save."""
    if heatmap.size == 0 or heatmap.max() <= 0:
        return
    h, w = image_bgr.shape[:2]
    norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-9)
    norm_8u = (norm * 255).astype(np.uint8)
    big = cv2.resize(norm_8u, (w, h), interpolation=cv2.INTER_NEAREST)
    color_map = cv2.applyColorMap(big, cv2.COLORMAP_JET)
    blend = cv2.addWeighted(image_bgr, 0.55, color_map, 0.45, 0)
    cv2.imwrite(str(out_path), blend)


# ── Grounding DINO (open-vocab text → bbox via HF transformers) ───────────────



def _save_clip_label_viz(
    image_bgr: np.ndarray,
    segments: list,
    seg_by_id: dict,
    seg_labels: list,
    target_class: str,
    occluder_class: str,
    out_path: Path,
) -> None:
    """Save a coloured visualisation of CLIP segment labels.
    target = green, occluder = red, background = grey, other = yellow."""
    palette = {
        target_class:    (0, 255, 0),
        occluder_class:  (0, 0, 255),
        "background":    (128, 128, 128),
        "other object":  (0, 200, 200),
    }
    viz     = image_bgr.copy()
    overlay = np.zeros_like(viz)
    for sl in seg_labels:
        sid = sl["id"]
        if sid not in seg_by_id or sl["label"] is None:
            continue
        colour = palette.get(sl["label"], (200, 200, 200))
        overlay[seg_by_id[sid]["mask"] == 1] = colour
    viz = cv2.addWeighted(viz, 0.55, overlay, 0.45, 0)
    for sl in seg_labels:
        sid = sl["id"]
        if sid not in seg_by_id:
            continue
        seg_mask = seg_by_id[sid]["mask"]
        M = cv2.moments(seg_mask)
        if M["m00"] <= 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        lbl = (sl["label"] or "?")[:10]
        text = f"{sid}:{lbl}({sl['score']:.2f})"
        cv2.putText(viz, text, (cx - 36, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(viz, text, (cx - 36, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), viz)




def _clip_verified_strict(image_bgr: np.ndarray, mask: np.ndarray, label: str) -> tuple[float, bool]:
    """CLIP gate for text-first masks: unavailable CLIP is a rejection."""
    score, passed = _clip_verify_mask(image_bgr, mask, label)
    return score, bool(passed and score == score)


# ── SAM3 segment-everything helper ────────────────────────────────────────────



def _clip_verify_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    target_class: str,
) -> tuple[float, bool]:
    """Score the masked region with CLIP against
    ``CLIP_VERIFY_PROMPT_TEMPLATE``. Returns (score, passed).

    ``passed`` is True iff USE_CLIP_VERIFY is on, CLIP is available, and
    score ≥ CLIP_VERIFY_THRESHOLD. If CLIP isn't installed the score is
    NaN and we return (nan, True) so the gate is effectively a no-op
    (fall-through to GPT-V), matching legacy behaviour.
    """
    if not bool(getattr(config, "USE_CLIP_VERIFY", False)):
        return float("nan"), True
    try:
        from metrics import clip_score_region                  # noqa: WPS433
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [CLIPVerify] metrics.clip_score_region unavailable: {exc!r}")
        return float("nan"), True
    tmpl = getattr(config, "CLIP_VERIFY_PROMPT_TEMPLATE", "a photo of a {target}")
    prompt = tmpl.format(target=target_class)
    model_name = getattr(config, "CLIP_MODEL_NAME", "ViT-B/32")
    score = clip_score_region(image_bgr, mask, prompt, model_name=model_name)
    if score != score:  # NaN
        print(f"  [CLIPVerify] score unavailable (CLIP not loaded) — falling through to GPT-V")
        return score, True
    thr = float(getattr(config, "CLIP_VERIFY_THRESHOLD", 0.20))
    passed = score >= thr
    verdict = "PASS" if passed else "FAIL"
    print(f"  [CLIPVerify] '{prompt}' score={score:.3f} thr={thr:.2f}  → {verdict}")
    return score, passed




register_free_fn("clip", _free_clip)
