"""Occluder-mask candidate fusion, and head-noun extraction used to turn a
long GPT description ("large pale weathered wooden log/stump in front of
the bear") into a short SAM3-text-friendly label ("log").
"""
import re
from typing import Optional

import cv2
import numpy as np

# ── Class-label noun extraction (used by the SAM3-text fallback prompts) ─────
# GPT often returns long descriptive class names like "large pale weathered
# wooden log/stump in front of the bear".  CLIP text-image scoring works far
# better with 1–3 word labels ("a log", "wood"), so we strip qualifiers down
# to the head noun(s).

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
