"""
pipeline/state.py — the LangGraph State shared across all nodes.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class State(TypedDict):
    image_path:         str
    target:             str           # optional user hint — empty = auto-detect
    # Occlusion Agent outputs
    occluded_object:    str
    occluder:           str
    what_to_remove:     str
    bbox:               Optional[list]
    boundary_expansion: int
    region_desc:        str
    subject_description: str          # rich description of visible subject (species, colors, posture)
    visible_parts:      str           # body parts currently visible in frame
    missing_parts:      str           # body parts cut off / expected but absent
    frame_cropped:      bool          # True when subject is cut off at image boundary
    expansion_directions: list        # e.g. ["bottom"] or ["right", "bottom"]
    expansion_pixels:   Optional[dict]  # {top, bottom, left, right} pixels to add
    mask_path:              Optional[str]   # binary occluder mask
    visible_mask_path:      Optional[str]   # binary modal mask (visible part of occluded obj)
    hidden_mask_path:       Optional[str]   # binary amodal target region (missing part to generate)
    occluder_removed_path:  Optional[str]
    occluder_viz_path:      Optional[str]
    hidden_polygon:         Optional[list]   # Fix 2: exact polygon of hidden region to inpaint
    # Inpainting Agent outputs
    pix2gestalt_dir:    Optional[str]   # dir holding completed_N.png samples
    output_path:        str
    output_rgba_path:   str
    # Reviewer
    review_score:       float
    review_feedback:    str
    failure_code:       str
    attempt:            int
    mask_retry_count:   int
    best_attempt:       int
    best_score:         float
