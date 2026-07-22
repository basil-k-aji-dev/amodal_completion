"""Shared state type and GPT structured-output (JSON-schema) definitions
used by Agent 1 (occlusion_agent) and Agent 3 (the reviewer/retry loop).
"""
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



OCCLUSION_SCHEMA = {
    "name": "occlusion_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "occluded_object": {"type": "string"},
            "occluder":        {"type": "string"},
            "what_to_remove":  {"type": "string"},
            # Rich description used by the reviewer to verify correctness
            "subject_description": {"type": "string"},
            "visible_parts":       {"type": "string"},
            "missing_parts":       {"type": "string"},
            # Frame-crop completion fields
            "frame_cropped": {"type": "boolean"},
            "expansion_directions": {
                "type": "array",
                "items": {"type": "string"},
            },
            "expansion_pixels": {
                "type": "object",
                "properties": {
                    "top":    {"type": "integer"},
                    "bottom": {"type": "integer"},
                    "left":   {"type": "integer"},
                    "right":  {"type": "integer"},
                },
                "required": ["top", "bottom", "left", "right"],
                "additionalProperties": False,
            },
            # Classic occlusion fields
            "selected_segment_ids": {
                "type": "array",
                "items": {"type": "integer"}
            },
            "visible_segment_ids": {
                "type": "array",
                "items": {"type": "integer"}
            },
            "polygon_override": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}}
            },
            "visible_polygon_override": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}}
            },
            "hidden_region": {
                "type": "object",
                "properties": {
                    "x1":          {"type": "integer"},
                    "y1":          {"type": "integer"},
                    "x2":          {"type": "integer"},
                    "y2":          {"type": "integer"},
                    "description": {"type": "string"},
                },
                "required": ["x1", "y1", "x2", "y2", "description"],
                "additionalProperties": False,
            },
            "boundary_expansion": {"type": "integer"},
            # Fix 2: precise hidden-region polygon and SAM3 point-prompt coordinates
            "hidden_polygon": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
            },
            "occluder_click": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
                "additionalProperties": False,
            },
            "subject_click": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
                "additionalProperties": False,
            },
        },
        "required": [
            "occluded_object", "occluder", "what_to_remove",
            "subject_description", "visible_parts", "missing_parts",
            "frame_cropped", "expansion_directions", "expansion_pixels",
            "selected_segment_ids", "visible_segment_ids",
            "polygon_override", "visible_polygon_override", "hidden_region", "boundary_expansion",
            "hidden_polygon", "occluder_click", "subject_click",
        ],
        "additionalProperties": False,
    },
}

REVIEWER_SCHEMA = {
    "name": "review_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "score":                    {"type": "number"},
            "feedback":                 {"type": "string"},
            "failure_code":             {"type": "string"},
            "improved_prompt":          {"type": "string"},
            "improved_negative_prompt": {"type": "string"},
        },
        "required": [
            "score", "feedback", "failure_code",
            "improved_prompt", "improved_negative_prompt",
        ],
        "additionalProperties": False,
    },
}

_GPT_AMODAL_SCHEMA = {
    "name": "amodal_silhouette",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "full_subject_polygon": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "minItems": 8,
                "maxItems": 60,
            },
            "rationale": {"type": "string"},
        },
        "required": ["full_subject_polygon", "rationale"],
        "additionalProperties": False,
    },
}


