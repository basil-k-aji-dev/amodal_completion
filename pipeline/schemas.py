"""
pipeline/schemas.py — strict JSON schemas for the GPT calls used by the
occlusion / amodal-review / mask-review / reviewer nodes.
"""

from __future__ import annotations


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


# ── GPT vision call (Responses API + reasoning=xhigh + prompt caching) ────────
#
# OpenAI Responses API replaces chat.completions for reasoning models. Key
# upgrades vs. the old call site:
#   • `reasoning.effort = "high"` — the maximum the server-side `gpt-5`
#     model accepts (the SDK enum lists `xhigh` too but that's rejected by
#     gpt-5 with a 400; reserve xhigh for gpt-5-codex / 5.1+ if upgraded).
#     Gives Agent 1 / amodal-completion / reviewer more chain-of-thought
#     budget so the polygon predictions are less likely to hallucinate
#     anatomy in the wrong spatial direction.
#   • `prompt_cache_key` — static prompt caching. Each logical call site
#     should pass a stable key so the long instruction prefix gets cached
#     across images. Retention set to 24h.
#   • `text.format` with JSON-schema strict mode — same effective contract
#     as the old `response_format={"type": "json_schema", ...}` but using
#     the flat Responses-API shape.
#   • Image content items use {type: "input_image", image_url, detail}.



_MASK_REVIEW_SCHEMA = {
    "name": "mask_verification",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["ok", "rationale", "corrective_click_x", "corrective_click_y"],
        "properties": {
            "ok":                 {"type": "boolean"},
            "rationale":          {"type": "string"},
            "corrective_click_x": {"type": "integer"},
            "corrective_click_y": {"type": "integer"},
        },
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


_GPT_REVIEW_SCHEMA = {
    "name": "amodal_review",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["ok", "redraw"]},
            "rationale": {"type": "string"},
            "corrected_polygon": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "minItems": 0,
                "maxItems": 60,
            },
        },
        "required": ["verdict", "rationale", "corrected_polygon"],
        "additionalProperties": False,
    },
}


