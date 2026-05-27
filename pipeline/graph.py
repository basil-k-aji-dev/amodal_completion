"""
pipeline/graph.py — LangGraph runner for the amodal completion pipeline.

Graph:
    occlusion_agent → completion_agent → reviewer
          ↑                  ↑               │
          │ (retry_mask)     └─(retry_completion)─┤
          └─────────────────────────────────────  ↓
                                                  END

route():
    score ≥ SCORE_THRESHOLD            → end
    attempt ≥ MAX_RETRIES              → end (keep best)
    MASK_* failure (retries available) → retry_mask  (re-run occlusion_agent)
    any other non-accept failure       → retry_completion (re-run Flux, new seed)

run(image_path, target) builds the initial State, invokes the compiled graph,
and prints a short summary.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, StateGraph

import config
from runtime import BASE_DIR
from pipeline.state import State
from pipeline.nodes import occlusion_agent, completion_agent, reviewer


def route(state: State) -> str:
    score        = state.get("review_score", 0.0)
    failure_code = state.get("failure_code", "")
    attempt      = state.get("attempt", 0)
    mask_retries = state.get("mask_retry_count", 0)

    if score >= config.SCORE_THRESHOLD:
        print(f"\n✓ Accepted — score {score:.1f}")
        return "end"

    if attempt >= config.MAX_RETRIES:
        best = state.get("best_attempt", attempt)
        print(f"\n✗ Max retries ({config.MAX_RETRIES}) reached. Best: attempt {best} "
              f"(score {state.get('best_score', score):.1f})")
        return "end"

    if failure_code in config.MASK_FAILURE_CODES and mask_retries < config.MAX_MASK_RETRIES:
        print(f"\n↻ Mask failure ({failure_code}) — re-running occlusion_agent "
              f"(mask retry {mask_retries + 1}/{config.MAX_MASK_RETRIES})")
        return "retry_mask"

    print(f"\n↻ Re-running completion (new Flux seed) — score {score:.1f} "
          f"< {config.SCORE_THRESHOLD}  [{failure_code}]")
    return "retry_completion"


def build_graph():
    g = StateGraph(State)

    g.add_node("occlusion_agent",  occlusion_agent)
    g.add_node("completion_agent", completion_agent)
    g.add_node("reviewer",         reviewer)

    g.set_entry_point("occlusion_agent")
    g.add_edge("occlusion_agent",  "completion_agent")
    g.add_edge("completion_agent", "reviewer")
    g.add_conditional_edges(
        "reviewer",
        route,
        {
            "retry_mask":       "occlusion_agent",
            "retry_completion": "completion_agent",
            "end":              END,
        },
    )

    return g.compile()


def _initial_state(image_path: str, target: str) -> State:
    return {
        "image_path":            image_path,
        "target":                target,
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


def run(image_path: str, target: str = "") -> dict:
    """Run the full pipeline on one image. Returns the final State dict."""
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"image not found: {img_path}")

    target = target or getattr(config, "TARGET", "")
    pipeline = build_graph()
    final = pipeline.invoke(_initial_state(str(img_path), target))

    stem         = img_path.stem
    best_attempt = final.get("best_attempt", final.get("attempt", 0))
    best_score   = final.get("best_score",   final.get("review_score", 0.0))
    print(f"\n{'=' * 52}")
    print(f"  Image       : {stem}")
    print(f"  Output      : {final.get('output_path', '')}")
    print(f"  RGBA        : {final.get('output_rgba_path', '')}")
    print(f"  Best score  : {best_score:.1f}/10  (attempt {best_attempt})")
    print(f"  Review log  : {BASE_DIR / 'output' / stem / 'review_log.json'}")
    return final


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else config.IMAGE_PATH
    run(img, getattr(config, "TARGET", ""))
