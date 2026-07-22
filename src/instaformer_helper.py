"""InstaFormer holistic occlusion+depth-order helper.

InstaFormer (github.com/SNU-VGILab/InstaOrder, NeurIPS 2025) predicts
occlusion AND depth order for every instance in a scene in a single forward
pass, replacing the older pairwise InstaOrderNet approach this project used
before (see instaorder_helper.py, now unused).

It needs Python 3.8 + Detectron2 + a pinned torch/CUDA stack that conflicts
with this project's own environment, so it runs as a subprocess in its own
isolated venv (amodal_completion/.instaformer-venv/). Only plain numpy
arrays and dicts cross the process boundary — no torch/Detectron2 objects.
"""

from __future__ import annotations

import pickle
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def run_instaformer(
    image_path: str,
    out_dir: Path,
    venv_python: str,
    repo_dir: str,
    config_file: str,
    ckpt_path: str,
    timeout: int = 240,
) -> Optional[dict]:
    """Run InstaFormer on one image. Returns None on any failure.

    Returns:
      {
        'segments':  [{'id', 'category_id', 'isthing', 'area', 'mask' np.uint8 0/1}, ...],
        'occlusion': np.ndarray (N, N) int64, -1 on the diagonal,
        'depth':     np.ndarray (N, N) int64, -1 on the diagonal,
      }
    """
    stem = Path(image_path).stem
    work_dir = out_dir / "_instaformer"
    (work_dir / "prediction").mkdir(parents=True, exist_ok=True)
    (work_dir / "segmentation").mkdir(parents=True, exist_ok=True)

    pkl_path = work_dir / "prediction" / f"{stem}_od.pkl"
    cmd = [
        venv_python, "demo/demo.py",
        "--config-file", config_file,
        "--input", image_path,
        "--output", str(work_dir),
        "--opts",
        "MODEL.WEIGHTS", ckpt_path,
        "MODEL.DEVICE", "cuda",
        "TEST.OCCLUSION_EVALUATION", "False",
        "TEST.DEPTH_EVALUATION", "False",
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=repo_dir, capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            print(f"  [InstaFormer] subprocess failed (code={proc.returncode}): "
                  f"{proc.stderr[-2000:]}")
            return None
    except Exception as exc:                                      # noqa: BLE001
        print(f"  [InstaFormer] subprocess error: {exc!r}")
        return None

    if not pkl_path.exists():
        print(f"  [InstaFormer] no prediction file produced: {pkl_path}")
        return None

    try:
        with open(pkl_path, "rb") as f:
            preds = pickle.load(f)
        seg_map, segments_info = preds["panoptic_seg"]
        segments = []
        for info in segments_info:
            mask = (seg_map == info["id"]).astype(np.uint8)
            segments.append({**info, "mask": mask})
        print(f"  [InstaFormer] {len(segments)} segment(s), "
              f"occlusion/depth matrices: "
              f"{preds.get('occlusion') is not None}/{preds.get('depth') is not None}")
        return {
            "segments": segments,
            "occlusion": preds.get("occlusion"),
            "depth": preds.get("depth"),
        }
    except Exception as exc:                                      # noqa: BLE001
        print(f"  [InstaFormer] failed to parse predictions: {exc!r}")
        return None


def best_matching_segment(
    result: dict,
    target_mask: Optional[np.ndarray] = None,
    click_xy: Optional[tuple] = None,
    min_iou: float = 0.05,
) -> Optional[int]:
    """Find the segment index best matching a known target: a click point
    takes priority (point-containment), falling back to best IoU against
    target_mask. Returns None if nothing matches well enough."""
    segments = result["segments"]
    if not segments:
        return None
    if click_xy is not None:
        x, y = int(click_xy[0]), int(click_xy[1])
        for i, s in enumerate(segments):
            m = s["mask"]
            if 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and m[y, x] > 0:
                return i
    if target_mask is not None and target_mask.sum() > 0:
        best_i, best_iou = None, 0.0
        for i, s in enumerate(segments):
            m = s["mask"]
            if m.shape != target_mask.shape:
                m = cv2.resize(m, (target_mask.shape[1], target_mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            inter = int((m.astype(bool) & target_mask.astype(bool)).sum())
            union = int((m.astype(bool) | target_mask.astype(bool)).sum())
            iou = inter / max(union, 1)
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_i is not None and best_iou >= min_iou:
            return best_i
    return None


def occluders_above(
    result: dict,
    target_mask: Optional[np.ndarray] = None,
    click_xy: Optional[tuple] = None,
    adjacency_px: int = 15,
) -> Optional[np.ndarray]:
    """Union of segments InstaFormer judges to be in an occlusion relation
    with the target. Returns None if the target can't be matched to a
    segment or no occluders are found.

    Note on directionality: the occlusion matrix's row/column convention
    ("i occludes j" vs "j occludes i") is not fully disambiguated from a
    single model version, and adjacent instances often occlude each other
    mutually along different parts of their silhouettes anyway. We treat
    occ[target, j] == 1 OR occ[j, target] == 1 as "these two instances are
    in an occlusion relation" — sufficient to build the occluder mask
    regardless of exact row/column direction.

    Adjacency filter: InstaFormer's learned occlusion matrix can produce
    false positives between objects that are merely nearby (e.g. two
    items sitting on the same desk) but don't actually overlap the
    target's silhouette. A candidate is only accepted if its mask
    actually touches the target's mask (dilated by `adjacency_px`) —
    a real occluder must physically cover part of the target.
    """
    target_idx = best_matching_segment(result, target_mask, click_xy)
    if target_idx is None:
        return None
    occ = result.get("occlusion")
    if occ is None:
        return None
    segments = result["segments"]
    target_seg_mask = segments[target_idx]["mask"]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (adjacency_px * 2 + 1, adjacency_px * 2 + 1))
    target_dilated = cv2.dilate(target_seg_mask.astype(np.uint8), kernel, iterations=1)
    union = None
    for j, seg in enumerate(segments):
        if j == target_idx or not seg.get("isthing", False):
            # "stuff" (background/wall/sky/...) segments are context, not
            # occluders — only discrete "thing" instances can occlude.
            continue
        if occ[target_idx, j] != 1 and occ[j, target_idx] != 1:
            continue
        m = seg["mask"]
        touches = int((m.astype(bool) & target_dilated.astype(bool)).sum()) > 0
        if not touches:
            continue
        union = m.copy() if union is None else np.clip(union | m, 0, 1).astype(np.uint8)
    return union


def same_class_split(result: dict, target_idx: int) -> Optional[tuple]:
    """When another segment shares the target's category_id (same-class
    occlusion — zebra-on-zebra, cat-on-cat), rank the target + its
    same-class siblings by frontness and return (front_idx, back_idx).

    This replaces the old PSALM-class-split + connected-components +
    SAM3-dual-click machinery: InstaFormer's own panoptic segmentation
    already separates same-class instances into distinct segment ids, and
    its occlusion matrix already ranks them — no extra models needed.

    Returns None if there's no same-class sibling or no occlusion matrix.
    """
    segments = result["segments"]
    occ = result.get("occlusion")
    if occ is None:
        return None
    target_cat = segments[target_idx]["category_id"]
    siblings = [i for i, s in enumerate(segments)
                if i != target_idx and s["category_id"] == target_cat]
    if not siblings:
        return None
    group = [target_idx] + siblings
    scores = {gi: sum(1 for gj in group if gi != gj and occ[gi, gj] == 1)
              for gi in group}
    ranked = sorted(group, key=lambda i: scores[i], reverse=True)
    front_idx = ranked[0]
    back_idx = target_idx if front_idx != target_idx else ranked[1]
    return front_idx, back_idx
