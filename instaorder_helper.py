"""InstaOrder occlusion-order helper for test3.

Bridges to the InstaOrderNet_od model living in
amodal_k8xu/InstaOrder/ without pulling in inference.py's
skimage/sklearn/midas deps. Only the pairwise occlusion-order forward pass
is needed: given a list of candidate-occluder segment masks and a visible
(target) mask, return the union of segments InstaOrder predicts as
*above* the target. This is the only signal that resolves same-class
occlusion (zebra-on-zebra, cat-on-cat) which CLIP-text scoring cannot.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn


def _stub_optional_deps() -> None:
    """Install minimal stubs for the optional InstaOrder visualisation deps
    so `import utils` (which re-exports visualize_utils) doesn't crash on
    skimage / sklearn / matplotlib being absent.  None of these are needed
    for the pairwise occlusion-order inference we actually run.
    """
    for mod_name in (
        "skimage", "skimage.morphology", "skimage.io", "skimage.draw",
        "skimage.measure",
        "sklearn", "sklearn.metrics",
        "matplotlib", "matplotlib.pyplot",
        "pycocotools", "pycocotools.mask", "pycocotools.coco",
        "pycocotools.cocoeval",
    ):
        if mod_name in sys.modules:
            continue
        m = types.ModuleType(mod_name)
        m.__spec__ = types.SimpleNamespace(
            name=mod_name, loader=None, origin="stub", submodule_search_locations=None,
            cached=None, parent=mod_name.rpartition(".")[0] or None, has_location=False,
        )
        m.__path__ = []  # mark as package so 'from X import Y' works
        # Populate the symbols InstaOrder actually pulls.
        if mod_name == "skimage.morphology":
            m.convex_hull = types.SimpleNamespace(  # used only in inference.py
                convex_hull_image=lambda x: x,
            )
        if mod_name == "skimage.draw":
            m.polygon2mask = lambda shape, polygon: np.zeros(shape, dtype=bool)
        if mod_name == "skimage.io":
            m.imread = lambda *a, **k: None
            m.imsave = lambda *a, **k: None
        if mod_name == "sklearn.metrics":
            m.precision_score = lambda *a, **k: 0.0
            m.recall_score = lambda *a, **k: 0.0
            m.f1_score = lambda *a, **k: 0.0
        if mod_name == "matplotlib.pyplot":
            m.figure = lambda *a, **k: None
            m.imshow = lambda *a, **k: None
            m.savefig = lambda *a, **k: None
            m.close = lambda *a, **k: None
        sys.modules[mod_name] = m

# ImageNet mean/std (matches InstaOrder/utils/data_utils.py data_mean / data_std).
_DATA_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DATA_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _ensure_sys_path(repo_dir: str) -> None:
    """Add the InstaOrder repo to sys.path AFTER existing entries so its
    generic-named modules (`inference`, `utils`, `models`) don't shadow
    same-named modules from other repos in the same Python process.  Most
    notably, pix2gestalt has its own `inference.py` with
    `load_model_from_config` — putting InstaOrder first broke that import.
    """
    repo = str(Path(repo_dir).expanduser().resolve())
    if repo not in sys.path:
        sys.path.append(repo)


def _cleanup_instaorder_module_shadows() -> None:
    """Remove generic-named InstaOrder modules from sys.modules cache so a
    later `from inference import …` from a different repo (e.g. pix2gestalt)
    re-imports from its OWN path rather than re-using InstaOrder's loaded
    module."""
    for mod_name in list(sys.modules):
        if mod_name in ("inference", "models", "utils") or mod_name.startswith(
                ("inference.", "models.", "utils.")):
            mod = sys.modules[mod_name]
            mod_file = getattr(mod, "__file__", "") or ""
            if "InstaOrder" in mod_file:
                del sys.modules[mod_name]


def _transform_image(image_rgb: np.ndarray, input_size: int) -> torch.Tensor:
    """ImageNet-normalised, (1, 3, S, S) float32 tensor on CUDA."""
    resized = cv2.resize(image_rgb, (input_size, input_size),
                         interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    resized = (resized - _DATA_MEAN) / _DATA_STD
    tensor  = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float()
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def _resize_mask(mask: np.ndarray, input_size: int) -> torch.Tensor:
    """Nearest-resize a binary mask to (1, 1, S, S) on the same device as image."""
    m = (mask > 0).astype(np.float32)
    m = cv2.resize(m, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).float()
    if torch.cuda.is_available():
        t = t.cuda()
    return t


_INSTAORDER_PARAMS = {
    "algo": "InstaOrderNet_od",
    "total_iter": 60000,
    "lr_steps": [32000, 48000],
    "lr_mults": [0.1, 0.1],
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "optim": "SGD",
    "warmup_lr": [],
    "warmup_steps": [],
    "use_rgb": True,
    "backbone_arch": "resnet50_cls",
    "backbone_param": {"in_channels": 5, "num_classes": [2, 3]},
    "overlap_weight": 0.1,
    "distinct_weight": 0.9,
}


_MODEL_SINGLETON = None  # cached so a session reuses the loaded weights.


def get_instaorder_model(repo_dir: str, ckpt_path: str):
    """Lazy-load and cache the InstaOrderNet_od model. Returns None on failure."""
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is not None:
        return _MODEL_SINGLETON

    if not Path(ckpt_path).exists():
        print(f"  [InstaOrder] ckpt missing: {ckpt_path}")
        return None

    _ensure_sys_path(repo_dir)
    _stub_optional_deps()
    try:
        import models  # InstaOrder/models/__init__.py  exposes InstaOrderNet_od
    except Exception as exc:                                # noqa: BLE001
        print(f"  [InstaOrder] import models failed: {exc!r}")
        return None

    try:
        m = models.__dict__["InstaOrderNet_od"](_INSTAORDER_PARAMS)
        m.load_state(ckpt_path)
        m.switch_to("eval")
    except Exception as exc:                                # noqa: BLE001
        print(f"  [InstaOrder] model load failed: {exc!r}")
        return None

    _MODEL_SINGLETON = m
    print(f"  [InstaOrder] Loaded {Path(ckpt_path).name}")
    # Now that InstaOrder is in-memory, drop its generic-named modules from
    # sys.modules so a later `from inference import …` in a different repo
    # (e.g. pix2gestalt) finds its OWN inference.py rather than InstaOrder's.
    _cleanup_instaorder_module_shadows()
    return m


def _net_forward_occ(model, image_t: torch.Tensor,
                     modal_a: torch.Tensor, modal_b: torch.Tensor) -> tuple:
    """Return (prob_a_over_b, prob_b_over_a) ∈ [0,1].

    Mirrors InstaOrder/inference.py:net_forward_occ_depth but skips the depth
    head and averages both (A,B) and (B,A) input orderings for symmetry.
    """
    with torch.no_grad():
        occ_ab, _ = model.model(torch.cat([modal_a, modal_b, image_t], dim=1))
        occ_ba, _ = model.model(torch.cat([modal_b, modal_a, image_t], dim=1))
        # nn.functional.sigmoid is deprecated → torch.sigmoid
        occ_ab = torch.sigmoid(occ_ab)
        occ_ba = torch.sigmoid(occ_ba)
        prob_a_over_b = ((occ_ab[:, 1] + occ_ba[:, 0]) / 2).item()
        prob_b_over_a = ((occ_ab[:, 0] + occ_ba[:, 1]) / 2).item()
    return prob_a_over_b, prob_b_over_a


def rank_by_frontness(
    model,
    image_rgb: np.ndarray,
    masks: list,
    input_size: int = 384,
) -> list:
    """Pairwise rank a list of binary masks by how 'in front' each one is.

    For every pair (i, j) we query InstaOrder for P(i over j) and P(j over i)
    and accumulate a per-mask 'front score' = sum_j P(i over j).  Higher
    score = more in front.

    Inputs:
      masks  — list of H×W binary masks (uint8 or bool); each represents a
               candidate instance silhouette.

    Returns a list of dicts sorted descending by front_score:
      [{ 'index': i, 'front_score': float, 'over_counts': int }, ...]
    """
    if model is None or len(masks) < 2:
        return [{"index": i, "front_score": 0.0, "over_counts": 0}
                for i in range(len(masks))]

    H, W = masks[0].shape[:2]
    if image_rgb.shape[:2] != (H, W):
        image_rgb = cv2.resize(image_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    image_t = _transform_image(image_rgb, input_size)

    n = len(masks)
    front_score = [0.0] * n
    over_counts = [0]   * n

    # Pre-resize all masks to the input grid once.
    mask_tensors = [_resize_mask(m, input_size) for m in masks]

    for i in range(n):
        for j in range(i + 1, n):
            p_i_over_j, p_j_over_i = _net_forward_occ(
                model, image_t, mask_tensors[i], mask_tensors[j])
            front_score[i] += p_i_over_j
            front_score[j] += p_j_over_i
            if p_i_over_j > 0.5:
                over_counts[i] += 1
            if p_j_over_i > 0.5:
                over_counts[j] += 1

    ranked = [
        {"index": i, "front_score": front_score[i], "over_counts": over_counts[i]}
        for i in range(n)
    ]
    ranked.sort(key=lambda d: d["front_score"], reverse=True)
    return ranked


def occluders_above(
    model,
    image_rgb: np.ndarray,
    target_mask: np.ndarray,
    candidate_masks: list,
    input_size: int = 384,
    occ_prob_thresh: float = 0.5,
) -> Optional[np.ndarray]:
    """Return the union of candidate masks predicted to be *above* target_mask.

    image_rgb        : H×W×3 uint8 RGB.
    target_mask      : H×W binary, the query (visible subject) mask.
    candidate_masks  : list of (name: str, H×W binary mask) tuples — each is a
                       segment that might be in front of the target.
    Returns a H×W uint8 0/1 mask (union of above-target candidates), or None
    if model is None, target is empty, or no candidate is judged above.
    """
    if model is None:
        return None
    if target_mask is None or int(target_mask.sum()) == 0:
        return None

    H, W = target_mask.shape[:2]
    if image_rgb.shape[:2] != (H, W):
        image_rgb = cv2.resize(image_rgb, (W, H), interpolation=cv2.INTER_LINEAR)

    image_t  = _transform_image(image_rgb, input_size)
    target_t = _resize_mask(target_mask, input_size)

    union = np.zeros((H, W), dtype=np.uint8)
    n_above = 0
    for name, cand in candidate_masks:
        if cand is None or int((cand > 0).sum()) == 0:
            continue
        if cand.shape[:2] != (H, W):
            cand = cv2.resize((cand > 0).astype(np.uint8), (W, H),
                              interpolation=cv2.INTER_NEAREST)
        # Skip candidates that almost entirely overlap the target — they ARE
        # the target, not occluders. (>= 80% of candidate inside target.)
        overlap = float(((cand > 0) & (target_mask > 0)).sum())
        if overlap / max(float((cand > 0).sum()), 1.0) >= 0.80:
            continue

        cand_t = _resize_mask(cand, input_size)
        p_cand_over_target, _ = _net_forward_occ(model, image_t, cand_t, target_t)
        if p_cand_over_target > occ_prob_thresh:
            union[(cand > 0)] = 1
            n_above += 1
    if n_above == 0:
        return None
    print(f"  [InstaOrder] {n_above} candidate(s) judged above target → "
          f"{int(union.sum())} px")
    return union
