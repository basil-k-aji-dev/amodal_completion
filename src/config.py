# ── Input ─────────────────────────────────────────────────────────────────────
IMAGE_PATH = "/home/basil-k-aji/Desktop/Workspace/RD/website/animal-8518802_640_deer.jpg"

# Optional hint telling Agent 1 what the occluded subject is (e.g. "horse").
# Default is empty — Agent 1 auto-detects the occluded object itself from
# the image. Only set this when you want to steer detection explicitly.
INPUT_PROMPT = "horse"

# Iterative off-frame canvas-extension loop (pad + outpaint when the subject
# touches a frame edge). Disabled for now — focus on getting the occluded-
# region generation right first; off-frame extension is a separate concern
# to revisit later.
USE_OFFFRAME_EXTENSION = False

# Agent 3 — GPT-vision reviewer/retry loop. Scores each Flux-Fill completion
# against the original photo and, on a low score, retries with GPT's own
# suggested prompt/negative-prompt fix (up to REVIEWER_MAX_RETRIES times).
USE_REVIEWER              = True
REVIEWER_SCORE_THRESHOLD  = 7.0   # score >= this = accepted, stop retrying
REVIEWER_MAX_RETRIES      = 2     # extra attempts beyond the first

# ── GPT (loaded from .env: OPENAI_MODEL, OPENAI_API_KEY) ─────────────────────
# Passed to the Responses API as max_output_tokens (see runtime.py:gpt_vision
# — previously defined here but never actually wired into the API call, so
# this had zero effect no matter how high it was set; now it's real).
# For a reasoning model with reasoning.effort="high", this budget covers
# BOTH the internal reasoning tokens AND the final visible output — with a
# low ceiling, "high" effort can burn the whole budget on reasoning and
# leave nothing for the actual JSON response (the empty-response failure
# this flag exists to prevent). It's a ceiling, not a target: the model
# stops when it's done, so raising this costs nothing in the common case
# and only matters for the tail case where reasoning legitimately needs
# more room — kept generous accordingly.
GPT_MAX_TOKENS = 32000

# ── SAM3 automatic segmentation ───────────────────────────────────────────────
SAM3_MODEL_ID        = "facebook/sam3"
SAM3_POINTS_PER_BATCH = 16     # points processed per forward pass
# Lowered from 0.75 → 0.50: the previous threshold was throwing away most of
# the image. Wood/log textures, walls, large flat occluders rarely score above
# 0.6 with SAM3 auto-segment; a stricter cut leaves the log unsegmented and
# the downstream CLIP labeller has nothing to label.
SAM3_SCORE_THRESH    = 0.50   # minimum mask confidence score to keep
SAM3_MIN_AREA        = 100    # minimum mask area in pixels (lowered from 200)

# pix2gestalt amodal completion was removed (permanently disabled via
# USE_PIX2GESTALT_AMODAL=False; superseded by direct Flux-Fill inpainting +
# a single GPT-V "draw the silhouette" call, _gpt_amodal_subject_mask).

# When False, skip amodal silhouette completion ENTIRELY. The "amodal mask"
# becomes the visible_mask directly — no extension into hidden / off-frame
# parts. Cutout shows just the visible subject on neutral gray. No Flux
# inpainting of hidden regions (since hidden = amodal - visible = 0).
# Use this when PSALM's visible segmentation is already sufficient and you
# don't want GPT-V over-extending the silhouette onto the occluder.
USE_AMODAL_COMPLETION  = False  # Jiang Ao-style: skip GPT polygon; Flux + post-SAM3 decide the silhouette

# ── GPU memory cap ────────────────────────────────────────────────────────────
# Hard limit on VRAM usage per process. Set to None to disable.
# 44.0 leaves ~4 GB headroom on a 48 GB card (RTX A6000) for other processes.
GPU_MEMORY_LIMIT_GB = 44.0

# LaMa background inpainting was removed (only ever called from the
# Mixed-Context-Diffusion-Sampling path and the ControlNet cascade, both
# permanently disabled/removed).

# ── ControlNet-Inpaint fallback was removed — FLUX.1-Fill-dev is the only
# inpainting backend and always will be; there is no fallback path.
# ── Flux-Fill specific config ─────────────────────────────────────────────────
FLUX_FILL_MODEL_ID         = "black-forest-labs/FLUX.1-Fill-dev"
FLUX_FILL_STEPS            = 30
FLUX_FILL_GUIDANCE_SCALE   = 30.0    # BFL's own documented default/example for FLUX.1-Fill-dev
                                     # specifically (diffusers pipeline_flux_fill.py's `guidance_scale`
                                     # default + EXAMPLE_DOC_STRING both use 30). Was 45.0 — above the
                                     # tool's own recommended value with no evidence it helped; the
                                     # DUPLICATE_SUBJECT/background-replacement failures that motivated
                                     # raising it were traced to the prompt wording, not guidance scale.
FLUX_FILL_MAX_SEQUENCE_LEN = 512
FLUX_FILL_CPU_OFFLOAD      = False   # enable_model_cpu_offload — ~10 GB peak; not needed on 48 GB cards
# Disk-offload + low-cpu-mem experiment (disabled — kept the code paths in
# main.py so they can be re-enabled later, but neither has been validated
# end-to-end yet).
FLUX_FILL_GROUP_OFFLOAD     = False
FLUX_FILL_GROUP_BLOCKS      = 2
FLUX_FILL_GROUP_USE_STREAM  = False
FLUX_FILL_OFFLOAD_DISK_PATH = "/tmp/flux_fill_offload"
FLUX_FILL_LOW_CPU_MEM_USAGE = False

# ── mmgp offload (alternative to diffusers block/model offload) ────────
# https://github.com/deepbeepmeep/mmgp — smarter memory manager that keeps
# Flux resident across multi-iter runs, slices adaptively, and uses async
# transfers to overlap PCIe with compute. Takes precedence over
# GROUP_OFFLOAD / CPU_OFFLOAD when True.
# pinnedMemory=True can 2× transfer speed but uses more RAM.
USE_MMGP_OFFLOAD            = False  # not needed on 48 GB cards — Flux fits fully on device
MMGP_PROFILE                = 5
MMGP_PINNED_MEMORY          = False

# ── Debug overrides (override what GPT returned in Agent 1) ──────────────────
# Set FORCE_FRAME_CROPPED=True to bypass GPT's frame-crop judgment and force
# the off-frame extension code path on every image.  Use FORCE_EXPANSION_PIXELS
# to set how much to pad each side (in pixels of the ORIGINAL image).  This
# is purely for testing the padded-canvas / off-frame mask flow on images
# GPT correctly judged as in-frame.  Set FORCE_FRAME_CROPPED=False to disable.
FORCE_FRAME_CROPPED        = False   # off-frame extension OFF — focus on in-frame shoulder inpaint only
FORCE_EXPANSION_PIXELS     = {"top": 0, "bottom": 280, "left": 80, "right": 300}  # legacy fixed-padding path

# ── Auto-budget padding + post-hoc crop ───────────────────────────────────────
# When USE_AUTO_PADDING_BUDGET=True, the off-frame helper IGNORES
# FORCE_EXPANSION_PIXELS and instead pads each side by FORCE_FRAME_CROPPED_BUDGET_PX,
# capped so neither dim exceeds MAX_PADDED_CANVAS_DIM (to keep Flux VRAM in budget).
# After Flux + SAM3 segmentation, crops the padded canvas + masks to the tight
# subject bbox + AUTO_CROP_MARGIN_PX. This avoids per-image padding tuning.
USE_AUTO_PADDING_BUDGET    = True
FORCE_FRAME_CROPPED_BUDGET_PX = 400   # preferred padding per side (px); reduced if it would blow the canvas cap
MAX_PADDED_CANVAS_DIM      = 1280     # cap on each padded dim (Flux VRAM-bounded)
AUTO_CROP_MARGIN_PX        = 30       # margin around segmented subject when cropping

# ── Narrow Flux inpaint mask via bbox×multiplier (skip generating in corners) ─
# Restricts Flux inpaint to: (gray padding) ∩ (visible_bbox × EXTENSION_MULTIPLIER).
# Saves 2-4× Flux runtime by not generating in regions we'd crop away anyway.
USE_EXTENSION_BBOX         = False    # v8 narrowing broke Flux denoising for thin L-shape masks;
                                       # v7 (full L-shape) produced better tight-cropped results.
EXTENSION_MULTIPLIER       = 2.0      # kept for future experimentation
MIN_EXTENSION_PX           = 100      # kept for future experimentation

# ── Bound the IN-FRAME hidden (inpaint) region to the subject's plausible
# extent ─────────────────────────────────────────────────────────────────
# Unrelated to USE_EXTENSION_BBOX above (that one narrows the OFF-FRAME
# padded-canvas mask). This one clips `hidden = dilate(occluder) − visible`
# (src/pipeline.py) to an expanded visible-subject bbox. Needed because a
# large occluder (e.g. a snowbank a rabbit sits behind) otherwise makes
# `hidden` the ENTIRE occluder — observed at 6-7x the visible subject's
# area — which Flux then "fills" with multiple/oversized subjects instead
# of one plausible continuation. A compact occluder (already close to
# subject-sized, e.g. a person) is barely affected by this clip.
BOUND_HIDDEN_TO_SUBJECT_BBOX  = True
HIDDEN_REGION_BBOX_MULTIPLIER = 2.0   # expand visible bbox by this factor about its center
HIDDEN_REGION_MIN_EXPANSION_PX = 100  # floor on the expansion in px per side

# Mixed Context Diffusion Sampling (Xu et al., CVPR 2024) was removed —
# permanently disabled via USE_MIXED_CONTEXT=False, and its ControlNet→Flux
# refinement cascade via USE_FLUX_REFINEMENT=False. See mixed_context.py in
# git history for the original implementation.

# ── Mask dilation ─────────────────────────────────────────────────────────────
MASK_EXPAND = 20    # px for elliptical dilation of the occluder mask

# CLIP-grounded occluder discovery, CLIP-on-grid fallback, and Grounding DINO
# were all removed (permanently disabled via USE_CLIP_GROUNDING /
# USE_CLIP_GRID_FALLBACK / USE_GROUNDING_DINO = False — PSALM/InstaFormer
# dominated mask fusion every run, so none of these ever changed an outcome).

# The old LangGraph-based reviewer/retry agent (`inpainting_agent`/`reviewer`/
# `route`/`build_graph` in the pre-restructure main.py) was removed; its
# scoring/retry behavior lives on as the inline Agent 3 loop in
# src/pipeline.py, gated by USE_REVIEWER above.

# ── AISFormer feature extraction (SAM3 backbone) ──────────────────────────────
# AISFORMER_ENABLED = True → SAM3 image features are extracted and saved to
#   output/<stem>/sam3_features.pt after segmentation, for offline training of
#   an AISFormerHead. The AISFormerHead inference path itself (which used to
#   read AISFORMER_CKPT/_THRESHOLD/_HIDDEN_DIM/_NUM_HEADS/_LAYERS and override
#   the shape prior) was removed — it was permanently unreachable since no
#   checkpoint has ever been trained/configured.
AISFORMER_ENABLED    = False   # MINIMAL pipeline: AISFormer OFF (no SAM3 feature extraction for downstream head)

# PSALM (referring-expression segmentation) was removed — permanently
# disabled via USE_PSALM=False (was over-segmenting nature scenes:
# "snowbank", "leaves", "post"); falls back to SAM3+GPT-V+InstaFormer.

# ── InstaFormer holistic occlusion+depth-order predictor ─────────────────────
# github.com/SNU-VGILab/InstaOrder (NeurIPS 2025) — single forward pass
# predicts occlusion AND depth order for every instance in the scene at once,
# via its own panoptic segmentation. Replaces BOTH the older pairwise
# InstaOrderNet (instaorder_helper.py, now unused) AND the separate
# Depth-Anything-V2 corroboration ranker — InstaFormer's own panoptic
# segmentation already separates same-class instances (zebra/zebra) into
# distinct segments, and its occlusion matrix already ranks them, so the old
# PSALM-class-split + connected-components + SAM3-dual-click + depth-rank
# machinery is no longer needed.
# Runs as a subprocess in its own isolated venv (Python 3.8 + Detectron2 +
# torch 2.1.0/cu118 — incompatible with this project's own environment).
USE_INSTAFORMER         = True
INSTAFORMER_REPO_DIR    = "/home/ubuntu/Workspace/amodal_completion/InstaFormer"
INSTAFORMER_VENV_PYTHON = "/home/ubuntu/Workspace/amodal_completion/.instaformer-venv/bin/python"
INSTAFORMER_CONFIG      = ("configs/instaorder/occlusion_depth/all/swin/"
                           "maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
INSTAFORMER_CKPT        = ("/home/ubuntu/Workspace/amodal_completion/InstaFormer/"
                           "checkpoints/instaformer_od_swinl_200.pth")

# ── Multi-model occluder-mask fusion ──────────────────────────────────────────
# Each candidate source produces a binary occluder-mask. Available sources
# (psalm / clip_segments / gdino_sam / clip_grid_sam were removed along with
# PSALM/CLIP/GroundingDINO — see history for the old voter implementations):
#   instaformer_pair — same-class occlusion via InstaFormer's own panoptic
#                      segmentation + occlusion matrix
#   instaformer      — holistic occlusion order over InstaFormer's own
#                      panoptic segments
#
# (The SAM3 text-prompt fallback used when no candidate above fires is a
# separate code path in occlusion_agent, not a fusion voter.)
#
# Modes:
#   "majority"     — pixel kept when ≥ MASK_FUSION_MIN_AGREE candidates agree
#   "union"        — OR of every non-empty candidate (max recall, may over-extend)
#   "intersection" — AND of every non-empty candidate (max precision, may miss)
#   "priority"     — original behaviour: first non-empty candidate wins
#                    (priority order = MASK_FUSION_PRIORITY)
MASK_FUSION_MODE      = "priority"
MASK_FUSION_MIN_AGREE = 2
MASK_FUSION_PRIORITY  = [
    "instaformer_pair",    # same-class occlusion via InstaFormer's own
                           # panoptic segmentation + occlusion matrix
                           # (zebra/zebra, cat/cat) — resolved natively
    "instaformer",         # holistic occlusion order over InstaFormer's
                           # own panoptic segments — primary signal
]
