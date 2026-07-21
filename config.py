# ── Input ─────────────────────────────────────────────────────────────────────
IMAGE_PATH = "/home/basil-k-aji/Desktop/Workspace/RD/website/animal-8518802_640_deer.jpg"

# Optional hint telling Agent 1 what the occluded subject is (e.g. "horse").
# Default is empty — Agent 1 auto-detects the occluded object itself from
# the image. Only set this when you want to steer detection explicitly.
INPUT_PROMPT = "rabbit"

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
GPT_MAX_TOKENS = 10192

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

# ── ControlNet-Inpaint (shape-prior guided completion) ────────────────────────
# pip install diffusers transformers accelerate xformers
# ── Inpainter backend selector ────────────────────────────────────────────────
# "controlnet_sd15" — ControlNet-Inpaint v1.1 + SD 1.5-inpaint base.
#                     Lightweight (~6GB VRAM), weaker anatomy priors. Not used
#                     directly any more — kept only as _run_flux_fill_inpaint's
#                     load-failure fallback.
# "flux_fill"       — FLUX.1-Fill-dev.  Much stronger anatomy / texture
#                     priors.  Requires ~16-24 GB VRAM and the model
#                     checkpoint (`black-forest-labs/FLUX.1-Fill-dev`).
#                     Memory mode is `enable_model_cpu_offload` so it fits
#                     in ~12 GB.  This is the ONLY live inpainting backend.
# "sd3_inpaint"     — placeholder for SD3-Inpaint; not implemented yet.
INPAINT_BACKEND            = "flux_fill"         # MINIMAL pipeline: Flux-Fill only (ControlNet+SD-1.5 OFF)

# ── Flux-Fill specific config (only used when INPAINT_BACKEND="flux_fill") ────
FLUX_FILL_MODEL_ID         = "black-forest-labs/FLUX.1-Fill-dev"
FLUX_FILL_STEPS            = 30
FLUX_FILL_GUIDANCE_SCALE   = 45.0    # Flux uses high CFG (10-50 range); higher = sharper detail
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

# CONTROLNET_MODEL_ID / SD_INPAINT_* below are only used by the ControlNet
# fallback path in _run_controlnet_inpaint (fires if FluxFillPipeline fails
# to load) — not part of the live inpainting path.
CONTROLNET_MODEL_ID        = "lllyasviel/control_v11p_sd15_inpaint"
SD_INPAINT_MODEL_ID        = "runwayml/stable-diffusion-inpainting"
SD_INPAINT_STEPS           = 50      # DDIM steps for SD inpainting (raised from 30
                                     # → 50 for finer detail in small completion
                                     # regions like a shoulder/arm strip)
SD_INPAINT_GUIDANCE_SCALE  = 7.5     # CFG scale
CONTROLNET_CONDITIONING_SCALE = 1.4  # ControlNet weight; raised from 1.0 to force the
                                     # shape prior to dominate over the
                                     # native UNet's bias toward the original (occluder)
                                     # pixels still in the inpaint region.
SD_INPAINT_STRENGTH        = 1.0     # full repaint of the masked region — no retention
                                     # of original (horse/leaf/etc) latent, otherwise
                                     # the occluder leaks into the result.

# Mixed Context Diffusion Sampling (Xu et al., CVPR 2024) was removed —
# permanently disabled via USE_MIXED_CONTEXT=False, and its ControlNet→Flux
# refinement cascade via USE_FLUX_REFINEMENT=False. See mixed_context.py in
# git history for the original implementation.

# ── Shape prior threshold ─────────────────────────────────────────────────────
# Used by the (fallback-only) ControlNet path's control-image compositing:
# the shape-prior hint is a white-background RGB; pixels below this
# brightness are treated as "object".
SHAPE_PRIOR_THRESH = 240   # per-channel min; pixel is "object" if any channel < thresh

# ── Mask dilation ─────────────────────────────────────────────────────────────
MASK_EXPAND = 20    # px for elliptical dilation of the occluder mask

# CLIP-grounded occluder discovery, CLIP-on-grid fallback, and Grounding DINO
# were all removed (permanently disabled via USE_CLIP_GROUNDING /
# USE_CLIP_GRID_FALLBACK / USE_GROUNDING_DINO = False — PSALM/InstaFormer
# dominated mask fusion every run, so none of these ever changed an outcome).

# The reviewer/retry agent (score threshold, retry counts, geometry
# self-check thresholds, structured failure-code taxonomy) was removed along
# with `inpainting_agent`/`reviewer`/`route`/`build_graph` in main.py — none
# of it is reachable from the live entry point. See git history if reviving.

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
