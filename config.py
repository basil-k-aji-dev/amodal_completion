# ── Input ─────────────────────────────────────────────────────────────────────
IMAGE_PATH = "/home/basil-k-aji/Desktop/Workspace/RD/website/bear-8845470_640_bear.jpg"

# Optional: the occluded object to reveal.
# Leave empty ("") to let the model automatically detect what is occluded.
TARGET = "bear"

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

# ── pix2gestalt amodal completion ─────────────────────────────────────────────
# Clone https://github.com/cvlab-columbia/pix2gestalt and set these paths.
# Checkpoint: hf_hub_download("cvlab/pix2gestalt-weights", "epoch=000005.ckpt")
from pathlib import Path
_HERE = Path(__file__).parent

PIX2GESTALT_REPO  = str(_HERE / "pix2gestalt" / "pix2gestalt")
PIX2GESTALT_CFG   = str(_HERE / "pix2gestalt" / "pix2gestalt" / "configs" /
                        "sd-finetune-pix2gestalt-c_concat-256.yaml")
PIX2GESTALT_CKPT  = str(_HERE / "ckpt" / "epoch=000005.ckpt")

PIX2GESTALT_SCALE     = 2.0    # CFG guidance scale (1–20)
PIX2GESTALT_STEPS     = 200    # DDIM steps — published quality numbers use 200
PIX2GESTALT_ETA       = 1.0    # DDIM eta (1.0 = full DDPM-level stochasticity)
PIX2GESTALT_N_SAMPLES = 1      # 1 sample per run keeps peak VRAM under 5 GB

# When False, skip pix2gestalt + GPT-V amodal review entirely and use a single
# GPT-V "draw the silhouette" call (_gpt_amodal_subject_mask) instead. Saves
# ~15-20 sec/image + ~3 GB VRAM. Required for the minimal-pipeline mode
# (SAM3 + GPT-5 + PSALM + InstaOrder + Flux-Fill only).
USE_PIX2GESTALT_AMODAL = False  # MINIMAL pipeline: pix2gestalt OFF; use GPT-V single silhouette call instead

# When False, skip amodal silhouette completion ENTIRELY. The "amodal mask"
# becomes the visible_mask directly — no extension into hidden / off-frame
# parts. Cutout shows just the visible subject on neutral gray. No Flux
# inpainting of hidden regions (since hidden = amodal - visible = 0).
# Use this when PSALM's visible segmentation is already sufficient and you
# don't want GPT-V over-extending the silhouette onto the occluder.
USE_AMODAL_COMPLETION  = False  # Jiang Ao-style: skip GPT polygon; Flux + post-SAM3 decide the silhouette

# ── GPU memory cap ────────────────────────────────────────────────────────────
# Hard limit on VRAM usage per process. Set to None to disable.
# Example: 10.0 leaves ~1.6 GB headroom on a 12 GB card for model swapping.
GPU_MEMORY_LIMIT_GB = 10.0

# ── LaMa background inpainting ────────────────────────────────────────────────
# pip install simple-lama-inpainting
# Runs on CPU or GPU; no download needed beyond pip.
LAMA_DEVICE = None   # None = auto (cuda if available, else cpu)

# ── ControlNet-Inpaint (shape-prior guided completion) ────────────────────────
# pip install diffusers transformers accelerate xformers
# ── Inpainter backend selector ────────────────────────────────────────────────
# "controlnet_sd15" — default; ControlNet-Inpaint v1.1 + SD 1.5-inpaint base.
#                     Lightweight (~6GB VRAM), weaker anatomy priors.  Works
#                     with all MCDS hooks.
# "flux_fill"       — FLUX.1-Fill-dev.  Much stronger anatomy / texture
#                     priors, no UNet so MCDS hooks DON'T apply.  Requires
#                     ~16-24 GB VRAM and the model checkpoint
#                     (`black-forest-labs/FLUX.1-Fill-dev`).  Memory mode
#                     is `enable_model_cpu_offload` so it fits in ~12 GB.
# "sd3_inpaint"     — placeholder for SD3-Inpaint; not implemented yet.
INPAINT_BACKEND            = "flux_fill"         # MINIMAL pipeline: Flux-Fill only (ControlNet+SD-1.5 OFF)

# ── Flux-Fill specific config (only used when INPAINT_BACKEND="flux_fill") ────
FLUX_FILL_MODEL_ID         = "black-forest-labs/FLUX.1-Fill-dev"
FLUX_FILL_STEPS            = 50
FLUX_FILL_GUIDANCE_SCALE   = 45.0    # Flux uses high CFG (10-50 range); higher = sharper detail
FLUX_FILL_MAX_SEQUENCE_LEN = 512
FLUX_FILL_CPU_OFFLOAD      = True    # enable_model_cpu_offload — ~10 GB peak
FLUX_FILL_SEQUENTIAL_OFFLOAD = True  # enable_sequential_cpu_offload — ~6 GB peak (3-4× slower)
                                       # Takes precedence over FLUX_FILL_CPU_OFFLOAD when True.
                                       # Required for sub-12 GB GPUs (RTX 3060, 3060 Ti, etc).
# Disk-offload + low-cpu-mem experiment (disabled — kept the code paths in
# main.py so they can be re-enabled later, but neither has been validated
# end-to-end yet).
FLUX_FILL_GROUP_OFFLOAD     = False
FLUX_FILL_GROUP_BLOCKS      = 2
FLUX_FILL_GROUP_USE_STREAM  = False
FLUX_FILL_OFFLOAD_DISK_PATH = "/tmp/flux_fill_offload"
FLUX_FILL_LOW_CPU_MEM_USAGE = False

# ── mmgp offload (alternative to diffusers sequential offload) ─────────
# https://github.com/deepbeepmeep/mmgp — smarter memory manager that keeps
# Flux resident across multi-iter runs, slices adaptively, and uses async
# transfers to overlap PCIe with compute. Takes precedence over
# FLUX_FILL_SEQUENTIAL_OFFLOAD / GROUP_OFFLOAD / CPU_OFFLOAD when True.
# pinnedMemory=True can 2× transfer speed but uses more RAM — disabled
# here since we only have 14 GB RAM and 24 GB Flux already strains swap.
USE_MMGP_OFFLOAD            = False  # tested: mmgp setup-tax + per-call warmup
                                       # made total Flux time +22% slower than
                                       # sequential offload on our load/free
                                       # access pattern. Would only win if Flux
                                       # were kept resident across iters.
MMGP_PROFILE                = 5
MMGP_PINNED_MEMORY          = False
                                       # Takes precedence over FLUX_FILL_CPU_OFFLOAD when True.
                                       # Required for sub-12 GB GPUs (RTX 3060, 3060 Ti, etc).

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

CONTROLNET_MODEL_ID        = "lllyasviel/control_v11p_sd15_inpaint"
SD_INPAINT_MODEL_ID        = "runwayml/stable-diffusion-inpainting"
SD_INPAINT_STEPS           = 50      # DDIM steps for SD inpainting (raised from 30
                                     # → 50 for finer detail in small completion
                                     # regions like a shoulder/arm strip)
SD_INPAINT_GUIDANCE_SCALE  = 7.5     # CFG scale
CONTROLNET_CONDITIONING_SCALE = 1.4  # ControlNet weight; raised from 1.0 to force the
                                     # pix2gestalt shape prior to dominate over the
                                     # native UNet's bias toward the original (occluder)
                                     # pixels still in the inpaint region.
SD_INPAINT_STRENGTH        = 1.0     # full repaint of the masked region — no retention
                                     # of original (horse/leaf/etc) latent, otherwise
                                     # the occluder leaks into the result.
N_SAMPLES                  = 4       # variations to cache per run

# ── Mixed Context Diffusion Sampling (co-occurrence bias suppression) ─────────
# Port of Xu et al., CVPR 2024 "Amodal Completion via Progressive Mixed Context
# Diffusion" (see amodal_k8xu/). Strips scene context from the SD inpainter so
# it stops filling hidden regions with objects that co-occur in the visible
# scene (the classic "road → car" failure mode).
#
# When enabled, _run_controlnet_inpaint:
#   1. greys out pixels outside (visible_mask ∪ inpaint_mask) before SD sees them
#   2. runs an extra LaMa pass to build an object-removed background image
#   3. registers a callback_on_step_end that swaps background latents with
#      noisy clean-bg latents until MIXED_CONTEXT_TIMESTEP_FRAC of denoising
#      has elapsed.
USE_MIXED_CONTEXT             = False   # MINIMAL pipeline: MCDS OFF
USE_FLUX_REFINEMENT           = False   # MINIMAL pipeline: no ControlNet→Flux cascade (Flux is the primary inpainter)
FLUX_REFINEMENT_STRENGTH      = 0.6     # how much Flux re-paints (0.4-0.7 typical; lower = preserve more)
MIXED_CONTEXT_TIMESTEP_FRAC   = 0.3   # fraction of steps over which latents are swapped.
                                      # Lowered from 0.7 → 0.3 so MCDS fires EARLY in
                                      # denoising (around step 8 of 27), before the
                                      # UNet's coarse structural decisions lock in the
                                      # occluder's silhouette.
MIXED_CONTEXT_GRAY            = 127   # neutral grey level for scene swap (0–255)

# Full-MCDS hook controls (mixed_context.register_mcds_unet_hook). Switched on
# the test3 quality regression was caused by (1) NOT swapping the 9-channel
# masked-image latents and (2) NOT doing KMeans refinement of the query mask
# mid-denoise.  These flags turn both back on.
MCDS_USE_UP_FT_KMEANS         = True
MCDS_NUM_CLUSTERS             = 8
MCDS_UP_BLOCK_IDX             = 2     # k8xu uses up_ft[2] — UNet decoder block 2
MCDS_INTERSECT_THRESH         = 0.2   # cluster ∩ query > 0.2 * query  →  merge into refined mask

# ── Shape prior threshold ─────────────────────────────────────────────────────
# pix2gestalt outputs a white-background RGB; pixels below this brightness are object.
SHAPE_PRIOR_THRESH = 240   # per-channel min; pixel is "object" if any channel < thresh

# ── Mask dilation ─────────────────────────────────────────────────────────────
MASK_EXPAND = 20    # px for elliptical dilation of the occluder mask

# ── Agent settings ────────────────────────────────────────────────────────────
SCORE_THRESHOLD  = 7.0   # reviewer score >= this = accepted (raised from 5.0)
MAX_RETRIES      = 3     # max total retry attempts
MAX_MASK_RETRIES = 1     # max times to re-analyse the occlusion mask

# Geometry self-check: re-prompt GPT when its occluder_click / polygons land on
# the subject instead of the occluder.  Each retry costs one extra GPT call
# (and one cheap SAM3 point-prompt run) so keep the limit small.
MAX_GPT_GEOMETRY_RETRIES = 2

# Tunable thresholds for the geometry cross-check (fraction of the candidate
# region that lies inside the GPT-supplied visible polygon).  Above the limit,
# the candidate is considered to be on the subject, not the occluder.
GEO_OCCLUDER_VS_VISIBLE_MAX = 0.50   # SAM3(occluder_click) ∩ visible polygon
GEO_HIDDEN_VS_VISIBLE_MAX   = 0.50   # hidden_polygon ∩ visible polygon

# ── CLIP-grounded occluder discovery (CVPR'25 amodal-style) ───────────────────
# Replaces GPT-supplied occluder polygons with vision-grounded segment labels.
# For each SAM3 segment we run OpenAI CLIP scoring against {target_class,
# occluder_class, "background", "other"} and assign each segment its best label.
# The CLIP-grounded occluder mask is then filtered by adjacency to the visible
# target — this drops occluder-class segments that are unrelated to the subject
# (e.g. a separate log somewhere else in the scene).
USE_CLIP_GROUNDING        = False   # disabled: PSALM dominates fusion every run; CLIP rejected
CLIP_MODEL_NAME           = "ViT-B/32"
CLIP_MIN_SCORE            = 0.30   # softmax probability threshold for accepting a segment label
CLIP_ADJACENCY_PX         = 60     # how many px around visible_mask count as "touching"
CLIP_OCCLUDER_MIN_AREA    = 200    # smaller candidate occluders are dropped

# CLIP-on-grid fallback (used when no SAM3 segment matches the occluder class).
# Score CLIP on a coarse grid of image patches → highest cell becomes a SAM3
# point-prompt anchor → forces SAM3 to produce a mask at that location.
USE_CLIP_GRID_FALLBACK    = False   # disabled with CLIP_GROUNDING; PSALM dominates
CLIP_GRID_N               = 16     # grid resolution (16 × 16 = 256 patches)
CLIP_GRID_MIN_SCORE       = 0.40   # patch must score >= this for the occluder label
CLIP_GRID_TOP_K           = 3      # how many top-scoring cells to use as SAM3 prompts

# ── Grounding DINO (HF transformers, no separate install needed) ──────────────
# Open-vocabulary text → bounding boxes. Used as the strongest fallback when
# both CLIP-on-segments and CLIP-on-grid don't find the occluder. Driven by
# the SHORT noun extracted from the GPT occluder description.
USE_GROUNDING_DINO        = False   # disabled: PSALM does this job in one step; GDINO never changed an outcome across test runs
GROUNDING_DINO_MODEL_ID   = "IDEA-Research/grounding-dino-tiny"   # ~700 MB
GROUNDING_DINO_BOX_THRESH = 0.30
GROUNDING_DINO_TEXT_THRESH = 0.25

# ── Structured failure taxonomy ───────────────────────────────────────────────
# MASK_*  failures   → re-run occlusion_agent (regenerate masks from scratch)
# SHAPE_* failures   → keep masks, regenerate shape prior + ControlNet samples
# PROMPT_* failures  → keep masks + shape prior, cycle to next ControlNet sample
#
# Note on OCCLUDER_*:  the old single 'OCCLUDER_REMNANT' code conflated two
# very different failures — (a) occluder pixels still opaque/present in the
# output (a mask-recall problem → re-segment), and (b) occluder removed but
# fill region has translucent ghost/halo artifacts (a fill-quality problem
# → cycle SD sample, do NOT shrink the mask). Splitting them lets the router
# avoid the regression where a good broad mask gets thrown away because the
# reviewer's "ghost" complaint pulled the mask narrower.
# 'OCCLUDER_REMNANT' is kept as a backwards-compat alias for SOLID so old GPT
# responses don't crash.
MASK_FAILURE_CODES   = {"MASK_INACCURATE", "OCCLUDER_REMNANT_SOLID",
                        "OCCLUDER_REMNANT"}
SHAPE_FAILURE_CODES  = {"SHAPE_PRIOR_BAD", "OCCLUDER_GHOST"}
PROMPT_FAILURE_CODES = {"ANATOMY_WRONG", "COLOR_MISMATCH", "SEAM_VISIBLE",
                        "PROMPT_WEAK", "BLURRY_OUTPUT"}
ALL_FAILURE_CODES    = ({"ACCEPTED"} | MASK_FAILURE_CODES
                                     | SHAPE_FAILURE_CODES
                                     | PROMPT_FAILURE_CODES)

# ── AISFormer (SAM3 backbone + amodal head) ───────────────────────────────────
# AISFORMER_ENABLED = True  → SAM3 image features are extracted and saved to
#   output/<stem>/sam3_features.pt after segmentation.  These can be used for
#   offline training of the AISFormerHead.
# AISFORMER_CKPT = ""       → feature extraction only; head inference is skipped.
# AISFORMER_CKPT = "/path"  → load head weights and use AISFormer as the shape
#   prior (replaces pix2gestalt thresholding; appearance hint is still pix2gestalt).
AISFORMER_ENABLED    = False   # MINIMAL pipeline: AISFormer OFF (no SAM3 feature extraction for downstream head)
AISFORMER_CKPT       = ""      # path to AISFormerHead .pt checkpoint; "" = skip inference
AISFORMER_THRESHOLD  = 0.5     # sigmoid threshold for binarising the amodal mask
AISFORMER_HIDDEN_DIM = 256     # hidden dimension for the head
AISFORMER_NUM_HEADS  = 8       # multi-head attention heads
AISFORMER_LAYERS     = 2       # transformer decoder layers

# ── PSALM (referring-expression segmentation, LISA-alternative) ───────────────
# Clone https://github.com/zamling/PSALM into amodal_test3/PSALM/.
# Separate venv at amodal_test3/.psalm-venv (created by setup_psalm.sh) keeps
# PSALM's pinned transformers fork from clashing with SAM3/diffusers.
# Each PSALM invocation is a subprocess — when it exits, the GPU is released
# automatically, so no _free_psalm() bookkeeping is needed.
USE_PSALM            = False  # disabled: was over-segmenting nature scenes ("snowbank", "leaves", "post"); fall back to SAM3+GPT-V+InstaOrder
PSALM_REPO_DIR       = str(_HERE / "PSALM")
PSALM_VENV_PYTHON    = str(_HERE / ".psalm-venv" / "bin" / "python")
PSALM_CHECKPOINT_DIR = str(_HERE / "ckpt" / "PSALM")    # HF snapshot dir
PSALM_INFER_SCRIPT   = str(_HERE / "scripts" / "run_psalm_seg.py")
PSALM_TIMEOUT        = 240     # seconds — kill subprocess after this
PSALM_MIN_AREA       = 200     # drop PSALM masks smaller than this many px

# ── Depth-Anything-V2 front/back ranker ───────────────────────────────────────
# Pure-vision signal: closer pixels (higher predicted_depth) → in front.
# Used in the InstaOrder-pair block as a SECOND independent ranker that
# corroborates (or overrides) the learned InstaOrder model.  Doesn't depend
# on any GPT-supplied text.
USE_DEPTH_RANK    = True
DEPTH_MODEL_ID    = "depth-anything/Depth-Anything-V2-Small-hf"

# ── InstaOrder occlusion-order reasoner ───────────────────────────────────────
# Learned pairwise occlusion order over SAM3 segments — the only signal that
# resolves same-class occlusion (zebra-on-zebra, cat-on-cat) which CLIP-text
# scoring cannot. Repo + checkpoint live under amodal_k8xu/ (we don't duplicate
# the 3.8 GB tree); the loader sys.path-adds the repo at first call.
USE_INSTAORDER      = True
INSTAORDER_REPO_DIR = "/home/basil-k-aji/Desktop/Workspace/RD/amodal_k8xu/InstaOrder"
INSTAORDER_CKPT     = ("/home/basil-k-aji/Desktop/Workspace/RD/amodal_k8xu/"
                       "InstaOrder/InstaOrder_ckpt/"
                       "InstaOrder_InstaOrderNet_od.pth.tar")
INSTAORDER_INPUT_SIZE  = 384
INSTAORDER_OCC_THRESH  = 0.5    # P(candidate over target) cutoff for inclusion

# ── Multi-model occluder-mask fusion ──────────────────────────────────────────
# Each candidate source produces a binary occluder-mask. Available sources:
#   psalm           — PSALM referring-expression mask
#   clip_segments   — primary CLIP-on-SAM3-segments path
#   gdino_sam       — GroundingDINO-tiny → SAM3 bbox-prompt fallback
#   clip_grid_sam   — CLIP-on-grid → SAM3 point-prompt fallback
#   gpt_polygon     — GPT polygon_override
#   gpt_segments    — union of GPT-selected SAM3 segment IDs
#
# Modes:
#   "majority"     — pixel kept when ≥ MASK_FUSION_MIN_AGREE candidates agree
#   "union"        — OR of every non-empty candidate (max recall, may over-extend)
#   "intersection" — AND of every non-empty candidate (max precision, may miss)
#   "priority"     — original behaviour: first non-empty candidate wins
#                    (priority order = MASK_FUSION_PRIORITY)
MASK_FUSION_MODE      = "priority"
MASK_FUSION_MIN_AGREE = 2
# Priority order rationale: PSALM (referring-expression) is the most reliable
# learned signal; GPT polygon is the human-language reasoning fallback; CLIP-
# on-segments is brittle for same-class occluders (returns ~empty on zebra-on-
# zebra, cat-on-cat) so it's demoted below GPT polygon.
MASK_FUSION_PRIORITY  = [
    "instaorder_pair",     # PSALM-class split via InstaOrder front-ranking
                           # — the only signal that resolves same-class
                           # occlusion (zebra/zebra, cat/cat)
    "psalm",               # referring-expression seg (subject-class-pinned)
                           # — primary signal for different-class occlusion
    "instaorder",          # learned occlusion-order over SAM3 segments
                           # — kept as a fallback voter, but BELOW psalm
                           # because it often picks tiny SAM3 fragments
                           # (e.g. the building/fountain case where
                           # InstaOrder produced a 1k px mask vs PSALM's
                           # correct 19k px fountain).
    "clip_segments",       # CLIP-on-segments  (vision-grounded)
    "gdino_sam",           # GroundingDINO+SAM3 fallback
    "clip_grid_sam",       # CLIP-grid+SAM3 fallback
    # GPT polygon/segments REMOVED — they're correlated with the text signals
    # above (same GPT call drives them), so they don't add independent votes.
]
