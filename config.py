# ── Input ─────────────────────────────────────────────────────────────────────
IMAGE_PATH = "/home/basil-k-aji/Desktop/Workspace/RD/website/bear-8845470_640_bear.jpg"

# Optional: the occluded object to reveal.
# Leave empty ("") to let the model automatically detect what is occluded.
TARGET = "bear"

# ── GPT (loaded from .env: OPENAI_MODEL, OPENAI_API_KEY) ─────────────────────
GPT_MAX_TOKENS = 10192

# ── SAM3 automatic segmentation ───────────────────────────────────────────────
SAM3_MODEL_ID         = "facebook/sam3"
SAM3_POINTS_PER_BATCH = 16     # points processed per forward pass
SAM3_SCORE_THRESH     = 0.50   # minimum mask confidence score to keep
SAM3_MIN_AREA         = 100    # minimum mask area in pixels

# ── SAM3 text-prompt-first segmentation ──────────────────────────────────────
# Agent 1 asks SAM3 to segment the GPT-derived target/occluder text first.
# CLIP verifies these masks. Automatic SAM3 segmentation + GPT numbered
# candidate selection only runs if CLIP rejects the text-prompt masks.
SAM3_TEXT_SCORE_THRESH = 0.30
SAM3_TEXT_MASK_THRESH  = 0.50
SAM3_TEXT_MIN_AREA     = 100

# ── Amodal completion toggle ──────────────────────────────────────────────────
# When False, skip amodal silhouette completion: the amodal mask becomes the
# visible_mask directly (Flux + post-SAM3 decide the final silhouette).
USE_AMODAL_COMPLETION  = False

# ── Mask-review-first verification ────────────────────────────────────────────
# After Agent 1 produces the visible mask, GPT-V reviews the green-overlay and
# returns a corrective click if the mask captured the wrong subject. Bounded by
# MASK_REVIEW_MAX_RETRIES.
USE_MASK_REVIEW_FIRST     = True
MASK_REVIEW_MIN_AREA_FRAC = 0.02   # re-review if visible mask < 2% of image area
MASK_REVIEW_MAX_RETRIES   = 1      # at most 1 corrective click

# ── CLIP-similarity verification gate ─────────────────────────────────────────
# Score the masked region against "a photo of a {target}" with CLIP; below the
# threshold, trigger GPT-V mask review. Cheap (~50 ms) false-segmentation catch.
USE_CLIP_VERIFY             = True
CLIP_VERIFY_THRESHOLD       = 0.20    # cosine similarity floor for ACCEPT
CLIP_VERIFY_PROMPT_TEMPLATE = "a photo of a {target}"

# ── GPT-provided point clicks (optional fallback) ─────────────────────────────
# Allow GPT to provide (x,y) coordinates as a fallback when auto-seg doesn't
# produce usable segments. When False, mask-review verifies but issues no clicks.
USE_GPT_CLICK            = True

# ── GPU memory cap ────────────────────────────────────────────────────────────
# Hard limit on VRAM usage per process. Set to None to disable.
GPU_MEMORY_LIMIT_GB = 10.0

# ── Inpainter backend selector ────────────────────────────────────────────────
# "flux_fill"   — local FLUX.1-Fill-dev (strong anatomy/texture priors).
# "remote_flux" — call a remote Flux server (server.py /inpaint); only the Flux
#                 denoise is offloaded, the rest of the pipeline runs locally.
INPAINT_BACKEND     = "flux_fill"
REMOTE_FLUX_URL     = ""
REMOTE_FLUX_TIMEOUT = 900                  # seconds; cover 3 off-frame iters + warmup

# ── Full-pipeline remote dispatch (client.py) ─────────────────────────────────
# RUN_ENV selects where the pipeline runs:
#   "local"     — everything runs locally (pipeline.graph.run).
#   "lightning" — uploads the image to LIGHTNING_URL (server.py on a Lightning
#                 Studio) and downloads the ZIP.
#   "colab"     — uploads to COLAB_URL (the same server.py running in a Colab
#                 notebook, exposed via ngrok / cloudflared). Server code is
#                 identical; only the tunnel URL differs.
#
# Legacy USE_LIGHTNING is still honoured: if RUN_ENV is unset and USE_LIGHTNING
# is True, the client behaves as if RUN_ENV="lightning".
RUN_ENV           = "local"          # "local" | "lightning" | "colab"
USE_LIGHTNING     = False            # deprecated; prefer RUN_ENV
LIGHTNING_URL     = ""
LIGHTNING_TIMEOUT = 1800             # seconds; whole-pipeline upper bound
COLAB_URL         = ""               # e.g. "https://xxxx-xx-xx.ngrok-free.app"
COLAB_TIMEOUT     = 1800             # seconds; whole-pipeline upper bound

# ── Flux-Fill config (used when INPAINT_BACKEND in {flux_fill, remote_flux}) ──
FLUX_FILL_MODEL_ID         = "black-forest-labs/FLUX.1-Fill-dev"
FLUX_FILL_STEPS            = 50
FLUX_FILL_GUIDANCE_SCALE   = 45.0    # Flux uses high CFG (10-50); higher = sharper detail
FLUX_FILL_MAX_SEQUENCE_LEN = 512
# When AUTO_DETECT_OFFLOAD=True, runtime.py overrides the FLUX_FILL_*_OFFLOAD
# flags + GPU_MEMORY_LIMIT_GB based on detected VRAM at startup. Lets the same
# code run on a 12 GB local card or a 22 GB L4 without manual changes.
AUTO_DETECT_OFFLOAD          = False  # manual: set FLUX_FILL_*_OFFLOAD below
FLUX_FILL_CPU_OFFLOAD        = True   # enable_model_cpu_offload — ~10 GB peak
FLUX_FILL_SEQUENTIAL_OFFLOAD = True   # enable_sequential_cpu_offload — ~6 GB peak (3-4× slower)
                                      # Takes precedence over FLUX_FILL_CPU_OFFLOAD when True.
# Disk-offload + low-cpu-mem experiment (disabled; code paths kept in flux.py).
FLUX_FILL_GROUP_OFFLOAD     = False
FLUX_FILL_GROUP_BLOCKS      = 2
FLUX_FILL_GROUP_USE_STREAM  = False
FLUX_FILL_OFFLOAD_DISK_PATH = "/tmp/flux_fill_offload"
FLUX_FILL_LOW_CPU_MEM_USAGE = False

# ── mmgp offload (alternative to diffusers sequential offload) ─────────────────
# https://github.com/deepbeepmeep/mmgp — takes precedence over the FLUX_FILL_*
# offload flags when USE_MMGP_OFFLOAD=True.
USE_MMGP_OFFLOAD   = False
MMGP_PROFILE       = 5
MMGP_PINNED_MEMORY = False

# ── Off-frame extension + auto-budget padding ─────────────────────────────────
# FORCE_FRAME_CROPPED=True forces the off-frame extension path on every image.
FORCE_FRAME_CROPPED    = False
FORCE_EXPANSION_PIXELS = {"top": 0, "bottom": 280, "left": 80, "right": 300}  # legacy fixed-padding
# USE_AUTO_PADDING_BUDGET pads each side by FORCE_FRAME_CROPPED_BUDGET_PX, capped
# so neither dim exceeds MAX_PADDED_CANVAS_DIM, then crops to the subject bbox
# + AUTO_CROP_MARGIN_PX after Flux + SAM3.
USE_AUTO_PADDING_BUDGET       = True
FORCE_FRAME_CROPPED_BUDGET_PX = 400   # preferred padding per side (px)
MAX_PADDED_CANVAS_DIM         = 1280  # cap on each padded dim (Flux VRAM-bounded)
AUTO_CROP_MARGIN_PX           = 30    # margin around segmented subject when cropping
# Narrow the Flux inpaint mask to (gray padding) ∩ (visible_bbox × multiplier).
USE_EXTENSION_BBOX            = False
EXTENSION_MULTIPLIER          = 2.0
MIN_EXTENSION_PX              = 100

# ── Mask dilation ─────────────────────────────────────────────────────────────
MASK_EXPAND = 20    # px for elliptical dilation of the occluder mask

# ── Agent settings ────────────────────────────────────────────────────────────
SCORE_THRESHOLD  = 7.0   # reviewer score >= this = accepted
MAX_RETRIES      = 3     # max total retry attempts
MAX_MASK_RETRIES = 1     # max times to re-analyse the occlusion mask

# Geometry cross-check thresholds (fraction of the candidate region inside the
# GPT-supplied visible polygon; above the limit it's on the subject, not the occluder).
GEO_OCCLUDER_VS_VISIBLE_MAX = 0.50   # SAM3(occluder_click) ∩ visible polygon
GEO_HIDDEN_VS_VISIBLE_MAX   = 0.50   # hidden_polygon ∩ visible polygon

# ── CLIP-grounded occluder discovery (CVPR'25 amodal-style) ───────────────────
# For each SAM3 segment, CLIP-score against {target, occluder, background, other}
# and assign each segment its best label; filter by adjacency to the visible target.
USE_CLIP_GROUNDING     = False
CLIP_MODEL_NAME        = "ViT-B/32"
CLIP_MIN_SCORE         = 0.30   # softmax probability threshold for accepting a label
CLIP_ADJACENCY_PX      = 60     # px around visible_mask that count as "touching"
CLIP_OCCLUDER_MIN_AREA = 200    # smaller candidate occluders are dropped

# CLIP-on-grid fallback: score CLIP on a coarse patch grid → top cells become
# SAM3 point-prompt anchors.
USE_CLIP_GRID_FALLBACK = False
CLIP_GRID_N            = 16     # grid resolution (16 × 16 = 256 patches)
CLIP_GRID_MIN_SCORE    = 0.40   # patch must score >= this for the occluder label
CLIP_GRID_TOP_K        = 3      # how many top-scoring cells to use as SAM3 prompts

# ── Structured failure taxonomy ───────────────────────────────────────────────
# MASK_* failures → re-run occlusion_agent. Other non-accept failures → re-run
# completion (new Flux seed). 'OCCLUDER_REMNANT' kept as a back-compat alias.
MASK_FAILURE_CODES   = {"MASK_INACCURATE", "OCCLUDER_REMNANT_SOLID",
                        "OCCLUDER_REMNANT"}
SHAPE_FAILURE_CODES  = {"SHAPE_PRIOR_BAD", "OCCLUDER_GHOST"}
PROMPT_FAILURE_CODES = {"ANATOMY_WRONG", "COLOR_MISMATCH", "SEAM_VISIBLE",
                        "PROMPT_WEAK", "BLURRY_OUTPUT"}
ALL_FAILURE_CODES    = ({"ACCEPTED"} | MASK_FAILURE_CODES
                                     | SHAPE_FAILURE_CODES
                                     | PROMPT_FAILURE_CODES)

# ── Multi-model occluder-mask fusion ──────────────────────────────────────────
# Each candidate source produces a binary occluder-mask. Active sources:
#   sam3_text     — SAM3 native text prompt + CLIP verification
#   sam3_click    — SAM3 point-prompt from a text-derived click
#   gpt_segments  — GPT-selected SAM3 auto-segment IDs
#   clip_segments — CLIP-on-SAM3-segments
# Modes: "priority" | "union" | "intersection" | "majority".
MASK_FUSION_MODE      = "priority"
MASK_FUSION_MIN_AGREE = 2
MASK_FUSION_PRIORITY  = [
    "sam3_text",
    "sam3_click",
    "gpt_segments",
    "clip_segments",
]
