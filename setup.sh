#!/usr/bin/env bash
#
# setup.sh — one-shot environment provisioner for amodal_completion.
#
# Builds every environment this pipeline needs:
#   .venv               (Python 3.12, main pipeline: SAM3 / Flux.1-Fill / GPT-5)
#   .instaformer-venv    (Python 3.8,  InstaFormer occlusion/depth-order subprocess)
#   .lisa-venv           (Python 3.10, legacy LISA — not called by the current
#                         pipeline, built only for reference/rollback)
#   hunyuan3d (conda)    (Python 3.10, optional 3D generation subprocess)
#
# and clones the vendored sub-project repos (InstaFormer, detectron2, LISA,
# Hunyuan3D-2.1) that are gitignored/untracked because they are multi-GB.
#
# Usage:
#   ./setup.sh                 # do everything
#   ./setup.sh --skip-lisa     # skip the legacy LISA env (not used at runtime)
#   ./setup.sh --skip-hunyuan3d
#   ./setup.sh --only-main     # just .venv + .env, nothing else
#
# Safe to re-run: every step checks whether its target already exists and
# skips if so. Nothing here deletes or overwrites data outside this repo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── flags ─────────────────────────────────────────────────────────────────
DO_MAIN=1
DO_INSTAFORMER=1
DO_LISA=1
DO_HUNYUAN3D=1

for arg in "$@"; do
  case "$arg" in
    --skip-instaformer) DO_INSTAFORMER=0 ;;
    --skip-lisa)        DO_LISA=0 ;;
    --skip-hunyuan3d)   DO_HUNYUAN3D=0 ;;
    --only-main)        DO_INSTAFORMER=0; DO_LISA=0; DO_HUNYUAN3D=0 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown flag: $arg" >&2
      exit 1
      ;;
  esac
done

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

command -v git >/dev/null || die "git is required"
command -v nvidia-smi >/dev/null || warn "nvidia-smi not found — is this machine GPU-equipped? CPU-only will be extremely slow for Flux/SAM3."

# ── uv (manages .venv / .instaformer-venv / .lisa-venv, incl. their own
#    pinned Python interpreters — no system Python version juggling needed) ──
if ! command -v uv >/dev/null; then
  log "Installing uv (Python/venv manager)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
log "uv version: $(uv --version)"

# ═══════════════════════════════════════════════════════════════════════
# 1. Main pipeline env — .venv (Python 3.12)
# ═══════════════════════════════════════════════════════════════════════
if [[ $DO_MAIN -eq 1 ]]; then
  log "[.venv] Syncing main pipeline environment"
  uv sync
  log "[.venv] Done — $(.venv/bin/python --version)"
fi

# ── .env ────────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
  log "Creating .env from .env.example — YOU MUST FILL IN THE VALUES"
  cp .env.example .env
  warn "Edit .env now: OPENAI_API_KEY (Responses API + gpt-5 access), OPENAI_MODEL, HF_TOKEN (gated facebook/sam3 + FLUX.1-Fill-dev access)"
else
  log ".env already exists — leaving it untouched"
fi

# ═══════════════════════════════════════════════════════════════════════
# 2. InstaFormer — occlusion/depth-order subprocess (Python 3.8 + detectron2)
# ═══════════════════════════════════════════════════════════════════════
if [[ $DO_INSTAFORMER -eq 1 ]]; then
  log "[InstaFormer] Cloning repo"
  if [[ ! -d InstaFormer ]]; then
    git clone https://github.com/SNU-VGILab/InstaFormer InstaFormer
  else
    log "[InstaFormer] Already cloned, skipping"
  fi

  log "[detectron2] Cloning + pinning to commit 80307d2 (required by InstaFormer)"
  if [[ ! -d detectron2 ]]; then
    git clone https://github.com/facebookresearch/detectron2.git
    (cd detectron2 && git checkout 80307d2)
  else
    log "[detectron2] Already cloned, skipping"
  fi

  log "[.instaformer-venv] Building Python 3.8 environment"
  if [[ ! -x .instaformer-venv/bin/python ]]; then
    uv venv .instaformer-venv --python 3.8
    # CUDA 11.8 build of torch 2.1.0, matching InstaFormer's own quick_install.sh
    uv pip install --python .instaformer-venv/bin/python \
      torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
      --index-url https://download.pytorch.org/whl/cu118
    uv pip install --python .instaformer-venv/bin/python \
      opencv-python ipython shapely h5py scipy submitit scikit-image \
      cython timm einops scikit-learn wandb numpy
    uv pip install --python .instaformer-venv/bin/python \
      "git+https://github.com/cocodataset/panopticapi.git"
    # detectron2, built from the pinned commit above
    uv pip install --python .instaformer-venv/bin/python -e ./detectron2
    # Custom CUDA op (multi-scale deformable attention) used by Mask2Former
    log "[InstaFormer] Building MultiScaleDeformableAttention CUDA op"
    (
      cd InstaFormer/mask2former/modeling/pixel_decoder/ops
      ../../../../../.instaformer-venv/bin/python setup.py build install
    )
  else
    log "[.instaformer-venv] Already exists, skipping"
  fi

  log "[InstaFormer] Fetching occlusion+depth-order checkpoint (~1.2 GB)"
  mkdir -p InstaFormer/checkpoints
  CKPT=InstaFormer/checkpoints/instaformer_od_swinl_200.pth
  if [[ ! -f "$CKPT" ]]; then
    if command -v gdown >/dev/null; then
      gdown "1lH_cn1SqDl7jMv5BP8DhuRYtzME0PCZI" -O "$CKPT"
    else
      uv pip install --python .venv/bin/python gdown >/dev/null
      .venv/bin/python -m gdown "1lH_cn1SqDl7jMv5BP8DhuRYtzME0PCZI" -O "$CKPT"
    fi
  else
    log "[InstaFormer] Checkpoint already present, skipping download"
  fi

  # Wire absolute paths into src/config.py so it works on THIS machine
  log "[InstaFormer] Patching absolute paths into src/config.py"
  python3 - "$SCRIPT_DIR" <<'PYEOF'
import re, sys
root = sys.argv[1]
cfg_path = f"{root}/src/config.py"
with open(cfg_path) as f:
    cfg = f.read()
cfg = re.sub(r'INSTAFORMER_REPO_DIR\s*=\s*".*?"', f'INSTAFORMER_REPO_DIR    = "{root}/InstaFormer"', cfg)
cfg = re.sub(r'INSTAFORMER_VENV_PYTHON\s*=\s*".*?"', f'INSTAFORMER_VENV_PYTHON = "{root}/.instaformer-venv/bin/python"', cfg)
cfg = re.sub(r'INSTAFORMER_CKPT\s*=\s*\((.|\n)*?\)', f'INSTAFORMER_CKPT        = "{root}/InstaFormer/checkpoints/instaformer_od_swinl_200.pth"', cfg)
with open(cfg_path, "w") as f:
    f.write(cfg)
print(f"Patched {cfg_path}")
PYEOF
  log "[InstaFormer] Done"
fi

# ═══════════════════════════════════════════════════════════════════════
# 3. LISA — legacy scene-reasoning model (NOT called by the current
#    pipeline; GPT-5 replaced it. Built only for historical reference.)
# ═══════════════════════════════════════════════════════════════════════
if [[ $DO_LISA -eq 1 ]]; then
  log "[LISA] Legacy component — GPT-5 has replaced this at runtime."
  log "[LISA] Cloning repo (skip with --skip-lisa if you don't need it)"
  if [[ ! -d LISA ]]; then
    git clone https://github.com/dvlab-research/LISA.git LISA
    warn "[LISA] predict_single.py (this project's non-interactive CLI wrapper) is not part of the upstream clone — restore it from git history/backup if you need it."
  else
    log "[LISA] Already cloned, skipping"
  fi

  log "[.lisa-venv] Building Python 3.10 environment"
  if [[ ! -x .lisa-venv/bin/python ]]; then
    uv venv .lisa-venv --python 3.10
    uv pip install --python .lisa-venv/bin/python \
      torch==1.13.1 torchvision==0.14.1 \
      --index-url https://download.pytorch.org/whl/cu117
    uv pip install --python .lisa-venv/bin/python -r LISA/requirements.txt
    uv pip install --python .lisa-venv/bin/python flash-attn --no-build-isolation || \
      warn "[LISA] flash-attn build failed — LISA is legacy/unused, safe to ignore unless you specifically need it"
  else
    log "[.lisa-venv] Already exists, skipping"
  fi
  log "[LISA] Done (checkpoint xinlai/LISA-13B-llama2-v1-explanatory auto-downloads from HF on first use)"
fi

# ═══════════════════════════════════════════════════════════════════════
# 4. Hunyuan3D-2.1 — optional image-to-3D post-step (conda env, its own
#    CUDA 12.4 build; hardcoded by src/pipeline.py to live at
#    /home/ubuntu/miniconda3/envs/hunyuan3d — this script installs conda
#    at that exact path so the pipeline needs no further code edits.)
# ═══════════════════════════════════════════════════════════════════════
if [[ $DO_HUNYUAN3D -eq 1 ]]; then
  log "[Hunyuan3D-2.1] Cloning repo"
  if [[ ! -d Hunyuan3D-2.1 ]]; then
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
  else
    log "[Hunyuan3D-2.1] Already cloned, skipping"
  fi

  MINICONDA_DIR="$HOME/miniconda3"
  if [[ ! -x "$MINICONDA_DIR/bin/conda" ]]; then
    log "[conda] Installing Miniconda to $MINICONDA_DIR (pipeline.py hardcodes this path)"
    TMP_INSTALLER="$(mktemp --suffix=.sh)"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$TMP_INSTALLER"
    bash "$TMP_INSTALLER" -b -p "$MINICONDA_DIR"
    rm -f "$TMP_INSTALLER"
  else
    log "[conda] Miniconda already present at $MINICONDA_DIR"
  fi
  # shellcheck disable=SC1091
  source "$MINICONDA_DIR/etc/profile.d/conda.sh"

  if ! conda env list | grep -q '^hunyuan3d '; then
    log "[hunyuan3d] Creating conda env (Python 3.10)"
    conda create -n hunyuan3d python=3.10 -y
    conda activate hunyuan3d
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
      --index-url https://download.pytorch.org/whl/cu124
    pip install -r Hunyuan3D-2.1/requirements.txt
    (cd Hunyuan3D-2.1/hy3dpaint/custom_rasterizer && pip install -e .)
    (cd Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh)
    conda deactivate
  else
    log "[hunyuan3d] conda env already exists, skipping"
  fi

  log "[Hunyuan3D-2.1] Fetching RealESRGAN upscaler weight"
  mkdir -p Hunyuan3D-2.1/hy3dpaint/ckpt
  if [[ ! -f Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth ]]; then
    wget -q --show-progress \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
      -P Hunyuan3D-2.1/hy3dpaint/ckpt
  else
    log "[Hunyuan3D-2.1] RealESRGAN weight already present, skipping"
  fi
  log "[Hunyuan3D-2.1] Done (shape/paint weights auto-download from tencent/Hunyuan3D-2.1 on first run)"
  log "[Hunyuan3D-2.1] Still off by default — set RUN_3D_GENERATION_HUNYUAN3D=True in src/config.py to enable"
fi

# ═══════════════════════════════════════════════════════════════════════
log "All requested environments are set up."
cat <<'EOF'

Next steps:
  1. Edit .env — fill in OPENAI_API_KEY, OPENAI_MODEL, HF_TOKEN.
  2. Accept the gated model licenses on huggingface.co while logged in:
       - facebook/sam3
       - black-forest-labs/FLUX.1-Fill-dev
     then run: .venv/bin/huggingface-cli login   (or `hf auth login`)
  3. Smoke-test on a sample image:
       .venv/bin/python src/pipeline.py data/positive/bunny-7847028_640_rabbit.jpg rabbit
  4. See SETUP.md for full details, troubleshooting, and manual steps.
EOF
