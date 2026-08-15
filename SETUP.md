# SETUP.md — full environment & reproduction guide

This document is the complete reference for reproducing this project's runtime
environment from scratch: what every piece is, why it's isolated the way it
is, and exactly how to install it — by hand or via the automated
[`setup.sh`](setup.sh) script. Read this once end-to-end before running
`setup.sh` if you want to understand what it's about to do to your machine;
otherwise skip straight to [Quick start](#quick-start).

For what the pipeline *does* and how to run it day-to-day, see
[README.md](README.md). This file is only about getting the machine ready.

---

## Table of contents

1. [Quick start](#quick-start)
2. [Why four separate environments](#why-four-separate-environments)
3. [Environment 1 — `.venv` (main pipeline)](#environment-1--venv-main-pipeline)
4. [Environment 2 — `.instaformer-venv` (InstaFormer)](#environment-2--instaformer-venv-instaformer)
5. [Environment 3 — `.lisa-venv` (LISA, legacy)](#environment-3--lisa-venv-lisa-legacy)
6. [Environment 4 — `hunyuan3d` conda env (optional 3D)](#environment-4--hunyuan3d-conda-env-optional-3d)
7. [Environment variables (`.env`)](#environment-variables-env)
8. [Gated model weights (Hugging Face)](#gated-model-weights-hugging-face)
9. [Test images (`data/positive/`)](#test-images-datapositive)
10. [Verifying the install](#verifying-the-install)
11. [Troubleshooting](#troubleshooting)
12. [Known stale/hardcoded paths](#known-stalehardcoded-paths)

---

## Quick start

On a fresh machine with an NVIDIA GPU + driver already installed:

```bash
git clone https://github.com/basil-k-aji-dev/amodal_completion.git
cd amodal_completion
./setup.sh                     # provisions everything (~20-40 min, several GB downloaded)
cp .env.example .env           # (setup.sh does this for you if missing)
$EDITOR .env                   # fill in OPENAI_API_KEY, OPENAI_MODEL, HF_TOKEN
.venv/bin/huggingface-cli login   # or: hf auth login  — needed for gated models
.venv/bin/python src/pipeline.py data/positive/bunny-7847028_640_rabbit.jpg rabbit
```

Flags to skip parts you don't need:

```bash
./setup.sh --skip-lisa          # skip legacy LISA env (not used at runtime)
./setup.sh --skip-hunyuan3d     # skip optional 3D-generation env
./setup.sh --only-main          # just .venv + .env — no InstaFormer/LISA/Hunyuan3D
```

The script is idempotent — re-running it skips anything already built.

---

## Why four separate environments

Each sub-component pins an incompatible Python/PyTorch/CUDA combination that
cannot coexist in one environment with the main pipeline's modern stack:

| Env | Python | Torch / CUDA | Why it can't share `.venv` |
|---|---|---|---|
| `.venv` | 3.12 | 2.10.0 (latest) | Needs bleeding-edge `diffusers`/`transformers` for `FLUX.1-Fill-dev` and the very new, gated `facebook/sam3` model |
| `.instaformer-venv` | 3.8 | 2.1.0 / cu118 | Hard-pinned by **detectron2 0.6 built from a specific 2023 commit** (`80307d2`) — does not build against modern PyTorch or Python ≥3.9 |
| `.lisa-venv` | 3.10 | 1.13.1 / cu117 | LISA's LLaVA/CLIP vision-tower code + `transformers==4.31.0` predate the API the main pipeline's `transformers` now uses |
| `hunyuan3d` (conda) | 3.10 | 2.5.1 / cu124 | Needs a different CUDA **minor** version (12.4) plus a large 3D-mesh stack (`open3d`, `cupy-cuda12x`, a custom CUDA rasterizer extension) that would conflict with the main env's CUDA-linked packages |

Each sub-project runs as an **isolated subprocess**, invoked from the main
`.venv` process via `subprocess.run([venv_python, ...])` (InstaFormer) or a
`conda activate ... && python ...` shell command (Hunyuan3D-2.1). Only plain
files cross the process boundary — PNG masks, a pickled dict of numpy arrays
(InstaFormer), a `.glb`/`.obj` mesh (Hunyuan3D) — so no Python object or
library ABI ever has to match between environments.

```
.venv (Python 3.12)  ──subprocess──▶  .instaformer-venv (Python 3.8)
     │                                   src/instaformer_helper.py calls
     │                                   InstaFormer/demo/demo.py,
     │                                   reads back a pickled panoptic-seg +
     │                                   occlusion/depth-order matrix
     │
     └──subprocess──▶  conda env "hunyuan3d" (Python 3.10)
                         src/pipeline.py shells into
                         Hunyuan3D-2.1/generate_3d.py,
                         reads back a .glb mesh file

.lisa-venv exists on disk but nothing in src/ currently calls into it —
GPT-5 (via the OpenAI Responses API, called from .venv) replaced LISA-13B
for scene reasoning. Kept for historical reference only.
```

---

## Environment 1 — `.venv` (main pipeline)

**Role:** runs `src/pipeline.py` and everything under `src/` — SAM3
segmentation, the GPT-5 reasoning agents, and FLUX.1-Fill-dev inpainting.

**Manager:** [`uv`](https://docs.astral.us/uv/) — installs its own pinned
Python 3.12 interpreter, no system Python version required.

**Manual setup:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv isn't installed
uv sync
```

This reads [`pyproject.toml`](pyproject.toml) / [`uv.lock`](uv.lock) and
creates `.venv/` with the exact locked versions. Key resolved packages (see
`uv.lock` for the full pinned set):

| Package | Version |
|---|---|
| Python | 3.12.13 |
| torch | 2.10.0 |
| torchvision | 0.25.0 |
| diffusers | 0.37.0 |
| transformers | 5.14.1 |
| sam2 | 1.1.0 |
| openai | 2.28.0 |
| accelerate, Pillow, opencv-python, numpy, python-dotenv | see `uv.lock` |

> **Note:** `pyproject.toml`'s `requires-python = ">=3.10"` is a floor, not
> the tested version — this project is built and tested on **3.12**. The
> project's `name`/`description` fields ("amodal-test3", "GPT-5 + SDXL +
> LangGraph") and the `langgraph` dependency are stale leftovers from an
> earlier architecture iteration; the current pipeline is a plain sequential
> 3-agent script with no LangGraph/SDXL involved. Harmless, just don't be
> confused by them.

**Run anything in this env via:**

```bash
.venv/bin/python src/pipeline.py <image> [prompt]
.venv/bin/python test_pipeline.py <image-or-dir> [prompt]
```

---

## Environment 2 — `.instaformer-venv` (InstaFormer)

**Role:** occlusion-order + depth-order prediction (which instance in the
scene occludes which, and their relative depth), used to help Agent 1 pick
the correct occluder/subject pair. Called from `.venv` via
`src/instaformer_helper.py`, one `subprocess.run()` per image.

**Upstream:** [`SNU-VGILab/InstaFormer`](https://github.com/SNU-VGILab/InstaFormer)
(built on `SNU-VGILab/InstaOrder`, "Instance-wise Holistic Order Prediction
in Natural Scenes," extending Lee & Park CVPR 2022). Vendored unmodified
into `InstaFormer/` at the repo root — gitignored (multi-GB, environment-
specific), **not part of this git repo**, must be cloned separately.

### Manual setup

```bash
git clone https://github.com/SNU-VGILab/InstaFormer InstaFormer

# Vendored, unmodified detectron2, pinned to the exact commit InstaFormer
# was built against (its own README: "detectron2 0.6 built from source in
# commit 80307d2 due to import issues"):
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2 && git checkout 80307d2 && cd ..

# Python 3.8 venv (uv substitutes for InstaFormer's own conda-based
# quick_install.sh — same package set, uv-managed instead of conda-managed):
uv venv .instaformer-venv --python 3.8
uv pip install --python .instaformer-venv/bin/python \
  torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu118
uv pip install --python .instaformer-venv/bin/python \
  opencv-python ipython shapely h5py scipy submitit scikit-image \
  cython timm einops scikit-learn wandb numpy
uv pip install --python .instaformer-venv/bin/python \
  "git+https://github.com/cocodataset/panopticapi.git"
uv pip install --python .instaformer-venv/bin/python -e ./detectron2

# Custom CUDA op (multi-scale deformable attention, used by Mask2Former):
cd InstaFormer/mask2former/modeling/pixel_decoder/ops
../../../../../.instaformer-venv/bin/python setup.py build install
cd -
```

Confirmed installed versions in this project's own `.instaformer-venv`:
Python 3.8.20, torch 2.1.0+cu118, torchvision 0.16.0+cu118, detectron2 0.6
(editable install from the pinned commit), opencv-python, shapely, h5py,
scipy, scikit-image, scikit-learn, timm, einops, wandb, panopticapi, numpy
1.24.4.

### Checkpoint

Download the **InstaFormer^o,d SWIN-L†₂₀₀** joint occlusion+depth model
(~1.2 GB) into `InstaFormer/checkpoints/`:

```bash
mkdir -p InstaFormer/checkpoints
pip install gdown   # in any env
gdown "1lH_cn1SqDl7jMv5BP8DhuRYtzME0PCZI" \
  -O InstaFormer/checkpoints/instaformer_od_swinl_200.pth
```

(Google Drive file ID `1lH_cn1SqDl7jMv5BP8DhuRYtzME0PCZI`, from InstaFormer's
own MODEL_ZOO table.)

### Wiring it into the pipeline

Edit these four values in [`src/config.py`](src/config.py) to absolute paths
on your machine (`setup.sh` does this automatically):

```python
INSTAFORMER_REPO_DIR    = "/abs/path/to/amodal_completion/InstaFormer"
INSTAFORMER_VENV_PYTHON = "/abs/path/to/amodal_completion/.instaformer-venv/bin/python"
INSTAFORMER_CONFIG      = ("configs/instaorder/occlusion_depth/all/swin/"
                           "maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
INSTAFORMER_CKPT        = "/abs/path/to/amodal_completion/InstaFormer/checkpoints/instaformer_od_swinl_200.pth"
```

`INSTAFORMER_CONFIG` is a path relative to `INSTAFORMER_REPO_DIR` and doesn't
need editing — it ships inside the InstaFormer clone at that location.

To disable InstaFormer entirely (mask fusion falls back to SAM3 text-prompt
segmentation alone), set `USE_INSTAFORMER = False` in `src/config.py`.

---

## Environment 3 — `.lisa-venv` (LISA, legacy)

**Role:** historical only. [`dvlab-research/LISA`](https://github.com/dvlab-research/LISA)
("Reasoning Segmentation via Large Language Model," CVPR'24 Oral) — a 13B
LLaVA-based multimodal model that used to do the scene-reasoning job GPT-5
does now via the OpenAI Responses API. **Nothing in `src/` currently calls
into LISA** — it was superseded during the "restructure into a clean 3-agent
pipeline" refactor. It's kept in the tree for reference/rollback, not
required to run the pipeline.

Skip it entirely with `./setup.sh --skip-lisa` unless you specifically want
to experiment with the old approach.

### Manual setup (only if you need it)

```bash
git clone https://github.com/dvlab-research/LISA.git LISA

uv venv .lisa-venv --python 3.10
uv pip install --python .lisa-venv/bin/python \
  torch==1.13.1 torchvision==0.14.1 \
  --index-url https://download.pytorch.org/whl/cu117
uv pip install --python .lisa-venv/bin/python -r LISA/requirements.txt
uv pip install --python .lisa-venv/bin/python flash-attn --no-build-isolation
```

Confirmed installed versions: Python 3.10.12, torch 1.13.1+cu117,
torchvision 0.14.1+cu117, transformers 4.31.0, peft 0.4.0, gradio 3.39.0,
openai 0.27.8 (legacy pre-v1 SDK), opencv-python 4.8.0.74, pycocotools
2.0.6.

Model weights (e.g. `xinlai/LISA-13B-llama2-v1-explanatory`) auto-download
from Hugging Face on first use — no local checkpoint files are vendored.

This project added a non-interactive `LISA/predict_single.py` CLI wrapper
(`--image`/`--prompt`/`--out_dir`) around LISA's own `chat.py` model-loading
code, producing a binary mask PNG in the same format the rest of the
pipeline uses. It is **not part of upstream LISA** — if it's missing after a
fresh clone, pull it from an earlier commit/backup of this repo.

---

## Environment 4 — `hunyuan3d` conda env (optional 3D)

**Role:** optional post-step. Turns the finished 2D RGBA cutout into a
textured 3D mesh (`.glb`). Off by default
(`RUN_3D_GENERATION_HUNYUAN3D = False` in `src/config.py`) — enable only if
you want the 3D step; failure here is non-fatal to the main 2D pipeline
either way.

**Upstream:** [`Tencent-Hunyuan/Hunyuan3D-2.1`](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
— two-stage image-to-3D: shape generation (DiT flow-matching, 3.3B params)
then PBR texture synthesis (2B params). VRAM needs per Hunyuan3D's own docs:
~10 GB shape-only, ~21 GB texture-only, ~29 GB combined.

Unlike the other three, this uses **conda**, not `uv` — Hunyuan3D's own
install path relies on conda-style environment activation, and
`src/pipeline.py` invokes it with a hardcoded
`conda activate hunyuan3d && ...` shell command
(see [Known stale/hardcoded paths](#known-stalehardcoded-paths) — the
conda install *location* is also hardcoded, not just the env name).

### Manual setup

```bash
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git

# src/pipeline.py hardcodes /home/ubuntu/miniconda3 as the conda install
# path (LD_LIBRARY_PATH construction in _run_hunyuan3d_generation()) —
# install Miniconda at exactly that path, or see the note below to
# retarget it to your own conda install.
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda create -n hunyuan3d python=3.10 -y
conda activate hunyuan3d
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r Hunyuan3D-2.1/requirements.txt
cd Hunyuan3D-2.1/hy3dpaint/custom_rasterizer && pip install -e . && cd ../..
cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh && cd ../../..
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
  -P Hunyuan3D-2.1/hy3dpaint/ckpt
conda deactivate
```

Key pinned deps (from Hunyuan3D-2.1's own `requirements.txt`): torch 2.5.1
+cu124, transformers==4.46.0, diffusers==0.30.0, accelerate==1.1.1,
huggingface-hub==0.30.2, trimesh==4.4.7, open3d==0.18.0, gradio==5.33.0,
cupy-cuda12x==13.4.1, onnxruntime==1.16.3, deepspeed (unpinned).

Shape/texture model weights (`tencent/Hunyuan3D-2.1` — `hunyuan3d-dit-v2-1`
and `hunyuan3d-paintpbr-v2-1` subfolders) auto-download from Hugging Face on
first run; no manual download needed beyond the RealESRGAN weight above.

This project added `Hunyuan3D-2.1/generate_3d.py`, a thin CLI wrapper:

```bash
python generate_3d.py --image /path/to/subject_final.png \
  --out-dir /path/to/final --prefix subject_hunyuan3d
```

It's not part of upstream Hunyuan3D-2.1 either — same caveat as LISA's
`predict_single.py` if it's missing after a fresh clone.

### Enabling it

Set in `src/config.py`:

```python
RUN_3D_GENERATION_HUNYUAN3D = True
HUNYUAN3D_REPO_DIR = "/abs/path/to/amodal_completion/Hunyuan3D-2.1"
HUNYUAN3D_CONDA_ENV = "hunyuan3d"
```

---

## Environment variables (`.env`)

```bash
cp .env.example .env
```

Then fill in all three (no other variables are read):

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key. Must have **Responses API** access with **gpt-5 enabled** — used for all three GPT agents (occlusion reasoning, and the vision reviewer). |
| `OPENAI_MODEL` | Model name passed to the Responses API, e.g. `gpt-5.2`. |
| `HF_TOKEN` | Hugging Face access token. Required for gated model downloads — `facebook/sam3` and `black-forest-labs/FLUX.1-Fill-dev` both require accepting a license on huggingface.co and a token with access. |

`src/runtime.py` loads `.env` from the repo root and raises immediately if
`OPENAI_API_KEY` or `OPENAI_MODEL` is missing — you'll know right away if
this step was skipped.

---

## Gated model weights (Hugging Face)

Everything except InstaFormer/LISA/Hunyuan3D checkpoints (which are fetched
separately, see above) **auto-downloads to your local HF cache on first
pipeline run** — no manual download step. But two of them are gated and
require you to accept a license first, while logged in on huggingface.co:

1. **[facebook/sam3](https://huggingface.co/facebook/sam3)** — segmentation
   backbone used throughout Agent 1.
2. **[black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)** —
   the inpainting model (Agent 2).

After accepting both licenses, authenticate the machine:

```bash
.venv/bin/huggingface-cli login      # or: .venv/bin/hf auth login
# paste your HF_TOKEN when prompted
```

(`depth-anything/Depth-Anything-V2-Large-hf` is also used if
`USE_DEPTH_GUIDED_FILL=True`, but it's not gated.)

---

## Test images (`data/positive/`)

`data/positive/` contains 14 curated positive test cases (one clearly-named
occluded subject per image, e.g. `bunny-7847028_640_rabbit.jpg`,
`corgi-puppy-ball_dog.jpg`) — copied into this repo from the external
`website/positive/` dataset so this repo is self-contained and reproducible
without depending on a sibling project. The trailing `_<class>` suffix in
each filename is the expected-subject hint that `test_pipeline.py`,
`Makefile`, and `run_batch*.sh` all auto-derive prompts from.

Try the pipeline against any of them:

```bash
.venv/bin/python src/pipeline.py data/positive/corgi-puppy-ball_dog.jpg dog
# or let Agent 1 auto-detect the subject (no prompt hint):
.venv/bin/python src/pipeline.py data/positive/bunny-7847028_640_rabbit.jpg
# or run the whole folder:
.venv/bin/python test_pipeline.py data/positive/
```

> Note: `run_batch.sh`, `run_batch_20.sh`, and the `Makefile`'s
> `POSITIVES_DIR`/`DATA_DIR` still point at the original external location
> (`/home/ubuntu/Workspace/website/positive` and `/data`) — those are
> larger, ongoing datasets outside this repo used for bulk evaluation runs,
> not required for basic setup/reproduction. `data/positive/` here is a
> small, self-contained sample sufficient to verify your setup works.

---

## Verifying the install

```bash
# 1. Main env sanity check
.venv/bin/python -c "import torch, diffusers, transformers, openai; print(torch.__version__, torch.cuda.is_available())"

# 2. InstaFormer env sanity check
.instaformer-venv/bin/python -c "import torch, detectron2, cv2; print(torch.__version__, torch.cuda.is_available())"

# 3. Full pipeline smoke test (uses InstaFormer + SAM3 + GPT-5 + Flux end to end)
.venv/bin/python src/pipeline.py data/positive/flowers-8991384_640_vase.jpg vase
```

A successful run creates `output/flowers-8991384_640_vase/_flux_cutout_vase/`
containing `flux_completed_rgba.png` and `comparison.png`. If the run fails,
see [Troubleshooting](#troubleshooting) below.

---

## Troubleshooting

**`EnvironmentError: OPENAI_API_KEY not set`**
`.env` is missing or incomplete — see [Environment variables](#environment-variables-env).

**403 / gated-repo error downloading `facebook/sam3` or `FLUX.1-Fill-dev`**
You haven't accepted the license on huggingface.co for that model, or
haven't run `huggingface-cli login` with a token that has access. See
[Gated model weights](#gated-model-weights-hugging-face).

**InstaFormer subprocess fails / times out (`instaformer_helper.py`)**
- Confirm the four `INSTAFORMER_*` paths in `src/config.py` are absolute and
  correct for this machine (`setup.sh` patches these automatically; if you
  set up manually, don't skip this step).
- Confirm the checkpoint exists at `INSTAFORMER_CKPT`'s path (~1.2 GB —
  a partial/failed download will fail silently at load time).
- Try running InstaFormer's own `demo/demo.py` directly to isolate whether
  the fault is in InstaFormer itself or in how the pipeline invokes it:
  ```bash
  .instaformer-venv/bin/python InstaFormer/demo/demo.py \
    --config-file InstaFormer/configs/instaorder/occlusion_depth/all/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml \
    --input data/positive/corgi-puppy-ball_dog.jpg --output /tmp/instaformer_test \
    --opts MODEL.WEIGHTS InstaFormer/checkpoints/instaformer_od_swinl_200.pth \
           MODEL.DEVICE cuda TEST.OCCLUSION_EVALUATION False TEST.DEPTH_EVALUATION False
  ```
- If detectron2 fails to import: it must be built from the exact pinned
  commit `80307d2` — a newer detectron2 checkout will not build against
  torch 2.1.0/Python 3.8 the same way.

**CUDA OOM on Flux-Fill**
Set `FLUX_FILL_CPU_OFFLOAD = True` in `src/config.py` if your GPU has less
than ~24 GB VRAM (streams the model through system RAM via
`enable_model_cpu_offload`, much slower but fits smaller cards). Also check
`GPU_MEMORY_LIMIT_GB` isn't set higher than your card's actual VRAM.

**Hunyuan3D subprocess fails**
It's non-fatal by design — the 2D pipeline output is already saved
regardless. Check `HUNYUAN3D_REPO_DIR`/`HUNYUAN3D_CONDA_ENV` in
`src/config.py`, and that conda is installed at the path
`src/pipeline.py`'s `_run_hunyuan3d_generation()` expects (see next
section) — or just leave `RUN_3D_GENERATION_HUNYUAN3D = False`, it's
optional.

**`uv python install cpython-3.8` fails (no prebuilt interpreter for your
platform/glibc)**
Fall back to a system/conda Python 3.8 and point `uv venv --python
/path/to/python3.8` at it, or build `.instaformer-venv` with conda instead,
following InstaFormer's own `quick_install.sh`.

---

## Known stale/hardcoded paths

Worth knowing about so they don't cause confusing failures:

- **`src/pipeline.py`'s `_run_hunyuan3d_generation()`** hardcodes
  `/home/ubuntu/miniconda3` (both the `conda.sh` source path and the
  `LD_LIBRARY_PATH` entries for `torch/lib` and `nvidia/cuda_runtime/lib`)
  — not read from `src/config.py`. If your conda lives elsewhere, either
  install Miniconda at that exact path (what `setup.sh` does) or edit that
  function directly.
- **`src/pipeline.py`'s `DEFAULT_IMAGE`** and **`src/config.py`'s
  `IMAGE_PATH`** default to a different machine/user's path
  (`/home/basil-k-aji/Desktop/Workspace/RD/website/...`). Harmless — always
  pass an explicit image path as the first CLI argument and these are never
  used.
- **`Makefile`'s `POSITIVES_DIR`** and **`run_batch.sh`/`run_batch_20.sh`'s
  `DATA_DIR`** point at `/home/ubuntu/Workspace/website/...`, a sibling
  dataset project outside this repo, used for larger bulk-evaluation runs.
  Not required for basic setup — use `data/positive/` (bundled in this repo)
  or edit those paths to point at your own larger dataset.
- **`InstaFormer`'s git remote** is actually `SNU-VGILab/InstaOrder` on
  GitHub even though it's referred to as "InstaFormer" throughout this
  project and its own README/docs — same underlying project, just double-
  check you land on the right repo/branch if the clone URL above ever
  moves.
