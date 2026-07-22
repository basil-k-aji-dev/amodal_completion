# amodal_completion

Open-world amodal appearance completion. Given an image with an occluded object, this pipeline segments the occluder, infers what's hidden, and inpaints the missing pixels — producing a clean RGBA cutout of the fully-revealed subject.

Driven by **SAM3** (segmentation) + **GPT-5 Responses API** (scene reasoning) + **InstaOrder** (occlusion ordering) + **FLUX.1-Fill-dev** (inpainting), wrapped in an iterative off-frame extension loop adapted from Jiang Ao et al., CVPR 2025.

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
   image ────►      │  Agent 1 — occlusion_agent                   │
                    │   • SAM3 auto-segment                        │
                    │   • InstaOrder occlusion-order ranking       │
                    │   • GPT-5 (reasoning='high') picks the       │
                    │     occluder + visible target click          │
                    │   • SAM3 point-prompt refines visible mask   │
                    │  → visible_mask.png, occluder_mask.png       │
                    └──────────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────────┐
   visible+occluder │  Inpaint region (Jiang Ao iter-0 logic)      │
                    │   hidden = dilate(occluder, 5×5, 3) ∖ visible│
                    └──────────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────────┐
                    │  FLUX.1-Fill-dev                             │
                    │   • base = visible-only cutout on neutral    │
                    │     gray (5-px erosion to drop edge bleed)   │
                    │   • mask = hidden region                     │
                    │   • prompt = "complete <subject>, …"         │
                    └──────────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────────┐
                    │  SAM3 PostSeg (argmax-IoU vs visible_mask)   │
                    │  → final silhouette of the painted subject   │
                    └──────────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────────┐
                    │  Iterative off-frame extension               │
                    │   while final_mask touches a frame edge:     │
                    │     pad canvas 150 px on touched sides       │
                    │     Flux outpaints into the new gray strip   │
                    │     SAM3 re-segments                         │
                    │   (capped at MAX_OFFFRAME_ITERS=3)           │
                    └──────────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────────┐
                    │  Alpha-blended RGBA output                   │
                    │   (5-px distance-transform-weighted feather  │
                    │    at the visible-silhouette boundary)       │
                    └──────────────────────────────────────────────┘
```

## Why this stack

- **SAM3** for visible/occluder segmentation — newer than LISA / Grounded-SAM, no 13B LLM backbone needed.
- **GPT-5 via Responses API** with `reasoning='high'` replaces local language models (LISA-13B) for scene reasoning. Network call, ~2 GB GPU savings.
- **No predicted amodal polygon** — `USE_AMODAL_COMPLETION=False`. The Jiang Ao iter-0 inpaint mask (dilated occluder) lets Flux decide where hidden anatomy goes. Avoids the 60-vertex polygon truncating thin appendages like bird legs.
- **FLUX.1-Fill-dev** for inpainting — strongest anatomy/texture priors available. Runs fully on device on 24 GB+ cards; falls back to `enable_model_cpu_offload` (`FLUX_FILL_CPU_OFFLOAD=True`) on smaller GPUs.
- **Iterative off-frame extension** (Jiang Ao 2025) — silhouettes that touch the frame edge get padded canvas + outpaint until contained.
- **IoU-based PostSeg** + **alpha blending** ported from Jiang Ao's `filter_out_amodal_segmentation` + `alpha_blending` (`amodal/main.py`).

## Hardware

Designed and tested on:

| | |
|---|---|
| GPU | RTX 3060 12 GB (Ampere) |
| RAM | 14 GB (with 32 GB swap on NVMe) |
| OS | Linux |

Per Flux pass: ~9–12 s/step × 50 steps ≈ 7–10 min. Typical run with 1–2 off-frame iters: 15–25 min wall-clock. GPU is PCIe-bound, not compute-bound — Flux is 24 GB in bf16 and must stream through `enable_model_cpu_offload` on this hardware. On 24 GB+ cards Flux runs fully on device and this drops to a couple minutes per pass.

Quality-neutral Ampere perf flags (`cudnn.benchmark`, TF32 matmul, VAE tiling/slicing) are enabled in `src/pipeline.py`.

## Setup

### Python environment

```bash
uv sync
```

Requires Python 3.10+. Key deps: `torch`, `diffusers>=0.31`, `transformers`, `openai`, `python-dotenv`.

### External dependency: InstaOrder / InstaFormer

The pipeline calls into InstaFormer (SNU-VGILab), a fork of InstaOrder (POSTECH-CVLab), for occlusion+depth-order ranking. It runs as a subprocess in its own isolated venv (`.instaformer-venv/`, Python 3.8 + Detectron2 + torch 2.1.0/cu118 — incompatible with this project's own environment). Clone it and download the checkpoint:

```bash
# Adjust paths in src/config.py to match
git clone https://github.com/SNU-VGILab/InstaFormer InstaFormer
# Download the InstaFormer occlusion+depth-order checkpoint into InstaFormer/checkpoints/
```

Then edit `src/config.py`:

```python
INSTAFORMER_REPO_DIR    = "/abs/path/to/InstaFormer"
INSTAFORMER_VENV_PYTHON = "/abs/path/to/.instaformer-venv/bin/python"
INSTAFORMER_CKPT        = "/abs/path/to/InstaFormer/checkpoints/instaformer_od_swinl_200.pth"
```

### Environment variables

```bash
cp .env.example .env
# fill in OPENAI_API_KEY (Responses API + gpt-5 access required)
```

### Model weights

All other models (SAM3, Flux-Fill, Depth-Anything) auto-download to your Hugging Face cache on first run. Flux-Fill is gated — accept the license at `huggingface.co/black-forest-labs/FLUX.1-Fill-dev`.

## Running

Single image:

```bash
.venv/bin/python src/pipeline.py /path/to/photo.jpg [prompt]
```

`image_path` defaults to `DEFAULT_IMAGE` in `src/pipeline.py` if omitted. The
optional `prompt` argument overrides `config.INPUT_PROMPT` for just that run
(hints Agent 1 what the occluded subject is, e.g. `"rabbit"`); leave it out
to let Agent 1 auto-detect.

Single image or a whole directory, via the common test runner:

```bash
.venv/bin/python test_pipeline.py /path/to/photo.jpg [prompt]
.venv/bin/python test_pipeline.py /path/to/image_dir/
```

In directory mode every `.jpg`/`.jpeg`/`.png` inside is run in turn (each as
its own subprocess, to keep GPU memory clean between images); the prompt hint
per image defaults to the filename's trailing `_<class>` suffix (e.g.
`horse-123_640_horse.jpg` → `horse`).

`run_batch.sh` / `run_batch_20.sh` are the curated large-batch runners used
against the full dataset in `website/data/`.

Outputs land in `output/<image-stem>/_flux_cutout_<subject>/`:

- `flux_completed_rgba.png` — alpha-blended cutout (feathered edges)
- `flux_completed_white_bg.png` — same subject on white background
- `flux_completed_restored.png` — subject on neutral gray
- `offframe_final_*` — when the off-frame loop fires
- `comparison.png` — 5-panel A/B/C/D/E walkthrough

## Key config flags

All in `src/config.py`. Currently-validated defaults:

| Flag | Value | Notes |
|---|---|---|
| `INPAINT_BACKEND` | `flux_fill` | Flux is the only validated backend |
| `USE_AMODAL_COMPLETION` | `False` | Skip GPT polygon, let Flux+SAM3 decide silhouette |
| `USE_INSTAFORMER` | `True` | InstaFormer is the occluder/depth-order ranking signal (via subprocess) |
| `USE_REVIEWER` | `True` | Agent 3 — GPT-vision reviewer scores each Flux completion and retries on a low score |
| `REVIEWER_SCORE_THRESHOLD` | `7.0` | score ≥ this = accepted, stop retrying |
| `REVIEWER_MAX_RETRIES` | `2` | extra Flux attempts beyond the first if the reviewer rejects |
| `USE_OFFFRAME_EXTENSION` | `False` | iterative off-frame canvas extension; disabled while in-frame quality is being tuned |
| `FLUX_FILL_CPU_OFFLOAD` | `False` | Flux runs fully on device; enable on <24 GB cards |
| `GPU_MEMORY_LIMIT_GB` | `44.0` | hard cap, leaves OS headroom (tuned for a 48 GB card) |

## What's intentionally not in this repo

Disabled / experimental code paths from upstream that were not part of the working architecture:

- PSALM referring-expression seg
- pix2gestalt amodal review
- AISFormer head (inference path — flag plumbing for feature extraction remains)
- ControlNet-SD-1.5 inpaint backend — fully removed, no fallback; FLUX.1-Fill-dev is the only inpainting backend
- Mixed-Context Diffusion Sampling
- CLIP-grounded occluder discovery
- Grounding DINO occluder fallback
- mmgp / group offload / disk offload (tested, didn't beat sequential)

The flag plumbing for these still exists in `src/config.py` so they can be re-enabled, but no vendored model dirs are bundled.

## Credits

- Iterative off-frame logic + IoU PostSeg + alpha blending adapted from Jiang Ao et al., "Open-World Amodal Appearance Completion", CVPR 2025 (`github.com/saraao/amodal`).
- InstaOrder from Lee & Park, "Instance-wise Occlusion and Depth Orders in Natural Scenes", CVPR 2022.
- Flux.1-Fill from Black Forest Labs.
- SAM3 from Meta.
