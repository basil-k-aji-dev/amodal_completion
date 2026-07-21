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

Quality-neutral Ampere perf flags (`cudnn.benchmark`, TF32 matmul, VAE tiling/slicing) are enabled in `main.py`.

## Setup

### Python environment

```bash
uv sync
```

Requires Python 3.10+. Key deps: `torch`, `diffusers>=0.31`, `transformers`, `openai`, `langgraph`, `python-dotenv`.

### External dependency: InstaOrder

The pipeline calls into the InstaOrder repo (POSTECH-CVLab) for same-class occluder ranking. Clone it and download the checkpoint:

```bash
# Adjust paths in config.py to match
git clone https://github.com/POSTECH-CVLab/InstaOrder ../InstaOrder
# Download InstaOrder_InstaOrderNet_od.pth.tar into InstaOrder/InstaOrder_ckpt/
```

Then edit `config.py`:

```python
INSTAORDER_REPO_DIR = "/abs/path/to/InstaOrder"
INSTAORDER_CKPT     = "/abs/path/to/InstaOrder/InstaOrder_ckpt/InstaOrder_InstaOrderNet_od.pth.tar"
```

### Environment variables

```bash
cp .env.example .env
# fill in OPENAI_API_KEY (Responses API + gpt-5 access required)
```

### Model weights

All other models (SAM3, Flux-Fill, Depth-Anything) auto-download to your Hugging Face cache on first run. Flux-Fill is gated — accept the license at `huggingface.co/black-forest-labs/FLUX.1-Fill-dev`.

## Running

Set the target image and class in `config.py`:

```python
IMAGE_PATH = "/path/to/photo.jpg"
TARGET     = "rabbit"   # or pigeon, blackbird, sheep, etc.
```

Then:

```bash
.venv/bin/python test_flux_cutout_person.py
```

Outputs land in `output/<image-stem>/_flux_cutout_person/`:

- `flux_completed_rgba.png` — alpha-blended cutout (feathered edges)
- `flux_completed_white_bg.png` — same subject on white background
- `flux_completed_restored.png` — subject on neutral gray
- `offframe_final_*` — when the off-frame loop fires
- `comparison.png` — 5-panel A/B/C/D/E walkthrough

## Key config flags

All in `config.py`. Currently-validated defaults:

| Flag | Value | Notes |
|---|---|---|
| `INPAINT_BACKEND` | `flux_fill` | Flux is the only validated backend |
| `USE_AMODAL_COMPLETION` | `False` | Skip GPT polygon, let Flux+SAM3 decide silhouette |
| `USE_PIX2GESTALT_AMODAL` | `False` | pix2gestalt review path disabled |
| `USE_PSALM` | `False` | over-segmented nature scenes |
| `USE_INSTAORDER` | `True` | InstaOrder is the only useful occluder-ranking signal here |
| `USE_DEPTH_RANK` | `True` | corroborates InstaOrder on same-class occluders |
| `FLUX_FILL_CPU_OFFLOAD` | `False` | Flux runs fully on device; disable on <24 GB cards |
| `GPU_MEMORY_LIMIT_GB` | `44.0` | hard cap, leaves OS headroom (tuned for a 48 GB card) |

## What's intentionally not in this repo

Disabled / experimental code paths from upstream that were not part of the working architecture:

- PSALM referring-expression seg
- pix2gestalt amodal review
- AISFormer head
- ControlNet-SD-1.5 inpaint backend
- Mixed-Context Diffusion Sampling
- CLIP-grounded occluder discovery
- Grounding DINO occluder fallback
- mmgp / group offload / disk offload (tested, didn't beat sequential)
- Reviewer (Agent 3) + retry loop

The flag plumbing for these still exists in `config.py` and `main.py` so they can be re-enabled, but no vendored model dirs are bundled.

## Credits

- Iterative off-frame logic + IoU PostSeg + alpha blending adapted from Jiang Ao et al., "Open-World Amodal Appearance Completion", CVPR 2025 (`github.com/saraao/amodal`).
- InstaOrder from Lee & Park, "Instance-wise Occlusion and Depth Orders in Natural Scenes", CVPR 2022.
- Flux.1-Fill from Black Forest Labs.
- SAM3 from Meta.
