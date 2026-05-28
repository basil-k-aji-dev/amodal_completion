"""
server.py — full amodal pipeline server (runs on any GPU host).

Runs on a Lightning Studio, Colab (via ngrok/cloudflared), an EC2 box, etc.
The client.py dispatcher picks the URL from config.RUN_ENV
("lightning" → LIGHTNING_URL, "colab" → COLAB_URL).

Warms SAM3 + CLIP + Flux at startup so the first request doesn't pay the load
tax. GPT is an API call — no warmup needed.

Endpoints:
  GET  /healthz   — liveness + which models are loaded
  POST /process   — full pipeline (multipart image upload) → base64 ZIP of
                    output/<stem>/_flux_cutout_person/
  POST /inpaint   — Flux-only inpaint (image+mask+prompt) for clients using
                    config.INPAINT_BACKEND="remote_flux"

Launch:
  pip install fastapi uvicorn python-multipart requests
  python server.py            # binds 0.0.0.0:8000
"""

from __future__ import annotations

import base64
import io
import os
import time
import zipfile
from pathlib import Path
from typing import List

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

import config
import runtime
runtime.SERVER_MODE = True   # keep all models GPU-resident; disable free/unload
from models import clip as clip_model
from models import flux as flux_model
from models import sam3 as sam3_model
from pipeline.graph import run as run_pipeline

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))
UPLOAD_ROOT = Path(os.environ.get("UPLOAD_ROOT", "/tmp/amodal_uploads"))
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="amodal_server", version="1.0")
_WARM: dict = {"sam3": False, "clip": False, "flux": False}


@app.on_event("startup")
def _warmup():
    if not torch.cuda.is_available():
        print("[server] WARNING: no CUDA — pipeline will be very slow / may fail")
        return
    print(f"[server] CUDA OK ({torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB VRAM)")
    for name, loader in (("flux", flux_model._get_flux_fill_pipe),
                         ("sam3", sam3_model._get_sam3),
                         ("clip", clip_model._get_clip)):
        try:
            t0 = time.time()
            loader()
            _WARM[name] = True
            print(f"[server] {name} ready in {time.time() - t0:.1f}s")
        except Exception as exc:                              # noqa: BLE001
            print(f"[server] {name} warmup failed: {exc!r}")


@app.get("/healthz")
def healthz():
    return {
        "ok":         True,
        "warm":       _WARM,
        "cuda":       torch.cuda.is_available(),
        "device":     (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else "cpu"),
        "vram_gb":    (round(torch.cuda.get_device_properties(0).total_memory
                             / 1024 ** 3, 2) if torch.cuda.is_available() else 0),
        "model_id":   getattr(config, "FLUX_FILL_MODEL_ID", "?"),
        "run_env":    getattr(config, "RUN_ENV", "?"),
        "inpaint":    getattr(config, "INPAINT_BACKEND", "?"),
    }


@app.post("/process")
async def process(
    image:  UploadFile = File(..., description="Image to amodal-complete"),
    target: str        = Form(default="", description="Subject class, e.g. 'pigeon'"),
):
    """Full pipeline on the uploaded image. Returns a base64 ZIP of everything
    under output/<image-stem>/_flux_cutout_person/."""
    ts = int(time.time() * 1000)
    upload_dir = UPLOAD_ROOT / f"req_{ts}"
    upload_dir.mkdir(parents=True, exist_ok=True)
    img_path = upload_dir / (image.filename or "input.jpg")
    img_path.write_bytes(await image.read())

    if target:
        config.TARGET = target
    config.IMAGE_PATH = str(img_path)

    print(f"[server] /process  image={img_path.name}  target={target!r}")
    t0 = time.time()
    try:
        run_pipeline(str(img_path), target)
    except Exception as exc:                                  # noqa: BLE001
        print(f"[server] pipeline error: {exc!r}")
        return {"ok": False, "error": repr(exc),
                "duration_s": round(time.time() - t0, 1)}
    dt = time.time() - t0

    stem = img_path.stem
    out_dir = runtime.BASE_DIR / "output" / stem / "_flux_cutout_person"
    if not out_dir.exists():
        return {"ok": False, "error": f"pipeline finished but {out_dir} is missing",
                "duration_s": round(dt, 1)}
    files = sorted(p for p in out_dir.rglob("*") if p.is_file())
    if not files:
        return {"ok": False, "error": f"output dir is empty: {out_dir}",
                "duration_s": round(dt, 1)}

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=str(f.relative_to(out_dir)))
    print(f"[server] /process done  {len(files)} files  {dt:.1f}s  "
          f"zip={len(buf.getvalue())//1024} KB")
    return {
        "ok":         True,
        "duration_s": round(dt, 1),
        "stem":       stem,
        "n_files":    len(files),
        "zip_b64":    base64.b64encode(buf.getvalue()).decode(),
    }


def _round_16(n: int, cap: int = 1280) -> int:
    return max(16, min(round(n / 16) * 16, cap))


@app.post("/inpaint")
async def inpaint(
    image:    UploadFile = File(..., description="Base image PNG"),
    mask:     UploadFile = File(..., description="Binary mask PNG (white=inpaint)"),
    prompt:   str        = Form(...),
    steps:    int        = Form(50),
    guidance: float      = Form(45.0),
    samples:  int        = Form(1),
    seed:     int        = Form(0),
):
    """Flux-only inpaint for clients using INPAINT_BACKEND='remote_flux'."""
    pipe = flux_model._get_flux_fill_pipe()
    if pipe is None:
        return {"error": "Flux-Fill pipeline unavailable on this host"}

    pil_image = Image.open(io.BytesIO(await image.read())).convert("RGB")
    pil_mask  = Image.open(io.BytesIO(await mask.read())).convert("L")

    w0, h0 = pil_image.size
    flux_w, flux_h = _round_16(w0), _round_16(h0)
    if (flux_w, flux_h) != (w0, h0):
        pil_image = pil_image.resize((flux_w, flux_h), Image.LANCZOS)
        pil_mask  = pil_mask.resize((flux_w, flux_h), Image.NEAREST)

    print(f"[server] /inpaint  size={flux_w}×{flux_h}  steps={steps}  "
          f"guidance={guidance}  samples={samples}  seed={seed}")
    t0 = time.time()
    outs: List[Image.Image] = []
    for i in range(max(1, samples)):
        gen = torch.Generator(device="cuda").manual_seed(int(seed) + i)
        result = pipe(
            image               = pil_image,
            mask_image          = pil_mask,
            prompt              = prompt,
            num_inference_steps = int(steps),
            guidance_scale      = float(guidance),
            generator           = gen,
            width               = flux_w,
            height              = flux_h,
        ).images[0]
        if (w0, h0) != result.size:
            result = result.resize((w0, h0), Image.LANCZOS)
        outs.append(result)
    print(f"[server] /inpaint done in {time.time() - t0:.1f}s ({len(outs)} sample(s))")

    enc = []
    for im in outs:
        b = io.BytesIO(); im.save(b, "PNG")
        enc.append(base64.b64encode(b.getvalue()).decode())
    return {"images_b64": enc}


if __name__ == "__main__":
    uvicorn.run("server:app", host=HOST, port=PORT, log_level="info")
