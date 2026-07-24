"""Process-wide runtime setup: environment loading, device/GPU-cap setup,
the OpenAI client, and the two small utilities (RAM release, image encoding,
GPT-vision calls) every other module in this package depends on.
"""
import base64
import json
import os
import re

import ctypes
import gc
import torch
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

import config

BASE_DIR = Path(__file__).resolve().parent.parent  # amodal_completion/ (parent of src/)


def _release_ram():
    """gc.collect + malloc_trim: forces Python to return freed memory to the OS immediately."""
    gc.collect()
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(BASE_DIR / ".env")

API_KEY   = os.environ.get("OPENAI_API_KEY")
GPT_MODEL = os.environ.get("OPENAI_MODEL")
if not GPT_MODEL:
    raise EnvironmentError("OPENAI_MODEL is not set in .env")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hard-cap VRAM so model swaps don't OOM.
if DEVICE == "cuda" and config.GPU_MEMORY_LIMIT_GB:
    _total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    _fraction = min(config.GPU_MEMORY_LIMIT_GB / _total_gb, 1.0)
    torch.cuda.set_per_process_memory_fraction(_fraction, device=0)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"GPU cap  : {config.GPU_MEMORY_LIMIT_GB:.1f} GB / {_total_gb:.1f} GB  ({_fraction:.0%})")

# ── Free Ampere perf wins (no quality impact) ────────────────────────
# All three are mathematically benign on bf16/fp32 inference workloads.
# Combined effect on RTX 3060 + Flux: ~5–10 % per-step.
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True          # auto-tune conv kernels for static shapes
    torch.backends.cuda.matmul.allow_tf32 = True   # TF32 matmul (Ampere)
    torch.backends.cudnn.allow_tf32 = True         # TF32 in cuDNN convolutions
    torch.set_float32_matmul_precision("high")     # newer API for the same TF32 path

if not API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in .env")

gpt = OpenAI(api_key=API_KEY)
print(f"Model : {GPT_MODEL}  |  Device: {DEVICE}")


def _encode(path) -> dict:
    p = Path(path)
    mime = "png" if p.suffix.lower() == ".png" else "jpeg"
    b64  = base64.b64encode(p.read_bytes()).decode()
    return {
        "type": "input_image",
        "image_url": f"data:image/{mime};base64,{b64}",
        "detail": "high",
    }


# ── GPT vision call (Responses API + reasoning=xhigh + prompt caching) ────────
#
# OpenAI Responses API replaces chat.completions for reasoning models. Key
# upgrades vs. the old call site:
#   • `reasoning.effort = "high"` — the maximum the server-side `gpt-5`
#     model accepts (the SDK enum lists `xhigh` too but that's rejected by
#     gpt-5 with a 400; reserve xhigh for gpt-5-codex / 5.1+ if upgraded).
#     Gives Agent 1 / amodal-completion / reviewer more chain-of-thought
#     budget so the polygon predictions are less likely to hallucinate
#     anatomy in the wrong spatial direction.
#   • `prompt_cache_key` — static prompt caching. Each logical call site
#     should pass a stable key so the long instruction prefix gets cached
#     across images. Retention set to 24h.
#   • `text.format` with JSON-schema strict mode — same effective contract
#     as the old `response_format={"type": "json_schema", ...}` but using
#     the flat Responses-API shape.
#   • Image content items use {type: "input_image", image_url, detail}.

def gpt_vision(images: list, prompt: str, schema: dict = None,
               max_retries: int = 2, cache_key: str = None) -> dict:
    current_prompt = prompt
    for attempt in range(max_retries + 1):
        content = [{"type": "input_text", "text": current_prompt}] + [_encode(p) for p in images]

        if schema:
            # Existing schemas are {name, schema, strict}. Responses API wants
            # those flat inside text.format.
            text_cfg = {"format": {
                "type":   "json_schema",
                "name":   schema["name"],
                "schema": schema["schema"],
                "strict": schema.get("strict", True),
            }}
        else:
            text_cfg = {"format": {"type": "json_object"}}

        kwargs: dict = {
            "model":     GPT_MODEL,
            "input":     [{"role": "user", "content": content}],
            "reasoning": {"effort": "high", "summary": "concise"},
            "text":      text_cfg,
            "max_output_tokens": config.GPT_MAX_TOKENS,
        }
        if cache_key:
            kwargs["prompt_cache_key"]       = cache_key
            kwargs["prompt_cache_retention"] = "24h"

        try:
            resp = gpt.responses.create(**kwargs)
        except Exception as exc:                              # noqa: BLE001
            if attempt < max_retries:
                print(f"  [GPT] API error (attempt {attempt + 1}/{max_retries + 1}): {exc!r}, retrying…")
                continue
            raise

        # Walk output items to detect refusals (Responses API surfaces
        # refusals as content items with type='refusal' inside a message).
        refusal_msg = None
        for item in (resp.output or []):
            if getattr(item, "type", None) != "message":
                continue
            for c in (getattr(item, "content", None) or []):
                if getattr(c, "type", None) == "refusal":
                    refusal_msg = getattr(c, "refusal", None) or "refused"
                    break
            if refusal_msg:
                break
        if refusal_msg:
            if attempt < max_retries:
                print(f"  [GPT] Refusal (attempt {attempt + 1}/{max_retries + 1}), retrying…")
                current_prompt = f"Please analyze this image technically and objectively. {current_prompt}"
                continue
            raise RuntimeError(f"[GPT] Model refused after {max_retries + 1} attempts: {refusal_msg}")

        raw = (resp.output_text or "").strip()
        if not raw:
            if attempt < max_retries:
                print(f"  [GPT] Empty response (status={resp.status!r}, attempt {attempt + 1}/{max_retries + 1}), retrying…")
                continue
            raise RuntimeError(
                f"[GPT] Empty response (status={resp.status!r}). "
                f"Try increasing GPT_MAX_TOKENS in config.py (currently {config.GPT_MAX_TOKENS})."
            )

        if resp.status == "incomplete":
            reason = getattr(getattr(resp, "incomplete_details", None), "reason", None)
            print(f"  [GPT] WARNING: response incomplete (reason={reason!r}) — partial output:\n{raw[:300]}")

        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            if attempt < max_retries:
                print(f"  [GPT] JSON parse error (attempt {attempt + 1}/{max_retries + 1}), retrying…")
                continue
            raise RuntimeError(f"[GPT] JSON parse error: {e}\nRaw response:\n{raw[:500]}")

    raise RuntimeError("[GPT] Exhausted all retries without a valid response")
