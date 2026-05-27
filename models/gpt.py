"""
models/gpt.py — OpenAI GPT vision client + helpers.

Owns the OpenAI client singleton, image encoding, and gpt_vision (the
Responses-API call with schema / refusal / JSON-retry handling).
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from openai import OpenAI

import config
from runtime import API_KEY, GPT_MODEL

if not API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in .env")
if not GPT_MODEL:
    raise EnvironmentError("OPENAI_MODEL is not set in .env")

gpt = OpenAI(api_key=API_KEY)
print(f"GPT      : {GPT_MODEL}  client ready")


# ── GPT vision helper ─────────────────────────────────────────────────────────
def _encode(path) -> dict:
    """Encode an image file as a Responses-API `input_image` content item."""
    p    = Path(path)
    mime = "jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "png"
    b64  = base64.b64encode(p.read_bytes()).decode()
    return {
        "type": "input_image",
        "image_url": f"data:image/{mime};base64,{b64}",
        "detail": "high",
    }


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
