"""
client.py — dispatcher.

Selects where the amodal pipeline runs based on config.RUN_ENV:

    "local"     — runs the LangGraph pipeline in-process (pipeline.graph.run).
    "lightning" — uploads to config.LIGHTNING_URL (server.py on a Lightning
                  Studio) and extracts the returned ZIP locally.
    "colab"     — uploads to config.COLAB_URL (server.py running in a Colab
                  notebook, exposed via ngrok / cloudflared).

The server-side code is the same for "lightning" and "colab" — only the
tunnel URL differs.

The legacy config.USE_LIGHTNING flag is still honoured: if RUN_ENV is unset
and USE_LIGHTNING is True, dispatch behaves as if RUN_ENV="lightning".

Usage:
    python client.py /path/to/image.jpg
"""

from __future__ import annotations

import base64
import io
import sys
import time
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import config                                    # noqa: E402


_VALID_ENVS = {"local", "lightning", "colab"}


def _resolve_run_env() -> str:
    """Return the effective run env: 'local' | 'lightning' | 'colab'.

    Resolution order:
      1. config.RUN_ENV (if set and valid)
      2. Legacy config.USE_LIGHTNING=True → 'lightning'
      3. 'local'
    """
    env = (getattr(config, "RUN_ENV", "") or "").strip().lower()
    if env in _VALID_ENVS:
        return env
    if env:
        print(f"[client] unknown RUN_ENV={env!r}; falling back to legacy USE_LIGHTNING",
              file=sys.stderr)
    if bool(getattr(config, "USE_LIGHTNING", False)):
        return "lightning"
    return "local"


def _remote_target(env: str) -> tuple[str, float]:
    """Return (url, timeout) for the remote env. Raises if URL is empty."""
    if env == "lightning":
        url = (getattr(config, "LIGHTNING_URL", "") or "").strip().rstrip("/")
        timeout = float(getattr(config, "LIGHTNING_TIMEOUT", 1800))
        if not url:
            raise RuntimeError("RUN_ENV='lightning' but LIGHTNING_URL is empty in config.py")
        return url, timeout
    if env == "colab":
        url = (getattr(config, "COLAB_URL", "") or "").strip().rstrip("/")
        timeout = float(getattr(config, "COLAB_TIMEOUT", 1800))
        if not url:
            raise RuntimeError("RUN_ENV='colab' but COLAB_URL is empty in config.py "
                               "(set it to your ngrok/cloudflared tunnel URL)")
        return url, timeout
    raise ValueError(f"_remote_target called with non-remote env: {env!r}")


def _run_local(image_path: str) -> int:
    """Run the LangGraph pipeline in-process."""
    from pipeline.graph import run as run_pipeline   # noqa: WPS433
    run_pipeline(image_path, getattr(config, "TARGET", ""))
    return 0


def _run_remote(image_path: str, env: str) -> int:
    """Upload image to the remote server for `env`, save returned ZIP locally."""
    import requests                              # noqa: WPS433

    url, timeout = _remote_target(env)
    target = (getattr(config, "TARGET", "") or "").strip()

    img = Path(image_path)
    if not img.exists():
        print(f"[client] image not found: {img}", file=sys.stderr)
        return 1

    print(f"[client] uploading {img.name}  →  {url}/process  (target={target!r})")
    t0 = time.time()
    with img.open("rb") as f:
        try:
            resp = requests.post(
                f"{url}/process",
                files={"image": (img.name, f, "image/jpeg")},
                data={"target": target},
                timeout=timeout,
            )
        except Exception as exc:                              # noqa: BLE001
            print(f"[client] request failed: {exc!r}", file=sys.stderr)
            return 3

    if resp.status_code != 200:
        print(f"[client] server returned HTTP {resp.status_code}: {resp.text[:300]}",
              file=sys.stderr)
        return 4

    data = resp.json()
    if not data.get("ok"):
        print(f"[client] server error: {data.get('error')}", file=sys.stderr)
        return 5

    # Extract ZIP into output/<stem>/_flux_cutout_person/  to match local layout.
    stem = data.get("stem") or img.stem
    out_dir = _HERE / "output" / stem / "_flux_cutout_person"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_bytes = base64.b64decode(data["zip_b64"])
    extracted = 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.infolist():
            zf.extract(member, out_dir)
            extracted += 1

    dt = time.time() - t0
    print(f"[client] done in {dt:.1f}s  (server: {data['duration_s']}s, "
          f"network+zip: {dt - data['duration_s']:.1f}s)")
    print(f"[client] {extracted} file(s) → {out_dir}/")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: client.py <image_path>", file=sys.stderr)
        return 1
    image_path = sys.argv[1]
    env = _resolve_run_env()
    label = {"local": "LOCAL", "lightning": "REMOTE (Lightning)", "colab": "REMOTE (Colab)"}[env]
    print(f"[client] dispatching in {label} mode  (RUN_ENV={env})")
    if env == "local":
        return _run_local(image_path)
    try:
        return _run_remote(image_path, env)
    except RuntimeError as exc:
        print(f"[client] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
