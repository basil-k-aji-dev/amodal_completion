"""
run.py — run the amodal completion pipeline on an image or a folder of images.

Auto-detects whether the input is a single image or a directory and processes
accordingly. Target label is extracted from the filename suffix when possible
(e.g. bear-8845470_640_bear.jpg → "bear"), falling back to config.TARGET.

Usage:
    python run.py /path/to/image.jpg
    python run.py /path/to/folder/
    python run.py /path/to/folder/ --url http://localhost:8000
    python run.py /path/to/image.jpg --env local
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import time
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import config  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_VALID_ENVS = {"local", "lightning", "colab"}


def _resolve_env() -> str:
    env = (getattr(config, "RUN_ENV", "") or "").strip().lower()
    if env in _VALID_ENVS:
        return env
    if bool(getattr(config, "USE_LIGHTNING", False)):
        return "lightning"
    return "local"


def _target_from_filename(path: Path) -> str:
    parts = path.stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isalpha():
        return parts[1]
    return (getattr(config, "TARGET", "") or "").strip()


def _remote_url(env: str) -> tuple[str, float]:
    if env == "lightning":
        url = (getattr(config, "LIGHTNING_URL", "") or "").strip().rstrip("/")
        if not url:
            raise RuntimeError("RUN_ENV='lightning' but LIGHTNING_URL is empty in config.py")
        return url, float(getattr(config, "LIGHTNING_TIMEOUT", 1800))
    if env == "colab":
        url = (getattr(config, "COLAB_URL", "") or "").strip().rstrip("/")
        if not url:
            raise RuntimeError("RUN_ENV='colab' but COLAB_URL is empty in config.py")
        return url, float(getattr(config, "COLAB_TIMEOUT", 1800))
    raise ValueError(f"not a remote env: {env!r}")


def _run_local(image_path: str, target: str) -> bool:
    from pipeline.graph import run as run_pipeline
    config.IMAGE_PATH = image_path
    config.TARGET = target
    run_pipeline(image_path, target)
    return True


def _run_remote(image_path: str, target: str, url: str) -> bool:
    import requests
    img = Path(image_path)
    url = url.rstrip("/")
    print(f"[run] uploading {img.name} → {url}/process  (target={target!r})")
    t0 = time.time()
    with img.open("rb") as f:
        try:
            resp = requests.post(
                f"{url}/process",
                files={"image": (img.name, f, "image/jpeg")},
                data={"target": target},
                timeout=1800,
            )
        except Exception as exc:
            print(f"[run] request failed: {exc!r}", file=sys.stderr)
            return False

    if resp.status_code != 200:
        print(f"[run] server HTTP {resp.status_code}: {resp.text[:300]}", file=sys.stderr)
        return False

    data = resp.json()
    if not data.get("ok"):
        print(f"[run] server error: {data.get('error')}", file=sys.stderr)
        return False

    stem = data.get("stem") or img.stem
    out_dir = _HERE / "output" / stem / "_flux_cutout_person"
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(data["zip_b64"]))) as zf:
        zf.extractall(out_dir)

    dt = time.time() - t0
    print(f"[run] done in {dt:.1f}s  → {out_dir}/")
    return True


def process_one(img: Path, env: str, url: str | None) -> tuple[str, float]:
    target = _target_from_filename(img)
    print(f"\n[run] {img.name}  target={target!r}  env={env}")
    t0 = time.time()
    try:
        if env == "local":
            ok = _run_local(str(img), target)
        else:
            ok = _run_remote(str(img), target, url or "http://localhost:8000")
        status = "ok" if ok else "failed"
    except Exception as exc:
        status = f"error: {exc!r}"
    return status, time.time() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="Amodal completion — image or folder")
    parser.add_argument("input", type=Path, help="Image file or folder of images")
    parser.add_argument("--env", choices=["local", "remote"], default=None,
                        help="Override run env (default: from config.RUN_ENV)")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Server URL when env=remote (default: http://localhost:8000)")
    args = parser.parse_args()

    # Resolve env — CLI flag overrides config
    if args.env == "remote":
        env = "remote"
    elif args.env == "local":
        env = "local"
    else:
        cfg_env = _resolve_env()
        env = "remote" if cfg_env in ("lightning", "colab") else "local"

    inp = args.input
    if not inp.exists():
        print(f"[run] not found: {inp}", file=sys.stderr)
        sys.exit(1)

    # Auto-detect: file or folder
    if inp.is_file():
        images = [inp]
    else:
        images = sorted(p for p in inp.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not images:
            print(f"[run] no images found in {inp}", file=sys.stderr)
            sys.exit(1)
        print(f"[run] {len(images)} image(s) found in {inp}")

    results: list[tuple[str, str, float]] = []
    for img in images:
        status, dt = process_one(img, env, args.url)
        results.append((img.name, status, dt))

    if len(results) > 1:
        print("\n─── Summary ─────────────────────────────────────")
        ok_count = sum(1 for _, s, _ in results if s == "ok")
        for name, status, dt in results:
            mark = "✓" if status == "ok" else "✗"
            print(f"  {mark}  {name:<45}  {status:<20}  {dt:.1f}s")
        print(f"\n  {ok_count}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
