#!/usr/bin/env python3
"""Run the pipeline against a single image or a batch/directory of images.

Single image:
    .venv/bin/python test_pipeline.py path/to/photo.jpg [prompt]

Batch (every .jpg/.jpeg/.png in a directory):
    .venv/bin/python test_pipeline.py path/to/dir/

Each image runs as its own `src/pipeline.py` subprocess (keeps GPU memory
clean between runs — same approach as run_batch.sh/run_batch_20.sh). In
batch mode the per-image prompt hint defaults to the filename's trailing
"_<class>" suffix (e.g. "horse-123_640_horse.jpg" -> "horse"); filenames
without that suffix run with no hint, so Agent 1 auto-detects instead.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "src" / "pipeline.py"
PYTHON = ROOT / ".venv" / "bin" / "python"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _prompt_from_filename(path: Path) -> str:
    stem = path.stem
    if "_" not in stem:
        return ""
    return stem.rsplit("_", 1)[-1]


def _run_one(image: Path, prompt: str) -> int:
    args = [str(PYTHON), str(PIPELINE), str(image)]
    if prompt:
        args.append(prompt)
    print(f"\n{'=' * 60}\n[{image.name}] prompt={prompt or '(auto-detect)'}\n{'=' * 60}")
    return subprocess.run(args).returncode


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Path not found: {target}")
        return 1

    if target.is_dir():
        images = sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not images:
            print(f"No images found in {target}")
            return 1
        failures = []
        for image in images:
            status = _run_one(image, _prompt_from_filename(image))
            print(f"[{image.name}] exit status: {status}")
            if status != 0:
                failures.append(image.name)
        print(f"\nBatch complete: {len(images) - len(failures)}/{len(images)} succeeded")
        if failures:
            print(f"Failed: {', '.join(failures)}")
        return 1 if failures else 0

    prompt = sys.argv[2] if len(sys.argv) > 2 else ""
    return _run_one(target, prompt)


if __name__ == "__main__":
    sys.exit(main())
