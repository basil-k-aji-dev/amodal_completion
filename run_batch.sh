#!/usr/bin/env bash
# Runs test_flux_cutout_person.py over every image in website/data/,
# setting config.INPUT_PROMPT from the filename's trailing _<class> suffix
# (as a hint to Agent 1 — leave DATA_DIR filenames without a suffix to let
# Agent 1 auto-detect instead).
# Outputs land in amodal_completion/output/<stem>/_flux_cutout_<subject>/
# (BASE_DIR-relative, per test_flux_cutout_person.py).
set -uo pipefail
cd "$(dirname "$0")"

DATA_DIR="/home/ubuntu/Workspace/website/data"
mkdir -p logs output

# horse already validated with InstaFormer wired in; skip it here.
DONE_STEMS="african-american-7481724_640_horse"

# Wait for any currently-running pipeline invocation to finish first
# (single GPU — avoid two Flux/SAM3 processes contending for VRAM).
while pgrep -f "test_flux_cutout_person.py" > /dev/null; do
  sleep 5
done

for f in "$DATA_DIR"/*.jpg; do
  stem=$(basename "$f" .jpg)
  case " $DONE_STEMS " in
    *" $stem "*) echo "[skip] $stem already done"; continue ;;
  esac
  target="${stem##*_}"
  echo "=================================================="
  echo "[$(date +%T)] Image: $f   INPUT_PROMPT=$target"
  echo "=================================================="
  python - "$target" <<'PY'
import re, sys, pathlib
target = sys.argv[1]
p = pathlib.Path("config.py")
text = p.read_text()
text = re.sub(r'^INPUT_PROMPT\s*=\s*".*"', f'INPUT_PROMPT = "{target}"', text, count=1, flags=re.M)
p.write_text(text)
PY
  .venv/bin/python test_flux_cutout_person.py "$f" 2>&1 | tee "logs/${stem}.log"
  status=${PIPESTATUS[0]}
  echo "[$(date +%T)] Exit status for $stem: $status"
done

echo "BATCH COMPLETE"
