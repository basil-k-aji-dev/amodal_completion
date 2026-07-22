#!/usr/bin/env bash
# Runs src/pipeline.py over: bunny first, then 20 selected COCO-category
# candidates (cats, dogs, cows, sheep, giraffe, horse) chosen to be
# likely-easy cases for the current InstaFormer + Flux setup.
set -uo pipefail
cd "$(dirname "$0")"

DATA_DIR="/home/ubuntu/Workspace/website/data"
mkdir -p logs output

IMAGES="
bunny-7847028_640_rabbit.jpg
animal-8165466_640_cat.jpg
cat-1647775_640_cat.jpg
cat-7875506_640_cat.jpg
cat-8031938_640_cat.jpg
cat-8361048_640_cat.jpg
european-shorthair-8136129_640_cat.jpg
pet-9040516_640_cat.jpg
simba-8618301_640_cat.jpg
dog-8510901_640_dog.jpg
stray-8933778_640_dog.jpg
pexels-c-t-phat-546614745-24014245_dog.jpg
pexels-miami302-26793646_dog.jpg
french-bulldog-7514203_640_frenchbulldog.jpg
animal-8937149_640_sheep.jpg
dairy-cattle-9018750_640_cow.jpg
highland-cow-8678950_640_cow.jpg
pexels-leefinvrede-27015904_cow.jpg
pexels-witheline-27442491_cow.jpg
giraffe-6378717_640_giraffe.jpg
horse-1006570_640_horse.jpg
"

# Wait for any currently-running pipeline invocation to finish first
# (single GPU — avoid two Flux/SAM3 processes contending for VRAM).
while pgrep -f "src/pipeline.py" > /dev/null; do
  sleep 5
done

for stem_jpg in $IMAGES; do
  f="$DATA_DIR/$stem_jpg"
  stem=$(basename "$f" .jpg)
  target="${stem##*_}"
  echo "=================================================="
  echo "[$(date +%T)] Image: $f   INPUT_PROMPT=$target"
  echo "=================================================="
  .venv/bin/python src/pipeline.py "$f" "$target" 2>&1 | tee "logs/${stem}.log"
  status=${PIPESTATUS[0]}
  echo "[$(date +%T)] Exit status for $stem: $status"
done

echo "BATCH_20 COMPLETE"
