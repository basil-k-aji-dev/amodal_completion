# Amodal Completion — convenience targets
#
# Usage:
#   make run IMAGE=horse.jpg                      # resolved inside POSITIVES_DIR; correct prompt auto-picked (see the case statement below)
#   make run IMAGE=/abs/path/to/photo.jpg          # any absolute/relative path works too
#   make run IMAGE=horse.jpg PROMPT=horse          # explicit PROMPT always overrides the auto-picked one
#   make run-all                                   # run every image in POSITIVES_DIR, each with its own auto-picked prompt
#   make run-all PROMPT=horse                      # explicit PROMPT overrides auto-pick for every image
#   make list                                      # list images available in POSITIVES_DIR
#
# Output always lands in output/<image-stem>/ (src/pipeline.py's own default) —
# nothing in this Makefile changes where results are written.
#
# Auto-picked prompts below are the known-correct subject/occluder hint for
# each positives/ image, established from the actual runs that produced a
# correct response. Anything not listed falls through to "" — auto-detect,
# unchanged from today's behaviour.

POSITIVES_DIR := /home/ubuntu/Workspace/website/positive
PYTHON        := ../.venv/bin/python
PROMPT        ?=

.PHONY: run run-all list help

help:
	@echo "make run IMAGE=<filename-or-path> [PROMPT=<hint>]"
	@echo "  IMAGE may be a bare filename (resolved inside $(POSITIVES_DIR))"
	@echo "  or an absolute/relative path to any image."
	@echo "  If PROMPT is not given, the known-correct hint is auto-picked per image."
	@echo "make run-all [PROMPT=<hint>]  — run every image in $(POSITIVES_DIR), one at a time"
	@echo "make list   — list images currently in $(POSITIVES_DIR)"

list:
	@ls -1 $(POSITIVES_DIR)

run:
ifndef IMAGE
	$(error IMAGE is not set. Usage: make run IMAGE=horse.jpg)
endif
	@if [ -f "$(IMAGE)" ]; then \
		IMG_PATH="$(IMAGE)"; \
	elif [ -f "$(POSITIVES_DIR)/$(IMAGE)" ]; then \
		IMG_PATH="$(POSITIVES_DIR)/$(IMAGE)"; \
	else \
		echo "Image not found: $(IMAGE) (checked as given, and inside $(POSITIVES_DIR))"; \
		exit 1; \
	fi; \
	if [ -n "$(PROMPT)" ]; then \
		USE_PROMPT="$(PROMPT)"; \
	else \
		case "$$(basename "$$IMG_PATH")" in \
			african-american-7481724_640_horse.jpg) USE_PROMPT="horse" ;; \
			bunny-7847028_640_rabbit.jpg)           USE_PROMPT="rabbit" ;; \
			cat-eating-food-tongue-out_cat.jpg)     USE_PROMPT="cat" ;; \
			corgi-puppy-ball_dog.jpg)               USE_PROMPT="dog" ;; \
			flowers-8991384_640_vase.jpg)           USE_PROMPT="vase" ;; \
			mathias-reding-w-U7ktLC4ic-unsplash_flowerpot.jpg) USE_PROMPT="flowerpot" ;; \
			pexels-patagonia-savage-107018019-26707538_fox.jpg) USE_PROMPT="fox" ;; \
			*)                                       USE_PROMPT="" ;; \
		esac; \
	fi; \
	echo "Running pipeline on $$IMG_PATH"; \
	cd src && $(PYTHON) pipeline.py "$$IMG_PATH" "$$USE_PROMPT"

run-all:
	@for f in "$(POSITIVES_DIR)"/*; do \
		[ -f "$$f" ] || continue; \
		if [ -n "$(PROMPT)" ]; then \
			USE_PROMPT="$(PROMPT)"; \
		else \
			case "$$(basename "$$f")" in \
				african-american-7481724_640_horse.jpg) USE_PROMPT="horse" ;; \
				bunny-7847028_640_rabbit.jpg)           USE_PROMPT="rabbit" ;; \
				cat-eating-food-tongue-out_cat.jpg)     USE_PROMPT="cat" ;; \
				corgi-puppy-ball_dog.jpg)               USE_PROMPT="dog" ;; \
				flowers-8991384_640_vase.jpg)           USE_PROMPT="vase" ;; \
				mathias-reding-w-U7ktLC4ic-unsplash_flowerpot.jpg) USE_PROMPT="flowerpot" ;; \
				pexels-patagonia-savage-107018019-26707538_fox.jpg) USE_PROMPT="fox" ;; \
				*)                                       USE_PROMPT="" ;; \
			esac; \
		fi; \
		echo "=================================================="; \
		echo "Running pipeline on $$f"; \
		echo "=================================================="; \
		(cd src && $(PYTHON) pipeline.py "$$f" "$$USE_PROMPT"); \
		echo "Exit status for $$f: $$?"; \
	done; \
	echo "run-all complete"
