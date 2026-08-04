# Amodal Completion — convenience targets
#
# Usage:
#   make run IMAGE=horse.jpg                      # resolved inside POSITIVES_DIR
#   make run IMAGE=/abs/path/to/photo.jpg          # any absolute/relative path works too
#   make run IMAGE=horse.jpg PROMPT=horse          # optional subject/occluder text hint
#   make list                                      # list images available in POSITIVES_DIR
#
# Output always lands in output/<image-stem>/ (src/pipeline.py's own default) —
# nothing in this Makefile changes where results are written.

POSITIVES_DIR := /home/ubuntu/Workspace/website/possitives
PYTHON        := ../.venv/bin/python
PROMPT        ?=

.PHONY: run list help

help:
	@echo "make run IMAGE=<filename-or-path> [PROMPT=<hint>]"
	@echo "  IMAGE may be a bare filename (resolved inside $(POSITIVES_DIR))"
	@echo "  or an absolute/relative path to any image."
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
	echo "Running pipeline on $$IMG_PATH (prompt='$(PROMPT)')"; \
	cd src && $(PYTHON) pipeline.py "$$IMG_PATH" "$(PROMPT)"
