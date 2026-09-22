# vc-multimodal --- common tasks.
#
# `uv run` always executes inside this project's .venv, so these targets work
# even if a conda base environment is active in your shell. See README for why
# you should still deactivate conda base.

UV ?= uv
PILOT_CONFIG ?= config/pilot.yaml

.DEFAULT_GOAL := help
.PHONY: help setup env lint format typecheck test test-fast cov pilot clean

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create the uv environment, install hooks, write .env if missing
	$(UV) sync --all-groups
	$(UV) run pre-commit install
	@$(MAKE) --no-print-directory env

env: ## Write .env from .env.example, pinning the ffmpeg/ffprobe found now
	@if [ -f .env ]; then \
		echo ".env already exists; leaving it alone."; \
	else \
		sed -e "s|^VC_FFMPEG=.*|VC_FFMPEG=$$(command -v ffmpeg || echo '')|" \
		    -e "s|^VC_FFPROBE=.*|VC_FFPROBE=$$(command -v ffprobe || echo '')|" \
		    .env.example > .env; \
		echo "Wrote .env. Review the data roots before running any stage."; \
		echo "  ffmpeg:  $$(command -v ffmpeg || echo 'NOT FOUND')"; \
		echo "  ffprobe: $$(command -v ffprobe || echo 'NOT FOUND')"; \
	fi

lint: ## Lint and check formatting
	$(UV) run ruff check src tests scripts
	$(UV) run ruff format --check src tests scripts

format: ## Apply formatting and autofixes
	$(UV) run ruff check --fix src tests scripts
	$(UV) run ruff format src tests scripts

typecheck: ## Type check src/ in strict mode
	$(UV) run mypy

test: ## Run the full test suite with coverage
	$(UV) run pytest --cov --cov-report=term-missing

test-fast: ## Run only the fast tests (skip decoding / model inference)
	$(UV) run pytest -m "not slow and not integration"

cov: ## Write an HTML coverage report
	$(UV) run pytest --cov --cov-report=html
	@echo "Open htmlcov/index.html"

pilot: ## Run every stage over the sessions listed in config/pilot.yaml
	$(UV) run vc run-all --config $(PILOT_CONFIG)

clean: ## Remove caches and build artifacts (never touches data roots)
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage coverage.xml build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
