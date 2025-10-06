.PHONY: install test format

UV_ENV=UV_CACHE_DIR=.uv-cache
PYTHON=.venv/bin/python

install:
	UV_CACHE_DIR=.uv-cache uv venv .venv
	$(UV_ENV) uv pip install -p $(PYTHON) -e .[dev]

test:
	$(UV_ENV) uv run --no-sync -p $(PYTHON) pytest -q

format:
	$(UV_ENV) uv run --no-sync -p $(PYTHON) black ibdnet tests
	$(UV_ENV) uv run --no-sync -p $(PYTHON) isort ibdnet tests
