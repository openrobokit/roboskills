.PHONY: install format lint check check-loc test ci build check-package

install:
	uv sync --group dev

format:
	uvx ruff format

lint:
	uvx ruff check

check:
	uv run --no-project python -m compileall -q -x '/\.venv/' src
	! git grep -InE '[a-z]_internal\b|roboskills-internal|archive/artifacts|jianglong-org' -- ':!Makefile'

check-loc:
	python3 scripts/count_loc.py

test:
	uv run --group dev pytest -q tests

ci: lint check check-loc test check-package

build:
	uv build

check-package: build
	uv run --no-project python tests/packaging/check_wheel.py dist/roboskills-0.1.0-py3-none-any.whl
