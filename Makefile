.PHONY: test lint typecheck counts validate

test:
	python -m pytest

lint:
	ruff check packages tests tools
	ruff format --check packages tests tools

typecheck:
	mypy packages

counts:
	python tools/count_params.py

validate: lint typecheck test counts
