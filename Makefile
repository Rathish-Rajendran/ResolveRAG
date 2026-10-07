.DEFAULT_GOAL := help

.PHONY: help sync format lint typecheck test test-unit check benchmark-smoke benchmark-overnight benchmark-resume

help:
	@echo "Available commands:"
	@echo "  make sync       Install locked dependencies"
	@echo "  make format     Apply automatic formatting and safe lint fixes"
	@echo "  make lint       Check formatting and lint rules"
	@echo "  make typecheck  Run static type checking"
	@echo "  make test       Run the complete pytest suite"
	@echo "  make test-unit  Run tests that do not require external infrastructure"
	@echo "  make check      Run the standard local quality gate"
	@echo "  make benchmark-smoke     Validate the 30-cell retrieval matrix"
	@echo "  make benchmark-overnight Build full indexes and benchmark 30 configurations"
	@echo "  make benchmark-resume    Resume evaluation without rebuilding full indexes"

sync:
	uv sync --locked

format:
	uv run ruff check --fix .
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

test-unit:
	uv run pytest -m "not integration"

check: lint typecheck test-unit

benchmark-smoke:
	uv run resolverag evaluate retrieval --index-manifest data/processed/techqa/indexes/smoke_25/manifest.json --sample-size 2 --no-resume

benchmark-overnight:
	uv run resolverag index build --config configs/indexes/local.yaml
	uv run resolverag evaluate retrieval --config configs/evaluation/retrieval.yaml --resume

benchmark-resume:
	uv run resolverag evaluate retrieval --config configs/evaluation/retrieval.yaml --resume
