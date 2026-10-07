.DEFAULT_GOAL := help

.PHONY: help sync format lint typecheck test test-unit check qdrant-up qdrant-down benchmark-smoke benchmark-overnight benchmark-resume

help:
	@echo "Available commands:"
	@echo "  make sync       Install locked dependencies"
	@echo "  make format     Apply automatic formatting and safe lint fixes"
	@echo "  make lint       Check formatting and lint rules"
	@echo "  make typecheck  Run static type checking"
	@echo "  make test       Run the complete pytest suite"
	@echo "  make test-unit  Run tests that do not require external infrastructure"
	@echo "  make check      Run the standard local quality gate"
	@echo "  make qdrant-up  Start and health-check the local Qdrant server"
	@echo "  make qdrant-down Stop the local Qdrant server without deleting data"
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

qdrant-up:
	docker compose up -d qdrant
	@curl --retry 30 --retry-delay 1 --retry-connrefused --fail --silent --show-error http://127.0.0.1:6333/healthz > /dev/null
	@echo "Qdrant is ready at http://127.0.0.1:6333"

qdrant-down:
	docker compose stop qdrant

benchmark-smoke: qdrant-up
	uv run resolverag index build --config configs/indexes/qdrant.yaml --limit 25
	uv run resolverag evaluate retrieval --index-manifest data/processed/techqa/indexes/smoke_25/manifest.json --sample-size 2 --no-resume

benchmark-overnight: qdrant-up
	uv run resolverag index build --config configs/indexes/qdrant.yaml
	uv run resolverag evaluate retrieval --config configs/evaluation/retrieval.yaml --resume

benchmark-resume: qdrant-up
	uv run resolverag evaluate retrieval --config configs/evaluation/retrieval.yaml --resume
