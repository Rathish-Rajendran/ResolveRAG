.DEFAULT_GOAL := help

.PHONY: help sync format lint typecheck test test-unit check

help:
	@echo "Available commands:"
	@echo "  make sync       Install locked dependencies"
	@echo "  make format     Apply automatic formatting and safe lint fixes"
	@echo "  make lint       Check formatting and lint rules"
	@echo "  make typecheck  Run static type checking"
	@echo "  make test       Run the complete pytest suite"
	@echo "  make test-unit  Run tests that do not require external infrastructure"
	@echo "  make check      Run the standard local quality gate"

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