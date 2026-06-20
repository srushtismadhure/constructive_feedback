.PHONY: install dev deploy lint format check clean

install:
	uv sync

dev:
	uv run uvicorn main:web_app --host 0.0.0.0 --port 8000 --reload

deploy:
	uv run modal deploy main.py

lint:
	uv run ruff check .

format:
	uv run ruff format .

check:
	uv run pyright .

clean:
	rm -rf .venv __pycache__ .ruff_cache
