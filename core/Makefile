.PHONY: install test dev deploy lint format check clean

install:
	uv sync

# End-to-end smoke test: bridge reward + rover controller + Brain-drives-rover.
test:
	uv run python tests/smoke_test.py

# FastAPI backend, local (no Modal).
dev:
	uv run uvicorn main:web_app --host 0.0.0.0 --port 8000 --reload

# Deploy the Modal app.
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
