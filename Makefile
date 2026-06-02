# RRS dev tasks. `make dev` runs the API + frontend together in one terminal;
# Ctrl-C stops both. See README.md / CLAUDE.md for the full per-phase commands.

# Ports (backend must match the frontend's NEXT_PUBLIC_API_BASE_URL default).
API_PORT ?= 8011
WEB_PORT ?= 3000

# uv eats the env-var editable-install flakiness; PYTHONPATH=src is the reliable path.
RUN := PYTHONPATH=src uv run

.DEFAULT_GOAL := help
.PHONY: help dev backend frontend install test lint

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

dev: ## Run API + frontend together (Ctrl-C stops both)
	@echo "→ API   http://localhost:$(API_PORT)  (docs at /docs)"
	@echo "→ Web   http://localhost:$(WEB_PORT)"
	@echo "  (first request waits ~6s while the model loads)"
	@trap 'kill 0' INT TERM EXIT; \
		$(RUN) uvicorn rrs.api.app:app --port $(API_PORT) & \
		(cd frontend && NEXT_PUBLIC_API_BASE_URL=http://localhost:$(API_PORT) npm run dev -- --port $(WEB_PORT)) & \
		wait

backend: ## Run only the FastAPI backend
	$(RUN) uvicorn rrs.api.app:app --reload --port $(API_PORT)

frontend: ## Run only the Next.js frontend
	cd frontend && npm run dev -- --port $(WEB_PORT)

install: ## Install all deps (Python ml+api extras, frontend npm)
	uv sync --extra ml --extra api
	cd frontend && npm install

test: ## Run the Python test suite
	$(RUN) pytest -q

lint: ## Lint the Python source
	uv run ruff check src tests scripts
