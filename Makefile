.PHONY: help setup runner-install api-install web-install lint test dev-api dev-web

help:
	@echo "OESB developer tasks"
	@echo "  make setup         - install all components (runner, api, web)"
	@echo "  make lint          - run linters across the repo"
	@echo "  make test          - run all test suites"
	@echo "  make dev-api       - run the API in dev mode"
	@echo "  make dev-web       - run the web app in dev mode"

setup: runner-install api-install web-install

runner-install:
	cd runner && pip install -e ".[dev]"

api-install:
	cd api && pip install -e ".[dev]"

web-install:
	cd web && npm install

lint:
	cd runner && ruff check . || true
	cd api && ruff check . || true
	cd web && npm run lint || true

test:
	cd runner && pytest -q || true
	cd api && pytest -q || true

dev-api:
	cd api && uvicorn oesb_api.main:app --reload

dev-web:
	cd web && npm run dev
