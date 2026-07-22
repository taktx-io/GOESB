.PHONY: help setup runner-install lint test

help:
	@echo "OESB developer tasks"
	@echo "  make setup         - install the runner"
	@echo "  make lint          - run linters"
	@echo "  make test          - run the test suite"
	@echo ""
	@echo "The leaderboard/API product (api/, web/) lives in the separate"
	@echo "taktx-io/oesb-platform repo — see docs/adr/0006-split-platform-repo.md."

setup: runner-install

runner-install:
	cd runner && pip install -e ".[dev]"

lint:
	cd runner && ruff check .

test:
	cd runner && pytest -q
