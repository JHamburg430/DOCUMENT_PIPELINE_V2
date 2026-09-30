PYTHON ?= python
COMPOSE ?= docker compose -f infra/compose/docker-compose.yml

.PHONY: test unit integration live up down logs repair-regressions

REVISION ?= $(shell git rev-parse HEAD)

repair-regressions:
	@git diff --quiet && git diff --cached --quiet || (echo "Commit source changes before attributing replay to $(REVISION)"; exit 1)
	$(COMPOSE) exec -T ui python scripts/benchmark/agent_repair_regressions.py check --revision $(REVISION)

test:
	$(PYTHON) -m pytest

unit:
	$(PYTHON) -m pytest tests/unit

integration:
	$(PYTHON) -m pytest tests/integration

live:
	$(PYTHON) -m pytest -o addopts='' -m live tests/integration

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down -v

logs:
	$(COMPOSE) logs -f
