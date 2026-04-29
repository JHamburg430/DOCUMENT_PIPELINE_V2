PYTHON ?= python
COMPOSE ?= docker compose -f infra/compose/docker-compose.yml

.PHONY: test unit integration live up down logs

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
