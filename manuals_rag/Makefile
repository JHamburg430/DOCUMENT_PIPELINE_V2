PYTHON=/home/john/Desktop/Programming/Document_Pipeline/.venv/bin/python
COMPOSE=docker compose -f infra/compose/docker-compose.yml

.PHONY: test unit integration up down logs

test:
	cd manuals_rag && $(PYTHON) -m pytest

unit:
	cd manuals_rag && $(PYTHON) -m pytest tests/unit

integration:
	cd manuals_rag && $(PYTHON) -m pytest tests/integration

up:
	cd manuals_rag && $(COMPOSE) up -d --build

down:
	cd manuals_rag && $(COMPOSE) down -v

logs:
	cd manuals_rag && $(COMPOSE) logs -f
