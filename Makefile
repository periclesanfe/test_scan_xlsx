.PHONY: check-python dev run test

PYTHON ?= python3
PIP := .venv/bin/pip

check-python:
	@$(PYTHON) -c "import sys; req=(3, 11); cur=sys.version_info[:2]; \
print(f'Python detectado: {sys.version.split()[0]}'); \
raise SystemExit(0 if cur >= req else 1)" || \
	(echo "Erro: este projeto requer Python 3.11+."; \
	echo "Instale Python 3.11 e execute: make dev PYTHON=python3.11"; exit 1)

dev:
	@$(MAKE) check-python PYTHON=$(PYTHON)
	$(PYTHON) -m venv --clear .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	FLASK_ENV=development SECRET_KEY=dev-secret-key .venv/bin/python app.py

test:
	.venv/bin/pytest -q
