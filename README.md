# test_scan_xlsx

Aplicacao Flask para cadastro de perguntas, montagem de provas, geracao de PDF, registro de scans e consolidacao de resultados.

## Funcionalidades

- CRUD de perguntas (`yes_no`, `likert`, `custom`)
- Importacao/exportacao de perguntas em XLSX
- Criacao de provas com ordenacao de questoes
- Geracao de PDF imprimivel da prova
- Registro de scan com preenchimento manual e tentativa de OMR
- Relatorio individual e relatorio geral por prova
- Exportacao JSON para integracao:
  - `GET /questions/export.json`
  - `POST /questions/import.json` (payload JSON ou arquivo `.json`)
  - `GET /tests/<tid>/manifest.json`
  - `GET /tests/<tid>/results.json`

## Persistencia de dados

- Banco local SQLite (padrao `quiz.db`)
- Uploads locais em `uploads/`

## Requisitos

- Python 3.11+

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Executar

```bash
export FLASK_ENV=development
export SECRET_KEY=dev-secret-key
python app.py
```

App em `http://127.0.0.1:5000`.

## Testes

```bash
pytest -q
```

## Variaveis de ambiente

- `SECRET_KEY`: obrigatoria em producao (`FLASK_ENV=production`)
- `DATABASE_URL`: string de conexao SQLAlchemy (padrao `sqlite:///quiz.db`)
- `FLASK_ENV`: `development` ou `production`
- `FLASK_DEBUG`: `1` para debug local

## Integracao

Documentacao de formatos em [docs/integration_formats.md](docs/integration_formats.md).
Plano de teste OMR em [docs/omr_test_plan.md](docs/omr_test_plan.md).

## Ferramentas OMR

Gerar dataset sintetico:

```bash
python scripts/omr_synthetic_dataset.py /tmp/omr_dataset --samples 20
```

Avaliar acuracia:

```bash
python scripts/omr_metrics.py /tmp/omr_dataset/dataset.json
```
