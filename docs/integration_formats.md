# Formatos de Integracao

## Objetivo
Padronizar exportacao para consumo por sistemas externos sem depender de planilha manual.

## Formatos suportados atualmente

- XLSX:
  - uso principal: cadastro manual de perguntas
  - endpoints:
    - `GET /questions/export`
    - `POST /questions/import`

- JSON:
  - uso principal: integracao com outros sistemas
  - endpoints:
    - `GET /questions/export.json`
    - `POST /questions/import.json`
    - `GET /tests/<tid>/manifest.json`
    - `GET /tests/<tid>/results.json`

## Recomendacao de uso

- Operacao humana: manter XLSX.
- Integracao tecnica: usar JSON.
- XML: implementar apenas se houver requisito explicito de legado.

## Estrutura esperada para OMR (manifest)

`/tests/<tid>/manifest.json` retorna:

- metadados da prova
- questoes em ordem
- opcoes por questao
- gabarito (quando houver)

## Estrutura de resultados

`/tests/<tid>/results.json` retorna:

- lista de scans
- respostas detectadas/manuais por `question_id`
- score por scan (`correct`, `total`, `percent`)
- referencia de imagem enviada
