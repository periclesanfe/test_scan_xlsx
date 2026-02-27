# Plano de Teste OMR

## Meta
Medir e evoluir confiabilidade da leitura de marcacoes em cenarios reais.

## Dataset Golden

Criar conjunto de imagens com gabarito verdadeiro contendo:

- folha reta e limpa
- folha inclinada/rotacionada
- sombra e variacao de iluminacao
- baixa resolucao e ruido
- marcacao fraca
- multipla marcacao

## Criterios de avaliacao

- acuracia global por questao
- precision/recall por alternativa
- taxa de falso positivo
- taxa de "sem resposta" indevida

## Metas iniciais sugeridas

- scans limpos: >= 98% acuracia
- scans degradados: >= 93% acuracia

## Automacao

- testes sinteticos para regressao rapida
- execucao em CI para detectar queda de qualidade
- relatorio com comparacao entre versoes de algoritmo

## Scripts de apoio

- gerar dataset sintetico:
  - `python scripts/omr_synthetic_dataset.py /tmp/omr_dataset --samples 20`
- medir acuracia:
  - `python scripts/omr_metrics.py /tmp/omr_dataset/dataset.json`
