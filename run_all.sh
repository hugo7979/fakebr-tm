#!/usr/bin/env bash
# Reproduz todas as tabelas com um único comando:  bash run_all.sh   (ou  bash run_all.sh --final)
# --final avalia no conjunto de teste. Usar UMA vez, no fim, com as configurações já fixadas no dev.
set -euo pipefail
cd "$(dirname "$0")/src"
python3 data.py          # descarrega Fake.Br (commit fixo) e constrói data/corpus.csv
python3 split.py         # divisão dev/teste por grupos de pares (semente 42) -> data/split.json
python3 characterize.py  # estatísticas, Zipf, vocabulário por pipeline, log-odds
python3 tune.py           # afinação por CV no dev (em cache em results/tuning.csv)
python3 classify.py "$@" # baselines, TF-IDF, ablation, leave-one-topic-out [, teste final]
python3 topics.py         # tópicos LDA (seleção de K em cache em results/topicos_k_selecao.csv)
python3 ie_extract.py     # extração de informação: avaliação no ouro (se já anotado) + falsas vs verdadeiras
python3 near_duplicates.py # verificação de quase-duplicados entre folds
python3 make_tables.py     # tabelas do artigo (paper/tables)
