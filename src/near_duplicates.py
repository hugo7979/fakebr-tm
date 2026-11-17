"""Verificação de fuga por notícias QUASE repetidas entre folds (a divisão só agrupa repetições exatas).

Para cada documento do dev, procura o mais semelhante (cosseno TF-IDF no texto cortado) que esteja
noutro fold da CV. Se um modelo acerta sobretudo porque "já viu" uma quase-cópia, o seu F1 deve
cair quando se retiram esses documentos. Recalcula o F1 dos modelos com previsões out-of-fold
guardadas (BERTimbau fine-tuned) só nos documentos sem vizinho acima do limiar.

Saída: results/near_duplicates.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from classify import get_folds
from data import ROOT
from split import load_split

RES = ROOT / "results"
THRESHOLDS = [0.5, 0.7, 0.8, 0.9]

if __name__ == "__main__":
    dev, _ = load_split()
    fold = np.zeros(len(dev), int)
    for k, (_, va) in enumerate(get_folds(dev)):
        fold[va] = k
    X = TfidfVectorizer(min_df=2, sublinear_tf=True).fit_transform(dev.text_cut)
    S = (X @ X.T).tocsr()
    S.setdiag(0)
    S.eliminate_zeros()
    best = np.zeros(len(dev))
    same = np.zeros(len(dev), bool)
    for i in range(len(dev)):
        row = S.getrow(i)
        m = fold[row.indices] != fold[i]
        if m.any():
            j = row.indices[m][row.data[m].argmax()]
            best[i], same[i] = row.data[m].max(), dev.label.iloc[j] == dev.label.iloc[i]

    oofs = {f.stem.replace("oof_", ""): f for f in sorted(RES.glob("oof_bertimbau_ft_*.csv"))}
    rows = []
    for th in THRESHOLDS:
        has = best >= th
        r = {"limiar_cosseno": th, "docs_com_vizinho_noutro_fold": int(has.sum()),
             "dos_quais_falsas": int(dev.label[has].sum()),
             "pct_vizinho_mesma_classe": same[has].mean() * 100 if has.any() else np.nan}
        for name, f in oofs.items():
            o = pd.read_csv(f).set_index("doc_id").loc[dev.doc_id]
            pred = (o.prob_falsa.values > .5).astype(int)
            r[f"F1_{name}_todos"] = f1_score(dev.label, pred, average="macro")
            r[f"F1_{name}_sem_vizinhos"] = f1_score(dev.label[~has], pred[~has], average="macro")
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "near_duplicates.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
