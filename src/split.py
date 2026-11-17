"""Divisão desenvolvimento/teste, feita UMA vez antes de qualquer modelo.

Unidade de agrupamento: componente ligada de pares que partilham o mesmo
texto ou o mesmo link (há notícias verdadeiras reutilizadas em vários pares).
Assim, nenhum par (nem texto repetido) fica dos dois lados da divisão.
A estratificação é feita pelo tema do par.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from data import ROOT, load

SEED = 42
SPLIT_FILE = ROOT / "data" / "split.json"


def make_groups(df: pd.DataFrame) -> pd.Series:
    parent = {p: p for p in df.pair_id.unique()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for key in ("text", "link"):
        for _, ids in df.groupby(key).pair_id:
            ids = list(ids)
            for other in ids[1:]:
                parent[find(other)] = find(ids[0])
    return df.pair_id.map(lambda p: find(p))


def topic_bucket(t: str) -> str:
    return t if t in {"politica", "tv_celebridades", "sociedade_cotidiano"} else "outros"


def make_split() -> dict:
    df = load()
    df["group"] = make_groups(df)
    df["topic4"] = df.topic.map(topic_bucket)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    _, test_idx = next(sgkf.split(df, df.topic4, groups=df.group))
    test_docs = set(df.iloc[test_idx].doc_id)
    split = {
        "seed": SEED,
        "test": sorted(test_docs),
        "groups": dict(zip(df.doc_id, df.group.astype(int))),
    }
    SPLIT_FILE.write_text(json.dumps(split))
    return split


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devolve (dev, test). O teste só deve ser usado na avaliação final."""
    split = json.loads(SPLIT_FILE.read_text()) if SPLIT_FILE.exists() else make_split()
    df = load()
    df["group"] = df.doc_id.map(split["groups"])
    df["topic4"] = df.topic.map(topic_bucket)
    is_test = df.doc_id.isin(set(split["test"]))
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


if __name__ == "__main__":
    dev, test = load_split()
    print("dev", dev.shape, "test", test.shape)
    print(pd.crosstab(test.topic4, test.label))
    shared = set(dev.group) & set(test.group)
    print("grupos partilhados dev/teste:", len(shared))
    print("grupos com >1 par:", (dev.groupby("group").pair_id.nunique() > 1).sum()
          + (test.groupby("group").pair_id.nunique() > 1).sum())
