"""Download e carregamento do corpus Fake.Br.

Os dados em bruto NÃO são incluídos no repositório (regra do enunciado):
este módulo clona o repositório oficial num commit fixo e constrói uma
tabela única `data/corpus.csv` (uma linha por notícia).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

REPO_URL = "https://github.com/roneysco/Fake.br-Corpus.git"
COMMIT = "780f5516c4ae070761632d98ac3368f3ded09d35"  # fixado para reprodutibilidade
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
OUT = ROOT / "data" / "corpus.csv"

META_FIELDS = [
    "author", "link", "category", "date", "n_tokens", "n_words_no_punct",
    "n_types", "n_links", "n_upper_words", "n_verbs", "n_subj_imp_verbs",
    "n_nouns", "n_adjectives", "n_adverbs", "n_modal_verbs",
    "n_sing_1_2_pronouns", "n_plural_1_pronouns", "n_pronouns", "pausality",
    "n_chars", "avg_sentence_len", "avg_word_len", "spelling_errors_pct",
    "emotiveness", "diversity",
]


def download() -> None:
    if (RAW / "full_texts").exists():
        return
    subprocess.run(["git", "clone", REPO_URL, str(RAW)], check=True)
    subprocess.run(["git", "-C", str(RAW), "checkout", COMMIT], check=True)


def _read(p: Path) -> str | None:
    if not p.exists():  # a versão normalizada tem ficheiros em falta/renumerados
        return None
    return p.read_text(encoding="utf-8", errors="replace").lstrip("﻿").strip()


def _meta(p: Path) -> dict:
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    lines += [""] * (len(META_FIELDS) - len(lines))
    return dict(zip(META_FIELDS, (l.strip() for l in lines)))


def build() -> pd.DataFrame:
    download()
    rows = []
    for label, name in [(1, "fake"), (0, "true")]:
        for f in sorted((RAW / "full_texts" / name).glob("*.txt"), key=lambda p: int(p.stem)):
            pid = int(f.stem)
            row = {
                "pair_id": pid,
                "label": label,  # 1 = falsa, 0 = verdadeira
                "text": _read(f),
                "text_norm": _read(RAW / "size_normalized_texts" / name / f.name),
            }
            row.update(_meta(RAW / "full_texts" / f"{name}-meta-information" / f"{pid}-meta.txt"))
            rows.append(row)
    df = pd.DataFrame(rows)
    num = META_FIELDS[4:]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df["domain"] = df["link"].str.extract(r"https?://(?:www\.)?([^/]+)")[0]
    # tema do par = tema da notícia falsa (os pares foram alinhados por assunto)
    fake_cat = df[df.label == 1].set_index("pair_id")["category"]
    df["topic"] = df["pair_id"].map(fake_cat)
    df["doc_id"] = df["label"].map({1: "fake", 0: "true"}) + "_" + df["pair_id"].astype(str)
    return df


CUT_WORDS, CUT_MARGIN = 100, 5


def cut_text(text) -> str:
    """Versão "cortada": primeiras 100 palavras do texto normalizado (ou n-5 nos textos mais curtos).

    Porquê: na versão normalizada do corpus, 98% das verdadeiras foram cortadas a meio da frase e
    77% das falsas acabam em pontuação; só "como acaba o texto" dá F1 ~0,89. Cortando TODOS os
    textos, nenhum acaba naturalmente e o atalho desaparece (os pares já têm o mesmo comprimento).
    """
    w = str(text or "").split()
    return " ".join(w[:max(1, min(CUT_WORDS, len(w) - CUT_MARGIN))])


def load() -> pd.DataFrame:
    if not OUT.exists():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        build().to_csv(OUT, index=False)
    df = pd.read_csv(OUT, keep_default_na=False, na_values=[""])
    df["text_cut"] = df.text_norm.map(cut_text)
    return df


if __name__ == "__main__":
    d = load()
    print(d.shape)
    print(d.groupby(["label", "topic"]).size().unstack(0))
