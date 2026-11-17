"""Pré-processamento: tokenização, minúsculas, stopwords, stemming RSLP.

Cada passo é opcional para permitir a ablation e medir o efeito no vocabulário.
"""
from __future__ import annotations

import re
from functools import lru_cache

import nltk

for pkg in ("stopwords", "rslp"):
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg == "stopwords" else f"stemmers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords  # noqa: E402
from nltk.stem import RSLPStemmer  # noqa: E402

STOP = set(stopwords.words("portuguese"))
_stemmer = RSLPStemmer()
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:-[A-Za-zÀ-ÖØ-öø-ÿ]+)*|\d+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")


@lru_cache(maxsize=200_000)
def stem(w: str) -> str:
    return _stemmer.stem(w)


def tokenize(text: str, lower=True, rm_stop=False, do_stem=False, rm_digits=False) -> list[str]:
    text = URL_RE.sub(" URL ", text or "")
    toks = TOKEN_RE.findall(text)
    if lower:
        toks = [t.lower() for t in toks]
    if rm_digits:
        toks = [t for t in toks if not t.isdigit()]
    if rm_stop:
        toks = [t for t in toks if t.lower() not in STOP]
    if do_stem:
        toks = [stem(t.lower()) for t in toks]
    return toks


PIPELINES = {
    "raw": dict(lower=False),
    "lower": dict(lower=True),
    "lower+stop": dict(lower=True, rm_stop=True),
    "lower+stop+rslp": dict(lower=True, rm_stop=True, do_stem=True),
}


def analyzer(name: str):
    cfg = PIPELINES[name]
    return lambda text: tokenize(text, **cfg)
