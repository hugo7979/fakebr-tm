"""Caracterização do corpus (calculada só no conjunto de desenvolvimento).

Gera: results/corpus_stats.csv, results/vocab_by_pipeline.csv,
results/zipf.csv, results/top_logodds.csv e figuras em results/figs/.
"""
from __future__ import annotations

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import ROOT  # noqa: E402
from preprocess import PIPELINES, tokenize  # noqa: E402
from split import load_split  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figs"
FIG.mkdir(parents=True, exist_ok=True)
NAMES = {0: "verdadeira", 1: "falsa"}


def corpus_stats(dev: pd.DataFrame) -> pd.DataFrame:
    dev = dev.assign(n_words=dev.text.map(lambda t: len(tokenize(t))),
                     n_words_norm=dev.text_norm.fillna("").map(lambda t: len(tokenize(t))))
    g = dev.groupby("label")
    stats = pd.DataFrame({
        "documentos": g.size(),
        "palavras_total": g.n_words.sum(),
        "palavras_mediana": g.n_words.median(),
        "palavras_p10": g.n_words.quantile(.1),
        "palavras_p90": g.n_words.quantile(.9),
        "palavras_norm_mediana": g.n_words_norm.median(),
        "maiusculas_mediana": g.n_upper_words.median(),
        "erros_ortograficos_media_pct": g.spelling_errors_pct.mean() * 100,
        "frase_media_mediana": g.avg_sentence_len.median(),
        "dominio_mais_frequente_pct": g.domain.agg(lambda s: s.value_counts(normalize=True).iloc[0] * 100),
    }).rename(index=NAMES).T
    stats.to_csv(RES / "corpus_stats.csv")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    bins = np.logspace(1, 4, 40)
    for lab, c in [(0, "#3b6fb6"), (1, "#d1495b")]:
        ax.hist(dev[dev.label == lab].n_words, bins=bins, alpha=.6, label=NAMES[lab], color=c)
    ax.set_xscale("log"); ax.set_xlabel("palavras por documento (escala log)"); ax.set_ylabel("documentos")
    ax.legend(); ax.set_title("Comprimento por classe (dev)")
    fig.tight_layout(); fig.savefig(FIG / "comprimento.png", dpi=150); plt.close(fig)

    pd.crosstab(dev.topic, dev.label).rename(columns=NAMES).to_csv(RES / "topicos.csv")
    return stats


def vocab_by_pipeline(dev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, cfg in PIPELINES.items():
        c = Counter(t for txt in dev.text for t in tokenize(txt, **cfg))
        rows.append({"pipeline": name, "tokens": sum(c.values()), "vocabulario": len(c),
                     "hapax": sum(1 for v in c.values() if v == 1)})
    out = pd.DataFrame(rows)
    out["reducao_vocab_%"] = (1 - out.vocabulario / out.vocabulario.iloc[0]) * 100
    out.to_csv(RES / "vocab_by_pipeline.csv", index=False)
    return out


def zipf(dev: pd.DataFrame) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(5, 3.6))
    rows = []
    for lab, c in [(0, "#3b6fb6"), (1, "#d1495b")]:
        cnt = Counter(t for txt in dev[dev.label == lab].text for t in tokenize(txt))
        f = np.array(sorted(cnt.values(), reverse=True), dtype=float)
        r = np.arange(1, len(f) + 1)
        mask = (r >= 10) & (r <= 10_000)  # ajuste na zona central (evita cabeça e cauda de hapax)
        slope, intercept = np.polyfit(np.log10(r[mask]), np.log10(f[mask]), 1)
        rows.append({"classe": NAMES[lab], "alpha": -slope, "log10_C": intercept, "vocab": len(f)})
        ax.loglog(r, f, color=c, lw=1.2, label=f"{NAMES[lab]} (α={-slope:.2f})")
        ax.loglog(r[mask], 10 ** (intercept + slope * np.log10(r[mask])), "--", color=c, lw=.8)
    ax.set_xlabel("ordem (rank)"); ax.set_ylabel("frequência"); ax.legend(); ax.set_title("Lei de Zipf (dev)")
    fig.tight_layout(); fig.savefig(FIG / "zipf.png", dpi=150); plt.close(fig)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "zipf.csv", index=False)
    return out


def log_odds(dev: pd.DataFrame, k=25) -> pd.DataFrame:
    """Log-odds com prior de Dirichlet informativo (Monroe et al., 2008)."""
    cf = Counter(t for txt in dev[dev.label == 1].text_norm.fillna("") for t in tokenize(txt, rm_stop=True))
    ct = Counter(t for txt in dev[dev.label == 0].text_norm.fillna("") for t in tokenize(txt, rm_stop=True))
    prior = cf + ct
    a0, nf, nt = sum(prior.values()), sum(cf.values()), sum(ct.values())
    rows = []
    for w, a in prior.items():
        if a < 20:
            continue
        yf, yt = cf[w], ct[w]
        d = np.log((yf + a) / (nf + a0 - yf - a)) - np.log((yt + a) / (nt + a0 - yt - a))
        var = 1 / (yf + a) + 1 / (yt + a)
        rows.append((w, d / np.sqrt(var), yf, yt))
    df = pd.DataFrame(rows, columns=["palavra", "z", "freq_falsa", "freq_verdadeira"]).sort_values("z")
    out = pd.concat([df.tail(k).iloc[::-1].assign(lado="falsa"), df.head(k).assign(lado="verdadeira")])
    out.to_csv(RES / "top_logodds.csv", index=False)
    return out


if __name__ == "__main__":
    dev, _ = load_split()
    pd.set_option("display.width", 160)
    print(corpus_stats(dev).round(2))
    print(vocab_by_pipeline(dev).round(1))
    print(zipf(dev).round(3))
    lo = log_odds(dev)
    print("falsa:", ", ".join(lo[lo.lado == "falsa"].palavra.head(15)))
    print("verdadeira:", ", ".join(lo[lo.lado == "verdadeira"].palavra.head(15)))
