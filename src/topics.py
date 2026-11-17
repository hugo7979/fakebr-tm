"""Tarefa não supervisionada: modelação de tópicos com LDA (só conjunto de desenvolvimento).

Perguntas:
  1. Que assuntos existem no corpus?  (palavras de cada tópico)
  2. Os tópicos separam FALSAS de VERDADEIRAS, ou só os TEMAS?  (NMI, prevalência por classe)
  3. A distribuição de tópicos serve como atributo de classificação?  (F1 com CV, mesmos folds)

Escolha do número de tópicos K: coerência NPMI média dos top-10 termos, 3 sementes por K
(estabilidade); maior K a menos de 1 dp da melhor coerência (ver comentário no main). A seleção fica em cache em results/topicos_k_selecao.csv (--reselect para refazer).

Usa o texto normalizado (para o comprimento não influenciar) e contagens (o LDA modela contagens,
não TF-IDF). Remove stopwords e números: aqui queremos CONTEÚDO, não estilo.
"""
from __future__ import annotations

import argparse
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import LatentDirichletAllocation  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, normalized_mutual_info_score  # noqa: E402

from classify import get_folds, metrics  # noqa: E402
from data import ROOT  # noqa: E402
from preprocess import tokenize  # noqa: E402
from split import SEED, load_split  # noqa: E402

warnings.filterwarnings("ignore")
RES = ROOT / "results"
FIG = RES / "figs"
K_GRID = [5, 10, 15, 20, 30]
SEEDS = [42, 7, 2026]
TOPN = 10


def identity(x):
    return x


def doc_tokens(texts):
    return [[t for t in tokenize(x, rm_stop=True, rm_digits=True) if len(t) > 2] for x in texts]


def vectorizer():
    return CountVectorizer(analyzer=identity, min_df=5, max_df=0.5)


def fit_lda(X, k, seed):
    return LatentDirichletAllocation(n_components=k, learning_method="batch", max_iter=30,
                                     random_state=seed, n_jobs=-1).fit(X)


def npmi_coherence(model, Xbin, topn=TOPN):
    """NPMI médio entre pares dos top-n termos de cada tópico (co-ocorrência no documento)."""
    N = Xbin.shape[0]
    df = np.asarray(Xbin.sum(0)).ravel()
    per_topic = []
    for comp in model.components_:
        top = comp.argsort()[::-1][:topn]
        sub = Xbin[:, top]
        co = (sub.T @ sub).toarray()
        vals = []
        for i in range(topn):
            for j in range(i + 1, topn):
                pij = co[i, j] / N
                if pij == 0:
                    vals.append(-1.0)
                    continue
                pi, pj = df[top[i]] / N, df[top[j]] / N
                vals.append(np.log(pij / (pi * pj)) / -np.log(pij))
        per_topic.append(np.mean(vals))
    return np.array(per_topic)


def select_k(X, Xbin):
    rows = []
    for k in K_GRID:
        for s in SEEDS:
            t0 = time.time()
            m = fit_lda(X, k, s)
            coh = npmi_coherence(m, Xbin)
            rows.append({"K": k, "semente": s, "npmi_medio": coh.mean(), "npmi_min": coh.min(),
                         "perplexidade": m.perplexity(X), "segundos": time.time() - t0})
            print(f"  K={k:<3} semente={s:<5} NPMI={coh.mean():.4f}  ({rows[-1]['segundos']:.0f}s)", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "topicos_k_selecao.csv", index=False)
    return out


def plot_selection(sel):
    g = sel.groupby("K").npmi_medio.agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=3, color="#3b6fb6")
    ax.set_xlabel("número de tópicos K")
    ax.set_ylabel("coerência NPMI (média ± dp, 3 sementes)")
    ax.set_title("Escolha de K (dev)")
    fig.tight_layout()
    fig.savefig(FIG / "topicos_coerencia.png", dpi=150)
    plt.close(fig)


def describe(model, cv, theta, dev, coh):
    vocab = np.array(cv.get_feature_names_out())
    dom = theta.argmax(1)
    rows = []
    for k, comp in enumerate(model.components_):
        in_k = dom == k
        rows.append({
            "topico": k,
            "palavras": " ".join(vocab[comp.argsort()[::-1][:12]]),
            "npmi": coh[k],
            "peso_medio_falsas": theta[dev.label.values == 1, k].mean(),
            "peso_medio_verdadeiras": theta[dev.label.values == 0, k].mean(),
            "docs_dominante": int(in_k.sum()),
            "pct_falsas_entre_dominantes": dev.label.values[in_k].mean() * 100 if in_k.any() else np.nan,
            "tema_mais_comum": dev.topic.values[in_k].tolist() and pd.Series(dev.topic.values[in_k]).mode()[0],
        })
    out = pd.DataFrame(rows)
    out["razao_F_V"] = out.peso_medio_falsas / out.peso_medio_verdadeiras
    out = out.sort_values("razao_F_V", ascending=False)
    out.to_csv(RES / "topicos_palavras.csv", index=False)

    # documentos representativos: os 3 com maior peso em cada tópico (para interpretar e citar no artigo)
    reps = []
    for k in range(theta.shape[1]):
        for rank, i in enumerate(theta[:, k].argsort()[::-1][:3], 1):
            reps.append({"topico": k, "rank": rank, "peso": theta[i, k], "doc_id": dev.doc_id.iloc[i],
                         "classe": "falsa" if dev.label.iloc[i] == 1 else "verdadeira",
                         "inicio_do_texto": " ".join(str(dev.text_norm.iloc[i]).split()[:40])})
    pd.DataFrame(reps).to_csv(RES / "topicos_documentos.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(out) + 1))
    lab = [f"T{r.topico}: {' '.join(r.palavras.split()[:4])}" for r in out.itertuples()]
    y = np.arange(len(out))
    ax.barh(y - .2, out.peso_medio_falsas, .4, color="#d1495b", label="falsas")
    ax.barh(y + .2, out.peso_medio_verdadeiras, .4, color="#3b6fb6", label="verdadeiras")
    ax.set_yticks(y, lab, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("peso médio do tópico nos documentos")
    ax.legend()
    ax.set_title("Prevalência dos tópicos por classe (dev)")
    fig.tight_layout()
    fig.savefig(FIG / "topicos_por_classe.png", dpi=150)
    plt.close(fig)
    return out


def separation(theta, dev, X, k_best):
    """NMI do tópico dominante com o rótulo e com o tema; F1 de classificação com os tópicos."""
    dom = theta.argmax(1)
    rows = [
        {"medida": "NMI(tópico dominante, falsa/verdadeira)", "valor": normalized_mutual_info_score(dev.label, dom)},
        {"medida": "NMI(tópico dominante, tema do corpus)", "valor": normalized_mutual_info_score(dev.topic, dom)},
    ]
    # classificação: LDA reajustado DENTRO de cada fold (sem fuga), LogReg sobre as proporções de tópicos
    oof = np.zeros(len(dev), dtype=int)
    fold_f1 = []
    for tr, va in get_folds(dev):
        cv = vectorizer()
        Xtr = cv.fit_transform(dev.tokens.iloc[tr])
        lda = fit_lda(Xtr, k_best, SEED)
        Ttr, Tva = lda.transform(Xtr), lda.transform(cv.transform(dev.tokens.iloc[va]))
        clf = LogisticRegression(C=10, max_iter=2000, random_state=SEED).fit(Ttr, dev.label.iloc[tr])
        oof[va] = clf.predict(Tva)
        fold_f1.append(f1_score(dev.label.iloc[va], oof[va], average="macro"))
    m = metrics(dev.label.values, oof)
    rows += [{"medida": f"F1-macro LogReg sobre {k_best} tópicos (CV 5 folds, média)", "valor": np.mean(fold_f1)},
             {"medida": "  desvio-padrão entre folds", "valor": np.std(fold_f1)},
             {"medida": "  IC95% bootstrap (out-of-fold)", "valor": f"[{m['ci95_lo']:.3f}, {m['ci95_hi']:.3f}]"}]
    out = pd.DataFrame(rows)
    out.to_csv(RES / "topicos_metricas.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reselect", action="store_true", help="refazer a seleção de K (~8 min)")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 90)
    dev, _ = load_split()
    dev["tokens"] = doc_tokens(dev.text_norm.fillna(""))
    cv = vectorizer()
    X = cv.fit_transform(dev.tokens)
    Xbin = (X > 0).astype(np.int32).tocsc()
    print(f"matriz documentos x termos: {X.shape}")

    sel_file = RES / "topicos_k_selecao.csv"
    if a.reselect or not sel_file.exists():
        print("== Seleção de K (coerência NPMI, 3 sementes) ==")
        sel = select_k(X, Xbin)
    else:
        sel = pd.read_csv(sel_file)
        print("== Seleção de K: lida de results/topicos_k_selecao.csv (--reselect para refazer) ==")
    plot_selection(sel)
    summary = sel.groupby("K").agg(npmi=("npmi_medio", "mean"), npmi_dp=("npmi_medio", "std"),
                                   perplexidade=("perplexidade", "mean"), segundos=("segundos", "mean"))
    print(summary.round(4).to_string())
    # Regra "um desvio-padrão": as diferenças de coerência entre K são da ordem da variação entre
    # sementes, por isso escolhemos o MAIOR K (tópicos mais finos, mais interpretáveis) cuja coerência
    # média fica a menos de 1 dp da melhor. Análoga à regra 1-SE de Breiman et al. (1984).
    best = summary.npmi.idxmax()
    limiar = summary.loc[best, "npmi"] - summary.loc[best, "npmi_dp"]
    k_best = int(summary[summary.npmi >= limiar].index.max())
    print(f"melhor média: K={best}; limiar (média - 1 dp) = {limiar:.4f} -> K escolhido = {k_best}")

    model = fit_lda(X, k_best, SEED)
    theta = model.transform(X)
    coh = npmi_coherence(model, Xbin)
    print(f"== {k_best} tópicos, ordenados de 'mais típico das falsas' para 'mais típico das verdadeiras' ==")
    d = describe(model, cv, theta, dev, coh)
    print(d[["topico", "razao_F_V", "pct_falsas_entre_dominantes", "tema_mais_comum", "palavras"]]
          .round(2).to_string(index=False))
    print("== Os tópicos separam classes ou temas? ==")
    print(separation(theta, dev, X, k_best).to_string(index=False))
