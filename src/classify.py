"""Classificação: baselines, TF-IDF + modelos lineares, ablation e protocolo por tema.

Todas as escolhas são feitas com validação cruzada no conjunto de desenvolvimento
(StratifiedGroupKFold por grupo de pares). O teste é usado uma única vez, no fim,
com a configuração escolhida no dev (flag --final).

Saídas: results/cv_*.csv, results/test_*.csv, results/test_predictions.csv
"""
from __future__ import annotations

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from data import ROOT
from preprocess import PIPELINES, tokenize
from split import SEED, load_split

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
STYLE = ["n_words_no_punct", "n_types", "n_links", "n_upper_words", "n_verbs", "n_subj_imp_verbs",
         "n_nouns", "n_adjectives", "n_adverbs", "n_modal_verbs", "n_sing_1_2_pronouns",
         "n_plural_1_pronouns", "n_pronouns", "pausality", "n_chars", "avg_sentence_len",
         "avg_word_len", "spelling_errors_pct", "emotiveness", "diversity"]
# atributos de proporção (independentes do comprimento) para a versão "sem comprimento"
STYLE_RATIO = ["pausality", "avg_sentence_len", "avg_word_len", "spelling_errors_pct",
               "emotiveness", "diversity"]


# ---------------------------------------------------------------- utilitários
def identity(x):
    return x


def tfidf(ngram=(1, 2), min_df=2, max_features=200_000):
    return TfidfVectorizer(analyzer=identity, min_df=min_df, max_features=max_features,
                           sublinear_tf=True) if ngram == (1, 1) else \
        TfidfVectorizer(tokenizer=identity, preprocessor=identity, token_pattern=None,
                        lowercase=False, ngram_range=ngram, min_df=min_df,
                        max_features=max_features, sublinear_tf=True)


def bootstrap_ci(y, p, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = [f1_score(y[s], p[s], average="macro") for s in (rng.choice(idx, len(idx)) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])


def metrics(y, p, prefix=""):
    f = f1_score(y, p, average=None, labels=[0, 1])
    lo, hi = bootstrap_ci(np.asarray(y), np.asarray(p))
    return {f"{prefix}f1_macro": f1_score(y, p, average="macro"), f"{prefix}ci95_lo": lo,
            f"{prefix}ci95_hi": hi, f"{prefix}f1_verdadeira": f[0], f"{prefix}f1_falsa": f[1]}


EMB_DIR = ROOT / "data"


def load_embeddings(name):
    """Embeddings pré-calculados no Colab (notebooks/bertimbau.ipynb): data/emb_<name>.npz."""
    z = np.load(EMB_DIR / f"emb_{name}.npz", allow_pickle=False)
    return dict(zip(z["doc_id"].tolist(), z["X"]))


class Featurizer:
    """Constrói a matriz de atributos para uma configuração (ajustada só no treino)."""

    def __init__(self, kind, text_col="text_norm", pipeline="lower", ngram=(1, 2), min_df=None,
                 char_ngram=(2, 5)):
        self.kind, self.text_col, self.pipeline, self.ngram = kind, text_col, pipeline, ngram
        self.min_df, self.char_ngram = min_df, tuple(char_ngram)

    def _tokens(self, df):
        cfg = PIPELINES[self.pipeline]
        return [tokenize(t, **cfg) for t in df[self.text_col].fillna("")]

    @staticmethod
    def _ending(df, col):
        t = df[col].fillna("").str.rstrip()
        return np.c_[t.str.contains(r"[.!?…]$"), t.str.contains(r"[\"”»)]$"),
                     t.str.contains(r"[,;:]$"), t.str.contains(r"\w$")].astype(float)

    def fit_transform(self, df):
        if self.kind == "ending":
            return self._ending(df, self.text_col)
        if self.kind == "length":
            self.sc = StandardScaler()
            return self.sc.fit_transform(np.log1p(df[["n_words_no_punct"]].values))
        if self.kind in ("style", "style_ratio"):
            cols = STYLE if self.kind == "style" else STYLE_RATIO
            self.sc = StandardScaler()
            return self.sc.fit_transform(df[cols].fillna(0).values)
        if self.kind == "emb":  # text_col = nome do ficheiro de embeddings
            self.emb = load_embeddings(self.text_col)
            self.sc = StandardScaler()
            return self.sc.fit_transform(np.vstack([self.emb[d] for d in df.doc_id]))
        if self.kind == "char":
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=self.char_ngram,
                                       min_df=self.min_df or 3, max_features=300_000, sublinear_tf=True)
            return self.vec.fit_transform(df[self.text_col].fillna(""))
        self.vec = tfidf(tuple(self.ngram), min_df=self.min_df or 2)
        return self.vec.fit_transform(self._tokens(df))

    def transform(self, df):
        if self.kind == "ending":
            return self._ending(df, self.text_col)
        if self.kind == "length":
            return self.sc.transform(np.log1p(df[["n_words_no_punct"]].values))
        if self.kind in ("style", "style_ratio"):
            cols = STYLE if self.kind == "style" else STYLE_RATIO
            return self.sc.transform(df[cols].fillna(0).values)
        if self.kind == "emb":
            return self.sc.transform(np.vstack([self.emb[d] for d in df.doc_id]))
        if self.kind == "char":
            return self.vec.transform(df[self.text_col].fillna(""))
        return self.vec.transform(self._tokens(df))


def make_model(name, C=None):
    """C=None usa os valores por defeito do projeto; src/tune.py escolhe C por CV no dev."""
    return {
        "majority": lambda: DummyClassifier(strategy="most_frequent"),
        "logreg": lambda: LogisticRegression(C=C or 10, max_iter=2000, random_state=SEED),
        "svm": lambda: LinearSVC(C=C or 1.0, random_state=SEED),
        "logreg_emb": lambda: LogisticRegression(C=C or 0.1, max_iter=5000, random_state=SEED),
    }[name]()


def get_folds(dev, n_splits=5):
    """Folds da validação cruzada. Partilhados com o notebook do BERTimbau."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(cv.split(dev, dev.label, groups=dev.group))


# ---------------------------------------------------------------- experiências
def run_cv(dev, configs, n_splits=5, tag="cv"):
    """configs: lista de dicts {nome, feat: dict p/ Featurizer, model}."""
    folds = get_folds(dev, n_splits)
    rows = []
    for c in configs:
        t0 = time.time()
        oof = np.zeros(len(dev), dtype=int)
        fold_f1 = []
        for tr, va in folds:
            fz = Featurizer(**c["feat"])
            Xtr, Xva = fz.fit_transform(dev.iloc[tr]), fz.transform(dev.iloc[va])
            m = make_model(c["model"], c.get("C")).fit(Xtr, dev.label.iloc[tr])
            oof[va] = m.predict(Xva)
            fold_f1.append(f1_score(dev.label.iloc[va], oof[va], average="macro"))
        r = {"config": c["nome"], "f1_macro_mean": np.mean(fold_f1), "f1_macro_std": np.std(fold_f1),
             **metrics(dev.label.values, oof, prefix="oof_"), "segundos": time.time() - t0}
        rows.append(r)
        print(f"  {c['nome']:<42} F1m={r['f1_macro_mean']:.4f} ±{r['f1_macro_std']:.4f} "
              f"({r['segundos']:.0f}s)", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / f"{tag}.csv", index=False)
    return out


MAIN = [
    {"nome": "baseline: classe maioritária", "feat": {"kind": "length"}, "model": "majority"},
    {"nome": "baseline: só comprimento (texto completo)", "feat": {"kind": "length"}, "model": "logreg"},
    {"nome": "estilo NILC (20 atributos)", "feat": {"kind": "style"}, "model": "logreg"},
    {"nome": "estilo NILC só proporções (6)", "feat": {"kind": "style_ratio"}, "model": "logreg"},
    {"nome": "TF-IDF palavras 1-2 + LogReg (completo)", "feat": {"kind": "word", "text_col": "text"}, "model": "logreg"},
    {"nome": "TF-IDF palavras 1-2 + SVM (completo)", "feat": {"kind": "word", "text_col": "text"}, "model": "svm"},
    {"nome": "TF-IDF palavras 1-2 + LogReg (normalizado)", "feat": {"kind": "word"}, "model": "logreg"},
    {"nome": "TF-IDF palavras 1-2 + SVM (normalizado)", "feat": {"kind": "word"}, "model": "svm"},
    {"nome": "TF-IDF caracteres 2-5 + SVM (normalizado)", "feat": {"kind": "char"}, "model": "svm"},
    # (índices 0-8 usados no leave-one-topic-out: acrescentar sempre DEPOIS daqui)
    # Atalho do final do texto: na versão normalizada as verdadeiras foram cortadas a meio da frase
    {"nome": "baseline: como acaba o texto (normalizado)", "feat": {"kind": "ending", "text_col": "text_norm"}, "model": "logreg"},
    {"nome": "baseline: como acaba o texto (cortado)", "feat": {"kind": "ending", "text_col": "text_cut"}, "model": "logreg"},
    {"nome": "TF-IDF palavras 1-2 + SVM (cortado)", "feat": {"kind": "word", "text_col": "text_cut"}, "model": "svm"},
    {"nome": "TF-IDF caracteres 2-5 + SVM (cortado)", "feat": {"kind": "char", "text_col": "text_cut"}, "model": "svm"},
]
# Representação neural: só entra se os embeddings já tiverem sido gerados no Colab
for _name, _label in [("bertimbau_norm", "normalizado"), ("bertimbau_full", "completo, 512 tokens"),
                      ("bertimbau_cut", "cortado")]:
    if (EMB_DIR / f"emb_{_name}.npz").exists():
        MAIN.append({"nome": f"BERTimbau congelado + LogReg ({_label})",
                     "feat": {"kind": "emb", "text_col": _name}, "model": "logreg_emb"})

ABLATION = [
    {"nome": f"SVM normalizado | {p} | {'1-2' if ng == (1, 2) else '1'}-gramas",
     "feat": {"kind": "word", "pipeline": p, "ngram": ng}, "model": "svm"}
    for p in PIPELINES for ng in [(1, 1), (1, 2)]
]


def leave_one_topic_out(dev, configs):
    """Treina sem um tema e testa nesse tema (dentro do dev). Compara com CV no mesmo tema."""
    rows = []
    for c in configs:
        for topic in ["politica", "tv_celebridades", "sociedade_cotidiano", "outros"]:
            tr, te = dev[dev.topic4 != topic], dev[dev.topic4 == topic]
            fz = Featurizer(**c["feat"])
            m = make_model(c["model"], c.get("C")).fit(fz.fit_transform(tr), tr.label)
            p = m.predict(fz.transform(te))
            rows.append({"config": c["nome"], "tema_excluido": topic, "n_teste": len(te),
                         **metrics(te.label.values, p)})
            print(f"  {c['nome']:<42} sem {topic:<20} F1m={rows[-1]['f1_macro']:.4f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "cv_leave_one_topic_out.csv", index=False)
    return out


def load_tuned():
    """Configurações escolhidas por src/tune.py (CV no dev). Vazio se ainda não houve afinação."""
    import json
    f = RES / "best_params.json"
    return json.loads(f.read_text()) if f.exists() else []


def final_test(dev, test, configs):
    """Avaliação ÚNICA no teste. Correr apenas depois de fixar as configurações no dev."""
    rows, preds = [], test[["doc_id", "pair_id", "label", "topic"]].copy()
    for c in configs:
        t0 = time.time()
        fz = Featurizer(**c["feat"])
        m = make_model(c["model"], c.get("C")).fit(fz.fit_transform(dev), dev.label)
        p = m.predict(fz.transform(test))
        cm = confusion_matrix(test.label, p, labels=[0, 1])
        rows.append({"config": c["nome"], **metrics(test.label.values, p),
                     "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1],
                     "segundos_treino+teste": time.time() - t0})
        preds[c["nome"]] = p
    out = pd.DataFrame(rows)
    out.to_csv(RES / "test_main_table.csv", index=False)
    preds.to_csv(RES / "test_predictions.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="avaliar no conjunto de teste (uma vez!)")
    ap.add_argument("--skip-ablation", action="store_true")
    a = ap.parse_args()
    dev, test = load_split()
    pd.set_option("display.width", 200)
    print("== Comparação principal (CV 5 folds, dev) ==")
    run_cv(dev, MAIN, tag="cv_main")
    if not a.skip_ablation:
        print("== Ablation do pré-processamento ==")
        run_cv(dev, ABLATION, tag="cv_ablation")
    print("== Leave-one-topic-out (dev) ==")
    leave_one_topic_out(dev, [MAIN[3], MAIN[7], MAIN[8]])
    if a.final:
        print("== TESTE FINAL ==")
        print(final_test(dev, test, MAIN + load_tuned()).round(4).to_string(index=False))
