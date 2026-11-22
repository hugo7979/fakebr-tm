"""Demo para a apresentação: aplica o pipeline a uma notícia NOVA.

    python src/demo.py --file minha_noticia.txt     # notícia num ficheiro de texto (UTF-8)
    python src/demo.py                              # colar a notícia no terminal; terminar com linha vazia
    python src/demo.py --exemplo falha              # exemplos do corpus: falsa | verdadeira | falha

Mostra: (1) a previsão do classificador de estilo com a probabilidade; (2) os pedaços de texto que mais
pesaram; (3) os sinais de ancoragem factual (datas, locais, pessoas, atribuição) face à média das falsas
e das verdadeiras; (4) as entidades extraídas; (5) o tópico LDA dominante.

Modelo: TF-IDF de caracteres + regressão logística na versão "cortada" (100 primeiras palavras),
treinado SÓ no conjunto de desenvolvimento (o teste nunca é usado aqui). Dá probabilidades e é
explicável. Os modelos ficam em cache em models/ (o primeiro arranque demora ~1 min).
É uma ferramenta de TRIAGEM: ordena o que verificar primeiro, não decide o que é falso.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data import ROOT, cut_text, load
from split import SEED, load_split

MODELS, RES = ROOT / "models", ROOT / "results"
EXEMPLOS = ROOT / "demo" / "exemplos.json"
TYPES = ["PESSOA", "ORGANIZACAO", "LOCAL", "DATA"]


def char_params():
    """Usa a afinação de src/tune.py, se existir; senão, os valores por defeito do projeto."""
    f = RES / "best_params.json"
    if f.exists():
        for b in json.loads(f.read_text(encoding="utf-8")):
            if b["feat"]["kind"] == "char" and b["model"] == "logreg":
                return tuple(b["feat"]["char_ngram"]), b["feat"]["min_df"], b["C"]
    return (2, 5), 3, 10


def held_out():
    """Os exemplos da demo ficam FORA do treino, para a previsão ser honesta (o modelo nunca os viu)."""
    return {e["doc_id"] for e in json.loads(EXEMPLOS.read_text(encoding="utf-8")).values()} if EXEMPLOS.exists() else set()


def get_classifier(dev):
    ng, mdf, C = char_params()
    f = MODELS / f"demo_char_logreg_{ng[0]}{ng[1]}_{mdf}_{C}.joblib"
    if f.exists():
        return joblib.load(f)
    print("  (a treinar o classificador no conjunto de desenvolvimento...)", flush=True)
    dev = dev[~dev.doc_id.isin(held_out())]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ng, min_df=mdf, max_features=300_000, sublinear_tf=True)
    X = vec.fit_transform(dev.text_cut)
    clf = LogisticRegression(C=C, max_iter=3000, random_state=SEED).fit(X, dev.label)
    MODELS.mkdir(exist_ok=True)
    joblib.dump((vec, clf), f)
    return vec, clf


def get_topics(dev):
    from topics import doc_tokens, fit_lda, vectorizer
    k = int(pd.read_csv(RES / "topicos_palavras.csv").shape[0])
    f = MODELS / f"demo_lda_{k}.joblib"
    if f.exists():
        return joblib.load(f)
    print("  (a treinar o modelo de tópicos...)", flush=True)
    cv = vectorizer()
    X = cv.fit_transform(doc_tokens(dev.text_norm.fillna("")))
    lda = fit_lda(X, k, SEED)  # mesmos dados, parâmetros e semente que topics.py -> mesmos tópicos
    joblib.dump((cv, lda), f)
    return cv, lda


def explain(vec, clf, text, top=8):
    x = vec.transform([text])
    contrib = x.multiply(clf.coef_[0]).tocsr()
    idx, val = contrib.indices, contrib.data
    names = vec.get_feature_names_out()
    order = np.argsort(val)
    to_true = [(names[idx[i]], val[i]) for i in order[:top] if val[i] < 0]
    to_fake = [(names[idx[i]], val[i]) for i in order[::-1][:top] if val[i] > 0]
    return to_fake, to_true


def show(text, dev):
    from ie_extract import ATTRIB_RE, Extractor

    cut = cut_text(text)
    vec, clf = get_classifier(dev)
    p = clf.predict_proba(vec.transform([cut]))[0, 1]
    bar = "█" * int(round(p * 20)) + "░" * (20 - int(round(p * 20)))
    print("\n" + "=" * 72)
    print(f" 1. PREVISÃO: {'FALSA' if p >= .5 else 'VERDADEIRA'}   prob. de ser falsa = {p:.2f}  [{bar}]")
    print("=" * 72)
    if len(text.split()) > len(cut.split()):
        print(f"   (o classificador lê as primeiras {len(cut.split())} palavras, como no treino)")

    to_fake, to_true = explain(vec, clf, cut)
    print("\n 2. PEDAÇOS DE TEXTO QUE MAIS PESARAM (n-gramas de caracteres)")
    print("    -> falsa:     " + "  ".join(f"«{g}»" for g, _ in to_fake))
    print("    -> verdadeira: " + "  ".join(f"«{g}»" for g, _ in to_true))

    ext = Extractor()
    ents = ext([text])[0]
    nw = max(len(text.split()), 1)
    dens = {t: sum(e[0] == t for e in ents) / nw * 100 for t in TYPES}
    dens["ATRIBUICAO"] = len(ATTRIB_RE.findall(text)) / nw * 100
    ref = pd.read_csv(RES / "ie_falsas_vs_verdadeiras.csv").set_index("medida_por_100_palavras")
    print("\n 3. ANCORAGEM FACTUAL (por 100 palavras)       esta   média falsas   média verdadeiras")
    for t in ["DATA", "LOCAL", "PESSOA", "ORGANIZACAO", "ATRIBUICAO"]:
        print(f"    {t:<14}                              {dens[t]:5.2f}      {ref.loc[t, 'media_falsas']:5.2f}"
              f"            {ref.loc[t, 'media_verdadeiras']:5.2f}")
    if dens["DATA"] == 0:
        print("    ! nenhuma data concreta: sinal para subir na fila de verificação (as verdadeiras têm ~5x mais datas)")

    print("\n 4. ENTIDADES EXTRAÍDAS")
    for t in TYPES:
        vals = list(dict.fromkeys(x for tt, x in ents if tt == t))
        print(f"    {t:<12} " + (", ".join(vals) if vals else "—"))

    from topics import doc_tokens
    cv, lda = get_topics(dev)
    theta = lda.transform(cv.transform(doc_tokens([text])))[0]
    k = int(theta.argmax())
    tp = pd.read_csv(RES / "topicos_palavras.csv").set_index("topico").loc[k]
    print(f"\n 5. TÓPICO DOMINANTE: T{k} (peso {theta[k]:.2f}) — {' '.join(tp.palavras.split()[:8])}")
    print(f"    neste tópico, o peso nas falsas é {tp.razao_F_V:.1f}x o peso nas verdadeiras")
    print("\n Ferramenta de TRIAGEM: indica o que verificar primeiro; a decisão é de um verificador humano.\n")


def read_input(a):
    if a.exemplo:
        ex = json.loads(EXEMPLOS.read_text(encoding="utf-8"))[a.exemplo]
        doc = load().set_index("doc_id").loc[ex["doc_id"]]
        print(f"Exemplo '{a.exemplo}': {ex['doc_id']} (classe real: {'falsa' if doc.label else 'verdadeira'})"
              f" — {ex['porque']}")
        return doc.text
    if a.file:
        return Path(a.file).read_text(encoding="utf-8")
    print("Cola a notícia e termina com uma linha vazia (Enter duas vezes):")
    lines = []
    for line in sys.stdin:
        if not line.strip() and lines:
            break
        lines.append(line.rstrip("\n"))
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--exemplo", choices=["falsa", "verdadeira", "falha"])
    a = ap.parse_args()
    text = read_input(a)
    if len(text.split()) < 20:
        sys.exit("Texto demasiado curto para uma previsão fiável (menos de 20 palavras).")
    dev, _ = load_split()
    show(text, dev)
