"""Extração de informação: entidades (PESSOA, ORGANIZACAO, LOCAL) com spaCy + DATA por regras.

Três usos:
  python ie_extract.py                 # avalia no ouro anotado (annotation/*.xlsx) + análise falsas vs verdadeiras
  python ie_extract.py --model pt_core_news_sm   # outro modelo (comparação de custo/qualidade)
  python ie_extract.py --models-compare          # sm, md (e lg se instalado) → results/ie_modelos.csv

Avaliação ao nível do documento: o conjunto de entidades distintas (tipo, texto normalizado)
previsto para cada documento é comparado com o conjunto anotado. Reportamos
  - estrito: o texto tem de coincidir exatamente (após normalização);
  - relaxado: basta um conter o outro ("Lula" vs "Luiz Inácio Lula da Silva"), mesmo tipo.
"""
from __future__ import annotations

import argparse
import re
import time
import unicodedata

import numpy as np
import pandas as pd

from data import ROOT, load
from split import SEED, load_split

ANN = ROOT / "annotation"
RES = ROOT / "results"
TYPES = ["PESSOA", "ORGANIZACAO", "LOCAL", "DATA"]
SPACY_MAP = {"PER": "PESSOA", "ORG": "ORGANIZACAO", "LOC": "LOCAL"}  # MISC é ignorado

# ------------------------------------------------------------------ DATA por regras
_MES = r"(?:janeiro|fevereiro|mar[çc]o|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)"
_DIA_SEM = r"(?:segunda|ter[çc]a|quarta|quinta|sexta)(?:-feira)?|s[áa]bado|domingo"
DATE_PATTERNS = [
    rf"\b(?:{_DIA_SEM})\s*\(\s*\d{{1,2}}(?:/\d{{1,2}})?\s*\)",               # quarta-feira (12)
    rf"\b\d{{1,2}}º?\s+de\s+{_MES}(?:\s+de\s+\d{{4}})?",                      # 12 de março [de 2017]
    rf"\b{_MES}\s+de\s+\d{{4}}",                                              # março de 2017
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",                                           # 12/03/2017
    rf"\b(?:{_DIA_SEM})\b",                                                   # quarta-feira
    rf"\b{_MES}\b",                                                           # março
    r"\b(?:19[5-9]\d|20[0-2]\d)\b",                                           # 2017
]
DATE_RE = re.compile("|".join(f"(?:{p})" for p in DATE_PATTERNS), re.IGNORECASE)
# atribuição de fontes (para a análise, não é anotada): "segundo X", "de acordo com", verbos de dizer
ATTRIB_RE = re.compile(r"\b(?:segundo (?:o|a|os|as)\b|de acordo com|conforme|afirm(?:ou|aram)|"
                       r"declar(?:ou|aram)|informou|disse(?:ram)?|explic(?:ou|aram)|em nota)", re.IGNORECASE)


# Regra do guia (12/11): em "PSDB-MG" anota-se só o partido; a sigla do estado NÃO se anota.
# O extrator segue a mesma regra, para não contar como erro algo que os anotadores omitem de propósito.
UF = {"AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR",
      "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"}
PARTY_STATE_RE = re.compile(r"\b[A-Za-zÀ-ÿ]+\s*-\s*(" + "|".join(sorted(UF)) + r")\b")


def party_state(text: str, start: int, end: int) -> bool:
    """True se a entidade em text[start:end] é a sigla de estado de um padrão "PARTIDO-UF"."""
    return any(m.start(1) <= start and end <= m.end(1) for m in PARTY_STATE_RE.finditer(text))


def strip_state(ent: str) -> str:
    """"PSDB-MG" -> "PSDB" (o spaCy junta partido e estado numa só entidade)."""
    return re.sub(r"\s*-\s*(?:" + "|".join(sorted(UF)) + r")$", "", ent)


def dates(text: str) -> list[str]:
    text = text or ""
    # "Janeiro" em "Rio de Janeiro" não é um mês
    return [m.group(0) for m in DATE_RE.finditer(text)
            if not re.search(r"Rio\s+de\s+$", text[max(0, m.start() - 8):m.start()])]


# ------------------------------------------------------------------ normalização e conjuntos
def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s)).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^(?:o|a|os|as|do|da|dos|das|de|no|na|em)\s+", "", s)  # artigo/preposição colado
    return s.strip(" .,;:!?\"'«»“”()[]-–")


def to_set(ents) -> set[tuple[str, str]]:
    return {(t, norm(x)) for t, x in ents if norm(x)}


class Extractor:
    def __init__(self, model="pt_core_news_md"):
        import spacy
        self.name = model
        self.nlp = spacy.load(model, disable=["lemmatizer", "morphologizer", "parser", "attribute_ruler"])

    def __call__(self, texts):
        out = []
        for doc, t in zip(self.nlp.pipe(texts, batch_size=64), texts):
            ents = [(SPACY_MAP[e.label_], strip_state(e.text)) for e in doc.ents
                    if e.label_ in SPACY_MAP and not party_state(t, e.start_char, e.end_char)]
            ents += [("DATA", d) for d in dates(t)]
            out.append(ents)
        return out


# ------------------------------------------------------------------ ouro anotado
def load_gold():
    """Lê annotation/anotacao_P*.xlsx. Devolve (ouro_principal, ouro_segundo) como dicts doc_id -> set."""
    sel = pd.read_csv(ANN / "selecao.csv")
    code2doc = dict(zip(sel.doc_code, sel.doc_id))
    prim, sec, done = {}, {}, set()
    for f in sorted(ANN.glob("anotacao_P*.xlsx")):
        annot = f.stem.split("_")[1]
        x = pd.read_excel(f, sheet_name=None, dtype=str)
        a = x["anotacoes"].iloc[:, :3].dropna(subset=[x["anotacoes"].columns[0]])
        a.columns = ["doc_code", "tipo", "entidade"]
        a = a.dropna(subset=["tipo", "entidade"])
        a["tipo"] = a.tipo.str.strip().str.upper()
        bad = a[~a.tipo.isin(TYPES) | ~a.doc_code.isin(code2doc)]
        if len(bad):
            print(f"  aviso {f.name}: {len(bad)} linhas com tipo/doc inválido ignoradas")
        a = a.drop(bad.index)
        fin = x["feito"]
        fin = set(fin[fin.iloc[:, 1].astype(str).str.upper().str.startswith("S")].iloc[:, 0])
        for code in fin:
            d = code2doc[code]
            ents = to_set(a[a.doc_code == code][["tipo", "entidade"]].itertuples(index=False))
            row = sel[sel.doc_code == code].iloc[0]
            (prim if row.anotador == annot else sec)[d] = ents
            done.add((code, annot))
    return prim, sec


# ------------------------------------------------------------------ métricas
def _match(p, g, relaxed):
    if p[0] != g[0]:
        return False
    return p[1] == g[1] or (relaxed and (p[1] in g[1] or g[1] in p[1]))


def counts(pred: set, gold: set, relaxed=False, tipo=None):
    pred = {p for p in pred if tipo is None or p[0] == tipo}
    gold = {g for g in gold if tipo is None or g[0] == tipo}
    tp_p = sum(any(_match(p, g, relaxed) for g in gold) for p in pred)   # previsões corretas
    tp_g = sum(any(_match(p, g, relaxed) for p in pred) for g in gold)   # ouro encontrado
    return tp_p, len(pred), tp_g, len(gold)


def prf(docs, pred, gold, relaxed=False, tipo=None, n_boot=1000):
    c = np.array([counts(pred[d], gold[d], relaxed, tipo) for d in docs], dtype=float)

    def score(cc):
        s = cc.sum(0)
        p = s[0] / s[1] if s[1] else np.nan
        r = s[2] / s[3] if s[3] else np.nan
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    P, R, F = score(c)
    rng = np.random.default_rng(SEED)
    boot = np.array([score(c[rng.integers(0, len(c), len(c))]) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boot[:, 2], [2.5, 97.5])
    return {"precisao": P, "revocacao": R, "f1": F, "f1_ci95_lo": lo, "f1_ci95_hi": hi,
            "n_previstas": int(c[:, 1].sum()), "n_ouro": int(c[:, 3].sum()), "n_docs": len(docs)}


def evaluate(pred, gold, tag=""):
    docs = sorted(gold)
    rows = []
    for relaxed in (False, True):
        for tipo in TYPES + [None]:
            rows.append({"modelo": tag, "correspondencia": "relaxada" if relaxed else "estrita",
                         "tipo": tipo or "TODOS (micro)", **prf(docs, pred, gold, relaxed, tipo)})
    return pd.DataFrame(rows)


def gold_density(prim, corpus):
    """Falsas vs verdadeiras com o OURO humano (50 pares): confirma ou não o que o extrator sugere."""
    from scipy.stats import wilcoxon
    rows = []
    d = corpus.loc[sorted(prim)]
    nw = d.text_norm.fillna("").str.split().str.len().clip(lower=1)
    for tp in TYPES:
        dens = pd.Series({doc: sum(t == tp for t, _ in prim[doc]) for doc in d.index}) / nw * 100
        piv = pd.DataFrame({"pair": d.pair_id, "label": d.label, "v": dens}).pivot_table(
            index="pair", columns="label", values="v").dropna()
        st = wilcoxon(piv[1], piv[0]) if (piv[1] != piv[0]).any() else None
        rows.append({"tipo": tp, "ouro_media_falsas": piv[1].mean(), "ouro_media_verdadeiras": piv[0].mean(),
                     "razao_V_F": piv[0].mean() / piv[1].mean() if piv[1].mean() else np.nan,
                     "pares": len(piv), "wilcoxon_p": st.pvalue if st else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "ie_ouro_falsas_vs_verdadeiras.csv", index=False)
    return out


def type_confusions(docs, pred, gold):
    """Para onde vão as previsões erradas: tipo previsto -> tipo no ouro (mesmo texto) ou 'não anotado'."""
    import collections
    conf = collections.Counter()
    for d in docs:
        by_txt = collections.defaultdict(set)
        for t, x in gold[d]:
            by_txt[x].add(t)
        for t, x in pred[d]:
            if (t, x) not in gold[d]:
                g = by_txt.get(x)
                conf[(t, "/".join(sorted(g)) if g else "não anotado")] += 1
    out = pd.DataFrame([{"previsto": t, "no_ouro": g, "n": n} for (t, g), n in conf.most_common()])
    out.to_csv(RES / "ie_confusoes.csv", index=False)
    return out


def agreement(prim, sec):
    """Concordância entre anotadores = F1 entre os dois (simétrico)."""
    docs = sorted(set(prim) & set(sec))
    if not docs:
        return None
    return evaluate(sec, {d: prim[d] for d in docs}, tag="concordancia_anotadores")


# ------------------------------------------------------------------ análise falsas vs verdadeiras
def analysis(dev, ext):
    """Densidade de entidades/datas/atribuições por 100 palavras, comparação emparelhada por par."""
    from scipy.stats import wilcoxon
    t0 = time.time()
    texts = dev.text_norm.fillna("").tolist()
    ents = ext(texts)
    secs = time.time() - t0
    nw = np.array([max(len(t.split()), 1) for t in texts])
    feats = pd.DataFrame({tp: [sum(e[0] == tp for e in es) for es in ents] for tp in TYPES})
    feats["ATRIBUICAO"] = [len(ATTRIB_RE.findall(t)) for t in texts]
    dens = feats.div(nw, axis=0) * 100
    dens[["pair_id", "label"]] = dev[["pair_id", "label"]].values
    rows = []
    for col in TYPES + ["ATRIBUICAO"]:
        piv = dens.pivot_table(index="pair_id", columns="label", values=col).dropna()
        stat = wilcoxon(piv[1], piv[0])
        rows.append({"medida_por_100_palavras": col, "media_falsas": piv[1].mean(),
                     "media_verdadeiras": piv[0].mean(), "mediana_dif_(F-V)": (piv[1] - piv[0]).median(),
                     "pares": len(piv), "wilcoxon_p": stat.pvalue})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "ie_falsas_vs_verdadeiras.csv", index=False)
    return out, secs


def model_costs(texts, models):
    rows = []
    for m in models:
        try:
            t0 = time.time()
            ext = Extractor(m)
            load_s = time.time() - t0
        except OSError:
            print(f"  {m} não instalado — ignorado")
            continue
        t0 = time.time()
        ext(texts)
        rows.append({"modelo": m, "carregar_s": load_s, "processar_s": time.time() - t0,
                     "docs_por_s": len(texts) / (time.time() - t0)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pt_core_news_md")
    ap.add_argument("--models-compare", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    RES.mkdir(exist_ok=True)
    corpus = load().set_index("doc_id")

    prim, sec = load_gold()
    models = ["pt_core_news_sm", "pt_core_news_md", "pt_core_news_lg"] if a.models_compare else [a.model]
    if prim:
        print(f"== Avaliação no ouro: {len(prim)} docs anotados (+{len(sec)} com 2.º anotador) ==")
        if len(prim) < 100:
            print(f"  ATENÇÃO: só {len(prim)}/100 documentos marcados como terminados na folha 'feito'.")
        docs = sorted(prim)
        texts = corpus.loc[docs, "text_norm"].tolist()
        tabs = []
        for m in models:
            try:
                ext = Extractor(m)
            except OSError:
                print(f"  {m} não instalado — ignorado")
                continue
            pred = {d: to_set(e) for d, e in zip(docs, ext(texts))}
            tabs.append(evaluate(pred, prim, tag=m))
            pd.DataFrame([(d, t, x, "previsto") for d in docs for t, x in sorted(pred[d])] +
                         [(d, t, x, "ouro") for d in docs for t, x in sorted(prim[d])],
                         columns=["doc_id", "tipo", "entidade", "origem"]) \
                .to_csv(RES / f"ie_entidades_{m}.csv", index=False)
        ev = pd.concat(tabs)
        ev.to_csv(RES / "ie_avaliacao.csv", index=False)
        print(ev[ev.correspondencia == "estrita"].round(3).to_string(index=False))
        md = {d: to_set(e) for d, e in zip(docs, Extractor("pt_core_news_md")(texts))}
        print("== Erros de tipo do extrator (md) ==")
        print(type_confusions(docs, md, prim).head(8).to_string(index=False))
        print("== Falsas vs verdadeiras no OURO (por 100 palavras, pares emparelhados) ==")
        print(gold_density(prim, corpus).round(4).to_string(index=False))
        ag = agreement(prim, sec)
        if ag is not None:
            ag.to_csv(RES / "ie_concordancia.csv", index=False)
            print("== Concordância entre anotadores (F1) ==")
            print(ag[["correspondencia", "tipo", "f1", "n_docs"]].round(3).to_string(index=False))
    else:
        print("== Ainda não há documentos anotados (folha 'feito' com S). Salto a avaliação. ==")

    print(f"== Falsas vs verdadeiras no dev ({a.model}, texto normalizado) ==")
    dev, _ = load_split()
    res, secs = analysis(dev, Extractor(a.model))
    print(res.round(4).to_string(index=False), f"\n  ({len(dev)} docs em {secs:.0f}s)")
    if a.models_compare:
        costs = model_costs(dev.text_norm.fillna("").head(1000).tolist(), models)
        costs.to_csv(RES / "ie_modelos_custo.csv", index=False)
        print(costs.round(2).to_string(index=False))
