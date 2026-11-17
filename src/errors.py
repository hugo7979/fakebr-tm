"""Análise de erros (enunciado: >= 30 erros agrupados em categorias).

Passo 1 — gerar a folha de análise (depois de abrir o teste com run_all.py --final):
    python errors.py                          # erros do BERTimbau fine-tuned (cortado) no TESTE, se existir;
                                              # senão, do melhor modelo clássico no teste
    python errors.py --model "nome exato"     # outro modelo de results/test_predictions.csv
    python errors.py --source dev             # treino/ensaio: erros out-of-fold no dev (não vai para o artigo)
  -> annotation/analise_erros[_dev].xlsx: 40 erros (metade falsos positivos, metade falsos negativos),
     com o texto que o modelo viu, a classe real e uma coluna "categoria" com lista pendente.

Passo 2 — preencher a coluna "categoria" (e "notas") no Excel. Aqui pode ver-se a classe real:
     é análise, não anotação às cegas.

Passo 3 — resumir:
    python errors.py --summary [--source dev]
  -> results/erros_categorias.csv (contagens por categoria e tipo de erro) e tabela no artigo.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from data import ROOT, load
from split import SEED

RES, ANN = ROOT / "results", ROOT / "annotation"
N_ERRORS = 40  # >= 30 pedidos; margem para descartar casos duvidosos
CATEGORIES = [
    ("Falsa com estilo jornalístico", "falsa escrita como notícia séria: datas, fontes, registo formal"),
    ("Verdadeira opinativa ou em discurso direto", "coluna, comentário, entrevista, muitas citações"),
    ("Pouco texto informativo no corte", "o início cortado tem pouco conteúdo (título, legenda, lista)"),
    ("Sátira ou humor", "texto humorístico ou satírico"),
    ("Marcas da fonte", "nome de jornal/rádio/site, assinatura, 'leia também' que puxam para a classe errada"),
    ("Tema típico da outra classe", "assunto muito associado à outra classe (ex.: política opinativa)"),
    ("Possível erro de rótulo no corpus", "a notícia parece estar mal rotulada"),
    ("Outro", "descrever na coluna notas"),
]
LABEL = {0: "verdadeira", 1: "falsa"}


def predictions(source: str, model: str | None):
    """Devolve (DataFrame doc_id/label/pred/prob, nome do modelo)."""
    if source == "dev":
        f = RES / "oof_bertimbau_ft_text_cut.csv"
        if not f.exists():
            sys.exit("Sem previsões out-of-fold do BERTimbau (results/oof_bertimbau_ft_text_cut.csv).")
        d = pd.read_csv(f)
        return d.assign(pred=(d.prob_falsa > .5).astype(int)), "BERTimbau fine-tuned (cortado) — dev, out-of-fold"
    bert = RES / "test_predictions_bertimbau_ft_text_cut.csv"
    if model is None and bert.exists():
        return pd.read_csv(bert), "BERTimbau fine-tuned (cortado) — teste"
    f = RES / "test_predictions.csv"
    if not f.exists():
        sys.exit("O teste ainda não foi aberto (falta results/test_predictions.csv). "
                 "Corre primeiro: python run_all.py --final   (uma única vez!)")
    d = pd.read_csv(f)
    if model is None:  # melhor modelo não-baseline no teste
        t = pd.read_csv(RES / "test_main_table.csv")
        t = t[~t.config.str.startswith("baseline")]
        model = t.sort_values("f1_macro", ascending=False).config.iloc[0]
    if model not in d.columns:
        sys.exit(f"Modelo '{model}' não existe. Opções:\n  " + "\n  ".join(d.columns[4:]))
    return d.assign(pred=d[model]), f"{model} — teste"


def combined_test_errors():
    """Teste: o BERTimbau afinado erra poucos (<30). Junta TODOS os erros dele aos do melhor clássico
    (caracteres + SVM afinado), até N_ERRORS, e indica que modelo(s) erraram cada notícia."""
    bert = pd.read_csv(RES / "test_predictions_bertimbau_ft_text_cut.csv").set_index("doc_id")
    t = pd.read_csv(RES / "test_predictions.csv").set_index("doc_id")
    col = [c for c in t.columns if "caracteres + SVM (cortado, afinado" in c][0]
    eb = bert[bert.pred != bert.label]
    ec = t[t[col] != t.label]
    rows = [{"doc_id": d, "label": int(r.label), "pred": int(r.pred), "prob_falsa": r.prob_falsa,
             "modelo": "ambos" if d in ec.index else "BERTimbau"} for d, r in eb.iterrows()]
    rest = ec[~ec.index.isin(eb.index)]
    need = max(0, N_ERRORS - len(rows))
    fp, fn = rest[rest.label == 0], rest[rest.label == 1]
    n_fp = min(len(fp), need // 2 + need % 2)
    n_fn = min(len(fn), need - n_fp)
    pick = pd.concat([fp.sample(n_fp, random_state=SEED), fn.sample(n_fn, random_state=SEED)])
    rows += [{"doc_id": d, "label": int(r.label), "pred": int(r[col]), "prob_falsa": float("nan"),
              "modelo": "caracteres+SVM"} for d, r in pick.iterrows()]
    s = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    name = (f"BERTimbau fine-tuned (cortado): {len(eb)} erros, todos incluídos; completado com "
            f"{len(pick)} dos {len(rest)} restantes erros de '{col}' — teste")
    nfp = int((s.label == 0).sum())
    return s, name, nfp, len(s) - nfp


def sample_errors(pred: pd.DataFrame) -> pd.DataFrame:
    err = pred[pred.pred != pred.label]
    fp, fn = err[err.label == 0], err[err.label == 1]  # FP: verdadeira prevista falsa
    k = N_ERRORS // 2
    take_fp, take_fn = min(k, len(fp)), min(k, len(fn))
    # se um dos tipos tem poucos erros, completa com o outro
    take_fn = min(len(fn), N_ERRORS - take_fp) if take_fp < k else take_fn
    take_fp = min(len(fp), N_ERRORS - take_fn) if take_fn < k else take_fp
    s = pd.concat([fp.sample(take_fp, random_state=SEED), fn.sample(take_fn, random_state=SEED)])
    return s.sample(frac=1, random_state=SEED).reset_index(drop=True), len(fp), len(fn)


def write_sheet(s: pd.DataFrame, path, model_name, n_fp, n_fn):
    corpus = load().set_index("doc_id")
    wb = Workbook()
    ws = wb.active
    ws.title = "erros"
    has_model = "modelo" in s.columns
    head = ["id", "doc_id", "modelo que errou" if has_model else "modelo", "classe real", "previsão",
            "tipo de erro", "prob. falsa", "tema", "domínio",
            "palavras (texto completo)", "texto que o modelo viu (cortado)", "início do texto completo",
            "categoria", "notas"]
    ws.append(head)
    for i, r in enumerate(s.itertuples(), 1):
        c = corpus.loc[r.doc_id]
        ws.append([f"E{i:02d}", r.doc_id, r.modelo if has_model else model_name.split(" — ")[0],
                   LABEL[r.label], LABEL[r.pred],
                   "falso positivo" if r.label == 0 else "falso negativo",
                   round(float(r.prob_falsa), 3) if hasattr(r, "prob_falsa") and pd.notna(r.prob_falsa) else "",
                   c.topic, c.domain, int(c.n_words_no_punct) if pd.notna(c.n_words_no_punct) else "",
                   c.text_cut, " ".join(str(c.text).split()[:80]), "", ""])
    widths = [5, 11, 14, 11, 11, 15, 9, 16, 18, 10, 70, 70, 34, 30]
    for col, w in zip("ABCDEFGHIJKLMN", widths):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    wc = wb.create_sheet("categorias")
    wc.append(["categoria", "quando usar"])
    for name, desc in CATEGORIES:
        wc.append([name, desc])
    wc.column_dimensions["A"].width = 42
    wc.column_dimensions["B"].width = 90
    dv = DataValidation(type="list", formula1=f"=categorias!$A$2:$A${len(CATEGORIES) + 1}", allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"M2:M{len(s) + 1}")
    wi = wb.create_sheet("info")
    for line in [f"Modelo: {model_name}", f"Erros totais: {n_fp} falsos positivos (verdadeira -> falsa), "
                 f"{n_fn} falsos negativos (falsa -> verdadeira)", f"Amostra: {len(s)} erros (semente {SEED})",
                 "Podem acrescentar categorias na folha 'categorias' (a lista pendente atualiza-se)."]:
        wi.append([line])
    wi.column_dimensions["A"].width = 110
    for sh in (ws, wc):
        for cell in sh[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="3B6FB6")
        sh.freeze_panes = "A2"
    wb.save(path)


def summary(path, source):
    x = pd.read_excel(path, sheet_name="erros", dtype=str)
    x = x[x["categoria"].notna() & (x["categoria"].str.strip() != "")]
    x["categoria"] = x["categoria"].str.strip()
    if x.empty:
        sys.exit(f"Nenhum erro com categoria preenchida em {path.name}.")
    if "modelo que errou" in x.columns:
        print(pd.crosstab(x["categoria"], x["modelo que errou"]).to_string(), "\n")
    tab = pd.crosstab(x["categoria"], x["tipo de erro"], margins=True, margins_name="total")
    tab = tab.sort_values("total", ascending=False)
    exemplos = x.groupby("categoria")["id"].apply(lambda s: ", ".join(s.head(3)))
    tab["exemplos"] = exemplos.reindex(tab.index).fillna("")
    out = RES / ("erros_categorias_dev.csv" if source == "dev" else "erros_categorias.csv")
    tab.to_csv(out)
    print(f"{len(x)} erros categorizados -> {out.name}")
    print(tab.to_string())
    if len(x) < 30:
        print(f"ATENÇÃO: o enunciado pede pelo menos 30 erros analisados (tens {len(x)}).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["test", "dev"], default="test")
    ap.add_argument("--model", default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--force", action="store_true", help="regenerar a folha (apaga categorias já escritas)")
    a = ap.parse_args()
    path = ANN / ("analise_erros_dev.xlsx" if a.source == "dev" else "analise_erros.xlsx")
    if a.summary:
        summary(path, a.source)
        raise SystemExit
    if path.exists() and not a.force:
        sys.exit(f"{path.name} já existe — não o regenero para não apagar o trabalho feito. "
                 "Usa --summary para resumir, ou --force para regenerar.")
    if a.source == "test" and a.model is None and (RES / "test_predictions_bertimbau_ft_text_cut.csv").exists():
        s, name, n_fp, n_fn = combined_test_errors()
    else:
        pred, name = predictions(a.source, a.model)
        s, n_fp, n_fn = sample_errors(pred)
    write_sheet(s, path, name, n_fp, n_fn)
    print(f"{name}\n  -> {len(s)} erros na folha ({n_fp} FP, {n_fn} FN): {path}")
