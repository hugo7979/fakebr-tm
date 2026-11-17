"""Seleção dos 100 documentos para anotação manual (extração de informação).

Escolhemos 50 PARES (falsa + verdadeira sobre o mesmo assunto) do conjunto de
desenvolvimento, estratificados por tema. Assim a comparação "falsas citam menos
entidades/fontes?" é controlada pelo assunto. Usamos o texto normalizado
(~155 palavras), para a anotação ser exequível e os dois lados terem o mesmo tamanho.

Cada um dos 5 anotadores recebe 10 pares (20 docs) + 4 docs de outro anotador
(dupla anotação de 20 docs no total → concordância entre anotadores).

Gera annotation/anotacao_P1.xlsx ... anotacao_P5.xlsx (folha "textos" e folha
"anotacoes" com lista pendente de tipos) e annotation/selecao.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from data import ROOT
from split import SEED, load_split

ANN = ROOT / "annotation"
TYPES = ["PESSOA", "ORGANIZACAO", "LOCAL", "DATA"]
N_PAIRS, N_ANNOT, N_DOUBLE = 50, 5, 4


def select(dev: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    # pares completos, com texto normalizado nos dois lados, e só pares "simples" (grupo = 1 par)
    ok = dev.groupby("pair_id").filter(
        lambda g: len(g) == 2 and g.text_norm.notna().all() and g.group.nunique() == 1)
    single = ok.groupby("group").pair_id.nunique()
    ok = ok[ok.group.isin(single[single == 1].index)]
    pairs = ok.drop_duplicates("pair_id")[["pair_id", "topic4"]]
    share = pairs.topic4.value_counts(normalize=True)
    quota = (share * N_PAIRS).round().astype(int)
    quota[quota.idxmax()] += N_PAIRS - quota.sum()  # acerto para somar 50
    chosen = []
    for topic, q in quota.items():
        ids = pairs[pairs.topic4 == topic].pair_id.to_numpy()
        chosen += rng.choice(ids, size=q, replace=False).tolist()
    sel = ok[ok.pair_id.isin(chosen)].copy()
    # distribuição: pares baralhados → 10 pares por anotador
    order = rng.permutation(chosen)
    owner = {p: f"P{i // (N_PAIRS // N_ANNOT) + 1}" for i, p in enumerate(order)}
    sel["anotador"] = sel.pair_id.map(owner)
    # dupla anotação: cada anotador revê 4 docs do anotador seguinte (P1→P2, ..., P5→P1)
    sel["segundo_anotador"] = ""
    for i in range(N_ANNOT):
        a, nxt = f"P{i + 1}", f"P{(i + 1) % N_ANNOT + 1}"
        docs = sel[sel.anotador == nxt].doc_id.to_numpy()
        pick = rng.choice(docs, size=N_DOUBLE, replace=False)
        sel.loc[sel.doc_id.isin(pick), "segundo_anotador"] = a
    return sel.sort_values(["anotador", "pair_id", "label"]).reset_index(drop=True)


def write_workbook(sel: pd.DataFrame, annot: str) -> None:
    mine = pd.concat([sel[sel.anotador == annot], sel[sel.segundo_anotador == annot]])
    # baralhar a ordem para o anotador não saber qual é a falsa e qual a verdadeira do par
    mine = mine.sample(frac=1, random_state=SEED).reset_index(drop=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "textos"
    ws.append(["doc_id", "texto"])
    for d, t in zip(mine.doc_code, mine.text_norm):
        ws.append([d, t])
    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 140
    for row in ws.iter_rows(min_row=2):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
        row[0].alignment = Alignment(vertical="top")

    wa = wb.create_sheet("anotacoes")
    wa.append(["doc_id", "tipo", "entidade (copiar exatamente do texto)", "notas"])
    for c, w in zip("ABCD", [10, 16, 50, 40]):
        wa.column_dimensions[c].width = w
    dv_t = DataValidation(type="list", formula1='"' + ",".join(TYPES) + '"', allow_blank=False)
    dv_d = DataValidation(type="list", formula1=f"=textos!$A$2:$A${len(mine) + 1}", allow_blank=False)
    wa.add_data_validation(dv_t)
    wa.add_data_validation(dv_d)
    dv_t.add("B2:B2000")
    dv_d.add("A2:A2000")

    wf = wb.create_sheet("feito")
    wf.append(["doc_id", "terminado? (S/N)", "documento sem entidades? (S)"])
    for d in mine.doc_code:
        wf.append([d, "N", ""])
    wf.column_dimensions["B"].width = 18
    wf.column_dimensions["C"].width = 28

    for sh in (ws, wa, wf):
        for cell in sh[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="3B6FB6")
        sh.freeze_panes = "A2"
    wb.save(ANN / f"anotacao_{annot}.xlsx")


if __name__ == "__main__":
    import sys
    ANN.mkdir(exist_ok=True)
    if list(ANN.glob("anotacao_P*.xlsx")) and "--force" not in sys.argv:
        sys.exit("Os ficheiros de anotação já existem. NÃO os regeneres (apagarias as anotações).\n"
                 "Só com --force, e apenas se ainda ninguém começou a anotar.")
    dev, _ = load_split()
    sel = select(dev)
    # códigos opacos (D001...) para o anotador não ver o rótulo no doc_id
    codes = {d: f"D{i + 1:03d}" for i, d in enumerate(sel.sample(frac=1, random_state=SEED).doc_id)}
    sel["doc_code"] = sel.doc_id.map(codes)
    sel[["doc_code", "doc_id", "pair_id", "label", "topic", "anotador", "segundo_anotador"]] \
        .to_csv(ANN / "selecao.csv", index=False)
    for i in range(N_ANNOT):
        write_workbook(sel, f"P{i + 1}")
    print(sel.groupby(["topic4", "label"]).size().unstack())
    print(sel.groupby("anotador").size().to_dict(), "| dupla anotação:",
          (sel.segundo_anotador != "").sum(), "docs")
    print("palavras por anotador (~):",
          sel.assign(n=sel.text_norm.str.split().str.len())
          .pipe(lambda s: pd.concat([s.groupby("anotador").n.sum(),
                                     s[s.segundo_anotador != ""].groupby("segundo_anotador").n.sum()], axis=1)
                .sum(axis=1)).astype(int).to_dict())
