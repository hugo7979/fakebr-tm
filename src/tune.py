"""Afinação de hiperparâmetros dos modelos clássicos — SÓ com validação cruzada no dev.

Afinamos na versão "cortada" (text_cut), a única sem os atalhos de comprimento e de final de texto,
os três modelos mais fortes: TF-IDF palavras + SVM, TF-IDF caracteres + SVM, TF-IDF caracteres + LogReg.
As melhores configurações ficam em results/best_params.json e entram no TESTE FINAL
(classify.py --final). Os valores de CV da escolhida estão ligeiramente otimistas (foram usados para
escolher); o teste dá a estimativa sem viés.

    python tune.py            # ~12 min em 2 núcleos (resultado em cache)
    python tune.py --redo     # refazer
"""
from __future__ import annotations

import argparse
import itertools
import json

import pandas as pd

from classify import RES, run_cv
from split import load_split

TEXT = "text_cut"
GRIDS = {
    "TF-IDF palavras + SVM": [
        {"feat": {"kind": "word", "text_col": TEXT, "pipeline": p, "ngram": (1, 2), "min_df": m},
         "model": "svm", "C": c}
        for p, m, c in itertools.product(["raw", "lower"], [1, 2, 5], [0.1, 0.3, 1, 3])],
    "TF-IDF caracteres + SVM": [
        {"feat": {"kind": "char", "text_col": TEXT, "char_ngram": ng, "min_df": 3}, "model": "svm", "C": c}
        for ng, c in itertools.product([(2, 5), (1, 4), (3, 6)], [0.1, 0.3, 1, 3])],
    "TF-IDF caracteres + LogReg": [
        {"feat": {"kind": "char", "text_col": TEXT, "char_ngram": (2, 5), "min_df": 3}, "model": "logreg", "C": c}
        for c in [1, 10, 100]],
}


def describe(cfg):
    f = cfg["feat"]
    if f["kind"] == "word":
        return f"pipeline={f['pipeline']}, min_df={f['min_df']}, C={cfg['C']}"
    return f"n-gramas={tuple(f['char_ngram'])}, min_df={f['min_df']}, C={cfg['C']}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    out_file, best_file = RES / "tuning.csv", RES / "best_params.json"
    if out_file.exists() and best_file.exists() and not a.redo:
        print("Afinação em cache (results/tuning.csv). Use --redo para refazer.")
        raise SystemExit
    dev, _ = load_split()
    configs = []
    for fam, grid in GRIDS.items():
        for g in grid:
            configs.append({**g, "nome": f"{fam} | {describe(g)}", "familia": fam})
    res = run_cv(dev, configs, tag="tuning")
    res["familia"] = [c["familia"] for c in configs]
    res["parametros"] = [describe(c) for c in configs]
    res.to_csv(out_file, index=False)

    best = []
    for fam in GRIDS:
        sub = res[res.familia == fam]
        top = sub.f1_macro_mean.max()
        # empate dentro de 0,001 → preferir a configuração mais regularizada (C menor) = mais simples
        cand = [configs[i] for i in sub.index if sub.loc[i, "f1_macro_mean"] >= top - 0.001]
        chosen = min(cand, key=lambda c: c["C"])
        best.append({"nome": f"{fam} (cortado, afinado: {describe(chosen)})",
                     "feat": {k: list(v) if isinstance(v, tuple) else v for k, v in chosen["feat"].items()},
                     "model": chosen["model"], "C": chosen["C"]})
        print(f"{fam:<28} melhor F1={top:.4f} -> escolhido: {describe(chosen)}")
    best_file.write_text(json.dumps(best, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 200)
