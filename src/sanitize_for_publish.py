"""Prepara uma cópia do repositório para publicar (GitHub), SEM textos do corpus.

O enunciado proíbe incluir os dados em bruto; os ficheiros de anotação e de análise contêm cópias
dos textos das notícias. Esta cópia mantém as anotações, categorias e nomes, e apaga as colunas com
texto (o código obtém sempre os textos do corpus através de src/data.py).

    python src/sanitize_for_publish.py DESTINO      # cria DESTINO/fakebr-tm sem textos
"""
import shutil
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"data_raw", "models", "__pycache__", ".git"}
SKIP_FILES = {"corpus.csv"}
SKIP_SUFFIX = {".aux", ".log", ".blg", ".bbl", ".fls", ".fdb_latexmk", ".out", ".zip"}
NOTE = "(texto omitido: ver o corpus Fake.Br, descarregado por src/data.py)"


def blank_columns(path, sheet, headers):
    wb = load_workbook(path)
    ws = wb[sheet]
    head = [c.value for c in ws[1]]
    for h in headers:
        if h in head:
            col = head.index(h) + 1
            for r in range(2, ws.max_row + 1):
                if ws.cell(r, col).value:
                    ws.cell(r, col).value = NOTE
    wb.save(path)


def main(dest):
    out = Path(dest) / "fakebr-tm"
    if out.exists():
        sys.exit(f"{out} já existe — escolham outra pasta de destino.")

    def ignore(d, names):
        return [n for n in names if n in SKIP or n in SKIP_FILES or Path(n).suffix in SKIP_SUFFIX]
    shutil.copytree(ROOT, out, ignore=ignore)
    ann, res = out / "annotation", out / "results"
    for f in ann.glob("anotacao_P*.xlsx"):
        blank_columns(f, "textos", ["texto"])
    for f in ann.glob("analise_erros*.xlsx"):
        blank_columns(f, "erros", ["texto que o modelo viu (cortado)", "início do texto completo"])
    tn = ann / "topicos_nomes.xlsx"
    if tn.exists():
        blank_columns(tn, "topicos", ["documento 1 (início)", "documento 2 (início)", "documento 3 (início)"])
    td = res / "topicos_documentos.csv"
    if td.exists():
        pd.read_csv(td).drop(columns=["inicio_do_texto"], errors="ignore").to_csv(td, index=False)
    print(f"Cópia sem textos do corpus em: {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
