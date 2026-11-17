"""Reproduz todos os resultados com um único comando, em Windows, Mac ou Linux.

    python run_all.py           # desenvolvimento: caracterização, CV, ablation, por tema
    python run_all.py --final   # + avaliação ÚNICA no teste (só no fim do projeto!)
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
STEPS = [["data.py"], ["split.py"], ["characterize.py"], ["tune.py"], ["classify.py", *sys.argv[1:]], ["topics.py"], ["ie_extract.py"], ["near_duplicates.py"], ["cost.py"], ["make_tables.py"]]

for step in STEPS:
    print(f"\n=== {' '.join(step)} ===", flush=True)
    # sys.executable = o Python que está a correr este script (o do .venv, se estiver ativo)
    r = subprocess.run([sys.executable, *step], cwd=SRC)
    if r.returncode != 0:
        sys.exit(f"Falhou em {step[0]} (código {r.returncode}). Os passos seguintes não foram corridos.")
print("\nConcluído. Resultados em results/")
