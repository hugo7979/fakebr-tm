"""Custo computacional (enunciado): tempo de treino e de inferência, memória e tamanho dos modelos.

Mede, na MESMA máquina e com o mesmo protocolo, os modelos finais que correm em CPU:
treino no dev completo (5 756 notícias) e inferência em 1 000 notícias do dev
(o teste não é usado aqui). Memória: pico de RAM do processo durante cada passo.
O BERTimbau (GPU, Colab) é lido de results/custo_bertimbau_*.json.

    python src/cost.py        # ~3 min; -> results/custo.csv e results/custo_hardware.json
Correr no computador onde querem reportar os tempos (o artigo indica o hardware).
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import platform
import time
import tracemalloc

import numpy as np
import pandas as pd

from classify import RES, Featurizer, load_tuned, make_model
from split import load_split

N_INFER = 1000


def hardware():
    info = {"sistema": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
            "processador": platform.processor() or platform.machine(), "nucleos": os.cpu_count()}
    try:
        import psutil  # opcional
        info["ram_GB"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        try:  # Linux
            with open("/proc/meminfo") as f:
                info["ram_GB"] = round(int(f.readline().split()[1]) / 1e6, 1)
        except OSError:
            info["ram_GB"] = None
    return info


def measure(fn):
    """Executa fn() e devolve (resultado, segundos, pico de memória Python em MB)."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    secs = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    return out, secs, peak


def size_mb(obj):
    return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)) / 1e6


def classic_rows(dev):
    configs = [
        {"nome": "Só comprimento (baseline)", "feat": {"kind": "length"}, "model": "logreg"},
        *load_tuned(),
    ]
    sample = dev.sample(N_INFER, random_state=42)
    rows = []
    for c in configs:
        def train():
            fz = Featurizer(**c["feat"])
            m = make_model(c["model"], c.get("C")).fit(fz.fit_transform(dev), dev.label)
            return fz, m
        (fz, m), t_train, mem_train = measure(train)
        _, t_inf, mem_inf = measure(lambda: m.predict(fz.transform(sample)))
        nfeat = getattr(getattr(fz, "vec", None), "vocabulary_", None)
        rows.append({"modelo": c["nome"], "hardware": "CPU", "treino_s": t_train,
                     "inferencia_ms_por_doc": t_inf / N_INFER * 1000,
                     "memoria_pico_MB": max(mem_train, mem_inf),
                     "tamanho_modelo_MB": size_mb((fz, m)),
                     "n_atributos": len(nfeat) if nfeat is not None else 1})
        print(f"  {c['nome'][:60]:<60} treino {t_train:6.1f}s | {rows[-1]['inferencia_ms_por_doc']:.2f} ms/doc"
              f" | RAM {rows[-1]['memoria_pico_MB']:.0f} MB | {rows[-1]['tamanho_modelo_MB']:.1f} MB", flush=True)
    return rows


def spacy_rows(dev):
    from ie_extract import Extractor
    texts = dev.text_norm.fillna("").sample(N_INFER, random_state=42).tolist()
    rows = []
    for name in ["pt_core_news_sm", "pt_core_news_md", "pt_core_news_lg"]:
        try:
            ext, t_load, _ = measure(lambda: Extractor(name))
        except OSError:
            continue
        _, t_inf, mem = measure(lambda: ext(texts))
        import spacy
        path = spacy.util.get_package_path(name)
        disk = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
        rows.append({"modelo": f"Extração de entidades ({name})", "hardware": "CPU", "treino_s": np.nan,
                     "inferencia_ms_por_doc": t_inf / N_INFER * 1000, "memoria_pico_MB": mem,
                     "tamanho_modelo_MB": disk, "n_atributos": np.nan})
        print(f"  {name:<60} {rows[-1]['inferencia_ms_por_doc']:.2f} ms/doc | {disk:.0f} MB", flush=True)
    return rows


def bert_rows():
    rows = []
    for f, label in [("custo_bertimbau_text_cut.json", "BERTimbau fine-tuned (cortado), CV 5 folds"),
                     ("custo_bertimbau_text_cut_final.json", "BERTimbau fine-tuned (cortado), treino final + teste")]:
        p = RES / f
        if not p.exists():
            continue
        c = json.loads(p.read_text())
        t = c.get("finetune_cv_total_s") or c.get("final_treino_dev+teste_s")
        mem = c.get("gpu_mem_pico_GB") or c.get("final_gpu_mem_pico_GB")
        rows.append({"modelo": label, "hardware": f"GPU {c.get('gpu')}", "treino_s": t,
                     "inferencia_ms_por_doc": np.nan, "memoria_pico_MB": mem * 1000 if mem else np.nan,
                     # tamanho: pesos do bert-base-portuguese-cased (~436 MB); n_atributos: parâmetros do BERT-base
                     "tamanho_modelo_MB": 436.0, "n_atributos": 110_000_000})
    return rows


if __name__ == "__main__":
    dev, _ = load_split()
    hw = hardware()
    print("Hardware:", hw)
    print("== Modelos clássicos (treino no dev completo, inferência em 1 000 notícias) ==")
    rows = classic_rows(dev)
    print("== Extração de entidades ==")
    rows += spacy_rows(dev)
    rows += bert_rows()
    out = pd.DataFrame(rows)
    out.to_csv(RES / "custo.csv", index=False)
    (RES / "custo_hardware.json").write_text(json.dumps(hw, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 200)
    print(out.round(2).to_string(index=False))
    print("\nNenhum modelo de linguagem nem serviço externo pago foi usado no pipeline (0 tokens, 0 €).")
