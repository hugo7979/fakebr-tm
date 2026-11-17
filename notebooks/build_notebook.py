"""Gera notebooks/bertimbau.ipynb (as células ficam aqui, em texto simples, fáceis de rever)."""
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# BERTimbau no Fake.Br — 2.ª representação (Colab, GPU)

**Porque este notebook existe.** O enunciado pede TF-IDF *e* pelo menos mais uma representação.
O BERTimbau (Souza et al., 2020) é um BERT pré-treinado em português do Brasil: ao contrário do TF-IDF,
cada palavra é representada consoante o contexto. Precisa de GPU, por isso corre aqui e não no `run_all.sh`.

**O que faz (mesma divisão e mesmos folds que `src/classify.py`):**
- **Parte A — BERTimbau congelado:** o modelo não é treinado; usamos a média dos vetores da última camada
  como "resumo" de cada notícia (768 números) e treinamos uma regressão logística por cima. Barato (~3 min).
- **Parte B — fine-tuning:** o modelo inteiro é ajustado à tarefa, com validação cruzada de 5 folds no dev
  (~30–40 min numa T4).
- **Parte C — teste final:** desligada por defeito (`RUN_FINAL = False`). Só ligar uma vez, no fim, como no resto do projeto.

**Versão do texto (`FT_COL`).** Por defeito usa `text_cut`: primeiras 100 palavras do texto normalizado,
com TODOS os textos cortados a meio. Na versão normalizada do corpus, 98% das verdadeiras foram cortadas a
meio da frase e 77% das falsas acabam em pontuação — só "como acaba o texto" dá F1≈0,89, e o BERT (que
sabe onde o texto termina) aproveita esse atalho. A versão cortada remove-o.

**Antes de começar:** *Ambiente de execução → Alterar tipo de ambiente → GPU T4*.
No fim, a última célula descarrega um zip com os resultados para pôr no repositório."""),
code("""# ---- Configuração (é aqui que se mexe) ----
MODEL = "neuralmind/bert-base-portuguese-cased"
SEED = 42
MAX_LEN_EMB = 512   # parte A
MAX_LEN_FT = 384    # parte B (sobra para 100 palavras)
FT_COL = "text_cut" # texto do fine-tuning: "text_cut" (recomendado), "text_norm" ou "text"
# embeddings congelados a calcular: (nome do ficheiro, coluna). norm/full já foram feitos na 1.ª execução
EMB_VERSIONS = [("bertimbau_cut", "text_cut")]
NOMES = {"text_cut": "cortado", "text_norm": "normalizado", "text": "completo, 512 tokens"}
EPOCHS, BATCH, LR = 3, 16, 2e-5
RUN_FINETUNE = True
RUN_FINAL = False   # True só no fim do projeto, com tudo fixado!
SMOKE = False       # True = teste rápido com poucos documentos (para verificar que tudo corre)"""),
code("""# ---- Instalação (versões fixas, iguais às do requirements.txt) ----
!pip -q install scikit-learn==1.7.2 nltk==3.9.1 transformers==4.46.3
import torch; print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NENHUMA — mudar o ambiente para GPU!")"""),
code("""# ---- Obter o projeto: faz upload do fakebr-tm.zip (ou clona o repositório do grupo) ----
import os, sys, zipfile
if not os.path.exists("fakebr-tm"):
    try:
        from google.colab import files
        up = files.upload()  # escolher fakebr-tm.zip
        zipfile.ZipFile(next(iter(up))).extractall(".")
    except ImportError:
        raise SystemExit("Fora do Colab: coloca a pasta fakebr-tm ao lado do notebook")
    # alternativa: !git clone https://github.com/<grupo>/fakebr-tm.git
os.chdir("fakebr-tm") if os.path.basename(os.getcwd()) != "fakebr-tm" else None
sys.path.insert(0, "src")
print(os.getcwd())"""),
code("""# ---- Dados: reutiliza exatamente o código do repositório ----
import random, time, json
import numpy as np, pandas as pd
from data import load
from split import load_split
from classify import get_folds, metrics, run_cv, RES

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

dev, test = load_split()
if SMOKE:  # amostra pequena mas com grupos inteiros
    keep_d = set(dev.group.drop_duplicates().sample(60, random_state=SEED))
    keep_t = set(test.group.drop_duplicates().sample(15, random_state=SEED))
    dev = dev[dev.group.isin(keep_d)].reset_index(drop=True)
    test = test[test.group.isin(keep_t)].reset_index(drop=True)
allx = pd.concat([dev, test], ignore_index=True)
folds = get_folds(dev)
print("dev", len(dev), "test", len(test), "| tamanhos dos folds de validação:", [len(v) for _, v in folds])
device = "cuda" if torch.cuda.is_available() else "cpu\""""),
md("""## Parte A — BERTimbau congelado (embeddings + regressão logística)

Calculamos um vetor por notícia para **todas** as notícias (dev e teste). Isto não é batota: o modelo não
aprende nada com os nossos dados nesta fase. Os rótulos do teste só são usados no `--final`.
As versões a calcular estão em `EMB_VERSIONS` (primeira célula)."""),
code("""from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

tok = AutoTokenizer.from_pretrained(MODEL)

@torch.no_grad()
def embed(texts, max_len=MAX_LEN_EMB, bs=32):
    enc_model = AutoModel.from_pretrained(MODEL).to(device).eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i+bs], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            h = enc_model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())  # mean pooling sem padding
    del enc_model; torch.cuda.empty_cache()
    return np.vstack(out)

custo = {}
gerados = []
for name, col in EMB_VERSIONS:
    set_seed(); t0 = time.time()
    X = embed(allx[col].fillna("").tolist())
    custo[f"embeddings_{name}_s"] = time.time() - t0
    np.savez_compressed(f"data/emb_{name}.npz", doc_id=np.array(allx.doc_id.tolist(), dtype="U32"), X=X)
    gerados.append(f"data/emb_{name}.npz")
    print(name, X.shape, f"{custo[f'embeddings_{name}_s']:.0f}s")"""),
code("""# Mesma função de validação cruzada que o resto do projeto -> resultados comparáveis
cfg = [{"nome": f"BERTimbau congelado + LogReg ({NOMES[col]})", "feat": {"kind": "emb", "text_col": name}, "model": "logreg_emb"}
       for name, col in EMB_VERSIONS]
tag = "cv_bertimbau_frozen_" + "_".join(n for n, _ in EMB_VERSIONS)
run_cv(dev, cfg, tag=tag); gerados.append(f"results/{tag}.csv")"""),
md("""## Parte B — fine-tuning com validação cruzada (5 folds, texto `FT_COL`)

Usamos a versão cortada de propósito: queremos saber se o modelo aprende algo **além do comprimento e de
como o texto acaba**.
Em cada fold o modelo começa de novo a partir do BERTimbau original; guardamos as previsões
*out-of-fold* para a análise de erros e o tempo/memória para a secção de custo computacional."""),
code("""def finetune_predict(train_df, pred_df, col=FT_COL):
    set_seed()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)
    enc = lambda df: tok(df[col].fillna("").tolist(), truncation=True, max_length=MAX_LEN_FT)
    tr_enc, y = enc(train_df), torch.tensor(train_df.label.values)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(train_df) + BATCH - 1) // BATCH)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    g = torch.Generator().manual_seed(SEED)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(train_df), generator=g)
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i+BATCH].tolist()
            b = tok.pad({k: [tr_enc[k][j] for j in idx] for k in tr_enc}, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss = model(**b, labels=y[idx].to(device)).loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item() * len(idx)
        print(f"   época {ep+1}: loss={tot/len(train_df):.4f}")
    model.eval(); pe, probs = enc(pred_df), []
    with torch.no_grad():
        for i in range(0, len(pred_df), 64):
            b = tok.pad({k: pe[k][i:i+64] for k in pe}, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                probs.append(torch.softmax(model(**b).logits.float(), -1)[:, 1].cpu().numpy())
    del model; torch.cuda.empty_cache()
    return np.concatenate(probs)

if RUN_FINETUNE:
    oof = np.zeros(len(dev)); fold_f1, t0 = [], time.time()
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    for k, (tr, va) in enumerate(folds):
        print(f"fold {k+1}/5")
        oof[va] = finetune_predict(dev.iloc[tr], dev.iloc[va])
        fold_f1.append(metrics(dev.label.values[va], (oof[va] > .5).astype(int))["f1_macro"])
        print(f"   F1-macro = {fold_f1[-1]:.4f}")
    custo["finetune_cv_total_s"] = time.time() - t0
    custo["gpu_mem_pico_GB"] = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
    row = {"config": f"BERTimbau fine-tuned ({NOMES[FT_COL]})", "f1_macro_mean": np.mean(fold_f1),
           "f1_macro_std": np.std(fold_f1), **metrics(dev.label.values, (oof > .5).astype(int), prefix="oof_"),
           "segundos": custo["finetune_cv_total_s"]}
    pd.DataFrame([row]).to_csv(RES / f"cv_bertimbau_ft_{FT_COL}.csv", index=False)
    dev[["doc_id", "label", "topic"]].assign(prob_falsa=oof).to_csv(RES / f"oof_bertimbau_ft_{FT_COL}.csv", index=False)
    gerados += [f"results/cv_bertimbau_ft_{FT_COL}.csv", f"results/oof_bertimbau_ft_{FT_COL}.csv"]
    print(pd.DataFrame([row]).round(4).T)"""),
md("""## Parte C — teste final (UMA vez, no fim do projeto)

Treina com todo o dev e avalia no teste. Deixar `RUN_FINAL = False` até as configurações estarem fixadas."""),
code("""if RUN_FINAL:
    from sklearn.metrics import confusion_matrix
    t0 = time.time()
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    prob = finetune_predict(dev, test); pred = (prob > .5).astype(int)
    custo["final_treino_dev+teste_s"] = time.time() - t0
    custo["final_gpu_mem_pico_GB"] = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
    cm = confusion_matrix(test.label, pred, labels=[0, 1])
    row = {"config": f"BERTimbau fine-tuned ({NOMES[FT_COL]})", **metrics(test.label.values, pred),
           "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1], "segundos_treino+teste": time.time() - t0}
    pd.DataFrame([row]).to_csv(RES / f"test_bertimbau_ft_{FT_COL}.csv", index=False)
    test[["doc_id", "label", "topic"]].assign(prob_falsa=prob, pred=pred).to_csv(RES / f"test_predictions_bertimbau_ft_{FT_COL}.csv", index=False)
    gerados += [f"results/test_bertimbau_ft_{FT_COL}.csv", f"results/test_predictions_bertimbau_ft_{FT_COL}.csv"]
    print(pd.DataFrame([row]).round(4).T)
else:
    print("Teste final desligado (RUN_FINAL = False).")"""),
code("""# ---- Guardar custo e descarregar resultados para o repositório ----
custo.update({"modelo": MODEL, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
              "max_len_emb": MAX_LEN_EMB, "max_len_ft": MAX_LEN_FT, "epocas": EPOCHS, "batch": BATCH, "lr": LR,
              "ft_col": FT_COL, "smoke": SMOKE})
suf = "_final" if (RUN_FINAL and not RUN_FINETUNE) else ""  # não sobrescreve o custo do fine-tuning em CV
json.dump(custo, open(RES / f"custo_bertimbau_{FT_COL}{suf}.json", "w"), indent=2, default=str)
gerados.append(f"results/custo_bertimbau_{FT_COL}{suf}.json")
out = f"resultados_bertimbau_{FT_COL}.zip"
with zipfile.ZipFile(out, "w") as z:
    for f in gerados:  # só o que esta execução produziu
        z.write(f)
print(open(RES / f"custo_bertimbau_{FT_COL}{suf}.json").read())
try:
    from google.colab import files; files.download(out)
except ImportError:
    print("zip em", os.path.abspath(out))"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"accelerator": "GPU", "colab": {"gpuType": "T4"},
                                               "kernelspec": {"name": "python3", "display_name": "Python 3"}})
Path(__file__).with_name("bertimbau.ipynb").write_text(nbf.writes(nb))
print("ok")
