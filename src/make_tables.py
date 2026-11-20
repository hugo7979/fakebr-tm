"""Gera as tabelas LaTeX do artigo (paper/tables/*.tex) e copia as figuras (paper/figs/)
a partir de results/. Assim os números do artigo vêm sempre do código, nunca copiados à mão.

Tabelas cujos resultados ainda não existem (teste final, BERTimbau, ouro da EI) são geradas
como uma caixa "por preencher" para o artigo compilar na mesma.
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from data import ROOT

RES, PAPER = ROOT / "results", ROOT / "paper"
TAB, FIG = PAPER / "tables", PAPER / "figs"


def n(x, d=3):
    """Número em formato português (vírgula decimal)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{d}f}".replace(".", ",")


def tex_escape(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#").replace("|", r"$|$"))


def write(name, body):
    (TAB / f"{name}.tex").write_text(body, encoding="utf-8")


def pending(name, caption, label, what):
    write(name, rf"""\begin{{table}}[t]\centering
\caption{{{caption}}}\label{{{label}}}
\fbox{{\parbox{{0.9\linewidth}}{{\centering\textcolor{{red}}{{\textbf{{Por preencher:}} {what}}}}}}}
\end{{table}}
""")


def table(name, caption, label, cols, header, rows, note=None, size=r"\small", sep=None):
    if sep:  # mais espaço entre colunas (números com separador de milhares)
        size += rf"\setlength{{\tabcolsep}}{{{sep}}}"
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    foot = rf"\par\smallskip{{\footnotesize {note}}}" if note else ""
    write(name, rf"""\begin{{table}}[t]\centering{size}
\caption{{{caption}}}\label{{{label}}}
\begin{{tabular}}{{{cols}}}
\toprule
{header} \\
\midrule
{body}
\bottomrule
\end{{tabular}}{foot}
\end{{table}}
""")


def corpus():
    s = pd.read_csv(RES / "corpus_stats.csv", index_col=0)
    lab = {"documentos": "Documentos", "palavras_total": "Palavras (total)",
           "palavras_mediana": "Palavras por doc.\\ (mediana)", "palavras_p10": "\\quad percentil 10",
           "palavras_p90": "\\quad percentil 90", "palavras_norm_mediana": "Palavras, versão normalizada (mediana)",
           "maiusculas_mediana": "Palavras em maiúsculas (mediana)",
           "erros_ortograficos_media_pct": "Erros ortográficos (\\%, média)",
           "frase_media_mediana": "Palavras por frase (mediana)",
           "dominio_mais_frequente_pct": "Domínio mais frequente (\\% da classe)"}
    rows = []
    for k, v in lab.items():
        d = 0 if k in ("documentos", "palavras_total", "palavras_mediana", "palavras_p10", "palavras_p90",
                       "maiusculas_mediana") else 1
        if k == "erros_ortograficos_media_pct":
            d = 2
        f = (lambda x: f"{x:,.0f}".replace(",", r"\,")) if d == 0 else (lambda x, d=d: n(x, d))
        rows.append([v, f(s.loc[k, "verdadeira"]), f(s.loc[k, "falsa"])])
    table("tab_corpus", "Estatísticas descritivas do conjunto de desenvolvimento (2\\,878 pares).",
          "tab:corpus", "lrr", "& Verdadeiras & Falsas", rows,
          note="Metadados de estilo calculados pelos autores do corpus sobre o texto completo.", sep="10pt")


def vocab():
    v = pd.read_csv(RES / "vocab_by_pipeline.csv")
    names = {"raw": "Tokens originais", "lower": "+ minúsculas", "lower+stop": "+ remoção de stopwords",
             "lower+stop+rslp": "+ stemming RSLP"}
    rows = [[names[r.pipeline], f"{r.tokens:,}".replace(",", r"\,"), f"{r.vocabulario:,}".replace(",", r"\,"),
             f"{r.hapax:,}".replace(",", r"\,"), n(r._5, 1)] for r in v.itertuples()]
    table("tab_vocab", "Efeito de cada passo de pré-processamento no vocabulário (dev, texto completo).",
          "tab:vocab", "lrrrr", "Pipeline & Tokens & Vocabulário & Hapax & Redução (\\%)", rows, sep="10pt")


def short(name):
    import re
    name = re.sub(r", afinado: .*\)$", ", afinado)", name)
    for a, b in [("baseline: ", ""), ("BERTimbau congelado + LogReg (", "BERTimbau congelado ("),
                 ("completo, 512 tokens", "completo, 512 tok."), ("só comprimento (texto completo)", "só comprimento"),
                 ("TF-IDF palavras 1-2", "palavras"), ("TF-IDF caracteres 2-5", "caracteres"),
                 ("TF-IDF palavras", "palavras"), ("TF-IDF caracteres", "caracteres"),
                 ("estilo NILC só proporções", "estilo, só proporções"), ("estilo NILC", "estilo")]:
        name = name.replace(a, b)
    return name


def cv_main():
    c = pd.read_csv(RES / "cv_main.csv")
    rows = []
    for r in c.itertuples():
        rows.append([tex_escape(short(r.config)), f"{n(r.f1_macro_mean)} $\\pm$ {n(r.f1_macro_std)}",
                     f"[{n(r.oof_ci95_lo)}; {n(r.oof_ci95_hi)}]", n(r.oof_f1_verdadeira), n(r.oof_f1_falsa),
                     n(r.segundos, 0)])
    for extra in sorted(RES.glob("cv_bertimbau_ft_*.csv")):
        for r in pd.read_csv(extra).itertuples():
            rows.append([tex_escape(short(r.config)), f"{n(r.f1_macro_mean)} $\\pm$ {n(r.f1_macro_std)}",
                         f"[{n(r.oof_ci95_lo)}; {n(r.oof_ci95_hi)}]", n(r.oof_f1_verdadeira),
                         n(r.oof_f1_falsa), n(r.segundos, 0)])
    table("tab_cv_main", "Classificação: validação cruzada com 5 folds agrupados por par, no conjunto de "
          "desenvolvimento. F1-macro média $\\pm$ desvio-padrão entre folds; IC95\\% por bootstrap "
          "(1\\,000 reamostragens) sobre as previsões \\emph{out-of-fold}; F1 por classe; tempo total da CV.",
          "tab:cv", "lccccr", "Configuração & F1-macro & IC95\\% & F1$_V$ & F1$_F$ & s", rows,
          note="``Completo'': texto original; ``normalizado'': versão do corpus com os dois textos de cada par "
               "cortados ao mesmo comprimento.", size=r"\scriptsize")


def ablation():
    a = pd.read_csv(RES / "cv_ablation.csv")
    a["pp"] = a.config.str.split("|").str[1].str.strip()
    a["ng"] = a.config.str.split("|").str[2].str.strip()
    names = {"raw": "sem normalização", "lower": "minúsculas", "lower+stop": "+ sem stopwords",
             "lower+stop+rslp": "+ stemming RSLP"}
    rows = []
    for p in names:
        r1 = a[(a.pp == p) & (a.ng == "1-gramas")].iloc[0]
        r2 = a[(a.pp == p) & (a.ng == "1-2-gramas")].iloc[0]
        rows.append([names[p], f"{n(r1.f1_macro_mean)} $\\pm$ {n(r1.f1_macro_std)}",
                     f"{n(r2.f1_macro_mean)} $\\pm$ {n(r2.f1_macro_std)}"])
    table("tab_ablation", "Ablation do pré-processamento: TF-IDF + SVM linear no texto normalizado "
          "(F1-macro, CV 5 folds no dev).", "tab:ablation", "lcc", "Pré-processamento & 1-gramas & 1--2-gramas", rows)


def loto():
    t = pd.read_csv(RES / "cv_leave_one_topic_out.csv")
    temas = ["politica", "tv_celebridades", "sociedade_cotidiano", "outros"]
    nomes = {"politica": "política", "tv_celebridades": "TV/celebr.", "sociedade_cotidiano": "sociedade",
             "outros": "outros"}
    cfg = {"estilo NILC só proporções (6)": "Estilo (6 proporções)",
           "TF-IDF palavras 1-2 + SVM (normalizado)": "TF-IDF palavras + SVM",
           "TF-IDF caracteres 2-5 + SVM (normalizado)": "TF-IDF caracteres + SVM"}
    rows = []
    for c, lab in cfg.items():
        sub = t[t.config == c].set_index("tema_excluido")
        rows.append([lab] + [f"{n(sub.loc[x, 'f1_macro'])} {{\\tiny[{n(sub.loc[x, 'ci95_lo'], 2)}; "
                             f"{n(sub.loc[x, 'ci95_hi'], 2)}]}}" for x in temas])
    ntest = t.drop_duplicates("tema_excluido").set_index("tema_excluido").n_teste
    header = "Modelo & " + " & ".join(nomes[x] for x in temas)
    ns = ", ".join(f"{nomes[x]} {ntest[x]}" for x in temas)
    table("tab_loto", "Generalização para temas não vistos (\\emph{leave-one-topic-out} no dev): treino sem "
          f"o tema da coluna e teste nesse tema. F1-macro e IC95\\%. Documentos de teste: {ns}.",
          "tab:loto", "lcccc", header, rows, size=r"\scriptsize")


def topics():
    t = pd.read_csv(RES / "topicos_palavras.csv")
    nomes = {}
    fx = ROOT / "annotation" / "topicos_nomes.xlsx"   # nomes dados pelo grupo (opcional)
    if fx.exists():
        x = pd.read_excel(fx, dtype=str)
        nomes = {int(k): v.strip() for k, v in zip(x.iloc[:, 0], x.iloc[:, 1]) if isinstance(v, str) and v.strip()}
    rows = [[f"T{r.topico}" + (f" {tex_escape(nomes[r.topico])}" if r.topico in nomes else ""),
             n(r.razao_F_V, 2), n(r.pct_falsas_entre_dominantes, 0), str(r.docs_dominante),
             tex_escape(" ".join(r.palavras.split()[:8 if not nomes else 5]))] for r in t.itertuples()]
    m = pd.read_csv(RES / "topicos_metricas.csv").set_index("medida").valor
    note = (f"NMI(tópico dominante, classe) = {n(float(m.iloc[0]))}; NMI(tópico dominante, tema) = "
            f"{n(float(m.iloc[1]))}. Razão F/V: peso médio do tópico nas falsas a dividir pelo peso nas verdadeiras.")
    table("tab_topics", f"Os {len(t)} tópicos LDA (dev), ordenados do mais típico das falsas ao mais típico "
          "das verdadeiras.", "tab:topics", "lrrrp{4.6cm}" if nomes else "lrrrp{7.2cm}",
          "Tóp. & F/V & \\%F & Docs & Termos mais prováveis", rows, note=note, size=r"\scriptsize")
    k = pd.read_csv(RES / "topicos_k_selecao.csv").groupby("K").agg(m=("npmi_medio", "mean"),
                                                                    s=("npmi_medio", "std"),
                                                                    p=("perplexidade", "mean"))
    table("tab_k", "Escolha do número de tópicos: coerência NPMI média (3 sementes) e perplexidade.",
          "tab:k", "rccc", "K & NPMI & dp & Perplexidade",
          [[str(i), n(r.m), n(r.s), n(r.p, 0)] for i, r in k.iterrows()])


def ie():
    d = pd.read_csv(RES / "ie_falsas_vs_verdadeiras.csv")
    nomes = {"PESSOA": "Pessoas", "ORGANIZACAO": "Organizações", "LOCAL": "Locais", "DATA": "Datas",
             "ATRIBUICAO": "Expressões de atribuição"}
    rows = [[nomes[r[1]], n(r[2], 2), n(r[3], 2), n(r[4], 2),
             ("$<10^{-4}$" if r[6] < 1e-4 else n(r[6], 4))] for r in d.itertuples()]
    table("tab_ie_density", "Densidade de entidades por 100 palavras (dev, texto normalizado, "
          "\\texttt{pt\\_core\\_news\\_md} + regras para datas). Teste de Wilcoxon emparelhado por par "
          "($n=2\\,878$).", "tab:iedens", "lcccc", "Medida & Falsas & Verdadeiras & Mediana F$-$V & $p$", rows)
    ev = RES / "ie_avaliacao.csv"
    if ev.exists():
        e = pd.read_csv(ev)
        e = e[(e.modelo == "pt_core_news_md")]
        rows = [[r.correspondencia, tex_escape(r.tipo), n(r.precisao), n(r.revocacao), n(r.f1),
                 f"[{n(r.f1_ci95_lo)}; {n(r.f1_ci95_hi)}]", str(r.n_ouro)] for r in e.itertuples()]
        table("tab_ie_eval", "Extração de informação: avaliação no ouro anotado manualmente.", "tab:ieeval",
              "llccccr", "Correspondência & Tipo & P & R & F1 & IC95\\% & $n$ ouro", rows, size=r"\scriptsize")
        ag = RES / "ie_concordancia.csv"
        if ag.exists():
            g = pd.read_csv(ag)
            g = g[g.tipo == "TODOS (micro)"].set_index("correspondencia").f1
            txt = (f"Concordância entre anotadores (F1, 20 docs): {n(g.get('estrita'))} estrita, "
                   f"{n(g.get('relaxada'))} relaxada.")
            t = (TAB / "tab_ie_eval.tex").read_text()
            (TAB / "tab_ie_eval.tex").write_text(t.replace("\\end{tabular}", "\\end{tabular}\\par\\smallskip{\\footnotesize "
                                                           + txt + "}"))
        go = RES / "ie_ouro_falsas_vs_verdadeiras.csv"
        if go.exists():
            o = pd.read_csv(go)
            nomes = {"PESSOA": "Pessoas", "ORGANIZACAO": "Organizações", "LOCAL": "Locais", "DATA": "Datas"}
            rows = [[nomes[r.tipo], n(r.ouro_media_falsas, 2), n(r.ouro_media_verdadeiras, 2), n(r.razao_V_F, 1),
                     ("$<10^{-4}$" if r.wilcoxon_p < 1e-4 else n(r.wilcoxon_p, 3))] for r in o.itertuples()]
            table("tab_ie_gold", "Entidades por 100 palavras segundo a anotação HUMANA (50 pares do dev). "
                  "Wilcoxon emparelhado por par.", "tab:iegold", "lcccc",
                  "Tipo & Falsas & Verdadeiras & V/F & $p$", rows)
    else:
        pending("tab_ie_eval", "Extração de informação: precisão, revocação e F1 no ouro anotado.",
                "tab:ieeval", "anotação dos 100 documentos (prazo 20 nov.), depois correr "
                              "\\texttt{python run\\_all.py}.")


def test():
    t = RES / "test_main_table.csv"
    if t.exists():
        c = pd.concat([pd.read_csv(t)] + [pd.read_csv(f) for f in sorted(RES.glob("test_bertimbau_ft_*.csv"))],
                      ignore_index=True)
        rows = [[tex_escape(short(r.config)), n(r.f1_macro), f"[{n(r.ci95_lo)}; {n(r.ci95_hi)}]",
                 n(r.f1_verdadeira), n(r.f1_falsa), f"{r.TN}/{r.FP}/{r.FN}/{r.TP}"] for r in c.itertuples()]
        table("tab_test", "Resultados no conjunto de teste (1\\,444 docs, usado uma única vez). "
              "Matriz de confusão como VN/FP/FN/VP (positivo = falsa).", "tab:test", "lccccc",
              "Configuração & F1-macro & IC95\\% & F1$_V$ & F1$_F$ & VN/FP/FN/VP", rows, size=r"\scriptsize")
    else:
        pending("tab_test", "Resultados no conjunto de teste, com matriz de confusão.", "tab:test",
                "correr \\texttt{python run\\_all.py --final} uma única vez, com as configurações já fixadas.")


def errors_table():
    f = RES / "erros_categorias.csv"
    if not f.exists():
        pending("tab_errors", "Análise de erros do melhor modelo no teste, por categoria.", "tab:errors",
                "abrir o teste, correr \\texttt{python src/errors.py}, categorizar os erros no Excel e "
                "\\texttt{python src/errors.py --summary}.")
        return
    t = pd.read_csv(f, index_col=0)
    t = t.drop(index="total", errors="ignore")
    rows = [[tex_escape(cat), str(int(r["falso positivo"])) if "falso positivo" in t else "0",
             str(int(r["falso negativo"])) if "falso negativo" in t else "0", str(int(r["total"])),
             tex_escape(r.get("exemplos", ""))] for cat, r in t.iterrows()]
    table("tab_errors", f"Categorias de erro do melhor modelo no teste ({int(t.total.sum())} erros analisados). "
          "FP: verdadeira classificada como falsa; FN: falsa classificada como verdadeira.", "tab:errors",
          "lrrrl", "Categoria & FP & FN & Total & Exemplos", rows, size=r"\scriptsize")


def cost_table():
    f = RES / "custo.csv"
    if not f.exists():
        return
    import json
    c = pd.read_csv(f)
    hw = json.loads((RES / "custo_hardware.json").read_text()) if (RES / "custo_hardware.json").exists() else {}

    def fmt(x, d):
        return "--" if pd.isna(x) else n(x, d)
    def lab(m):
        m = short(m).replace(" (cortado, afinado)", "").replace("Extração de entidades (pt_core_news_", "spaCy (")
        return (m.replace("BERTimbau fine-tuned (cortado), CV 5 folds", "BERTimbau afinado: CV 5 folds")
                 .replace("BERTimbau fine-tuned (cortado), treino final + teste", "BERTimbau afinado: final + teste")
                 .replace("Só comprimento (baseline)", "Só comprimento"))
    rows = [[tex_escape(lab(r.modelo)), tex_escape(r.hardware.replace("GPU Tesla ", "GPU ")),
             fmt(r.treino_s, 1), fmt(r.inferencia_ms_por_doc, 2), fmt(r.memoria_pico_MB, 0),
             fmt(r.tamanho_modelo_MB, 1)] for r in c.itertuples()]
    where = (f"CPU: {hw.get('sistema', '?')}, {hw.get('nucleos', '?')} núcleos lógicos, {str(hw.get('ram_GB', '?')).replace('.', ',')} GB de RAM"
             if hw else "CPU não registada")
    table("tab_cost", "Custo computacional. Treino no dev completo (5\\,756 notícias); inferência medida em "
          f"1\\,000 notícias. Memória: pico de RAM do processo (GPU no caso do BERTimbau). {tex_escape(where)}; "
          "GPU: Google Colab gratuito. Nenhum serviço externo pago nem modelo de linguagem foi usado "
          "no pipeline (0 tokens).", "tab:cost", "llrrrr",
          "Modelo & Hardware & Treino (s) & ms/notícia & Memória (MB) & Tam. (MB)", rows,
          size=r"\scriptsize")


def tuning_table():
    f = RES / "tuning.csv"
    if not f.exists():
        return
    t = pd.read_csv(f)
    rows = []
    for fam, sub in t.groupby("familia", sort=False):
        b = sub.sort_values("f1_macro_mean", ascending=False).iloc[0]
        rows.append([tex_escape(fam), str(len(sub)), n(sub.f1_macro_mean.min()), n(sub.f1_macro_mean.max()),
                     tex_escape(b.parametros)])
    table("tab_tuning", "Afinação de hiperparâmetros por CV (5 folds, dev, texto cortado): número de "
          "configurações, F1-macro da pior e da melhor, e parâmetros escolhidos.", "tab:tuning", "lrccl",
          "Modelo & Config. & Pior & Melhor & Escolhida", rows, size=r"\scriptsize")


if __name__ == "__main__":
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (RES / "figs").glob("*.png"):
        shutil.copy(f, FIG / f.name)
    for fn in (corpus, vocab, cv_main, ablation, loto, topics, ie, test, errors_table, tuning_table, cost_table):
        fn()
    print("tabelas:", sorted(p.name for p in TAB.glob("*.tex")))
