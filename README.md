# Deteção de Notícias Falsas em Português (Fake.Br) — Trabalho de Grupo TM

**Pergunta de investigação.** Que marcas linguísticas distinguem notícias falsas de verdadeiras no Fake.Br,
e o desempenho dos detetores mantém-se (a) quando se neutraliza o comprimento dos textos e (b) em temas
não vistos no treino?

**Critério de sucesso (fixado antes do teste).** Um modelo baseado em conteúdo deve superar a baseline
"só comprimento" em F1-macro com intervalos de confiança a 95% não sobrepostos, e a queda em
leave-one-topic-out face à validação cruzada no mesmo tema deve ser quantificada.

## Reproduzir

Requer Python 3.12, 3.13 ou 3.14 e Git.

```bash
pip install -r requirements.txt
python run_all.py            # dev apenas (CV, ablation, por tema) ~5 min — funciona em Windows, Mac e Linux
python run_all.py --final    # + avaliação ÚNICA no teste — só no fim do projeto!
```

(`bash run_all.sh` faz o mesmo em Mac/Linux.)

### BERTimbau (2.ª representação, precisa de GPU → Google Colab)

1. Abrir `notebooks/bertimbau.ipynb` no Colab e escolher *Ambiente de execução → GPU T4*.
2. Correr todas as células; quando pedir, fazer upload do `fakebr-tm.zip` (ou clonar o repositório do grupo).
3. No fim descarrega `resultados_bertimbau.zip`: descompactar na raiz do repositório
   (cria `data/emb_bertimbau_*.npz` e `results/*bertimbau*`).
4. A partir daí, `python run_all.py` inclui automaticamente o BERTimbau congelado na tabela principal.

O notebook usa a mesma divisão (`data/split.json`) e os mesmos 5 folds (`get_folds`) que o resto do projeto.
Para verificar que tudo corre antes de gastar GPU: `SMOKE = True` na primeira célula.
O notebook é gerado a partir de `notebooks/build_notebook.py` (mais fácil de rever).

Os dados **não** estão no repositório: `src/data.py` clona o
[Fake.br-Corpus](https://github.com/roneysco/Fake.br-Corpus) no commit `780f551`.
O repositório do corpus **não tem ficheiro de licença**; os autores pedem citação de
Monteiro et al. (PROPOR 2018) e Silva et al. (ESWA 2020). Mencionar isto na secção de ética.

## Dados em bruto e textos

O repositório não inclui os textos do corpus (regra do enunciado e ausência de licença): `src/data.py`
descarrega-os. Os ficheiros de anotação e de análise publicados guardam as anotações, categorias e
nomes, mas não os textos das notícias (versão criada com `src/sanitize_for_publish.py`).

## Estrutura

| Ficheiro | Função |
|---|---|
| `src/data.py` | download (commit fixo) e tabela única com par, rótulo, tema, 25 metadados NILC |
| `src/split.py` | divisão dev/teste 80/20, **agrupada por par** e por textos/links repetidos, estratificada por tema, semente 42 |
| `src/preprocess.py` | tokenização, minúsculas, stopwords, RSLP — passos opcionais para ablation |
| `src/characterize.py` | estatísticas, comprimentos, Zipf, vocabulário por pipeline, palavras distintivas (log-odds) |
| `src/classify.py` | baselines, estilo, TF-IDF palavras/caracteres + LogReg/SVM, ablation, leave-one-topic-out, teste final |
| `notebooks/bertimbau.ipynb` | BERTimbau no Colab: (A) embeddings congelados + LogReg, (B) fine-tuning com CV de 5 folds, (C) teste final desligado |
| `src/topics.py` | tópicos LDA: escolha de K por coerência NPMI (3 sementes, regra de 1 dp), palavras por tópico, prevalência por classe, NMI com rótulo/tema, F1 com tópicos como atributos |
| `src/ie_select.py` | seleção (já feita) de 50 pares do dev para anotação e criação dos Excel por anotador — **não voltar a correr** |
| `src/ie_extract.py` | extração de entidades (spaCy + regras para datas), avaliação P/R/F1 no ouro, concordância, falsas vs verdadeiras |
| `annotation/` | `GUIA_ANOTACAO.md`, `anotacao_P1..P5.xlsx` (a preencher), `selecao.csv` (rótulos — anotadores não abrem) |
| `src/tune.py` | afinação de hiperparâmetros por CV no dev (texto cortado) → `results/best_params.json`, usados no teste final |
| `src/errors.py` | análise de erros: Excel com 40 erros para categorizar (`--source dev` para ensaio) e resumo `--summary` |
| `src/demo.py` | demo da apresentação (ver `demo/COMO_USAR.md`) |
| `run_all.py` | comando único (multiplataforma) que corre os 4 scripts por ordem |
| `results/` | todas as tabelas (CSV) e figuras |

## Decisões e descobertas sobre o corpus (dev, 5 756 docs)

1. **Pares alinhados → fuga de informação.** Cada notícia falsa tem uma verdadeira sobre o mesmo assunto.
   Além disso, 17 notícias verdadeiras aparecem em mais do que um par (mesmo link). Agrupámos por
   componente ligada (par + texto/link partilhado): 0 grupos partilhados entre dev e teste.
2. **Viés de comprimento forte.** Mediana de 919 palavras (verdadeiras) vs 156 (falsas).
   Na versão `size_normalized` as medianas ficam 157 vs 155.
3. **Viés de fonte.** 92,7% das falsas no dev vêm de um único site (diariodobrasil.org); 64% das
   verdadeiras vêm do G1. Parte do que o modelo aprende pode ser *estilo editorial do site*, não "falsidade".
4. **Ruído.** 4 ficheiros em falta / renumerados na versão normalizada (586, 1607 vs 697, 1468);
   1 texto verdadeiro duplicado; datas em formatos heterogéneos ("01 de abril de 2010", "9/8/2017").
5. **Zipf.** α ≈ 1,13 (verdadeiras) e 1,12 (falsas), ajuste nas ordens 10–10 000. Figura em `results/figs/zipf.png`.
6. **Pré-processamento.** Minúsculas reduzem o vocabulário 15%; RSLP 59%. Mas remover stopwords e
   fazer stemming **piora** a classificação (as palavras funcionais carregam estilo).
7. **Palavras distintivas (texto normalizado).** Falsas: *dilma, lava-jato, disse, coréia* (grafia
   pré-Acordo), *vídeo, lula, através, globo, povo, pra*. Verdadeiras: *nesta, tribunal, quarta-feira,
   2017, segundo, desta* — marcas de escrita jornalística (datas, atribuição de fontes).

## Resultados preliminares — validação cruzada no dev (5 folds, F1-macro média ± dp)

| Configuração | F1-macro |
|---|---|
| Classe maioritária | 0,333 ± 0,000 |
| **Só comprimento (1 atributo)** | **0,944 ± 0,005** |
| Estilo NILC (20 atributos) | 0,965 ± 0,003 |
| Estilo NILC, só proporções (6) | 0,927 ± 0,007 |
| TF-IDF 1-2 + LogReg, texto completo | 0,972 ± 0,005 |
| TF-IDF 1-2 + SVM, texto completo | 0,975 ± 0,006 |
| TF-IDF 1-2 + LogReg, normalizado | 0,940 ± 0,005 |
| TF-IDF 1-2 + SVM, normalizado | 0,943 ± 0,005 |
| TF-IDF caracteres 2-5 + SVM, normalizado | 0,956 ± 0,007 |

**Leitura:** com texto completo, ~0,97 parece excelente, mas um único número (o comprimento) já dá 0,944.
Com comprimento neutralizado, os n-gramas de caracteres são os mais robustos.

**Leave-one-topic-out (dev, F1-macro no tema excluído):**

| Modelo (normalizado) | sem política | sem TV | sem sociedade | sem outros |
|---|---|---|---|---|
| Estilo (proporções) | 0,920 | 0,936 | 0,936 | 0,873 |
| TF-IDF palavras + SVM | 0,888 | 0,939 | 0,913 | 0,860 |
| TF-IDF caracteres + SVM | 0,911 | 0,958 | 0,945 | 0,911 |

O **teste ainda não foi usado**. As tabelas acima são todas do conjunto de desenvolvimento.

## Atalho do final do texto e versão "cortada" (descoberto em 11/11)

Na versão `size_normalized` do corpus, 98% das verdadeiras foram cortadas a meio da frase (só 7% acabam em
pontuação, contra 77% das falsas). Quatro atributos sobre o último carácter dão **F1 = 0,890**.
O BERTimbau (que sabe onde o texto termina) obtém 0,970 congelado e **0,993 com fine-tuning** no texto
normalizado — números inflacionados por este atalho. Criámos `text_cut` (`src/data.py`): primeiras 100
palavras do normalizado (ou n−5), para TODOS os textos acabarem a meio. Aí a baseline do final desce para
0,395 e os clássicos mantêm-se (caracteres 0,934, palavras 0,921).

| BERTimbau (CV 5 folds, dev) | normalizado | **cortado** |
|---|---|---|
| congelado + LogReg | 0,970 | **0,945** |
| fine-tuned (3 épocas, T4) | 0,993 | **0,987** [0,984; 0,990] |

Retirar os 51 docs com uma quase-cópia (cosseno ≥ 0,7) noutro fold não muda o F1 do fine-tuned
(`src/near_duplicates.py` → `results/near_duplicates.csv`): o ganho não vem de memorizar repetições.
O valor do congelado (cortado) calculado aqui coincide com o do Colab (0,9448) — reprodutível entre ambientes.

## Tópicos (LDA, dev, texto normalizado, sem stopwords)

- **Escolha de K:** a coerência NPMI é quase plana entre K=5 e K=30 (0,174–0,189, diferenças dentro da
  variação entre sementes). Regra de 1 desvio-padrão → **K = 15**. Tabela: `results/topicos_k_selecao.csv`.
- **Tópicos típicos das falsas:** T10 *lula dilma disse país vai petista* (93% falsas entre os docs onde
  domina; peso 5,6× maior), T14 *vídeo redes sociais facebook família*.
- **Tópicos típicos das verdadeiras:** T8 *tribunal STF decisão julgamento defesa*, T6 *mil maior casos
  saúde segundo* (números/estatísticas), T0 *hospital acidente morte*, T13 *câmara senado impeachment*.
- **Os tópicos captam mais o tema do que a veracidade:** NMI(tópico, tema) = 0,104 vs NMI(tópico, classe) = 0,047.
- **Só o conteúdo (15 tópicos) dá F1-macro = 0,747** [0,736; 0,758] — muito abaixo do estilo (0,93–0,96):
  o que distingue as falsas no Fake.Br é sobretudo *como* escrevem, não *de que* falam.

## Extração de informação (avaliada em 100 notícias anotadas por 5 anotadores)

Extrator: spaCy `pt_core_news_md` + regras para datas. Ouro: `annotation/anotacao_P1..P5.xlsx`.

| | Precisão | Revocação | F1 |
|---|---|---|---|
| PESSOA | 0,776 | 0,909 | 0,837 |
| ORGANIZACAO | 0,681 | 0,409 | 0,511 |
| LOCAL | 0,377 | 0,938 | 0,537 |
| DATA | 0,716 | 0,765 | 0,739 |
| **todos** | 0,634 | 0,735 | **0,681** (0,818 ignorando o tipo) |

Concordância entre anotadores (20 docs): F1 0,875 estrita / 0,931 relaxada. O extrator confunde
instituições com locais (103 casos: STF, Congresso, PF...). Com o ouro humano (50 pares), só as
**datas** distinguem as classes: 1,24 vs 0,23 por 100 palavras nas verdadeiras vs falsas (≈5×, p<1e-4);
locais 1,02 vs 0,69 (p=0,067, não significativo). 10 inconsistências de formato (~1%) mantidas.

## Referências de partida (a comentar criticamente no artigo)

- Monteiro et al. (2018), PROPOR — corpus Fake.Br e primeiros resultados.
- Silva, Santos, Almeida & Pardo (2020), *Expert Systems with Applications* 146 — estudo extenso no Fake.Br.
- Souza, Nogueira & Lotufo (2020), BRACIS — BERTimbau.
- Monroe, Colaresi & Quinn (2008), *Political Analysis* — log-odds com prior de Dirichlet.
- Pérez-Rosas et al. (2018), COLING — deteção automática de fake news (inglês), comparação de atributos.
- Shu et al. (2017), *SIGKDD Explorations* — panorama da deteção de fake news.
- Zipf (1949) / Piantadosi (2014) — lei de Zipf.
- Trabalhos recentes sobre notícias falsas geradas por LLM em português (PROPOR 2026) — para discussão.
