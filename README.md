# Deteção de notícias falsas no corpus Fake.Br

Trabalho de grupo de Text Mining (Opção 2), Mestrado em Inteligência Artificial, Universidade do Minho.

Diogo Filipe Oliveira Gomes (PG63946) · Sérgio Paulo Vieira Carvalho (PG63711) · Hugo Novais Martins (PG64175) ·
Diogo Valério Borges (PG64174) · Tiago Alexandre Lopes Mendes (PG64031)

O artigo está em [`paper/main.pdf`](paper/main.pdf).

## Pergunta de investigação

Que características linguísticas distinguem as notícias falsas das verdadeiras num corpus real em português
do Brasil, e um detetor treinado com elas mantém o desempenho quando (a) se neutraliza o comprimento dos
textos e (b) é aplicado a temas que não viu no treino?

## Principais resultados

- **O corpus tem dois atalhos.** Só o comprimento do texto dá F1-macro de 0,944 (validação cruzada). Na
  versão normalizada do corpus, 98% das notícias verdadeiras acabam a meio de uma frase, e o último
  carácter basta para 0,890. Criámos uma versão **cortada** (primeiras 100 palavras), sem os dois atalhos.
- **Sem atalhos, no teste:** BERTimbau afinado 0,987 [0,980; 0,992]; n-gramas de caracteres + SVM 0,939;
  só comprimento 0,938. Só o BERTimbau supera a baseline do comprimento com intervalos não sobrepostos.
- **Estilo, não tema.** Remover *stopwords* e fazer *stemming* piora os resultados; os modelos de estilo
  perdem menos em temas não vistos; os tópicos LDA estão mais associados ao tema (NMI 0,104) do que à
  veracidade (0,047).
- **Extração de informação** (100 notícias anotadas pelo grupo): com a anotação humana, só as datas
  distinguem as classes; as verdadeiras têm cerca de cinco vezes mais datas (1,24 vs. 0,23 por 100 palavras).

## Reproduzir

Requer Python 3.13 ou 3.14 e Git. No Windows (PowerShell):

```powershell
git clone https://github.com/hugo7979/fakebr-tm.git
cd fakebr-tm
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_all.py --final
```

Em Mac ou Linux, a ativação do ambiente é `source .venv/bin/activate`.

`python run_all.py` corre tudo no conjunto de desenvolvimento; `--final` acrescenta a avaliação no teste
(já feita uma única vez; repeti-la dá os mesmos números). O resultado são os ficheiros em `results/` e as
tabelas do artigo em `paper/tables/`. Os tempos de execução variam com o computador; todos os outros valores
devem coincidir. A afinação (`src/tune.py`) e a escolha do número de tópicos (`src/topics.py`) ficam em cache;
para as refazer: `python src/tune.py --redo` e `python src/topics.py --reselect`.

### BERTimbau (Google Colab, GPU)

Os resultados do BERTimbau já estão no repositório (`data/emb_bertimbau_*.npz` e `results/*bertimbau*`) e
entram automaticamente nas tabelas. Para os refazer: abrir `notebooks/bertimbau.ipynb` no Colab com GPU T4,
correr todas as células e descompactar o zip resultante na raiz do repositório. O notebook usa a mesma
divisão e os mesmos folds que o resto do projeto, e é gerado a partir de `notebooks/build_notebook.py`.

### Demo

```powershell
python src/demo.py --exemplo falsa          # exemplos: falsa | verdadeira | falha
python src/demo.py --file noticia.txt       # uma notícia nova num ficheiro de texto
```

Ver `demo/COMO_USAR.md`.

## Dados

Os textos do corpus **não** estão no repositório (regra do enunciado). `src/data.py` descarrega o
[Fake.br-Corpus](https://github.com/roneysco/Fake.br-Corpus) no commit fixo `780f551`, para todos obterem
os mesmos dados. O corpus não tem ficheiro de licença; os autores pedem a citação de Monteiro et al.
(PROPOR 2018) e Silva et al. (Expert Systems with Applications, 2020). Os ficheiros de anotação publicados
guardam as anotações, categorias e nomes, mas não os textos das notícias (cópia criada com
`src/sanitize_for_publish.py`).

## Estrutura

| Ficheiro | Função |
|---|---|
| `run_all.py` | corre todos os passos por ordem (Windows, Mac e Linux) |
| `src/data.py` | descarrega o corpus (commit fixo) e cria a tabela única e a versão cortada |
| `src/split.py` | divisão desenvolvimento/teste 80/20, agrupada por par e estratificada por tema (semente 42) |
| `src/preprocess.py` | tokenização, minúsculas, *stopwords*, *stemming* RSLP (passos opcionais) |
| `src/characterize.py` | estatísticas do corpus, lei de Zipf, vocabulário por pré-processamento, palavras distintivas |
| `src/tune.py` | afinação de hiperparâmetros por validação cruzada no desenvolvimento |
| `src/classify.py` | baselines, estilo, TF-IDF + LogReg/SVM, ablation, *leave-one-topic-out*, teste final |
| `src/topics.py` | tópicos LDA: escolha de K por coerência NPMI, palavras, prevalência por classe, NMI |
| `src/ie_select.py` | seleção das 100 notícias para anotação (já usado; não reescreve sem `--force`) |
| `src/ie_extract.py` | extração de entidades e datas, avaliação no ouro, concordância, falsas vs. verdadeiras |
| `src/near_duplicates.py` | verificação de notícias quase repetidas entre folds |
| `src/errors.py` | folha de análise de erros e resumo das categorias |
| `src/cost.py` | custo computacional (tempo, memória, tamanho dos modelos) |
| `src/make_tables.py` | gera as tabelas LaTeX do artigo a partir de `results/` |
| `src/demo.py` | demo para a apresentação |
| `src/sanitize_for_publish.py` | cópia do repositório sem os textos do corpus |
| `notebooks/` | notebook do BERTimbau (Colab) |
| `annotation/` | guia de anotação, anotações do grupo, análise de erros, nomes dos tópicos |
| `results/` | todos os resultados (CSV) e figuras |
| `paper/` | artigo em LaTeX (modelo Springer LNCS) e PDF |
