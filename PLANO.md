# Plano de trabalho — 5 elementos, entrega 4 dez 2026 (tolerância até 6), apresentação semana de 7 dez

Estado (6 nov): repositório montado, corpus caracterizado, divisão dev/teste fixada, baselines e
TF-IDF corridos no dev. **Não voltar a gerar `data/split.json`** — comitem-no no repositório.

## O que já está feito ✅
- Download reprodutível, tabela do corpus, divisão agrupada por pares (sem fugas)
- Estatísticas, comprimentos, Zipf, efeito do pré-processamento no vocabulário, palavras distintivas
- Baselines (maioritária, comprimento), estilo NILC, TF-IDF palavras/caracteres × LogReg/SVM
- Ablation de pré-processamento (minúsculas / stopwords / RSLP / n-gramas) e texto completo vs normalizado
- Leave-one-topic-out
- `run_all.sh` com versões e semente fixadas
- Tópicos LDA (K=15) feitos — falta só interpretação/nomes dos tópicos no artigo (P4); BERTopic opcional
- Extração de informação: 100 docs selecionados (50 pares), Excel por anotador, guia de anotação, avaliação e análise prontas — **falta anotar** (todos, até 20 nov)
- Afinação dos clássicos feita (src/tune.py); ferramenta de análise de erros pronta (src/errors.py); demo pronta (src/demo.py, demo/COMO_USAR.md)
- BERTimbau corrido no Colab (normalizado e cortado), integrado na tabela principal e no artigo

## Divisão por pessoa

| Pessoa | Responsável por | Entregáveis |
|---|---|---|
| **P1 — Corpus & Ética** | Caracterização (já há base), datas, deduplicação, ética | Secções 3 (Dados) e 8 (Ética): RGPD (nomes de pessoas públicas nos textos), ausência de licença, viés político (58% política) e de fonte, risco de censura por falsos positivos, nível de risco no Regulamento da IA (UE 2024/1689), humano no circuito |
| **P2 — Clássicos** | Afinação (C, min_df, n-gramas) só com CV no dev; atributos estilísticos próprios calculados no texto normalizado; análise de coeficientes | Tabela principal; ablation final; custo computacional (tempo, memória) |
| **P3 — Representação neural** | BERTimbau (`neuralmind/bert-base-portuguese-cased`): (a) embeddings congelados + LogReg; (b) fine-tuning no Colab com GPU, só em `text_norm`, mesmos folds | 2.ª representação obrigatória; comparar custo vs ganho; extra opcional: LLM zero-shot (assinalar risco de contaminação, eventos 2016–18) |
| **P4 — Não supervisionado & erros** | LDA (gensim/sklearn) e/ou BERTopic; coerência; tópicos separam *classe* ou só *tema*? Pureza vs rótulo e vs tema | Secção de tópicos; **análise de ≥30 erros** do melhor modelo, agrupados em categorias (ex.: sátira, texto truncado, notícia verdadeira com estilo opinativo, falsa com estilo jornalístico) |
| **P5 — Extração de informação & repositório** | NER spaCy `pt_core_news_lg` (PER, ORG, LOC) + regex de datas e atribuição de fontes ("segundo X", "disse"); **anotar 100 docs** (50 falsas + 50 verdadeiras do dev) | Precisão/revocação por tipo de entidade; comparar densidade de entidades/fontes falsas vs verdadeiras; manter `run_all.sh` a funcionar e testar num clone limpo |

**Todos:** escrevem a sua secção no artigo; anotação dos 100 docs repartida (20 cada, 20 docs em comum
para medir concordância — kappa de Cohen, dá pontos).

## Calendário

| Semana | Marco |
|---|---|
| 6–13 nov | Cada um corre `run_all.sh`; guia de anotação escrito (P5) e anotação começada; LDA a correr; BERTimbau congelado |
| 14–20 nov | Anotação concluída; fine-tuning BERTimbau; afinação clássicos; tópicos interpretados |
| 21–27 nov | **Congelar configurações → `bash run_all.sh --final` uma única vez**; análise dos 30 erros sobre as previsões do teste; avaliação da extração de informação |
| 28 nov–4 dez | Artigo LNCS (≤25 pp), ≥8 referências comentadas, anexo de avaliação pelos pares (deltas somam 0; referir autorização do grupo de 5), declaração de uso de IA (com um caso em que o assistente errou), README final |
| até 7 dez | Demo: documento novo (notícia de 2026 + uma falsa verificada por agência de fact-checking) e mostrar um caso de falha |

## Pontos fortes para o artigo
1. "Um número chega": comprimento sozinho = 0,944. Qualquer resultado no texto completo tem de ser
   comparado com isto — muitos trabalhos citados não o fazem (crítica às referências).
2. Normalizar o tamanho baixa o TF-IDF de 0,975 para 0,943; os n-gramas de caracteres aguentam melhor (0,956).
3. Viés de fonte (93% de um site): limitação central — testar se o modelo aprende "estilo do site".
4. Generalização por tema: queda de até ~5 pontos sem política.
5. Remover stopwords/stemming piora — contra a prática habitual.

## Para o anexo "uso de IA"
O código inicial e esta análise foram gerados com um assistente de IA (Claude). Exemplos reais de erros
a documentar: (1) a primeira versão do carregador assumia que a versão `size_normalized` tinha os mesmos
ficheiros que a versão completa e falhou (ficheiros 586/1607 vs 697/1468); (2) o notebook do BERTimbau gravava os
identificadores como objetos Python, que o numpy recusa ler — só foi detetado ao testar o notebook com um modelo minúsculo. Verifiquem vocês tudo antes de submeter.
