# Guia de anotação — Extração de Informação (Fake.Br)

**Objetivo.** Criar o "ouro" (as respostas certas, feitas por humanos) para medir a precisão e a
revocação do extrator automático, como pede o enunciado (≥100 documentos anotados à mão).

**Prazo:** 20 de novembro. **Tempo estimado:** 1h30–2h por pessoa (24 notícias curtas).

---

## 1. O vosso ficheiro

Cada pessoa tem um ficheiro: `anotacao_P1.xlsx` … `anotacao_P5.xlsx` (P1–P5 conforme o `PLANO.md`).
Cada ficheiro tem 3 folhas:

| Folha | Para quê |
|---|---|
| `textos` | as notícias a anotar (código `D001`…, e o texto) |
| `anotacoes` | onde escrevem as entidades: **uma linha por entidade** |
| `feito` | marcar `S` quando terminarem cada notícia (só as marcadas com `S` contam!) |

Na folha `anotacoes`, as colunas `doc_id` e `tipo` têm listas pendentes — escolham da lista.

```
doc_id | tipo         | entidade (copiar exatamente do texto) | notas
D014   | PESSOA       | Sérgio Moro                            |
D014   | ORGANIZACAO  | Polícia Federal                        |
D014   | LOCAL        | Curitiba                               |
D014   | DATA         | quarta-feira (12)                      |
```

## 2. Regras gerais

1. **Não sabem (nem devem tentar saber) se a notícia é falsa ou verdadeira.** Os códigos são opacos de
   propósito. **Não abram o `selecao.csv`** (tem os rótulos) — isso enviesaria a anotação.
2. **Não usem o spaCy nem o ChatGPT/Claude para pré-anotar.** O ouro tem de ser independente do sistema
   que vai ser avaliado; senão a avaliação mede só "quanto concordo com a máquina".
3. **Cada entidade distinta uma vez por notícia.** Se "Lula" aparece 5 vezes, uma linha. Se aparecem
   formas diferentes ("Lula" e "Luiz Inácio Lula da Silva"), uma linha para cada forma.
4. **Copiar exatamente do texto** (copiar/colar), sem títulos nem artigos:
   "o presidente Michel Temer" → `Michel Temer`.
5. **Só nomes próprios** (excepto DATA). "o governo", "a polícia", "um deputado" **não** se anotam.
6. **Na dúvida, escrevam na coluna `notas`** e sigam em frente. Não discutam os documentos de dupla
   anotação com o colega antes de ambos terminarem.
7. Notícia sem nenhuma entidade: marquem `S` na coluna C da folha `feito`.

## 3. Os quatro tipos

### PESSOA
Nomes de pessoas reais ou personagens: `Dilma Rousseff`, `Lula`, `Neymar`, `Anitta`.
- Alcunhas que identificam uma pessoa contam: `Lula`.
- **Não** contam: cargos sozinhos ("o ministro"), grupos ("os petistas"), @utilizadores.

### ORGANIZACAO
Instituições, empresas, partidos, órgãos públicos, meios de comunicação, clubes:
`PT`, `PSDB`, `Supremo Tribunal Federal`, `STF`, `Polícia Federal`, `Petrobras`, `Globo`, `G1`,
`Câmara dos Deputados`, `Flamengo`, `ONU`.
- Sigla e nome por extenso, se ambos aparecem, são duas linhas.
- Uma organização com nome de lugar é **só** ORGANIZACAO: `Universidade de São Paulo` (não anotar
  `São Paulo` à parte, a menos que apareça sozinho noutro sítio).
- **Não** anotar: operações e eventos (`Lava Jato`, `Copa do Mundo`), leis, programas.

### LOCAL
Países, estados, cidades, bairros, regiões, edifícios conhecidos: `Brasil`, `Rio de Janeiro`,
`Curitiba`, `Nordeste`, `Palácio do Planalto`, `Estados Unidos`.
- `Brasil` é sempre LOCAL, mesmo quando significa "o governo brasileiro".
- Gentílicos ("brasileiro", "carioca") **não** se anotam.

### DATA
Referências temporais **concretas**: datas, dias da semana, meses, anos.
- Anotar a expressão completa: `12 de março de 2017`, `quarta-feira (12)`, `março de 2017`,
  `12/03/2017`, `quarta-feira`, `março`, `2017`.
- **Não** anotar referências relativas vagas: `hoje`, `ontem`, `amanhã`, `nesta semana`,
  `recentemente`, `há dois anos`.
- **Não** anotar horas ("às 15h") nem idades ("70 anos") nem números que não sejam anos
  ("2000 pessoas").

### Casos difíceis (decisões do grupo, 12/11 — valem para todos)

| Caso | Regra |
|---|---|
| Títulos colados ao nome: "**Dona** Marisa", "Dr. Fulano" | Anotar **só o nome**, sem o título: `Marisa` |
| Hospitais, universidades, empresas: "Hospital Sírio-Libanês" | **ORGANIZACAO** (são instituições), não LOCAL |
| Unidades e termos genéricos: "UTI", "Unidade de Terapia Intensiva", "AVC" | **Não** se anotam |
| Horas: "às 12h05" | **Não** são DATA |
| Dia com número entre parênteses: "terça-feira (23)" | DATA, **com** o parêntesis: `terça-feira (23)` |
| "desta", "nesta" antes do dia: "desta quinta-feira (26)" | Copiar só a partir do dia: `quinta-feira (26)` |
| Partido com estado: "(PSDB-MG)", "(DEM-RJ)" | Anotar **só o partido** (`PSDB`, ORGANIZACAO). A sigla do estado **não** se anota. Um estado sozinho ("viajou para MG") é LOCAL |
| Nomes de programas de rádio/TV: "Jornal Eldorado", "Jornal Nacional" | **Não** se anotam. A rádio/canal sim: `Rádio Eldorado`, `Globo` (ORGANIZACAO) |
| "Rio de Janeiro" | LOCAL. O "Janeiro" aqui **não** é DATA |
| Texto que acaba a meio da frase | Normal (versão normalizada do corpus). Anotar só o que lá está e **não tentar adivinhar se a notícia é falsa ou verdadeira** |

## 4. Dupla anotação (concordância)

Cada pessoa tem **4 notícias que também são anotadas por outro colega** (sem saberem quais).
Serve para medir a **concordância entre anotadores** — mostra quão difícil/ambígua é a tarefa e
dá um "teto" realista para o sistema. Por isso: anotem sozinhos e não comparem antes do fim.

## 5. Quando terminarem

1. Confirmem que todas as notícias têm `S` na folha `feito`.
2. Guardem o ficheiro **com o mesmo nome**, em formato `.xlsx`, e coloquem-no na pasta `annotation/`.
3. Registem **quanto tempo demoraram** (vai para a secção de custo do artigo).
4. Quem coordena (P5) corre: `python src/ie_extract.py`

O resultado aparece em `results/ie_avaliacao.csv` (precisão/revocação/F1 por tipo, com intervalo de
confiança) e `results/ie_concordancia.csv` (concordância entre anotadores).
