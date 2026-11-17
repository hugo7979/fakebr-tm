# Demo — como usar na apresentação

Correr a partir da pasta `fakebr-tm`, com o `.venv` ativo. O primeiro arranque demora ~1 min
(treina e guarda os modelos em `models/`); os seguintes são rápidos.

```powershell
python src\demo.py --exemplo falsa        # falsa típica (viral, política opinativa, sem datas) -> FALSA 1.00
python src\demo.py --exemplo verdadeira   # verdadeira típica (STF, fontes)                     -> VERDADEIRA 0.03
python src\demo.py --exemplo falha        # CASO DE FALHA obrigatório: falsa em estilo jornalístico -> VERDADEIRA 0.05
python src\demo.py --file noticia.txt     # notícia NOVA guardada num .txt (UTF-8)
python src\demo.py                        # colar a notícia no terminal e carregar Enter duas vezes
```

## Notícia nova (obrigatório na apresentação)
Escolham uma notícia **de 2026** (ex.: g1.globo.com ou um sítio de verificação como aosfatos.org /
lupa.uol.com.br, que publica desmentidos de notícias falsas reais). Copiem título + texto para
`noticia.txt` (bloco de notas → Guardar como → Codificação UTF-8).
Mostrem uma verdadeira e, se possível, uma falsa desmentida por verificadores.

## O que dizer em cada secção da saída
1. **Previsão** — probabilidade de ser falsa; lê só as primeiras 100 palavras (versão "cortada", sem atalhos).
2. **Pedaços de texto** — n-gramas de caracteres com mais peso: aspas, reticências e "[...]" puxam
   para falsa; " em ", vírgulas, "lava jato" para verdadeira. É estilo, não tema.
3. **Ancoragem factual** — datas por 100 palavras face às médias; com anotação humana, as verdadeiras têm
   ~5× mais datas (o único sinal confirmado; os locais não, porque o extrator confunde instituições com locais).
4. **Entidades** — o extrator também erra (no exemplo da falha: "Lava Jato" como PESSOA; noutros, "STF"
   como LOCAL) — é por isso que medimos precisão/revocação com anotação manual.
5. **Tópico** — o tópico ajuda pouco a distinguir (NMI 0,047): o sinal está no estilo.

**Porque falha o exemplo `falha`:** a notícia falsa tem tantas datas como uma verdadeira (1,29 vs 1,28),
cita um tribunal e números concretos e não usa marcas de estilo das falsas → o modelo é enganado.
É o risco que discutimos: quem produz desinformação pode imitar o estilo jornalístico.

Os três exemplos estão FORA do treino do modelo da demo (previsões honestas).
É uma ferramenta de **triagem**: ordena o que verificar primeiro; não decide o que é falso.
