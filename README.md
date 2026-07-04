# deep-acordao-tcu2

> Mestrado em Administração Pública — Ciência de Dados e IA
> Instituto Brasileiro de Ensino, Desenvolvimento e Pesquisa (IDP)
> **Autores:** Bruno Aires · Candice Trigueiro · Rafael Ayoroa

Classificação automática de desfechos em acórdãos do Tribunal de Contas da União (TCU)
nas áreas de **Saúde e Educação** — *Irregular*, *Regular com Ressalva* ou *Regular*.

Pipeline completo de NLP jurídico: desde a aquisição dos dados abertos do TCU até
a avaliação comparativa de cinco modelos de classificação, com **anti-vazamento
obrigatório** e **ponderação de classes** em todos os experimentos.

---

## A Jornada do Projeto

### 1. Ponto de partida

A classificação automatizada de acórdãos TCU com TF-IDF + Logistic Regression
sobre o campo `SUMARIO` atingiu F1-macro > 0.95 — resultado que parecia resolver
o problema de forma direta.

### 2. A descoberta do vazamento (data leakage)

Uma auditoria sistemática revelou que o campo `SUMARIO` dos acórdãos TCU é redigido
pela Secretaria do Tribunal **após a decisão** e contém, literalmente, o texto
do veredito:

> "TOMADA DE CONTAS ESPECIAL. **CONTAS IRREGULARES.** DÉBITO. MULTA."

Usar esse campo como feature é equivalente a entregar a resposta ao modelo antes
da prova. A auditoria automatizada quantificou o problema:

| Campo | Contaminação | Gate |
|-------|:---:|:---:|
| `SUMARIO` | **98,1%** dos documentos contêm termo de veredito | FALHA |
| `VOTO` completo (com dispositivo) | 26,0% | FALHA |
| `VOTO_LIMPO` (sem dispositivo + mask) | **0,0%** | PASSA |

Os resultados anteriores com SUMARIO não refletiam capacidade preditiva real.

### 3. A correção

O pipeline foi **redesenhado do zero** para eliminar qualquer possibilidade de
vazamento, incorporando três princípios inegociáveis:

1. **`SUMARIO` nunca é feature** — usado apenas como fonte auxiliar de rótulo.
2. **Feature única = `VOTO_LIMPO`** — o voto do relator, truncado antes do
   dispositivo e com termos residuais de veredito mascarados como `[DECISAO]`.
3. **Gate obrigatório** — `auditar_vazamento(df, 'VOTO_LIMPO').gate_passou == True`
   antes de qualquer treino. Pipeline aborta se falhar.

A solução anti-vazamento opera em três camadas (defesa em profundidade):

```
VOTO (campo bruto do acórdão)
  │
  ├── strip_html()                      → remove tags HTML
  ├── remover_dispositivo()             → trunca no "ACORDAM...", "Diante do exposto..."
  └── mask termos residuais → [DECISAO] → captura fragmentos de veredito no corpo
  │
  ▼
VOTO_LIMPO  →  auditar_vazamento() == 0.0%  →  ✓ GATE PASSOU
```

### 4. Expansão e comparativo de modelos

Com a feature auditada e livre de contaminação, o escopo foi expandido para
**2016–2024** (9 anos, 3.644 acórdãos) e cinco arquiteturas foram avaliadas
em condições idênticas para determinar qual abordagem é mais eficaz na
classificação de textos jurídicos longos e desbalanceados.

---

## Modelos Comparados

Cinco arquiteturas avaliadas sobre `VOTO_LIMPO`, todas com pesos de classe:

| Modelo | Notebook | Abordagem | Cobertura do voto |
|--------|:---:|-----------|-------------------|
| **TF-IDF + LogReg** | 02 | Bag-of-words (50k features, bigramas) + linear | Documento inteiro |
| **TextCNN** | 03 | Embeddings + convoluções 1D (Kim, 2014) | 300 palavras |
| **LegalBert-pt + LoRA** | 04 | Fine-tuning BERT jurídico com adapters LoRA | 512 tokens (truncagem à direita) |
| **LegalBert Head+Tail** | 05 | BERT com início + fim do documento | 256 + 254 tokens |
| **Hierárquico** | 05 | Encoder por sentença + attention sobre sentenças | 48 sentenças × 128 tokens |

---

## Resultados

**Corpus:** 3.644 acórdãos (Saúde/Educação, 2016–2024).
**Distribuição:** Irregular 90,8% | Regular com Ressalva 6,6% | Regular 2,6%.
**Avaliação:** K-Fold estratificado 5× com pesos de classe.

| # | Modelo | F1-macro | IC 95% | Acurácia |
|---|--------|:---:|:---:|:---:|
| 1 | TF-IDF + LogReg `balanced` | **0.491** | [0.404, 0.579] | 0.913 |
| 2 | TextCNN ponderado | 0.367 | — | 0.885 |
| 3 | LegalBert Head+Tail + LoRA | 0.335 | [0.311, 0.359] | 0.878 |
| 4 | LegalBert Truncado + LoRA | 0.329 | [0.312, 0.346] | 0.857 |

**Hold-out temporal (treino ≤ 2022, teste = 2024):** F1-macro = 0.412 (baseline).

### F1 por classe (baseline — melhor modelo)

| Classe | F1 | Suporte |
|--------|:---:|:---:|
| Irregular | 0.958 | 3.310 |
| Regular com Ressalva | 0.399 | 241 |
| Regular | 0.116 | 93 |

### Interpretação

O baseline linear (TF-IDF + LogReg) supera todos os modelos profundos.
Três fatores explicam:

1. **Corpus pequeno + desbalanceamento extremo** — apenas ~240 exemplos de Ressalva e
   ~93 de Regular. Insuficiente para fine-tuning robusto de 110M parâmetros.
2. **Sinal predominantemente lexical** — presença/ausência de termos específicos é mais
   discriminativo que representações semânticas densas neste corpus.
3. **TF-IDF vê o documento inteiro** — sem truncagem. BERT limitado a 512 tokens
   perde informação em votos longos (mediana ~1.800 tokens).

A estratégia Head+Tail (início + fim do voto) deu ganho marginal (+0.006) sobre
truncagem simples — evidenciando que o gargalo principal não é perda de informação
posicional, mas insuficiência de exemplos para fine-tuning.

### Impacto da correção de vazamento

| Cenário | Feature | F1-macro |
|---------|---------|:---:|
| Antes da correção | `SUMARIO` (contaminado) | ~0.99 |
| Após correção | `VOTO_LIMPO` (auditado) | **0.491** |
| Sem pesos de classe | `VOTO_LIMPO` | 0.317 (classes raras colapsam a 0) |

A queda F1≈0.99 → 0.491 demonstra que os resultados anteriores eram inteiramente
inflados por leakage. O F1=0.491 é o resultado **honesto** — medido sobre feature
auditada, sem acesso ao veredito, com classes minoritárias efetivamente aprendidas.

---

## Pipeline

```
Etapa 1:  Download CSVs TCU (2016–2024)             → data/raw/
Etapa 2:  Filtro temático (Saúde/Educação, 45+ termos)
Etapa 3:  Rotulagem via SITUACAO/SUMARIO/ACORDAO    → LABEL (SUMARIO nunca vira feature)
Etapa 4:  construir_feature_voto(VOTO) → VOTO_LIMPO
Etapa 5:  GATE auditar_vazamento(VOTO_LIMPO) == 0   → aborta se falhar
Etapa 6:  Split temporal (treino ≤ 2022 | val=2023 | teste=2024)
Etapa 7:  TF-IDF + LogReg com class_weight='balanced'
Etapa 8:  TextCNN com CrossEntropyLoss ponderada
Etapa 9:  LegalBert-pt + LoRA (WeightedCE / Focal Loss)
Etapa 10: Head+Tail + Hierárquico (documentos longos)
Etapa 11: Métricas → resultados/metricas_*.json + comparativo final
```

---

## Estrutura

```
deep-acordao-tcu2/
├── src/
│   ├── aquisicao/baixar_csvs.py         ← download 2016–2024
│   ├── preprocessamento/
│   │   ├── anti_vazamento.py            ← gate + remoção de dispositivo + mask
│   │   ├── filtrar_tematico.py          ← carga + tema + label (45+ termos)
│   │   ├── limpeza.py                   ← tokenização TF-IDF / BERT head+tail
│   │   └── split_temporal.py            ← split temporal 2016–2024
│   ├── modelos/
│   │   ├── pesos.py                     ← cálculo de pesos de classe (usado por todos)
│   │   ├── baseline.py                  ← TF-IDF + LogReg (class_weight='balanced')
│   │   ├── textcnn.py                   ← TextCNN (CE ponderada, Kim 2014)
│   │   ├── focal_loss.py                ← FocalLoss (Lin et al., 2017)
│   │   ├── transformer.py               ← LegalBert-pt + LoRA + Weighted/Focal
│   │   └── hierarquico.py               ← Head+Tail + Hierárquico (docs longos)
│   └── avaliacao/metricas.py            ← métricas, matriz de confusão, JSON
├── notebooks/
│   ├── 00_visao_geral.ipynb
│   ├── 01_pipeline_limpo.ipynb           ← aquisição, filtro, VOTO_LIMPO, gate, split
│   ├── 02_baseline_ponderado.ipynb       ← TF-IDF + LogReg (5-fold + hold-out temporal)
│   ├── 03_textcnn_ponderado.ipynb        ← TextCNN com CE ponderada (5-fold)
│   ├── 04_legalbert_ponderado.ipynb      ← LegalBert-pt + LoRA (requer GPU)
│   └── 05_hierarquico_comparativo.ipynb  ← Head+Tail, Hierárquico, comparativo final
├── tests/test_pipeline.py               ← testes unitários (sem GPU/CSV real)
├── resultados/metricas_*.json           ← métricas dos experimentos
├── docs/
│   ├── referencias.md                   ← referências bibliográficas
│   └── decisoes.md                      ← decisões arquiteturais
└── data/                                ← CSVs e parquets (não versionados por tamanho)
```

---

## Instalação e Reprodução

```bash
git clone https://github.com/bsousa7/deep-acordao-tcu2.git
cd deep-acordao-tcu2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v
```

**No Google Colab (recomendado para notebooks com GPU):** abra `notebooks/` e
execute na ordem 01 → 02 → 03 → 04 → 05. Cada notebook persiste dados no
Google Drive para permitir execução em sessões separadas.

---

## Anti-vazamento — o gate

```python
from src.preprocessamento.anti_vazamento import (
    construir_feature_voto, auditar_vazamento,
)

feats = df["VOTO"].apply(construir_feature_voto)
df["VOTO_LIMPO"] = feats.apply(lambda f: f.texto)

resultado = auditar_vazamento(df, "VOTO_LIMPO")
assert resultado["gate_passou"], "Gate de vazamento falhou"
```

`construir_feature_voto` executa em sequência: remoção de HTML, truncagem no
primeiro marcador de dispositivo (ACORDAM, "Diante do exposto...", "Pelo exposto..."),
e substituição por `[DECISAO]` de termos residuais de veredito — defesa em profundidade.

---

## Ponderação de classes — o requisito

O corpus é fortemente desbalanceado (91% Irregular). Sem ponderação, os modelos
colapsam nas classes raras (F1 = 0 para Regular e Ressalva).

```python
from src.modelos.pesos import pesos_balanceados, pesos_tensor

# Sklearn (dict {classe: peso})
class_weight = pesos_balanceados(y_train)

# PyTorch (tensor para CrossEntropyLoss)
w = pesos_tensor(y_train_ids)
```

Fórmula: `w_c = N / (K * n_c)`, normalizada pela média para estabilizar a magnitude do loss.
Alternativa para desbalanceamento severo: `Focal Loss` (Lin et al., 2017).

---

## Referências

Ver `docs/referencias.md` — organizadas por domínio:

- **Domínio A:** Jurimetria preditiva e controle externo
  (Aletras et al., 2016; Medvedeva et al., 2020; Lage-Freitas et al., 2022; Tveita & Hustad, 2025).
- **Domínio B:** Modelos de linguagem e NLP jurídico
  (Vaswani et al., 2017; Devlin et al., 2019; Kim, 2014; Souza et al., 2020; Domingues, 2022).
- **Domínio C:** Adaptação, desbalanceamento e otimização
  (Hu et al., 2022; Sun et al., 2019; Lin et al., 2017; King & Zeng, 2001).

---

## Fonte de Dados

Portal de Dados Abertos do TCU — acórdãos completos em CSV oficiais:
<https://sites.tcu.gov.br/dados-abertos/jurisprudencia/>.
Uso conforme a política de dados abertos, sem scraping.
CSVs não versionados (`.gitignore`) por tamanho (~200–500 MB / ano),
baixados automaticamente pelo pipeline (notebook 01).
