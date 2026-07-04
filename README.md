# deep-acordao-tcu2

> Mestrado em Administração Pública — Ciência de Dados e IA
> Instituto Brasileiro de Ensino, Desenvolvimento e Pesquisa (IDP)
> **Autores:** Bruno Aires · Candice Trigueiro · Rafael Ayoroa

Classificação automática de desfechos em acórdãos do Tribunal de Contas da União (TCU)
nas áreas de **Saúde e Educação** — *Irregular*, *Regular com Ressalva* ou *Regular*.

Versão independente, construída a partir das descobertas do projeto original
[`bsousa7/deep-acordao-tcu`](https://github.com/bsousa7/deep-acordao-tcu).
Pipeline redesenhado **sem vazamento desde o primeiro passo**, com ponderação de
classes obrigatória e escopo temporal expandido (2016–2024).

---

## A Jornada: da v1 à v2

### 1. Ponto de partida — o projeto original (v1)

O [`deep-acordao-tcu`](https://github.com/bsousa7/deep-acordao-tcu) alcançou
F1-macro ~0.79 com TF-IDF + Logistic Regression usando o campo `SUMARIO` como
feature — resultado promissor que motivou a investigação com modelos mais
sofisticados (LegalBert-pt + LoRA).

### 2. A descoberta do vazamento (data leakage)

Auditoria sistemática revelou que o `SUMARIO` dos acórdãos TCU é redigido
**após o julgamento** e contém, literalmente, o texto do veredito:

| Campo | Contaminação | Gate |
|-------|:---:|:---:|
| `SUMARIO` | **98,1%** dos documentos contêm termo de veredito | FALHA |
| `VOTO` completo (com dispositivo) | 26,0% | FALHA |
| `VOTO_LIMPO` (sem dispositivo + mask) | **0,0%** | PASSA |

O resultado de F1≈0.99 com SUMARIO não refletia capacidade preditiva real —
o modelo simplesmente lia a resposta que já estava no texto.

### 3. A correção — este projeto (v2)

Em vez de *consertar* o pipeline antigo, **redesenhamos do zero** com três
princípios inegociáveis:

1. **`SUMARIO` nunca é feature** — usado apenas como fonte auxiliar de rótulo.
2. **Feature única = `VOTO_LIMPO`** — o voto do relator, truncado antes do
   dispositivo e com termos residuais de veredito mascarados como `[DECISAO]`.
3. **Gate obrigatório** — `auditar_vazamento(df, 'VOTO_LIMPO').gate_passou == True`
   antes de qualquer treino. Pipeline aborta se falhar.

### 4. Evolução em relação à v1

| # | Aspecto | v1 (referência) | **v2 (este projeto)** |
|---|---|---|---|
| 1 | Papel do `SUMARIO` | Feature TF-IDF/BERT + fonte de rótulo | **Apenas** fonte de rótulo. Nunca vira feature. |
| 2 | Feature de treino | `SUMARIO`, `VOTO` completo, `VOTO_LIMPO` | **Única:** `VOTO_LIMPO`. |
| 3 | Anti-vazamento | Descoberto e aplicado como *fix* posterior | **Aplicado no primeiro passo.** Gate obrigatório. |
| 4 | Ponderação de classes | Configurável | **Requisito** em todos os modelos. |
| 5 | Escopo temporal | 2024 (1.622 acórdãos) | **2016–2024** (3.644 acórdãos, 9 anos). |
| 6 | Modelos | TF-IDF + LogReg, LegalBert-pt | TF-IDF + LogReg, **TextCNN**, LegalBert-pt, **Head+Tail**, **Hierárquico**. |
| 7 | Resultados | Gerados com leakage; depois corrigidos | **Nunca tiveram leakage** — honestos desde o início. |

---

## O Vazamento Corrigido — em detalhe

O campo `SUMARIO` do acórdão TCU é escrito pela Secretaria do Tribunal
**após a decisão**. Ele tipicamente contém frases como:

> "TOMADA DE CONTAS ESPECIAL. **CONTAS IRREGULARES.** DÉBITO. MULTA."

Usar esse campo como feature é equivalente a dar ao modelo a resposta antes da
prova. A auditoria automatizada (`auditar_vazamento`) busca por uma lista de
termos de veredito (ex.: "contas irregulares", "contas regulares com ressalva",
"contas regulares") e contabiliza a fração de documentos contaminados.

A solução em três camadas:

```
VOTO (campo bruto)
  │
  ├── strip_html()                      → remove tags HTML
  ├── remover_dispositivo()             → trunca no "ACORDAM...", "Diante do exposto..."
  └── mask termos residuais → [DECISAO] → defesa em profundidade
  │
  ▼
VOTO_LIMPO  →  auditar_vazamento() == 0.0%  →  ✓ GATE PASSOU
```

---

## Modelos Comparados

Cinco arquiteturas avaliadas sobre `VOTO_LIMPO`, todas com pesos de classe:

| Modelo | Notebook | Abordagem | Cobertura do voto |
|--------|:---:|-----------|-------------------|
| **TF-IDF + LogReg** | 02 | Bag-of-words (50k features) + linear | Documento inteiro |
| **TextCNN** | 03 | Embeddings + conv 1D (Kim, 2014) | 300 palavras (truncagem) |
| **LegalBert-pt + LoRA** | 04 | Fine-tuning BERT jurídico com adapters | 512 tokens (truncagem à direita) |
| **LegalBert Head+Tail** | 05 | BERT com início + fim do documento | 256 + 254 tokens (início + fim) |
| **Hierárquico** | 05 | Encoder por sentença + attention | 48 sentenças × 128 tokens |

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

### Interpretação

O baseline linear (TF-IDF + LogReg) supera todos os modelos profundos. Três fatores explicam:

1. **Corpus pequeno + desbalanceamento extremo** — apenas ~240 exemplos de Ressalva e
   ~93 de Regular. Insuficiente para fine-tuning de 110M parâmetros (mesmo com LoRA).
2. **Sinal predominantemente lexical** — presença/ausência de termos específicos é mais
   discriminativo que representações semânticas densas neste corpus.
3. **TF-IDF vê o documento inteiro** — sem truncagem. BERT limitado a 512 tokens perde
   70%+ dos votos longos.

A estratégia Head+Tail (início + fim do voto) deu ganho marginal (+0.006) sobre truncagem
simples — evidenciando que o gargalo não é perda de informação posicional, mas sim
quantidade de dados de treino.

### Contraste com a v1

| Versão | Feature | F1-macro | Leakage |
|--------|---------|:---:|:---:|
| v1 — SUMARIO | `SUMARIO` | ~0.99 | 98,1% contaminado |
| v1 — VOTO corrigido | `VOTO_sem_dispositivo` | 0.539 | 0% |
| **v2 (este projeto)** | `VOTO_LIMPO` | **0.491** | 0% |

A diferença v1 corrigido (0.539) → v2 (0.491) se deve ao escopo expandido (2016–2024 vs
apenas 2024) e ao filtro temático mais abrangente (45+ termos) que traz maior diversidade
de acórdãos — tornando a tarefa mais difícil e os resultados mais generalizáveis.

---

## Pipeline

```
Etapa 1:  Download CSVs TCU (2016–2024)             → data/raw/
Etapa 2:  Filtro temático (Saúde/Educação)          → ASSUNTO/SUMARIO
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
│   │   ├── limpeza.py                   ← TF-IDF / BERT head+tail
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
│   ├── 01_pipeline_limpo.ipynb
│   ├── 02_baseline_ponderado.ipynb
│   ├── 03_textcnn_ponderado.ipynb
│   ├── 04_legalbert_ponderado.ipynb      (requer GPU)
│   └── 05_hierarquico_comparativo.ipynb  (requer GPU, comparativo final)
├── tests/test_pipeline.py               ← testes unitários (sem GPU/CSV real)
├── resultados/metricas_*.json           ← métricas dos experimentos
├── docs/{referencias,decisoes}.md
└── data/                                ← CSVs e parquets (não versionados)
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

**No Google Colab (recomendado):** abra `notebooks/` e execute na ordem
01 → 02 → 03 → 04 → 05. Cada notebook persiste dados no Google Drive
para permitir execução em sessões separadas.

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

## Ponderação de classes — o requisito

O corpus é fortemente desbalanceado (91% Irregular). Sem ponderação, os modelos
colapsam nas classes raras (recall = 0 para Regular e Ressalva).

```python
from src.modelos.pesos import pesos_balanceados, pesos_tensor

# Sklearn (dict {classe: peso})
class_weight = pesos_balanceados(y_train)

# PyTorch (tensor para CrossEntropyLoss)
w = pesos_tensor(y_train_ids)
```

Fórmula: `w_c = N / (K * n_c)`, normalizada pela média.
Alternativa: `Focal Loss` (Lin et al., 2017) para desbalanceamento severo.

---

## Referências

Ver `docs/referencias.md` — organizadas por domínio:

- **Domínio A:** Jurimetria preditiva e controle externo.
- **Domínio B:** Modelos de linguagem e NLP jurídico.
- **Domínio C:** Adaptação, desbalanceamento e otimização.

## Fonte de Dados

Portal de Dados Abertos do TCU — acórdãos completos em CSV oficiais:
<https://sites.tcu.gov.br/dados-abertos/jurisprudencia/>.
Uso conforme a política de dados abertos, sem scraping.
CSVs não versionados (`.gitignore`) por tamanho (~200–500 MB / ano).
