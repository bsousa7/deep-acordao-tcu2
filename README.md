# deep-acordao-tcu2

> Mestrado em Administração Pública — Ciência de Dados e IA
> Instituto Brasileiro de Ensino, Desenvolvimento e Pesquisa (IDP)
> **Autores:** Bruno Aires · Candice Trigueiro · Rafael Ayoroa

Classificação automática de desfechos em acórdãos do Tribunal de Contas da União (TCU)
nas áreas de **Saúde e Educação** — *Irregular*, *Regular com Ressalva* ou *Regular*.

Refatoração limpa do projeto original [`bsousa7/deep-acordao-tcu`](https://github.com/bsousa7/deep-acordao-tcu),
sem vazamento desde o primeiro passo e com ponderação de classes como requisito.

---

## O que muda em relação à versão anterior

| # | Aspecto | v1 (referência) | **v2 (este projeto)** |
|---|---|---|---|
| 1 | Papel do `SUMARIO` | Feature TF-IDF/BERT + fonte de rótulo (leakage 100 %) | **Apenas** fonte de rótulo. Nunca vira feature. |
| 2 | Feature de treino | `SUMARIO`, `VOTO` completo, `VOTO_LIMPO` | **Única:** `VOTO_LIMPO`. |
| 3 | Anti-vazamento | Aplicado como *fix* posterior | **Aplicado no primeiro passo** do pipeline. Gate obrigatório. |
| 4 | Ponderação de classes | Configurável | **Requisito** em todos os modelos (baseline, TextCNN, LegalBert). |
| 5 | Escopo temporal | 2020–2024 (padrão), 2024 nos artefatos | **2016–2024** por padrão, com split temporal (val=2023, teste=2024). |
| 6 | Resultados anteriores | Versionados no repositório | **Regenerados** pelos notebooks — nada versionado em `resultados/`. |

---

## Pipeline

```
Etapa 1: Download CSVs TCU (2016–2024)             → data/raw/
Etapa 2: Filtro temático + rotulagem via SUMARIO   → LABEL (SUMARIO nunca vira feature)
Etapa 3: construir_feature_voto() → VOTO_LIMPO      → data/interim/acordaos_rotulados.parquet
Etapa 4: GATE auditar_vazamento(VOTO_LIMPO) == 0    → aborta se falhar
Etapa 5: Split temporal (treino ≤ 2022 | val=2023 | teste=2024) → data/processed/
Etapa 6: TF-IDF + LogReg com class_weight='balanced' (5-fold + hold-out temporal)
Etapa 7: TextCNN com CrossEntropyLoss ponderada (5-fold)
Etapa 8 (opcional): LegalBert-pt + LoRA com WeightedCE ou Focal Loss (GPU)
Etapa 9 (opcional): Head+Tail + Hierárquico para documentos longos (GPU)
Etapa 10: Métricas em resultados/metricas_*.json + comparativo final
```

---

## Estrutura

```
deep-acordao-tcu2/
├── src/
│   ├── aquisicao/baixar_csvs.py         ← download 2016–2024
│   ├── preprocessamento/
│   │   ├── anti_vazamento.py            ← gate + remoção de dispositivo + mask
│   │   ├── filtrar_tematico.py          ← carga + tema + label
│   │   ├── limpeza.py                   ← TF-IDF / BERT head+tail
│   │   └── split_temporal.py            ← split temporal 2016–2024
│   ├── modelos/
│   │   ├── pesos.py                     ← cálculo de pesos de classe (usado por todos)
│   │   ├── baseline.py                  ← TF-IDF + LogReg (class_weight='balanced')
│   │   ├── textcnn.py                   ← TextCNN (CE ponderada)
│   │   ├── focal_loss.py                ← FocalLoss (Lin et al., 2017)
│   │   ├── transformer.py               ← LegalBert-pt + LoRA + Weighted/Focal
│   │   └── hierarquico.py               ← Head+Tail + Hierárquico (docs longos)
│   └── avaliacao/metricas.py            ← métricas, matriz de confusão, JSON
├── notebooks/
│   ├── 00_visao_geral.ipynb
│   ├── 01_pipeline_limpo.ipynb
│   ├── 02_baseline_ponderado.ipynb
│   ├── 03_textcnn_ponderado.ipynb
│   ├── 04_legalbert_ponderado.ipynb      (opcional, requer GPU)
│   └── 05_hierarquico_comparativo.ipynb  (requer GPU, comparativo final)
├── tests/test_pipeline.py               ← 13 testes unitários (sem GPU/CSV real)
├── docs/{referencias,decisoes}.md
└── resultados/                          ← vazio até a execução dos notebooks
```

---

## Como reproduzir os resultados

```bash
git clone https://github.com/bsousa7/deep-acordao-tcu2.git
cd deep-acordao-tcu2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v          # 13 testes passam sem CSV/GPU

# Em Colab (recomendado para GPU): abra notebooks/ e execute na ordem 01 → 02 → 03 → 04 → 05.
# Cada notebook regenera seu resultados/metricas_*.json. O 05 consolida o comparativo final.
```

## Resultados (corpus: 3.644 acórdãos, 2016–2024, Saúde/Educação)

Distribuição de classes: **Irregular 91%** | Regular com Ressalva 6,6% | Regular 2,4%.

| # | Modelo (notebook) | F1-macro | IC 95% | Acurácia |
|---|---|---|---|---|
| 1 | TF-IDF + LogReg `balanced` (02) | **0.491** | [0.404, 0.579] | 0.913 |
| 2 | TextCNN ponderado (03) | 0.367 | — | 0.885 |
| 3 | LegalBert Head+Tail + LoRA (05) | 0.335 | [0.311, 0.359] | 0.878 |
| 4 | LegalBert Truncado + LoRA (04) | 0.329 | [0.312, 0.346] | 0.857 |

> Todos os modelos usam **exclusivamente `VOTO_LIMPO`** (sem leakage do SUMARIO),
> com **pesos de classe** obrigatórios. Avaliação por K-Fold estratificado (5×).

### Interpretação

- O baseline linear (TF-IDF + LogReg) supera modelos profundos neste corpus — resultado
  esperado em cenários de poucos dados + desbalanceamento extremo.
- O sinal discriminativo é predominantemente **lexical** (presença/ausência de termos),
  não semântico — favorece bag-of-words sobre o documento inteiro.
- A estratégia Head+Tail (+0.005 sobre truncagem simples) indica que o gargalo não é
  truncagem, mas sim insuficiência de exemplos para fine-tuning (110M parâmetros).
- Eliminar o vazamento (SUMARIO → VOTO_LIMPO) torna a tarefa genuinamente difícil:
  o SUMARIO sozinho dava F1≈0.99 na v1 — evidenciando que os resultados anteriores
  eram inflados por leakage.

Comprometimento com honestidade metodológica: o número que sai é o número que o
código produziu, medido sobre feature auditada como livre de vazamento.

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

`construir_feature_voto` faz três coisas em sequência: remove HTML, trunca no primeiro
marcador de dispositivo do voto (ACORDAM, "Diante do exposto…", "Pelo exposto…"), e
substitui por `[DECISAO]` qualquer termo de veredito residual — defesa em profundidade.

## Ponderação de classes — o requisito

```python
from src.modelos.pesos import pesos_balanceados, pesos_tensor

# Baseline (dict {classe: peso})
pipe = construir_pipeline(class_weight=pesos_balanceados(y_train))

# TextCNN / LegalBert (tensor por ID)
w = pesos_tensor(y_train_ids)              # np.ndarray[num_classes]
```

Todos os treinos deste projeto usam pesos calculados por classe. Alternativa:
`Focal Loss` (Lin et al., 2017) para desbalanceamento severo.

---

## Referências

Ver `docs/referencias.md` — herdadas do projeto original, agrupadas por
domínio (Jurimetria, NLP jurídico, Adaptação/Desbalanceamento).

## Fonte de Dados

Portal de Dados Abertos do TCU — acórdãos completos em CSV oficiais:
<https://sites.tcu.gov.br/dados-abertos/jurisprudencia/>. Uso conforme a política de
dados abertos, sem scraping. CSVs não versionados (`.gitignore`) por tamanho
(~200–500 MB / ano), baixados pelo pipeline.
