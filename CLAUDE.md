# deep-acordao-tcu2 — Guia para Claude Code

## O Projeto

Jurimetria preditiva em acórdãos TCU (Saúde e Educação): classificar desfecho em
`Irregular`, `Regular com Ressalva` ou `Regular` — usando **exclusivamente**
`VOTO_LIMPO` como feature.

Refatoração v2 do projeto `bsousa7/deep-acordao-tcu`: pipeline sem vazamento
desde o primeiro passo, com ponderação de classes como requisito, sobre 2016–2024.

## Guardrails (Regras Inegociáveis)

1. **`SUMARIO` nunca é feature.** É apenas fonte auxiliar de rótulo.
2. **Feature única = `VOTO_LIMPO`** = `construir_feature_voto(VOTO)`.
3. **Gate `auditar_vazamento(df, 'VOTO_LIMPO').gate_passou`** antes de qualquer treino.
4. **Ponderação de classes** em todo classificador — `class_weight='balanced'`
   (sklearn) ou `pesos_tensor(y)` (PyTorch).
5. **Escopo temporal padrão = 2016–2024**; split temporal (val=2023, teste=2024).
6. **`RANDOM_STATE = 42`** em tudo.
7. **Sem scraping**; apenas CSVs oficiais do Portal de Dados Abertos do TCU.
8. **`usecols=[...]` obrigatório** em `pd.read_csv` — CSVs pesam 175–445 MB.
9. **Sem inventar referências** — apenas as listadas em `docs/referencias.md`.
10. **Sem versionar resultados** — `resultados/*.json`, figuras e modelos são regenerados pelos notebooks.

## Comandos Principais

```bash
pip install -r requirements.txt
python -m pytest tests/ -v            # 13 testes passam sem GPU/CSV real
python -m pytest tests/ --cov=src

ruff check src/ tests/
black src/ tests/
```

## Estrutura de Pastas

```
src/aquisicao/        — download CSVs TCU (2016–2024)
src/preprocessamento/ — anti_vazamento, filtro temático, limpeza, split temporal
src/modelos/          — pesos, baseline (TF-IDF), textcnn, focal_loss, transformer
src/avaliacao/        — métricas, matriz de confusão, JSON
notebooks/            — 00 visão geral, 01 pipeline, 02 baseline, 03 textcnn
tests/                — testes unitários sem CSV real
docs/                 — referências e decisões
```

## Decisões v2 (D2-01 a D2-08) — Ver `docs/decisoes.md`

Não revisitar sem alinhamento explícito.

| # | Decisão |
|---|---|
| D2-01 | SUMARIO só serve para extrair rótulo. |
| D2-02 | Feature única = VOTO_LIMPO. |
| D2-03 | Gate anti-vazamento obrigatório. |
| D2-04 | Ponderação de classes é requisito. |
| D2-05 | Escopo temporal 2016–2024, split temporal. |
| D2-06 | Sem carregar resultados antigos. |
| D2-07 | RANDOM_STATE=42. |
| D2-08 | Sem scraping / sem inventar refs. |

## Estrutura do CSV TCU

| Coluna | Uso |
|---|---|
| `NUMACORDAO` | Identificador |
| `SITUACAO`   | Rótulo (formato antigo; formato atual=OFICIALIZADO) |
| `SUMARIO`    | Fonte auxiliar de rótulo — **nunca feature** |
| `ACORDAO`    | Dispositivo — fallback de rótulo |
| `VOTO`       | Texto longo — **entrada única do modelo (após limpeza)** |
| `ASSUNTO`    | Filtro temático |
| `ANO`        | Adicionado pelo pipeline (do nome do arquivo) |

## Parâmetros Testados

### TF-IDF baseline
```python
TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42, class_weight="balanced")
```

### TextCNN
```python
embed_dim=100, filter_sizes=(3,4,5), num_filters=64, dropout=0.5
max_vocab=20_000, max_len=300, epochs=8, batch_size=32, lr=1e-3
CrossEntropyLoss(weight=pesos_tensor(y_tr))
```

### LegalBert-pt + LoRA
```python
epocas=5, batch_size=16, lr=3e-5, weight_decay=0.01, warmup_ratio=0.10
LoRA(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["query","value"])
loss='weighted_ce' | 'focal'  (padrão: weighted_ce; focal para desbalanceamento severo)
```
