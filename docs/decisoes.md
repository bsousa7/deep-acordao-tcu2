# Decisões Arquiteturais — deep-acordao-tcu2

Registro conciso das decisões de projeto **desta versão v2**. As decisões da versão
anterior (D-01 a D-08) permanecem apenas como referência histórica.

## D2-01 — SUMARIO não é feature (definição)

O `SUMARIO` do acórdão TCU é redigido após o julgamento e contém o veredito escrito
literalmente ("CONTAS IRREGULARES", "CONTAS REGULARES COM RESSALVA"). Auditoria: 100%
dos sumários contêm termo de veredito.

**Decisão:** SUMARIO é **exclusivamente** fonte de rótulo (via `extrair_rotulo`).
Não existe pipeline `SUMARIO_TFIDF` nem `SUMARIO_BERT` neste projeto.

## D2-02 — Feature única: VOTO_LIMPO

`VOTO_LIMPO = construir_feature_voto(VOTO)`:
1. `strip_html()` — remove tags HTML e colapsa whitespace.
2. `remover_dispositivo()` — trunca no primeiro marcador de dispositivo (ACORDAM,
   "Diante do exposto…", "Pelo exposto…" etc.).
3. Substituição por `[DECISAO]` de qualquer termo de veredito residual (defesa em
   profundidade contra fragmentos citados no corpo analítico).

## D2-03 — Gate de vazamento obrigatório

Antes de qualquer treino: `auditar_vazamento(df, 'VOTO_LIMPO').gate_passou == True`.
`fracao_vazado > 0` interrompe o pipeline via `assert`.

## D2-04 — Ponderação de classes é requisito, não opção

O corpus é fortemente enviesado para *Irregular*. Sem ponderação, os classificadores
colapsam nas classes raras (recall ≈ 0).

**Requisitos:**
- **Baseline** (`src/modelos/baseline.py`) — TF-IDF + LogReg com `class_weight='balanced'`.
- **TextCNN** (`src/modelos/textcnn.py`) — `CrossEntropyLoss(weight=pesos_tensor(y_tr))`
  recalculado a cada fold.
- **LegalBert-pt** (`src/modelos/transformer.py`) — `WeightedTrainer` (cross-entropy
  ponderada) ou `FocalTrainer` (Lin et al., 2017) com `alpha` = pesos por frequência inversa.

Fórmula: `w_c = N / (K * n_c)`, normalizada pela média para estabilizar a magnitude do loss.

## D2-05 — Escopo temporal 2016–2024

Nove anos (2016–2024) executados em pipeline único. Split padrão:
- Treino: anos ≤ 2022.
- Val: 2023.
- Teste: 2024.

`dividir_estratificado()` continua disponível como fallback para corpora pequenos.

## D2-06 — Resultados versionados, figuras regeneradas

Arquivos `resultados/*.json` são versionados no git (métricas finais dos experimentos).
Figuras (`resultados/figuras/`) e checkpoints de modelos não são versionados —
regenerados pelos notebooks quando executados.

## D2-07 — Random state fixo

`RANDOM_STATE = 42` em todo split, treino e inicialização de modelos.

## D2-08 — Sem scraping, sem SUMARIO como feature, sem invenção de referências

Regras não-negociáveis herdadas do projeto original.
