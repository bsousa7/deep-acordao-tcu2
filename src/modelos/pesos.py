"""
Cálculo de pesos por classe para regular o desbalanceamento.

O corpus TCU é fortemente enviesado para 'Irregular' (>90% em alguns temas).
Sem ponderação, os classificadores colapsam nas classes raras (recall=0).
Este módulo centraliza a lógica de ponderação usada em todos os modelos.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

NOMES_CLASSES = ["Irregular", "Regular com Ressalva", "Regular"]
LABEL2ID = {c: i for i, c in enumerate(NOMES_CLASSES)}
ID2LABEL = {i: c for i, c in enumerate(NOMES_CLASSES)}


def pesos_balanceados(y: list[str] | list[int] | np.ndarray) -> dict[str, float]:
    """
    Retorna dict {classe_str: peso} para uso em `class_weight` do sklearn.

    Fórmula: n_amostras / (n_classes * n_amostras_da_classe) — normalizada
    pela média para manter a escala do loss próxima de 1.
    """
    y_arr = np.asarray(y)
    y_str = np.array([c if isinstance(c, str) else ID2LABEL[int(c)] for c in y_arr])
    classes = np.array([c for c in NOMES_CLASSES if c in set(y_str.tolist())])
    if len(classes) == 0:
        return {c: 1.0 for c in NOMES_CLASSES}
    pesos = compute_class_weight("balanced", classes=classes, y=y_str)
    pesos = pesos / pesos.mean()
    saida = {c: 1.0 for c in NOMES_CLASSES}
    for cls, w in zip(classes, pesos):
        saida[cls] = float(w)
    return saida


def pesos_tensor(y_ids: list[int] | np.ndarray) -> "np.ndarray":
    """Retorna np.ndarray[num_classes] com pesos balanceados por ID."""
    y_arr = np.asarray(y_ids).astype(int)
    counts = np.bincount(y_arr, minlength=len(NOMES_CLASSES)).astype(float)
    counts = np.clip(counts, 1.0, None)
    pesos = counts.sum() / (len(NOMES_CLASSES) * counts)
    pesos = pesos / pesos.mean()
    return pesos.astype(np.float32)
