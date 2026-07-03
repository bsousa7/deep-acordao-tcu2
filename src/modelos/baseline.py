"""
Baseline TF-IDF + Regressão Logística — sempre ponderado, sempre sobre VOTO_LIMPO.

Nunca treina sobre SUMARIO. Nunca treina sem `class_weight`. A intenção é que
os números baseline sejam honestos por construção.
"""

import logging
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .pesos import NOMES_CLASSES, pesos_balanceados

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_SPLITS = 5

TFIDF_PARAMS = dict(
    max_features=50_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)


def construir_pipeline(class_weight: dict[str, float] | str | None = "balanced") -> Pipeline:
    """
    Pipeline TF-IDF + LogisticRegression com pesos de classe.

    Aceita:
      * dict {classe: peso}      — típico da `pesos_balanceados(y)`.
      * 'balanced'               — sklearn calcula do próprio y (padrão).
      * None                     — sem ponderação (só para experimentos de contraste).
    """
    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    clf = LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="lbfgs",
        random_state=RANDOM_STATE,
        class_weight=class_weight,
    )
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def _ic95(scores: list[float]) -> tuple[float, float]:
    n = len(scores)
    mean = float(np.mean(scores))
    se = float(np.std(scores, ddof=1)) / sqrt(n)
    try:
        from scipy.stats import t as t_dist

        lo, hi = t_dist.interval(0.95, df=n - 1, loc=mean, scale=se)
    except ImportError:
        t_crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(n - 1, 2.0)
        lo, hi = mean - t_crit * se, mean + t_crit * se
    return float(lo), float(hi)


def treinar_kfold(
    X: pd.Series,
    y: pd.Series,
    n_splits: int = N_SPLITS,
    class_weight: dict[str, float] | str | None = "balanced",
) -> dict:
    """
    Treina TF-IDF + LogReg com StratifiedKFold. Retorna métricas consolidadas.

    Se `class_weight='balanced'`, o sklearn calcula os pesos por fold internamente.
    Se for dict, aplicamos os pesos globais (calculados do y total antes do CV).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    X_arr = X.values
    y_arr = y.values

    fold_f1: list[float] = []
    fold_acc: list[float] = []
    per_class_f1 = {c: [] for c in NOMES_CLASSES}
    cm_total = np.zeros((len(NOMES_CLASSES), len(NOMES_CLASSES)), dtype=int)

    for fold_idx, (tr, va) in enumerate(skf.split(X_arr, y_arr), start=1):
        pipe = construir_pipeline(class_weight=class_weight)
        pipe.fit(X_arr[tr], y_arr[tr])
        y_pred = pipe.predict(X_arr[va])
        y_true = y_arr[va]

        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        fold_f1.append(float(f1))
        fold_acc.append(float(acc))
        cm_total += confusion_matrix(y_true, y_pred, labels=NOMES_CLASSES)

        pc = f1_score(y_true, y_pred, average=None, labels=NOMES_CLASSES, zero_division=0)
        for i, cls in enumerate(NOMES_CLASSES):
            per_class_f1[cls].append(float(pc[i]))

        logger.info("Fold %d/%d — F1-macro=%.4f | Acurácia=%.4f", fold_idx, n_splits, f1, acc)

    mean_f1 = float(np.mean(fold_f1))
    std_f1 = float(np.std(fold_f1, ddof=1))
    lo, hi = _ic95(fold_f1)
    logger.info("Resultado — F1-macro %.4f ± %.4f | IC95 [%.4f, %.4f]", mean_f1, std_f1, lo, hi)

    return {
        "fold_scores": fold_f1,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "ci_95": (lo, hi),
        "mean_acc": float(np.mean(fold_acc)),
        "per_class_f1": {c: float(np.mean(v)) for c, v in per_class_f1.items()},
        "confusion_matrix": cm_total.tolist(),
        "class_weight": (
            class_weight if class_weight is None or isinstance(class_weight, str)
            else {k: round(v, 4) for k, v in class_weight.items()}
        ),
    }


def treinar_holdout(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    class_weight: dict[str, float] | str | None = "balanced",
) -> tuple[Pipeline, dict]:
    """Treino/teste hold-out (usado com split temporal). Retorna (pipeline, metricas)."""
    if class_weight == "balanced-explicito":
        class_weight = pesos_balanceados(y_train)

    pipe = construir_pipeline(class_weight=class_weight)
    pipe.fit(X_train.values, y_train.values)
    y_pred = pipe.predict(X_test.values)

    metricas = {
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_por_classe": {
            c: float(v)
            for c, v in zip(
                NOMES_CLASSES,
                f1_score(y_test, y_pred, average=None, labels=NOMES_CLASSES, zero_division=0),
            )
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=NOMES_CLASSES).tolist(),
        "class_weight": class_weight if isinstance(class_weight, (str, type(None))) else {
            k: round(v, 4) for k, v in class_weight.items()
        },
    }
    logger.info(
        "Hold-out — F1-macro=%.4f | Acurácia=%.4f",
        metricas["f1_macro"], metricas["accuracy"],
    )
    return pipe, metricas
