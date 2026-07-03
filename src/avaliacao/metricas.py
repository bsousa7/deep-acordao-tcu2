"""Métricas, matriz de confusão e persistência de resultados em JSON."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

NOMES_CLASSES = ["Irregular", "Regular com Ressalva", "Regular"]
RESULTADOS_DIR = Path(__file__).resolve().parents[2] / "resultados"


def relatorio(y_true, y_pred, titulo: str = "") -> dict:
    rep = classification_report(
        y_true, y_pred, target_names=NOMES_CLASSES, output_dict=True, zero_division=0,
    )
    if titulo:
        logger.info("=== %s ===", titulo)
        logger.info(
            "\n" + classification_report(
                y_true, y_pred, target_names=NOMES_CLASSES, zero_division=0
            )
        )
    return rep


def resumo_metricas(y_true, y_pred) -> dict:
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_por_classe": {
            c: float(v) for c, v in zip(
                NOMES_CLASSES,
                f1_score(y_true, y_pred, average=None, labels=NOMES_CLASSES, zero_division=0),
            )
        },
        "recall_por_classe": {
            c: float(v) for c, v in zip(
                NOMES_CLASSES,
                recall_score(y_true, y_pred, average=None, labels=NOMES_CLASSES, zero_division=0),
            )
        },
        "precision_por_classe": {
            c: float(v) for c, v in zip(
                NOMES_CLASSES,
                precision_score(y_true, y_pred, average=None, labels=NOMES_CLASSES, zero_division=0),
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=NOMES_CLASSES).tolist(),
    }


def plotar_matriz_confusao(y_true, y_pred, titulo: str = "", output_dir: Path | None = None):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn não disponíveis — pulando matriz de confusão")
        return

    if output_dir is None:
        output_dir = RESULTADOS_DIR / "figuras"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=NOMES_CLASSES)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, sfx in zip(axes, [cm, cm_norm], ["d", ".2f"], ["Absoluta", "Recall"]):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=NOMES_CLASSES, yticklabels=NOMES_CLASSES, ax=ax,
        )
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de Confusão — {sfx}" + (f" ({titulo})" if titulo else ""))

    fig.tight_layout()
    nome = f"cm_{titulo.replace(' ', '_').replace('/', '_')}.png" if titulo else "cm.png"
    fig.savefig(output_dir / nome, dpi=150)
    plt.close(fig)
    logger.info("Figura salva: %s", output_dir / nome)


def comparar_modelos(resultados: dict[str, dict]) -> pd.DataFrame:
    linhas = []
    for nome, res in resultados.items():
        ci = res.get("ci_95", (None, None))
        linhas.append(
            {
                "Modelo/Campo": nome,
                "F1-macro": round(res.get("mean_f1", res.get("f1_macro", 0)), 4),
                "± std": round(res.get("std_f1", 0), 4),
                "IC95": f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci[0] is not None else "—",
                "Acurácia": round(res.get("mean_acc", res.get("accuracy", 0)), 4),
            }
        )
    df = pd.DataFrame(linhas).sort_values("F1-macro", ascending=False).reset_index(drop=True)
    logger.info("\n=== Comparação de Modelos ===\n%s", df.to_string(index=False))
    return df


def salvar_json(dados: dict, caminho: Path):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"não serializável: {type(o)}")

    with open(caminho, "w", encoding="utf-8") as fh:
        json.dump(dados, fh, indent=2, ensure_ascii=False, default=_default)
    logger.info("Salvo: %s", caminho)
