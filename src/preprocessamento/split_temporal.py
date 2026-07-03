"""
Split temporal para o pipeline 2016-2024.

Estratégias:
  * `dividir_temporal_por_ano(df, ano_teste=2024, ano_val=2023)`:
      - treino = anos < ano_val
      - val   = ano_val (2023)
      - teste = ano_teste (2024)
      Split honesto, sem embaralhamento entre anos — evita que o modelo
      "veja o futuro" pelo random split estratificado.
  * `dividir_estratificado(df)`:
      - 70/15/15 estratificado por LABEL (útil para K-Fold e para corpora
        pequenos, quando o split temporal é inviável).

Nada de leakage adicional: sempre garanta que `construir_feature_voto()`
foi aplicado ANTES de qualquer split.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


def dividir_temporal_por_ano(
    df: pd.DataFrame,
    ano_teste: int = 2024,
    ano_val: int = 2023,
    col_ano: str = "ANO",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split temporal: treino = anos < ano_val, val = ano_val, teste = ano_teste.

    Assume df com coluna ANO. Não embaralha — a ordem cronológica é o próprio
    critério de generalização.
    """
    if col_ano not in df.columns:
        raise ValueError(f"Coluna '{col_ano}' ausente. Necessária para split temporal.")

    train = df[df[col_ano] < ano_val].reset_index(drop=True)
    val = df[df[col_ano] == ano_val].reset_index(drop=True)
    test = df[df[col_ano] == ano_teste].reset_index(drop=True)

    for nome, split in [("treino", train), ("val", val), ("teste", test)]:
        anos_split = sorted(split[col_ano].unique().tolist()) if len(split) else []
        logger.info(
            "Split temporal %s: %d amostras | anos=%s | classes=%s",
            nome, len(split), anos_split, split.get("LABEL", pd.Series()).value_counts().to_dict(),
        )
    return train, val, test


def dividir_estratificado(
    df: pd.DataFrame,
    col_label: str = "LABEL",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split estratificado 70/15/15, seed=42. Falha se classe tiver < 3 amostras."""
    contagens = df[col_label].value_counts()
    n_min = 3
    classes_pequenas = contagens[contagens < n_min]
    if not classes_pequenas.empty:
        raise ValueError(
            f"Classes com amostras insuficientes para estratificação "
            f"(mínimo {n_min}): {classes_pequenas.to_dict()}"
        )

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df[col_label], random_state=RANDOM_STATE,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df[col_label], random_state=RANDOM_STATE,
    )

    for nome, split in [("treino", train_df), ("val", val_df), ("teste", test_df)]:
        logger.info(
            "Split estratificado %s: %d amostras | %s",
            nome, len(split), split[col_label].value_counts().to_dict(),
        )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def salvar_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    data_dir: Path = DATA_PROCESSED,
) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for nome, split in [("train", train), ("val", val), ("test", test)]:
        destino = data_dir / f"{nome}.parquet"
        split.to_parquet(destino, index=False, engine="pyarrow")
        logger.info("Salvo: %s (%d linhas)", destino, len(split))
