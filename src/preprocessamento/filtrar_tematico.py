"""
Carga dos CSVs anuais do TCU, filtro temático (Saúde/Educação) e extração de
rótulo. A construção das features limpas é responsabilidade do módulo
`anti_vazamento` — este arquivo só entrega o corpus rotulado.
"""

import logging
import re
from pathlib import Path

import pandas as pd

from .anti_vazamento import extrair_rotulo, strip_html

logger = logging.getLogger(__name__)

# Só carregamos as colunas mínimas necessárias — os CSVs do TCU pesam 175–445 MB.
COLUNAS_USADAS = ["NUMACORDAO", "SITUACAO", "SUMARIO", "VOTO", "ACORDAO", "ASSUNTO"]

TERMOS_TEMATICOS = [
    # --- Saúde ---
    "saúde",
    "sus",
    "hospital",
    "hospitalar",
    "unidade básica de saúde",
    "ubs",
    "upa",
    "samu",
    "vigilância sanitária",
    "anvisa",
    "ans",
    "atenção básica",
    "assistência farmacêutica",
    "farmácia básica",
    "medicamento",
    "posto de saúde",
    "ambulatorial",
    "ministério da saúde",
    "secretaria de saúde",
    "secretaria municipal de saúde",
    "secretaria estadual de saúde",
    # --- Educação ---
    "educação",
    "fnde",
    "merenda",
    "escola",
    "escolar",
    "ensino",
    "universidade",
    "universitário",
    "creche",
    "fundeb",
    "pnae",
    "pnate",
    "transporte escolar",
    "mec",
    "capes",
    "alfabetização",
    "livro didático",
    "instituto federal",
    "ministério da educação",
    "secretaria de educação",
    "secretaria municipal de educação",
    "secretaria estadual de educação",
]

# Word-boundary (\b) evita falso-positivo de abreviações curtas dentro de outras
# palavras (ex.: "ubs" não deve casar com "subsídio"; "mec" não deve casar com
# "mecanismo"; "upa" não deve casar com "ocupação").
_RE_TERMOS_TEMATICOS = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in TERMOS_TEMATICOS) + r")\b",
    flags=re.IGNORECASE,
)

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_INTERIM = Path(__file__).resolve().parents[2] / "data" / "interim"


def inspecionar_colunas(caminho_csv: str | Path) -> pd.DataFrame:
    """Retorna DataFrame com nome/índice de cada coluna do cabeçalho do CSV."""
    caminho_csv = Path(caminho_csv)
    df_header = pd.read_csv(caminho_csv, sep="|", encoding="utf-8-sig", nrows=0)
    df_header.columns = df_header.columns.str.strip()
    info = pd.DataFrame({"coluna": df_header.columns, "indice": range(len(df_header.columns))})
    logger.info("CSV: %d colunas", len(info))
    return info


def _contem_termos(texto: str | None) -> bool:
    if not isinstance(texto, str):
        return False
    return bool(_RE_TERMOS_TEMATICOS.search(texto))


def filtrar_por_tema(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém apenas acórdãos de Saúde ou Educação (busca em SUMARIO e ASSUNTO)."""
    mascara = df["SUMARIO"].apply(_contem_termos) | df["ASSUNTO"].apply(_contem_termos)
    filtrado = df[mascara].copy()
    logger.info(
        "Filtro temático: %d → %d acórdãos (%.1f%%)",
        len(df),
        len(filtrado),
        100 * len(filtrado) / max(len(df), 1),
    )
    return filtrado


def _carregar_ano(ano: int, data_dir: Path) -> pd.DataFrame | None:
    caminho = data_dir / f"acordao-completo-{ano}.csv"
    if not caminho.exists():
        logger.warning("Arquivo não encontrado: %s", caminho)
        return None

    df = pd.read_csv(
        caminho,
        sep="|",
        encoding="utf-8-sig",
        usecols=COLUNAS_USADAS,
        dtype=str,
        low_memory=False,
        on_bad_lines="skip",
    )
    df.columns = df.columns.str.strip()
    df["ANO"] = ano
    logger.info("Ano %d: %d acórdãos carregados", ano, len(df))
    return df


def combinar_anos(
    anos: list[int],
    data_dir: Path = DATA_RAW,
    apenas_tema: bool = True,
) -> pd.DataFrame:
    """
    Carrega, concatena, extrai labels e (por padrão) filtra por tema.

    Colunas retornadas: NUMACORDAO, SITUACAO, SUMARIO, VOTO, ACORDAO, ASSUNTO,
    ANO, LABEL. IMPORTANTE: SUMARIO permanece no DataFrame APENAS para
    referência/diagnóstico — não deve ser usado como feature de treino.
    """
    dfs = []
    for ano in anos:
        df_ano = _carregar_ano(ano, data_dir)
        if df_ano is not None:
            dfs.append(df_ano)

    if not dfs:
        raise RuntimeError(f"Nenhum CSV encontrado em {data_dir} para os anos {anos}")

    df = pd.concat(dfs, ignore_index=True)
    df["SUMARIO"] = df["SUMARIO"].apply(strip_html)
    df["ACORDAO"] = df["ACORDAO"].apply(strip_html)
    df["ASSUNTO"] = df["ASSUNTO"].apply(strip_html)

    df["LABEL"] = df.apply(
        lambda r: extrair_rotulo(r.get("SITUACAO"), r.get("SUMARIO"), r.get("ACORDAO")),
        axis=1,
    )
    n_sem_label = int(df["LABEL"].isna().sum())
    df = df.dropna(subset=["LABEL"]).copy()
    logger.info(
        "Após extração de label: %d acórdãos (%d descartados sem label)",
        len(df),
        n_sem_label,
    )

    if apenas_tema:
        df = filtrar_por_tema(df)

    logger.info("Distribuição de classes:\n%s", df["LABEL"].value_counts().to_string())
    return df.reset_index(drop=True)


def salvar_parquet(
    df: pd.DataFrame,
    destino: Path = DATA_INTERIM / "acordaos_rotulados.parquet",
) -> None:
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino, index=False, engine="pyarrow")
    size_mb = destino.stat().st_size / 1_000_000
    logger.info("Salvo: %s (%.2f MB, %d linhas)", destino, size_mb, len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    anos = list(range(2016, 2025))
    df = combinar_anos(anos)
    salvar_parquet(df)
    print(df["LABEL"].value_counts())
