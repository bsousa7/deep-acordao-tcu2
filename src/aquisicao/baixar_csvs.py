"""
Download dos CSVs anuais de acórdãos do TCU (2016–2024).

Fonte oficial: Portal de Dados Abertos do TCU. Suporta retomada via HTTP Range e
retry com backoff exponencial. Nenhum scraping — apenas os CSVs oficiais.
"""

import logging
import time
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

BASE_URL = (
    "https://sites.tcu.gov.br/dados-abertos/jurisprudencia/"
    "arquivos/acordao-completo/acordao-completo-{year}.csv"
)
# Escopo do projeto: 2016–2024 (9 anos).
ANOS_PADRAO = list(range(2016, 2025))
DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
CHUNK_SIZE = 1024 * 1024  # 1 MB
MAX_RETRIES = 5
BACKOFF_BASE = 2.0


def _url_para_ano(ano: int) -> str:
    return BASE_URL.format(year=ano)


def _tamanho_remoto(url: str, session: requests.Session) -> int | None:
    try:
        resp = session.head(url, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        length = resp.headers.get("Content-Length")
        return int(length) if length else None
    except Exception as exc:
        logger.debug("HEAD falhou para %s: %s", url, exc)
        return None


def _baixar_arquivo(url: str, destino: Path, session: requests.Session) -> None:
    local_size = destino.stat().st_size if destino.exists() else 0
    remote_size = _tamanho_remoto(url, session)

    if remote_size is not None and local_size == remote_size:
        logger.info("Já completo: %s (%d MB)", destino.name, remote_size // 1_000_000)
        return

    headers = {}
    mode = "wb"
    if local_size > 0:
        headers["Range"] = f"bytes={local_size}-"
        mode = "ab"
        logger.info("Retomando %s a partir de %d MB", destino.name, local_size // 1_000_000)

    resp = session.get(url, headers=headers, stream=True, timeout=60)

    if resp.status_code == 200 and mode == "ab":
        logger.warning("Servidor não suporta Range — reiniciando download de %s", destino.name)
        mode = "wb"
        local_size = 0

    resp.raise_for_status()

    total = remote_size if remote_size else None
    with open(destino, mode) as fh, tqdm(
        total=total,
        initial=local_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=destino.name,
        ncols=100,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                fh.write(chunk)
                bar.update(len(chunk))


def _baixar_com_retry(
    url: str,
    destino: Path,
    session: requests.Session,
    max_retries: int = MAX_RETRIES,
) -> None:
    for attempt in range(1, max_retries + 1):
        try:
            _baixar_arquivo(url, destino, session)
            return
        except (requests.RequestException, OSError) as exc:
            if attempt == max_retries:
                logger.error("Falha definitiva ao baixar %s após %d tentativas", url, max_retries)
                raise
            wait = BACKOFF_BASE**attempt
            logger.warning(
                "Tentativa %d/%d falhou (%s). Aguardando %.0fs...",
                attempt, max_retries, exc, wait,
            )
            time.sleep(wait)


def baixar_todos(
    anos: list[int] | None = None,
    data_dir: Path = DATA_RAW,
) -> dict[int, Path]:
    """Baixa CSVs TCU para os anos especificados (padrão: 2016–2024)."""
    anos = anos or ANOS_PADRAO
    data_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "deep-acordao-tcu2/1.0 (pesquisa academica)"})

    resultados: dict[int, Path] = {}
    for ano in anos:
        url = _url_para_ano(ano)
        destino = data_dir / f"acordao-completo-{ano}.csv"
        logger.info("Processando ano %d → %s", ano, destino)
        try:
            _baixar_com_retry(url, destino, session)
            resultados[ano] = destino
        except Exception as exc:
            logger.error("Ano %d ignorado: %s", ano, exc)

    logger.info("Download concluído: %d/%d arquivos", len(resultados), len(anos))
    return resultados


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    baixar_todos()
