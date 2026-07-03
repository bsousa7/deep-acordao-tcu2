"""
Limpeza de texto e head+tail para BERT — sempre sobre a feature LIMPA do VOTO.

Não há função de limpeza para SUMARIO: nesta versão do projeto o SUMARIO é
apenas fonte de rótulo. A feature única do modelo é o VOTO após
`construir_feature_voto()` (anti_vazamento.py), aqui referida como VOTO_LIMPO.
"""

import logging
import re
import unicodedata

import pandas as pd

logger = logging.getLogger(__name__)

HEAD_TOKENS = 128
TAIL_TOKENS = 382  # 128 + 382 + 2 especiais = 512
MODEL_NAME = "dominguesm/legal-bert-base-cased-ptbr"

_STOPWORDS_PT: set[str] = set()


def _carregar_stopwords() -> set[str]:
    global _STOPWORDS_PT
    if _STOPWORDS_PT:
        return _STOPWORDS_PT
    try:
        import nltk
        from nltk.corpus import stopwords

        nltk.download("stopwords", quiet=True)
        _STOPWORDS_PT = set(stopwords.words("portuguese"))
    except Exception as exc:
        logger.warning("NLTK stopwords não disponíveis: %s", exc)
        _STOPWORDS_PT = set()
    return _STOPWORDS_PT


def limpar_texto(texto: str | None, modo: str = "tfidf") -> str:
    """
    Limpa texto para TF-IDF ou BERT.

    modo='tfidf': lowercase + remove pontuação + remove stopwords + colapsa espaços.
    modo='bert':  normaliza unicode, colapsa quebras, preserva case.
    """
    if not isinstance(texto, str):
        return ""

    texto = unicodedata.normalize("NFKC", texto).strip()
    texto = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", " ", texto)

    if modo == "tfidf":
        texto = texto.lower()
        texto = re.sub(r"[^\w\s]", " ", texto, flags=re.UNICODE)
        sw = _carregar_stopwords()
        if sw:
            texto = " ".join(t for t in texto.split() if t not in sw)
        return re.sub(r"\s+", " ", texto).strip()

    # modo bert
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return re.sub(r"[ \t]+", " ", texto).strip()


def truncar_head_tail(texto: str, tokenizer) -> str:
    """Trunca em head+tail (128 + 382 = 510 tokens BPE)."""
    ids = tokenizer.encode(texto, add_special_tokens=False)
    budget = HEAD_TOKENS + TAIL_TOKENS
    if len(ids) <= budget:
        return texto
    head = ids[:HEAD_TOKENS]
    tail = ids[-TAIL_TOKENS:]
    return tokenizer.decode(head + tail, skip_special_tokens=True)


def aplicar_limpeza(
    df: pd.DataFrame,
    campo: str = "VOTO_LIMPO",
    modo: str = "tfidf",
    tokenizer=None,
) -> pd.Series:
    """
    Limpa o campo especificado. Para modo='bert', passa também por head+tail
    (o chamador fornece o tokenizer para evitar downloads implícitos).
    """
    try:
        from tqdm import tqdm as tqdm_cls

        tqdm_cls.pandas(desc=f"Limpeza {modo}/{campo}")
        serie = df[campo].progress_apply(lambda t: limpar_texto(t, modo))
    except ImportError:
        serie = df[campo].apply(lambda t: limpar_texto(t, modo))

    if modo == "bert":
        if tokenizer is None:
            from transformers import AutoTokenizer

            logger.info("Carregando tokenizer %s para head+tail...", MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        try:
            from tqdm import tqdm as tqdm_cls

            tqdm_cls.pandas(desc=f"Head+tail {campo}")
            serie = serie.progress_apply(lambda t: truncar_head_tail(t, tokenizer))
        except ImportError:
            serie = serie.apply(lambda t: truncar_head_tail(t, tokenizer))

    return serie
