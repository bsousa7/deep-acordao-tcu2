"""
Anti-vazamento: extração de rótulo e construção de features limpas desde o início.

Princípio central desta versão
------------------------------
O SUMARIO do acórdão TCU é redigido APÓS o julgamento e contém o veredito escrito
literalmente ("CONTAS IRREGULARES", "CONTAS REGULARES COM RESSALVA"). Usá-lo como
feature é copiar o gabarito. Portanto, nesta versão:

  * SUMARIO -> APENAS fonte auxiliar de rótulo. NUNCA vira feature de treino.
  * VOTO    -> feature única, APÓS remover o dispositivo (sentença final) e
               auditar o vazamento residual.

O gate `auditar_vazamento()` é obrigatório antes de qualquer treino e exige 0%
de textos com termo de veredito no campo de feature.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Termos de veredito (usados no gate e na extração de rótulo)
# ---------------------------------------------------------------------------

TERMOS_VEREDITO: list[str] = [
    # Fillers curtos ("foram julgadas", "julgadas") são tolerados entre "contas" e o veredito.
    r"\bcontas?(?:\s+\w+){0,3}\s+irregulares?\b",
    r"\bregulares?\s+com\s+ressalva\b",
    r"\bcontas?(?:\s+\w+){0,3}\s+regulares?\b",
    r"\birregularidade\s+das\s+contas\b",
    r"\bjulg(?:o|ar|amos|am)\s+(?:as\s+)?(?:presentes?\s+)?contas?(?:\s+\w+){0,3}\s+irregulares?\b",
    r"\bjulg(?:o|ar|amos|am)\s+(?:as\s+)?(?:presentes?\s+)?contas?(?:\s+\w+){0,3}\s+regulares?\b",
    r"\bACORDAM\s+os\s+Ministros\b",
]

_RE_VEREDITO = re.compile("|".join(TERMOS_VEREDITO), flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# 2. Padrões que marcam o INÍCIO do dispositivo no VOTO
# ---------------------------------------------------------------------------

_PADROES_DISPOSITIVO: list[str] = [
    r"ACORDAM\s+os\s+Ministros",
    r"VISTOS,?\s+relatados\s+e\s+discutidos",
    r"(?:Diante|Ante|Em\s+face|Por\s+todo|Em\s+vista)\s+(?:do|de\s+todo\s+o|dos)\s+exposto",
    r"(?:Com\s+essas|Nessas|Com\s+tais|Nestas)\s+considera(?:c|ç)(?:o|ő)es[,.]?\s*(?:VOTO|proponho|manifesto)",
    r"Pelo\s+exposto[,.]",
    r"É\s+como\s+voto",
    r"(?:Assim|Portanto)[,.]?\s+(?:VOTO|proponho|entendo\s+que\s+(?:se\s+)?(?:devem?|deva))",
    r"\bVOTO\b\s*\n",
    r"(?:proponho|proponemos)\s+ao\s+(?:Plenário|Tribunal)",
    r"\bjulg(?:o|ar|amos|am)\s+(?:as\s+)?contas\b",
]

_RE_DISPOSITIVO = re.compile(
    "|".join(f"(?:{p})" for p in _PADROES_DISPOSITIVO),
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 3. Utilitários de limpeza de HTML (CSVs do TCU trazem tags)
# ---------------------------------------------------------------------------

_RE_TAG = re.compile(r"<[^>]+>")
_RE_WS = re.compile(r"\s+")


def strip_html(texto: str | None) -> str:
    """Remove tags HTML e colapsa whitespace. Robusto a valores não-string."""
    if not isinstance(texto, str):
        return ""
    if "<" in texto:
        texto = _RE_TAG.sub(" ", texto)
    return _RE_WS.sub(" ", texto).strip()


# ---------------------------------------------------------------------------
# 4. Extração de rótulo (a partir de SUMARIO/SITUACAO/ACORDAO)
# ---------------------------------------------------------------------------


def extrair_rotulo(situacao: str | None, sumario: str | None, acordao: str | None) -> str | None:
    """
    Extrai o rótulo (Irregular / Regular com Ressalva / Regular) a partir de:
      1. SITUACAO — formato antigo do CSV; texto = desfecho direto.
      2. SUMARIO  — no formato atual (2020+) SITUACAO="OFICIALIZADO",
                    então o desfecho aparece como keyword no cabeçalho da ementa.
      3. ACORDAO  — dispositivo formal ("julgar irregulares...").

    O SUMARIO é usado APENAS aqui (extração de rótulo), NUNCA como feature.
    """
    if isinstance(situacao, str):
        s = situacao.lower().strip()
        if s and s != "oficializado":
            if "irregular" in s:
                return "Irregular"
            if "ressalva" in s:
                return "Regular com Ressalva"
            if s.startswith("regular"):
                return "Regular"

    sumario_txt = strip_html(sumario).lower()
    if re.search(r"\bregulares?\s+com\s+ressalva\b", sumario_txt):
        return "Regular com Ressalva"
    if re.search(r"\b(?:contas?\s+)?irregulares?\b", sumario_txt):
        return "Irregular"
    if re.search(r"\b(?:contas?\s+)?regulares?\b", sumario_txt):
        return "Regular"

    acordao_txt = strip_html(acordao).lower()
    if re.search(r"\bjulg(?:o|ar|amos|am)\s+(?:as\s+)?contas?\s+regulares?\s+com\s+ressalva\b", acordao_txt):
        return "Regular com Ressalva"
    if re.search(r"\bjulg(?:o|ar|amos|am)\s+(?:as\s+)?contas?\s+irregulares?\b", acordao_txt):
        return "Irregular"
    if re.search(r"\bjulg(?:o|ar|amos|am)\s+(?:as\s+)?contas?\s+regulares?\b", acordao_txt):
        return "Regular"

    return None


# ---------------------------------------------------------------------------
# 5. Construção de features limpas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureLimpa:
    """Resultado da construção da feature ex-ante do VOTO."""

    texto: str
    dispositivo_encontrado: bool


def remover_dispositivo(voto: str | None) -> FeatureLimpa:
    """
    Trunca o VOTO no primeiro marcador de dispositivo.

    Retorna FeatureLimpa(texto_sem_dispositivo, encontrou_marcador).
    Se nenhum marcador for encontrado, o texto é mantido inteiro (o gate
    de auditoria detectará vazamento residual, se houver).
    """
    txt = strip_html(voto)
    if not txt:
        return FeatureLimpa("", False)

    m = _RE_DISPOSITIVO.search(txt)
    if m:
        return FeatureLimpa(txt[: m.start()].rstrip(), True)
    return FeatureLimpa(txt, False)


def construir_feature_voto(voto_raw: str | None) -> FeatureLimpa:
    """
    Constrói a feature ex-ante do VOTO:
      1. Remove o dispositivo (sentença final).
      2. Substitui termos de veredito residuais por '[DECISAO]' (defesa em profundidade).

    O passo 2 é redundante se o passo 1 acertou, mas garante robustez contra
    fragmentos de veredito citados no corpo analítico do voto.
    """
    resultado = remover_dispositivo(voto_raw)
    texto_mascarado = _RE_VEREDITO.sub("[DECISAO]", resultado.texto)
    return FeatureLimpa(texto_mascarado, resultado.dispositivo_encontrado)


# ---------------------------------------------------------------------------
# 6. Auditoria (gate obrigatório)
# ---------------------------------------------------------------------------


def contem_veredito(texto: str | None) -> bool:
    return bool(_RE_VEREDITO.search(texto)) if isinstance(texto, str) else False


def auditar_vazamento(df: pd.DataFrame, campo: str, verbose: bool = True) -> dict:
    """
    Audita a fração de textos do campo que contêm termos de veredito.

    Retorna dict com:
      campo, total, n_vazados, fracao_vazado, gate_passou (True se fração == 0).
    """
    textos = df[campo].fillna("").astype(str).tolist()
    n_vazados = sum(1 for t in textos if contem_veredito(t))
    total = len(textos)
    fracao = n_vazados / total if total > 0 else 0.0

    resultado = {
        "campo": campo,
        "total": total,
        "n_vazados": n_vazados,
        "fracao_vazado": round(fracao, 4),
        "gate_passou": fracao == 0.0,
    }

    if verbose:
        status = "PASSOU" if resultado["gate_passou"] else "FALHOU"
        print(f"[ANTI-VAZAMENTO] Campo: {campo} | Gate: {status}")
        print(f"  Total:    {total:,}")
        print(f"  Vazados:  {n_vazados:,}  ({fracao * 100:.2f}%)")
    return resultado
