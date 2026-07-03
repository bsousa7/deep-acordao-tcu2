"""
Testes unitários do pipeline anti-vazamento e dos modelos ponderados.

Não requerem GPU, download de CSVs reais nem checkpoint de BERT.
"""

from __future__ import annotations

import io
import textwrap

import numpy as np
import pandas as pd
import pytest

from src.preprocessamento.anti_vazamento import (
    auditar_vazamento,
    construir_feature_voto,
    contem_veredito,
    extrair_rotulo,
    remover_dispositivo,
    strip_html,
)
from src.preprocessamento.filtrar_tematico import filtrar_por_tema
from src.preprocessamento.split_temporal import (
    dividir_estratificado,
    dividir_temporal_por_ano,
)
from src.modelos.pesos import LABEL2ID, pesos_balanceados, pesos_tensor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def voto_com_dispositivo() -> str:
    return textwrap.dedent(
        """
        Trata-se de tomada de contas especial instaurada contra ex-gestor municipal
        do Fundo Nacional de Desenvolvimento da Educação (FNDE), tendo por
        objeto irregularidades na aplicação de recursos da merenda escolar.

        A instrução analisou os documentos apresentados pelo responsável.

        Diante do exposto, VOTO por que o Tribunal aprove o Acórdão que ora submeto
        ao Colegiado.

        ACORDAM os Ministros do Tribunal de Contas da União em julgar irregulares
        as presentes contas.
        """
    ).strip()


@pytest.fixture
def df_mock() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SITUACAO": ["OFICIALIZADO"] * 4,
            "SUMARIO": [
                "TOMADA DE CONTAS ESPECIAL. FNDE. MERENDA. CONTAS IRREGULARES.",
                "PRESTAÇÃO DE CONTAS. SUS. CONTAS REGULARES COM RESSALVA.",
                "PRESTAÇÃO DE CONTAS. EDUCAÇÃO. CONTAS REGULARES.",
                "PRESTAÇÃO DE CONTAS. FNDE. EDUCAÇÃO. CONTAS IRREGULARES.",
            ],
            "ACORDAM": ["ACORDAM os Ministros em julgar irregulares"] * 4,
            "ACORDAO": ["julgar irregulares as contas"] * 4,
            "ASSUNTO": ["saúde", "sus", "educação", "fnde"],
            "VOTO": [
                "Voto pela irregularidade. ACORDAM os Ministros em julgar irregulares as contas.",
                "Voto pela regularidade com ressalva. Diante do exposto, voto.",
                "Voto pela regularidade. Ante o exposto, voto.",
                "Voto pela irregularidade. Pelo exposto, julgo as contas irregulares.",
            ],
            "ANO": [2016, 2020, 2022, 2024],
        }
    )


# ---------------------------------------------------------------------------
# Anti-vazamento
# ---------------------------------------------------------------------------


def test_strip_html_remove_tags():
    txt = "<p><b>Contas</b> Irregulares</p>"
    assert strip_html(txt) == "Contas Irregulares"


def test_contem_veredito_positivo_e_negativo():
    assert contem_veredito("as contas foram julgadas irregulares")
    assert not contem_veredito("análise de mérito sobre a instrução processual")


def test_remover_dispositivo_corta_no_marcador(voto_com_dispositivo):
    resultado = remover_dispositivo(voto_com_dispositivo)
    assert resultado.dispositivo_encontrado is True
    assert "ACORDAM" not in resultado.texto
    assert "irregularidades na aplicação" in resultado.texto


def test_construir_feature_voto_mascara_veredito_residual():
    voto = "irregularidade das contas foi apurada. Diante do exposto, VOTO por julgar."
    feat = construir_feature_voto(voto)
    assert "[DECISAO]" in feat.texto
    assert not contem_veredito(feat.texto)


def test_auditar_vazamento_gate_passa_em_texto_limpo():
    df = pd.DataFrame({"VOTO_LIMPO": ["análise técnica sem veredito", "instrução processual"]})
    resultado = auditar_vazamento(df, "VOTO_LIMPO", verbose=False)
    assert resultado["gate_passou"] is True
    assert resultado["fracao_vazado"] == 0.0


def test_auditar_vazamento_gate_falha_em_texto_vazado():
    df = pd.DataFrame({"SUMARIO": ["CONTAS IRREGULARES.", "análise técnica sem veredito"]})
    resultado = auditar_vazamento(df, "SUMARIO", verbose=False)
    assert resultado["gate_passou"] is False
    assert resultado["fracao_vazado"] > 0


def test_extrair_rotulo_a_partir_do_sumario():
    assert extrair_rotulo(None, "CONTAS IRREGULARES", None) == "Irregular"
    assert extrair_rotulo(None, "REGULARES COM RESSALVA", None) == "Regular com Ressalva"
    assert extrair_rotulo(None, "CONTAS REGULARES.", None) == "Regular"


def test_extrair_rotulo_prioriza_ressalva_sobre_regular():
    """A ordem dos regex garante que 'regulares com ressalva' não é classificado como 'Regular'."""
    assert extrair_rotulo(None, "As contas foram julgadas regulares com ressalva.", None) == "Regular com Ressalva"


# ---------------------------------------------------------------------------
# Filtro temático + split
# ---------------------------------------------------------------------------


def test_filtrar_por_tema_mantem_saude_e_educacao(df_mock):
    filtrado = filtrar_por_tema(df_mock)
    assert len(filtrado) == 4


def test_dividir_temporal_respeita_anos(df_mock):
    df = df_mock.copy()
    df["LABEL"] = ["Irregular", "Regular com Ressalva", "Regular", "Irregular"]
    train, val, test = dividir_temporal_por_ano(df, ano_teste=2024, ano_val=2022)
    assert set(train["ANO"]) <= {2016, 2020}
    assert set(val["ANO"]) == {2022}
    assert set(test["ANO"]) == {2024}


# ---------------------------------------------------------------------------
# Pesos
# ---------------------------------------------------------------------------


def test_pesos_balanceados_soma_e_ordem():
    y = ["Irregular"] * 90 + ["Regular com Ressalva"] * 8 + ["Regular"] * 2
    p = pesos_balanceados(y)
    assert set(p.keys()) == {"Irregular", "Regular com Ressalva", "Regular"}
    assert p["Regular"] > p["Regular com Ressalva"] > p["Irregular"]


def test_pesos_tensor_forma():
    y = np.array([0, 0, 0, 1, 2])
    w = pesos_tensor(y)
    assert w.shape == (3,)
    assert w[2] > w[0]


# ---------------------------------------------------------------------------
# Baseline sanidade (sem CSV real)
# ---------------------------------------------------------------------------


def test_baseline_pipeline_treina_com_pesos():
    from src.modelos.baseline import construir_pipeline

    X = pd.Series([
        "análise da fnde sobre a merenda escolar",
        "sus prestação de contas do estado",
        "recursos federais aplicados em educação",
        "irregularidade documental na aplicação do fundo",
        "instrução processual sobre o convênio",
        "auditoria de saúde no município",
    ])
    y = pd.Series(["Irregular", "Regular", "Regular com Ressalva", "Irregular", "Regular", "Regular com Ressalva"])
    pipe = construir_pipeline(class_weight="balanced")
    pipe.fit(X.values, y.values)
    preds = pipe.predict(X.values)
    assert len(preds) == len(X)
