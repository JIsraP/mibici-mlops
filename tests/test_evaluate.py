import numpy as np
import pandas as pd
import pytest

from mibici_mlops.modeling.compare import comparison_table, select_best
from mibici_mlops.modeling.evaluate import (
    error_by,
    improvement_pct,
    peak_mask,
    score,
    score_frame,
    station_segments,
)


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "estacion_id": [1, 1, 2, 2],
            "hora": [8, 13, 8, 18],
            "es_fin_semana": [0, 0, 1, 0],
            "salidas": [4, 0, 2, 6],
            "llegadas": [1, 1, 0, 3],
        }
    )


def test_horas_pico_solo_en_dias_laborales(frame):
    assert peak_mask(frame).tolist() == [True, False, False, True]


def test_metricas(frame):
    s = score(frame["salidas"], [2, 0, 2, 6], peak_mask(frame))
    assert s["mae"] == pytest.approx(0.5)
    assert s["rmse"] == pytest.approx(1.0)
    assert s["mae_pico"] == pytest.approx(1.0)  # errores pico: 2 y 0
    assert s["poisson"] > 0


def test_predicciones_negativas_se_recortan_a_cero(frame):
    assert score(frame["salidas"], [4, -3, 2, 6], peak_mask(frame))["mae"] == 0


def test_nombres_listos_para_mlflow(frame):
    m = score_frame(frame, {"salidas": frame["salidas"], "llegadas": frame["llegadas"]})
    assert set(m) == {f"{t}_{k}" for t in ("salidas", "llegadas") for k in ("mae", "rmse", "mae_pico", "poisson")}
    assert m["salidas_mae"] == 0


def test_error_por_hora_y_por_segmento(frame):
    pred = np.array([2, 0, 2, 6])
    by_hour = error_by(frame, pred, "salidas", "hora").set_index("hora")["mae"]
    assert by_hour[8] == pytest.approx(1.0)
    segments = pd.Series({1: "baja", 2: "alta"}, name="segmento")
    by_seg = error_by(frame, pred, "salidas", segments).set_index("segmento")["mae"]
    assert by_seg["baja"] == pytest.approx(1.0) and by_seg["alta"] == 0


def test_segmentos_por_terciles():
    train = pd.DataFrame({"estacion_id": [1, 2, 3, 4, 5, 6], "salidas": [1, 2, 3, 4, 5, 6]})
    seg = station_segments(train)
    assert seg[1] == "baja" and seg[6] == "alta"


def test_mejora_porcentual():
    assert improvement_pct(0.75, 1.0) == pytest.approx(25.0)


def runs_df():
    return pd.DataFrame(
        {
            "run_id": ["a", "b", "c", "d"],
            "tags.tipo": ["linea_base", "linea_base", "modelo", "modelo"],
            "params.modelo": ["promedio", "semana", "ridge", "hgb"],
            "metrics.salidas_mae_pico": [1.4, 1.6, 1.3, 1.1],
            "metrics.salidas_mae": [0.85, 0.99, 0.80, 0.70],
        }
    )


def test_tabla_compara_contra_la_mejor_linea_base():
    table = comparison_table(runs_df()).set_index("modelo")
    assert table.loc["hgb", "mejora_vs_base_pct"] == pytest.approx(21.4)
    assert table.loc["promedio", "mejora_vs_base_pct"] == 0
    assert bool(table.loc["hgb", "cumple_meta"]) and not bool(table.loc["ridge", "cumple_meta"])


def test_solo_compiten_modelos():
    assert select_best(comparison_table(runs_df()))["modelo"] == "hgb"
