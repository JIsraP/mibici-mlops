import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline

from mibici_mlops.config import DEGRADATION_ALERT_PCT, PSI_ALERT, PSI_WARN
from mibici_mlops.drift import (
    LAG_COLUMNS,
    aggregate_demand,
    bias_pct,
    bias_status,
    build_monitor_features,
    categorical_psi,
    change_pct,
    decide_action,
    degradation_status,
    demand_growth,
    evaluate_window,
    feature_drift,
    missing_lag,
    prepare_trips,
    psi,
    psi_status,
    seasonal_drop,
    worst,
)
from mibici_mlops.features import FEATURE_COLUMNS, TARGETS, build_preprocessor
from mibici_mlops.modeling.predict import predict_frame

RNG = np.random.default_rng(42)


# --- PSI y umbrales --------------------------------------------------------------------------


def test_psi_es_casi_cero_con_la_misma_distribucion():
    x = RNG.normal(size=20_000)
    assert psi(x, RNG.normal(size=20_000)) < PSI_WARN


def test_psi_detecta_desplazamiento_de_media():
    x = RNG.normal(size=20_000)
    assert psi(x, x + 1.0) > PSI_ALERT


def test_psi_soporta_conteos_con_muchos_ceros():
    ref = RNG.poisson(0.5, 10_000)
    assert psi(ref, ref) == pytest.approx(0, abs=1e-9)
    assert psi(ref, RNG.poisson(2.0, 10_000)) > PSI_ALERT


def test_psi_categorico_con_categoria_nueva():
    ref = ["A"] * 50 + ["B"] * 50
    assert categorical_psi(ref, ref) == pytest.approx(0)
    assert categorical_psi(ref, ["A"] * 30 + ["C"] * 70) > PSI_ALERT


def test_psi_sin_datos_falla():
    with pytest.raises(ValueError):
        psi([], [1, 2, 3])


@pytest.mark.parametrize(
    "value, expected", [(0.05, "estable"), (0.15, "advertencia"), (0.40, "alerta")]
)
def test_estado_psi(value, expected):
    assert psi_status(value) == expected


def test_estado_degradacion_y_cambio_porcentual():
    assert change_pct(1.0, 1.25) == pytest.approx(25)
    assert degradation_status(change_pct(1.0, 0.9)) == "estable"
    assert degradation_status(DEGRADATION_ALERT_PCT) == "alerta"


@pytest.mark.parametrize(
    "performance, inputs, action",
    [
        ("alerta", "alerta", "pipeline"),
        ("alerta", "estable", "reentrenar"),
        ("advertencia", "alerta", "vigilar"),
        ("estable", "advertencia", "vigilar_entradas"),
        ("estable", "estable", "ninguna"),
    ],
)
def test_regla_de_accion(performance, inputs, action):
    assert decide_action(performance, inputs) == action


def test_estaciones_nuevas_piden_actualizar_catalogo():
    assert decide_action("estable", "estable", new_stations=True) == "catalogo"
    # Una alerta de desempeno sigue mandando
    assert decide_action("alerta", "estable", new_stations=True) == "reentrenar"


def test_sesgo_detecta_caidas_que_el_mae_no_ve():
    actual = np.array([6.0, 4.0, 10.0])
    assert bias_pct(actual, actual) == pytest.approx(0)
    assert bias_pct(actual * 0.6, actual) == pytest.approx(66.67, abs=0.01)
    assert bias_status(-25) == "alerta"
    assert bias_status(12) == "advertencia"
    assert worst("estable", "alerta", "advertencia") == "alerta"


def test_feature_drift_marca_solo_la_variable_que_cambia():
    ref = pd.DataFrame({"a": RNG.normal(size=5000), "zona": ["GDL"] * 5000})
    cur = ref.assign(a=ref["a"] + 2)
    table = feature_drift(ref, cur, ["a", "zona"]).set_index("variable")
    assert table.loc["a", "estado"] == "alerta"
    assert table.loc["zona", "estado"] == "estable"


# --- Datos de monitoreo ----------------------------------------------------------------------


def raw_trips():
    return pd.DataFrame(
        {
            "viaje_id": ["1", "2", "3", "3", "4"],
            "usuario_id": ["9"] * 5,
            "genero": ["M"] * 5,
            "anio_nacimiento": ["1990"] * 5,
            "inicio": [
                "2026-01-01 08:05:00",
                "2026-01-01 08:30:00",
                "2026-01-01 09:10:00",
                "2026-01-01 09:10:00",  # duplicado exacto
                "2026-01-01 10:00:00",
            ],
            "fin": [
                "2026-01-01 08:20:00",
                "2026-01-01 09:05:00",
                "2026-01-01 09:40:00",
                "2026-01-01 09:40:00",
                "2026-01-01 09:59:00",  # duracion negativa
            ],
            "origen_id": ["1", "1", "2", "2", "1"],
            "destino_id": ["2", "2", "1", "1", "2"],
        }
    )


def test_prepare_trips_aplica_el_contrato():
    trips, counts = prepare_trips(raw_trips())
    assert counts["duplicados_exactos"] == 1
    assert counts["duracion_no_positiva"] == 1
    assert counts["viajes_monitoreo"] == len(trips) == 3


def test_aggregate_demand_llena_horas_sin_viajes():
    trips, _ = prepare_trips(raw_trips())
    demand = aggregate_demand(trips, "2026-01-01 08:00", "2026-01-01 11:00")
    assert len(demand) == 2 * 3  # 2 estaciones x 3 horas
    s1 = demand[demand["estacion_id"] == 1].set_index("fecha_hora")
    assert s1["salidas"].tolist() == [2, 0, 0]
    assert s1["llegadas"].tolist() == [0, 1, 0]
    assert demand["salidas"].sum() == 3


def hourly(stations, start, end, value=1):
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    idx = pd.MultiIndex.from_product([stations, hours], names=["estacion_id", "fecha_hora"])
    return pd.DataFrame({"salidas": value, "llegadas": value}, index=idx).reset_index()


def catalog(ids):
    return pd.DataFrame(
        {
            "estacion_id": ids,
            "latitud": 20.6,
            "longitud": -103.3,
            "capacidad": 15,
            "zona": "GDL",
        }
    )


def test_build_monitor_features_usa_historia_y_excluye_estaciones_nuevas():
    history = hourly([1, 2], "2025-12-20", "2026-01-01")
    demand = hourly([1, 2, 3], "2026-01-01", "2026-01-03")  # la 3 es nueva
    profiles = pd.DataFrame(
        {"perfil_pc1": [0.1, 0.2], "perfil_pc2": [0.0, 0.1]}, index=pd.Index([1, 2])
    )
    features, counts = build_monitor_features(
        demand, history, catalog([1, 2, 3]), profiles, "2026-01-01"
    )
    assert counts["estaciones_nuevas"] == 1
    # La estacion nueva aporta un tercio de las salidas del mes
    assert counts["pct_salidas_estaciones_nuevas"][pd.Period("2026-01", "M")] == pytest.approx(
        33.33, abs=0.01
    )
    assert set(features["estacion_id"]) == {1, 2}
    assert features["fecha_hora"].min() == pd.Timestamp("2026-01-01")
    # Los rezagos de la primera hora de 2026 vienen de diciembre de 2025
    assert features[LAG_COLUMNS].notna().all().all()
    # El 1 de enero de 2026 es festivo aunque el calendario original solo tenga 2025
    assert features.loc[features["fecha_hora"].dt.day == 1, "es_festivo"].eq(1).all()
    assert set(FEATURE_COLUMNS) <= set(features.columns)


# --- Escenarios sinteticos -------------------------------------------------------------------


@pytest.fixture
def frame():
    n = 400
    hours = pd.date_range("2025-11-03", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "estacion_id": np.repeat([1, 2], n // 2),
            "fecha_hora": np.tile(hours[: n // 2], 2),
            "hora": np.tile(hours[: n // 2].hour, 2),
            "hora_sin": 0.0,
            "hora_cos": 1.0,
            "dia_semana": np.tile(hours[: n // 2].day_of_week, 2),
            "es_fin_semana": 0,
            "es_festivo": 0,
            "latitud": 20.6,
            "longitud": -103.3,
            "capacidad": 15,
            "zona": "GDL",
            "perfil_pc1": 0.1,
            "perfil_pc2": 0.2,
        }
    )
    for t in TARGETS:
        df[t] = RNG.poisson(3, n).astype(float)
        for c in ("lag_24", "lag_168", "media_7d"):
            df[f"{t}_{c}"] = RNG.poisson(3, n).astype(float)
    return df


def test_escenarios_no_modifican_la_entrada(frame):
    original = frame.copy()
    for fn in (demand_growth, missing_lag, seasonal_drop):
        fn(frame)
    pd.testing.assert_frame_equal(frame, original)


def test_escenarios_cambian_lo_que_dicen(frame):
    grown = demand_growth(frame, 1.5)
    assert grown["salidas"].sum() == pytest.approx(frame["salidas"].sum() * 1.5)
    assert grown["salidas_lag_168"].sum() == pytest.approx(frame["salidas_lag_168"].sum() * 1.5)

    missing = missing_lag(frame)
    assert missing["salidas_lag_24"].eq(0).all()
    assert missing["salidas_lag_168"].equals(frame["salidas_lag_168"])

    drop = seasonal_drop(frame, 0.5)
    assert drop["salidas"].sum() == pytest.approx(frame["salidas"].sum() * 0.5)
    assert drop[LAG_COLUMNS].equals(frame[LAG_COLUMNS])


# --- Integracion: modelo -> escenario -> metricas -> alerta ----------------------------------


def test_evaluate_window_de_extremo_a_extremo(frame):
    model = make_pipeline(
        build_preprocessor().set_params(sparse_threshold=0),
        MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=20, random_state=42)),
    ).fit(frame[FEATURE_COLUMNS], frame[TARGETS])
    preds = predict_frame(model, frame)
    from mibici_mlops.modeling.evaluate import score_frame

    ref_metrics = score_frame(frame, {t: preds[f"pred_{t}"].to_numpy() for t in TARGETS})
    context = (frame, preds, ref_metrics, frame)

    same, _ = evaluate_window("igual", model, frame, *context)
    assert same["degradacion_pct"] == pytest.approx(0)
    assert same["accion"] == "ninguna"

    worse, drift = evaluate_window("caida", model, seasonal_drop(frame, 0.3), *context)
    assert worse["sesgo_pct"] > 100  # pronostica mas del doble de lo que ocurre
    assert worse["estado_desempeno"] == "alerta"
    assert worse["accion"] == "reentrenar"  # las entradas no cambiaron
    assert set(drift["variable"]) == set(FEATURE_COLUMNS)
