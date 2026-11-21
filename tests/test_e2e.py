"""Prueba end-to-end del pipeline de MiBici con datos sintéticos."""

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from mibici_mlops.cleaning import CleaningRules, TripCleaner
from mibici_mlops.features import FeatureBuilder, build_demand
from mibici_mlops.modeling.evaluate import score_frame
from mibici_mlops.modeling.train import fit_and_score


def test_pipeline_end_to_end():
    # 1. Generar viajes sintéticos con patrones diarios y horarios
    records = []
    trip_id = 1

    dates = pd.date_range("2025-01-01", "2025-11-30", freq="D")

    # Horarios de demanda simulada: mañana, mediodía y tarde
    weekday_schedule = {
        7: 5,
        8: 8,
        9: 4,
        12: 2,
        17: 6,
        18: 9,
        19: 5,
    }

    weekend_schedule = {
        9: 3,
        10: 5,
        11: 4,
        13: 3,
        16: 4,
        17: 5,
        18: 3,
    }

    for date in dates:
        is_weekend = date.day_of_week >= 5
        schedule = weekend_schedule if is_weekend else weekday_schedule

        # Variación determinista de volumen entre días
        daily_factor = 0.8 + (date.day % 5) * 0.1

        for hour, base_count in schedule.items():
            count = max(1, round(base_count * daily_factor))

            for trip in range(count):
                start = date + pd.Timedelta(hours=hour)

                # Alternar sentido y variar ligeramente el patrón
                # para generar demanda distinta entre estaciones.
                direction = (trip + hour + date.day) % 2
                origin = 1 if direction == 0 else 2
                destination = 2 if origin == 1 else 1

                start += pd.Timedelta(minutes=(trip * 7) % 60)

                records.append(
                    {
                        "viaje_id": trip_id,
                        "usuario_id": 1000 + trip_id,
                        "origen_id": origin,
                        "destino_id": destination,
                        "inicio": start,
                        "fin": start + pd.Timedelta(minutes=15 + trip % 10),
                        "genero": "M" if trip_id % 2 == 0 else "F",
                        "anio_nacimiento": 1995 + trip_id % 10,
                    }
                )
                trip_id += 1

    raw = pd.DataFrame(records)

    # 2. Limpieza de datos
    cleaner = TripCleaner(
        CleaningRules.variant("laxa"),
        valid_station_ids={1, 2},
    )

    trips = cleaner.clean(raw)

    assert not trips.empty
    assert trips["duracion_min"].gt(0).all()
    assert trips["origen_id"].isin({1, 2}).all()
    assert trips["destino_id"].isin({1, 2}).all()

    # 3. Generar demanda por estación y hora
    demand = build_demand(trips)

    assert not demand.empty
    assert {"salidas", "llegadas"}.issubset(demand.columns)
    assert demand["salidas"].sum() > 0
    assert demand["llegadas"].sum() > 0

    # 4. Catálogo sintético de estaciones
    stations = pd.DataFrame(
        {
            "estacion_id": [1, 2],
            "latitud": [20.67, 20.68],
            "longitud": [-103.35, -103.36],
            "capacidad": [20, 25],
            "zona": ["centro", "norte"],
        }
    )

    # 5. Construir variables de calendario, rezagos y perfiles
    builder = FeatureBuilder(
        stations,
        train_end="2025-11-01",
    )

    features = builder.fit_transform(demand)

    train = features[features["particion"] == "entrenamiento"]
    valid = features[features["particion"] == "validacion"]

    assert not train.empty
    assert not valid.empty
    assert train["fecha_hora"].max() < valid["fecha_hora"].min()
    assert features["particion"].notna().all()

    # 6. Entrenamiento, predicción y métricas
    _model, predictions, metrics = fit_and_score(
        PoissonRegressor(alpha=0.01, max_iter=300),
        train,
        valid,
    )

    # 7. Validar predicciones
    assert len(predictions) == len(valid)
    assert {"pred_salidas", "pred_llegadas"}.issubset(predictions.columns)

    prediction_values = predictions[["pred_salidas", "pred_llegadas"]].to_numpy()

    assert np.isfinite(prediction_values).all()
    assert (prediction_values >= 0).all()

    # 8. Evaluación independiente de las predicciones
    evaluated = score_frame(
        valid,
        {
            "salidas": predictions["pred_salidas"],
            "llegadas": predictions["pred_llegadas"],
        },
    )

    assert metrics
    assert evaluated
    assert all(np.isfinite(value) for value in evaluated.values())
