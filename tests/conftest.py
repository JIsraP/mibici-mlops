import json

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from mibici_mlops.features import FEATURE_COLUMNS
from mibici_mlops.modeling.predict import KEYS
from mibici_mlops.modeling.train import fit_and_score


@pytest.fixture(scope="module")
def trained_serving_model():
    hours = pd.date_range("2025-11-03", periods=72, freq="h")
    frame = pd.DataFrame(
        {
            "estacion_id": np.tile([1, 2], 36),
            "fecha_hora": hours,
            "hora": hours.hour,
            "hora_sin": np.sin(2 * np.pi * hours.hour / 24),
            "hora_cos": np.cos(2 * np.pi * hours.hour / 24),
            "latitud": 20.67,
            "longitud": -103.35,
            "capacidad": 20,
            "perfil_pc1": 0.1,
            "perfil_pc2": -0.2,
            "es_fin_semana": 0,
            "es_festivo": 0,
            "dia_semana": hours.dayofweek,
            "zona": "CENTRO",
        }
    )
    for target, offset in [("salidas", 1), ("llegadas", 2)]:
        frame[target] = hours.hour % 7 + offset
        for name in ["lag_24", "lag_168", "media_7d"]:
            frame[f"{target}_{name}"] = hours.hour % 6 + offset
    model, predictions, metrics = fit_and_score(
        HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=5,
            min_samples_leaf=2,
            early_stopping=False,
            random_state=42,
        ),
        frame.iloc[:48],
        frame.iloc[48:],
    )
    return model, frame.iloc[48:].copy(), predictions, metrics


@pytest.fixture
def serving_model_path(tmp_path, trained_serving_model):
    model, *_ = trained_serving_model
    path = tmp_path / "serving"
    mlflow.sklearn.save_model(
        model,
        str(path),
        serialization_format="skops",
        skops_trusted_types=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],
        pip_requirements=[],
    )
    return path


@pytest.fixture
def request_payload(trained_serving_model):
    _, valid, *_ = trained_serving_model
    return {
        "records": json.loads(
            valid[KEYS + FEATURE_COLUMNS].head(2).to_json(orient="records", date_format="iso")
        )
    }
