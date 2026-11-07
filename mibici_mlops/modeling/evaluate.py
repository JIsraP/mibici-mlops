"""Metricas para evaluar modelos de demanda por estacion y hora.

Todas las corridas (lineas base y modelos) se evaluan con estas funciones para que sus
metricas sean comparables. Cada metrica se registra en MLflow como `{objetivo}_{metrica}`,
por ejemplo `salidas_mae_pico`.

- mae: error absoluto medio, en viajes por estacion y hora; es la que entiende la operacion.
- rmse: penaliza mas los errores grandes, como los de horas pico.
- mae_pico: MAE solo en horas pico de dias laborales, donde se decide la redistribucion.
- poisson: desviacion de Poisson, adecuada para conteos con muchos ceros (53 % de las horas).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error

from mibici_mlops.config import PEAK_HOURS

METRICS = ("mae", "rmse", "mae_pico", "poisson")
SEGMENTS = ("baja", "media", "alta")


def peak_mask(frame: pd.DataFrame) -> np.ndarray:
    """Horas pico de dias laborales segun `config.PEAK_HOURS`."""
    return (frame["hora"].isin(PEAK_HOURS) & (frame["es_fin_semana"] == 0)).to_numpy()


def score(y_true, y_pred, peak) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)  # no hay conteos negativos
    peak = np.asarray(peak, dtype=bool)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_pico": float(mean_absolute_error(y_true[peak], y_pred[peak])),
        # La desviacion de Poisson exige predicciones positivas
        "poisson": float(mean_poisson_deviance(y_true, np.clip(y_pred, 1e-6, None))),
    }


def score_frame(frame: pd.DataFrame, predictions: dict[str, np.ndarray]) -> dict[str, float]:
    """Metricas de cada objetivo con nombres listos para `mlflow.log_metrics`."""
    peak = peak_mask(frame)
    return {
        f"{target}_{name}": value
        for target, pred in predictions.items()
        for name, value in score(frame[target], pred, peak).items()
    }


def station_segments(train: pd.DataFrame, target: str = "salidas") -> pd.Series:
    """Clasifica cada estacion en uso bajo, medio o alto por terciles de su demanda media."""
    mean = train.groupby("estacion_id")[target].mean()
    return pd.qcut(mean.rank(method="first"), 3, labels=SEGMENTS).rename("segmento")


def error_by(frame: pd.DataFrame, pred, target: str, by: str | pd.Series) -> pd.DataFrame:
    """MAE agrupado por una columna (`hora`) o por un mapeo de estaciones (segmentos)."""
    err = (frame[target] - np.clip(pred, 0, None)).abs()
    keys = frame["estacion_id"].map(by) if isinstance(by, pd.Series) else frame[by]
    name = by.name if isinstance(by, pd.Series) else by
    return (
        err.groupby(keys.to_numpy(), observed=True)
        .mean()
        .rename("mae")
        .rename_axis(name)
        .reset_index()
    )


def improvement_pct(model: float, baseline: float) -> float:
    """Reduccion porcentual del error frente a la linea base (positivo = mejor)."""
    return 100 * (baseline - model) / baseline
