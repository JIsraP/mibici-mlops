"""Lineas base de demanda para comparar versiones de datos.

En la Fase 1 no se entrena aun el modelo final: se mide que tan bien predicen dos lineas base
construidas con cada variante de limpieza, siempre contra la misma verdad (validacion de la
variante estricta), para saber si la limpieza cambia la senal.

Uso:
    python -m mibici.modeling.train --variant laxa
"""

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
import typer

from mibici.config import EXPERIMENT_BASELINE, MLFLOW_TRACKING_URI, PROCESSED_DATA_DIR
from mibici.features import TARGETS
from mibici.tracking import file_sha256

app = typer.Typer()

KEYS = ["estacion_id", "fecha_hora"]
PEAK_HOURS = [7, 8, 9, 18, 19, 20]


def historical_mean(train: pd.DataFrame, target: str) -> pd.DataFrame:
    """Promedio de la misma estacion, dia de la semana y hora en entrenamiento."""
    return (
        train.groupby(["estacion_id", "dia_semana", "hora"])[target]
        .mean()
        .rename("pred")
        .reset_index()
    )


def scores(y: pd.Series, pred: pd.Series, peak: pd.Series) -> dict:
    err = y - pred
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mae_pico": float(err[peak].abs().mean()),
    }


def evaluate(features: pd.DataFrame, truth: pd.DataFrame) -> dict:
    train = features[features["particion"] == "entrenamiento"]
    valid = features[features["particion"] == "validacion"]
    peak = truth["hora"].isin(PEAK_HOURS) & (truth["es_fin_semana"] == 0)
    results = {}
    for t in TARGETS:
        hist = truth[["estacion_id", "dia_semana", "hora"]].merge(
            historical_mean(train, t), how="left"
        )
        lagged = truth[KEYS].merge(valid[KEYS + [f"{t}_lag_168"]], how="left")
        methods = {
            "promedio_historico": hist["pred"].fillna(0),
            "misma_hora_semana_pasada": lagged[f"{t}_lag_168"].fillna(0),
        }
        for method, pred in methods.items():
            for k, v in scores(truth[t], pred.values, peak.values).items():
                results[f"{t}_{method}_{k}"] = v
    return results


@app.command()
def main(variant: str = "estricta", truth_variant: str = "estricta"):
    path = PROCESSED_DATA_DIR / f"variables_{variant}.parquet"
    features = pd.read_parquet(path)
    truth_path = PROCESSED_DATA_DIR / f"variables_{truth_variant}.parquet"
    truth = pd.read_parquet(truth_path)
    truth = truth[truth["particion"] == "validacion"].reset_index(drop=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_BASELINE)
    with mlflow.start_run(run_name=f"lineas_base_{variant}"):
        mlflow.set_tags(
            {"dataset_sha256": file_sha256(path), "verdad_sha256": file_sha256(truth_path)}
        )
        mlflow.log_params({"variante_datos": variant, "variante_verdad": truth_variant})
        mlflow.log_input(
            mlflow.data.from_pandas(features.head(10_000), source=str(path), name=f"v3_{variant}"),
            context="entrenamiento",
        )
        metrics = evaluate(features, truth)
        mlflow.log_metrics(metrics)
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.3f}")


if __name__ == "__main__":
    app()
