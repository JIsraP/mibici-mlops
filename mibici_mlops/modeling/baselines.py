"""Lineas base de demanda: la referencia que cualquier modelo debe superar.

Cada linea base se registra como una corrida propia con la etiqueta `tipo = linea_base`, las
metricas de `evaluate.score_frame` y las predicciones de validacion como artefacto, igual que
se pide a las corridas de modelos. Asi la comparacion de `compare.py` las trata a todas igual.

Uso:
    python -m mibici_mlops.modeling.baselines                       # version oficial, Fase 2
    python -m mibici_mlops.modeling.baselines --variant estricta \
        --experiment MiBici_Fase1_Experimentos_Datos                # comparar limpiezas
"""

from pathlib import Path
import tempfile

from loguru import logger
import mlflow
import pandas as pd
import typer

from mibici_mlops.config import EXPERIMENT_MODELS, MLFLOW_TRACKING_URI, PROCESSED_DATA_DIR
from mibici_mlops.features import TARGETS
from mibici_mlops.modeling.evaluate import score_frame
from mibici_mlops.tracking import file_sha256

app = typer.Typer()

KEYS = ["estacion_id", "fecha_hora"]


def historical_mean(train: pd.DataFrame, valid: pd.DataFrame, target: str) -> pd.Series:
    """Promedio de la misma estacion, dia de la semana y hora en entrenamiento."""
    keys = ["estacion_id", "dia_semana", "hora"]
    means = train.groupby(keys)[target].mean().rename("pred").reset_index()
    return valid[keys].merge(means, how="left")["pred"].fillna(0)


def same_hour_last_week(train: pd.DataFrame, valid: pd.DataFrame, target: str) -> pd.Series:
    """Lo que paso a la misma hora siete dias antes."""
    return valid[f"{target}_lag_168"].fillna(0).reset_index(drop=True)


BASELINES = {
    "promedio_historico": historical_mean,
    "misma_hora_semana_pasada": same_hour_last_week,
}


def predict_baseline(name: str, train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    return {t: BASELINES[name](train, valid, t).to_numpy() for t in TARGETS}


def log_baseline(name: str, features: pd.DataFrame, truth: pd.DataFrame, tags: dict, params: dict):
    train = features[features["particion"] == "entrenamiento"]
    valid = features[features["particion"] == "validacion"].reset_index(drop=True)
    # La linea base se ajusta con la variante elegida y se evalua contra la verdad oficial
    aligned = truth[KEYS].merge(valid, on=KEYS, how="left")
    predictions = predict_baseline(name, train, aligned)
    metrics = score_frame(truth, predictions)

    with mlflow.start_run(run_name=f"linea_base_{name}"):
        mlflow.set_tags({"tipo": "linea_base", **tags})
        mlflow.log_params({"modelo": name, **params})
        mlflow.log_metrics(metrics)
        preds = truth[KEYS].assign(**{f"pred_{t}": p for t, p in predictions.items()})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "predicciones_validacion.parquet"
            preds.to_parquet(out, index=False)
            mlflow.log_artifact(str(out))
    return metrics


@app.command()
def main(
    variant: str = "oficial",
    truth_variant: str = "oficial",
    experiment: str = EXPERIMENT_MODELS,
):
    path = PROCESSED_DATA_DIR / f"variables_{variant}.parquet"
    truth_path = PROCESSED_DATA_DIR / f"variables_{truth_variant}.parquet"
    features = pd.read_parquet(path)
    truth = features if path == truth_path else pd.read_parquet(truth_path)
    truth = truth[truth["particion"] == "validacion"].reset_index(drop=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)
    dataset_sha256 = file_sha256(path)
    tags = {
        "dataset_sha256": dataset_sha256,
        "validacion_sha256": dataset_sha256 if path == truth_path else file_sha256(truth_path),
    }
    params = {"variante_datos": variant, "variante_verdad": truth_variant}
    for name in BASELINES:
        metrics = log_baseline(name, features, truth, tags, params)
        logger.info(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))


if __name__ == "__main__":
    app()
