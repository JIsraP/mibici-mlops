"""Prediccion por lote sobre variables preparadas por estacion y hora."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

from loguru import logger
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import typer

from mibici_mlops.config import MLFLOW_TRACKING_URI
from mibici_mlops.features import FEATURE_COLUMNS, TARGETS

app = typer.Typer(pretty_exceptions_show_locals=False)
KEYS = ["estacion_id", "fecha_hora"]


def predict_frame(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    values = np.clip(model.predict(features[FEATURE_COLUMNS]), 0, None)
    return features[KEYS].assign(
        **{f"pred_{target}": values[:, i] for i, target in enumerate(TARGETS)}
    )


@app.command()
def main(
    model_uri: str,
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_path: Path,
    date: Annotated[datetime | None, typer.Option(formats=["%Y-%m-%d"])] = None,
):
    if input_path.resolve() == output_path.resolve():
        raise typer.BadParameter("El archivo de salida debe ser distinto de la entrada")

    filters = None
    if date is not None:
        filters = [("fecha_hora", ">=", date), ("fecha_hora", "<", date + timedelta(days=1))]
    features = pd.read_parquet(input_path, columns=KEYS + FEATURE_COLUMNS, filters=filters)
    if features.empty:
        raise typer.BadParameter("No hay filas de entrada para predecir")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(model_uri)
    predictions = predict_frame(model, features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path, index=False)
    logger.success(f"{len(predictions):,} predicciones en {output_path}")


if __name__ == "__main__":
    app()
