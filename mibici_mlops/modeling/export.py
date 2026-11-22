"""Exporta un modelo versionado de MLflow y ejemplos para verificar el servicio."""

import json
from pathlib import Path
import re
import shutil

import mlflow
import mlflow.sklearn
import pyarrow.parquet as pq
import typer

from mibici_mlops.api import PredictionRequest, PredictionResponse
from mibici_mlops.config import MLFLOW_TRACKING_URI, MODELS_DIR
from mibici_mlops.features import FEATURE_COLUMNS
from mibici_mlops.modeling.predict import KEYS, predict_frame
from mibici_mlops.tracking import file_sha256

app = typer.Typer()


def export_model(model_uri: str, features_path: Path, output_path: Path) -> dict:
    if re.fullmatch(r"models:/[^/]+/[1-9][0-9]*", model_uri) is None:
        raise ValueError("Usa un URI con version numerica: models:/MiBici_Demanda/1")
    if output_path.exists():
        raise FileExistsError(f"La carpeta de destino ya existe: {output_path}")

    batches = pq.ParquetFile(features_path).iter_batches(
        batch_size=5, columns=KEYS + FEATURE_COLUMNS
    )
    batch = next(batches, None)
    if batch is None:
        raise ValueError("El archivo de variables esta vacio")
    frame = batch.to_pandas()
    payload = PredictionRequest.model_validate(
        {"records": json.loads(frame.to_json(orient="records", date_format="iso"))}
    )

    source = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri))
    model = mlflow.sklearn.load_model(str(source))
    result = PredictionResponse(predictions=predict_frame(model, frame).to_dict(orient="records"))
    metadata = {
        "model_uri": model_uri,
        "mlmodel_sha256": file_sha256(source / "MLmodel"),
        "features_sha256": file_sha256(features_path),
        "records": len(payload.records),
        "artifact_sha256": {
            str(path.relative_to(source)): file_sha256(path)
            for path in sorted(source.rglob("*"))
            if path.is_file()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output_path)
    (output_path / "example_request.json").write_text(
        payload.model_dump_json(indent=2), encoding="utf-8"
    )
    (output_path / "example_response.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    (output_path / "serving_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


@app.command()
def main(model_uri: str, features_path: Path, output_path: Path = MODELS_DIR / "serving"):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        metadata = export_model(model_uri, features_path, output_path)
    except (ValueError, FileExistsError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    app()
