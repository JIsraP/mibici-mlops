"""Compara la respuesta HTTP con la referencia del mismo modelo exportado."""

import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import typer

from mibici_mlops.api import PredictionRequest, PredictionResponse
from mibici_mlops.config import MODELS_DIR, REPORTS_DIR

app = typer.Typer()


def verify_serving(url: str, model_path: Path, output_path: Path) -> dict:
    request_text = (model_path / "example_request.json").read_text(encoding="utf-8")
    payload = PredictionRequest.model_validate_json(request_text)
    expected = PredictionResponse.model_validate_json(
        (model_path / "example_response.json").read_text(encoding="utf-8")
    )
    request = Request(
        url.rstrip("/") + "/predict",
        data=payload.model_dump_json().encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        actual = PredictionResponse.model_validate_json(response.read())

    if len(actual.predictions) != len(expected.predictions):
        raise ValueError("La cantidad de predicciones difiere de la referencia")
    for left, right in zip(actual.predictions, expected.predictions, strict=True):
        if (left.estacion_id, left.fecha_hora) != (right.estacion_id, right.fecha_hora):
            raise ValueError("Las estaciones o fechas difieren de la referencia")

    def values(rows):
        return np.array([[row.pred_salidas, row.pred_llegadas] for row in rows], dtype=float)

    reference_values = values(expected.predictions)
    actual_values = values(actual.predictions)
    identical = np.allclose(reference_values, actual_values, rtol=1e-6, atol=1e-8)
    result = {
        "url": url,
        "model_uri": json.loads(
            (model_path / "serving_metadata.json").read_text(encoding="utf-8")
        )["model_uri"],
        "records": len(actual.predictions),
        "rtol": 1e-6,
        "atol": 1e-8,
        "max_absolute_difference": float(np.max(np.abs(reference_values - actual_values))),
        "passed": bool(identical),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not identical:
        raise ValueError(f"Las predicciones exceden la tolerancia. Evidencia: {output_path}")
    return result


@app.command()
def main(
    url: str = "http://127.0.0.1:8000",
    model_path: Path = MODELS_DIR / "serving",
    output_path: Path = REPORTS_DIR / "reproducibilidad_serving.json",
):
    typer.echo(json.dumps(verify_serving(url, model_path, output_path), indent=2))


if __name__ == "__main__":
    app()
