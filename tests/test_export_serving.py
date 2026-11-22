import json

import pytest

from mibici_mlops.api import PredictionRequest, PredictionResponse
from mibici_mlops.features import FEATURE_COLUMNS
from mibici_mlops.modeling.export import export_model
from mibici_mlops.modeling.predict import KEYS
from mibici_mlops.modeling.verify_serving import verify_serving


def test_exportacion_incluye_artefacto_version_y_referencia(
    tmp_path, monkeypatch, serving_model_path, trained_serving_model
):
    _, valid, *_ = trained_serving_model
    features_path = tmp_path / "features.parquet"
    valid[KEYS + FEATURE_COLUMNS].to_parquet(features_path, index=False)
    monkeypatch.setattr(
        "mibici_mlops.modeling.export.mlflow.artifacts.download_artifacts",
        lambda **kwargs: str(serving_model_path),
    )
    output = tmp_path / "exported"
    metadata = export_model("models:/MiBici_Demanda/1", features_path, output)
    assert (output / "MLmodel").is_file()
    assert metadata["model_uri"] == "models:/MiBici_Demanda/1"
    assert metadata["artifact_sha256"]
    payload = PredictionRequest.model_validate_json((output / "example_request.json").read_text())
    reference = PredictionResponse.model_validate_json(
        (output / "example_response.json").read_text()
    )
    assert len(payload.records) == len(reference.predictions) == 5
    with pytest.raises(FileExistsError):
        export_model("models:/MiBici_Demanda/1", features_path, output)


def test_exportacion_exige_version_numerica(tmp_path):
    with pytest.raises(ValueError, match="version numerica"):
        export_model("models:/MiBici_Demanda/latest", tmp_path / "data", tmp_path / "out")


def test_comparacion_http_detecta_desviaciones(
    tmp_path, monkeypatch, request_payload, trained_serving_model
):
    from io import BytesIO

    _, valid, expected, _ = trained_serving_model
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "example_request.json").write_text(json.dumps(request_payload))
    response = PredictionResponse(
        predictions=expected.head(2).to_dict(orient="records")
    ).model_dump_json()
    (model_path / "example_response.json").write_text(response)
    (model_path / "serving_metadata.json").write_text(
        json.dumps({"model_uri": "models:/MiBici_Demanda/1"})
    )
    monkeypatch.setattr(
        "mibici_mlops.modeling.verify_serving.urlopen",
        lambda *args, **kwargs: BytesIO(response.encode()),
    )
    evidence = tmp_path / "evidence.json"
    assert verify_serving("http://localhost:8000", model_path, evidence)["passed"]
    changed = json.loads(response)
    changed["predictions"][0]["pred_salidas"] += 1
    monkeypatch.setattr(
        "mibici_mlops.modeling.verify_serving.urlopen",
        lambda *args, **kwargs: BytesIO(json.dumps(changed).encode()),
    )
    with pytest.raises(ValueError, match="exceden la tolerancia"):
        verify_serving("http://localhost:8000", model_path, evidence)
    assert not json.loads(evidence.read_text())["passed"]
