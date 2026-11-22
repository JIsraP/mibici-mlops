from fastapi.testclient import TestClient
import numpy as np
import pytest

from mibici_mlops.api import create_app


def test_api_coincide_con_prediccion_directa(
    serving_model_path, request_payload, trained_serving_model
):
    _, valid, expected, _ = trained_serving_model
    with TestClient(create_app(serving_model_path)) as client:
        response = client.post("/predict", json=request_payload)
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert [row["estacion_id"] for row in predictions] == valid["estacion_id"].head(2).tolist()
    np.testing.assert_allclose(
        [[row["pred_salidas"], row["pred_llegadas"]] for row in predictions],
        expected[["pred_salidas", "pred_llegadas"]].head(2).to_numpy(),
        rtol=1e-6,
        atol=1e-8,
    )


@pytest.mark.parametrize("invalid", ["missing", "negative", "empty", "extra", "hour"])
def test_api_rechaza_entradas_invalidas(serving_model_path, request_payload, invalid):
    if invalid == "missing":
        del request_payload["records"][0]["salidas_lag_24"]
    elif invalid == "negative":
        request_payload["records"][0]["llegadas_lag_168"] = -1
    elif invalid == "empty":
        request_payload["records"] = []
    elif invalid == "extra":
        request_payload["records"][0]["salidas"] = 10
    else:
        request_payload["records"][0]["hora"] = 24
    with TestClient(create_app(serving_model_path)) as client:
        assert client.post("/predict", json=request_payload).status_code == 422


def test_swagger_documenta_schema_y_endpoint(serving_model_path):
    with TestClient(create_app(serving_model_path)) as client:
        assert client.get("/docs").status_code == 200
        schema = client.get("/openapi.json").json()
    assert "post" in schema["paths"]["/predict"]
    assert {"PredictionRow", "PredictionRequest", "PredictionResponse"}.issubset(
        schema["components"]["schemas"]
    )


def test_api_falla_al_iniciar_si_falta_modelo(tmp_path):
    with pytest.raises(RuntimeError, match="Falta el artefacto completo"):
        with TestClient(create_app(tmp_path / "missing")):
            pass
