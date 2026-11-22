import numpy as np
import pandas as pd

from mibici_mlops.modeling.predict import KEYS, predict_frame


def test_entrenamiento_devuelve_modelo_y_metricas_finitas(trained_serving_model):
    model, valid, predictions, metrics = trained_serving_model
    assert model is not None
    assert len(predictions) == len(valid)
    assert len(metrics) == 8
    assert all(np.isfinite(value) and value >= 0 for value in metrics.values())


def test_prediccion_conserva_llaves_y_no_depende_del_orden_de_columnas(
    trained_serving_model,
):
    model, valid, expected, _ = trained_serving_model
    actual = predict_frame(model, valid[valid.columns[::-1]])
    pd.testing.assert_frame_equal(actual[KEYS], valid[KEYS])
    np.testing.assert_allclose(
        actual[["pred_salidas", "pred_llegadas"]],
        expected[["pred_salidas", "pred_llegadas"]],
    )
    assert (actual[["pred_salidas", "pred_llegadas"]].to_numpy() >= 0).all()


def test_artefacto_skops_conserva_predicciones(serving_model_path, trained_serving_model):
    import mlflow.sklearn

    _, valid, expected, _ = trained_serving_model
    loaded = mlflow.sklearn.load_model(str(serving_model_path))
    actual = predict_frame(loaded, valid)
    np.testing.assert_allclose(
        actual[["pred_salidas", "pred_llegadas"]],
        expected[["pred_salidas", "pred_llegadas"]],
        rtol=1e-6,
        atol=1e-8,
    )
