"""Ajuste temporal, entrenamiento y evaluacion de los modelos de demanda."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from loguru import logger
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
import typer

from mibici_mlops.config import (
    EXPERIMENT_MODELS,
    MLFLOW_TRACKING_URI,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    REGISTERED_MODEL_NAME,
    SELECTION_METRIC,
    TARGET_IMPROVEMENT_PCT,
    TRAIN_END,
    TUNING_START,
    YEAR_END,
    YEAR_START,
)
from mibici_mlops.features import (
    FEATURE_COLUMNS,
    STATION_NUMERIC,
    TARGETS,
    FeatureBuilder,
    build_preprocessor,
)
from mibici_mlops.modeling.compare import comparison_table, load_runs
from mibici_mlops.modeling.evaluate import improvement_pct, peak_mask, score_frame
from mibici_mlops.modeling.predict import KEYS, predict_frame
from mibici_mlops.tracking import file_sha256

app = typer.Typer(pretty_exceptions_show_locals=False)
CANDIDATES = {
    "poisson": (PoissonRegressor, {"alpha": [0.001, 0.01], "solver": ["newton-cholesky"]}),
    "hgb_poisson": (
        HistGradientBoostingRegressor,
        {
            "max_leaf_nodes": [31, 63],
            "loss": ["poisson"],
            "max_iter": [100],
            "early_stopping": [False],
            "random_state": [RANDOM_SEED],
        },
    ),
}


def fit_and_score(estimator, train: pd.DataFrame, valid: pd.DataFrame):
    preprocessor = build_preprocessor().set_params(sparse_threshold=0)
    pipeline = make_pipeline(preprocessor, MultiOutputRegressor(estimator))
    pipeline.fit(train[FEATURE_COLUMNS], train[TARGETS])
    predictions = predict_frame(pipeline, valid)
    metrics = score_frame(valid, {target: predictions[f"pred_{target}"] for target in TARGETS})
    return pipeline, predictions, metrics


def train_candidate(name, train, valid, tuning_train, tuning_valid, data_tags):
    estimator, grid = CANDIDATES[name]
    trials = []
    for params in ParameterGrid(grid):
        _, _, metrics = fit_and_score(estimator(**params), tuning_train, tuning_valid)
        trials.append({"params": params, "metrics": metrics})
        logger.info(f"{name} {params}: MAE pico de ajuste = {metrics[SELECTION_METRIC]:.3f}")

    best = min(
        trials,
        key=lambda trial: (trial["metrics"][SELECTION_METRIC], trial["metrics"]["salidas_mae"]),
    )
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"tipo": "modelo", **data_tags})
        mlflow.log_params(
            {
                "modelo": name,
                "semilla": RANDOM_SEED,
                "inicio_validacion_interna": TUNING_START,
                "inicio_validacion_final": TRAIN_END,
                **best["params"],
            }
        )
        mlflow.log_dict({"pruebas": trials}, "ajuste_hiperparametros.json")
        pipeline, predictions, metrics = fit_and_score(estimator(**best["params"]), train, valid)
        mlflow.log_metrics(metrics)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "predicciones_validacion.parquet"
            predictions.to_parquet(path, index=False)
            mlflow.log_artifact(str(path))

        trusted_types = (
            ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]
            if estimator is HistGradientBoostingRegressor
            else None
        )
        model = mlflow.sklearn.log_model(
            pipeline,
            name="modelo",
            input_example=train[FEATURE_COLUMNS].head(),
            serialization_format="skops",
            skops_trusted_types=trusted_types,
        )
        logger.success(f"{name}: MAE pico = {metrics[SELECTION_METRIC]:.3f} | {model.model_uri}")


def register_best(experiment: str, dataset_sha256: str):
    runs = load_runs(experiment)
    if "tags.dataset_sha256" not in runs:
        raise typer.BadParameter("No hay corridas con una versión de datos registrada")
    runs = runs[runs["tags.dataset_sha256"] == dataset_sha256]
    if runs.empty:
        raise typer.BadParameter("No hay corridas para este archivo de variables")

    table = comparison_table(runs)
    baselines = table[table["tipo"] == "linea_base"]
    if baselines.empty:
        raise typer.BadParameter("Ejecuta make baseline antes de registrar el modelo")
    baseline = baselines.iloc[0]
    improvement = improvement_pct(table[SELECTION_METRIC], baseline[SELECTION_METRIC])
    eligible = table[
        (table["tipo"] == "modelo")
        & (improvement >= TARGET_IMPROVEMENT_PCT)
        & (table["llegadas_mae_pico"] <= baseline["llegadas_mae_pico"])
    ]
    if eligible.empty:
        raise typer.BadParameter(
            "Ningún modelo cumple la meta de salidas y el criterio de llegadas"
        )

    client = mlflow.MlflowClient()
    run = client.get_run(eligible.iloc[0]["run_id"])
    versions = client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
    for version in versions:
        if version.run_id == run.info.run_id and version.status == "READY":
            return version

    model_id = run.outputs.model_outputs[0].model_id
    tags = {**run.data.params, **run.data.metrics, "dataset_sha256": dataset_sha256}
    return mlflow.register_model(f"models:/{model_id}", REGISTERED_MODEL_NAME, tags=tags)


@app.command()
def main(
    model: str = "all",
    features_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = (
        PROCESSED_DATA_DIR / "variables_oficial.parquet"
    ),
    demand_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = (
        PROCESSED_DATA_DIR / "demanda_estacion_hora_oficial.parquet"
    ),
    experiment: str = EXPERIMENT_MODELS,
    register_only: bool = False,
):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    dataset_sha256 = file_sha256(features_path)
    if register_only:
        version = register_best(experiment, dataset_sha256)
        logger.success(f"Modelo registrado: models:/{version.name}/{version.version}")
        return

    if model != "all" and model not in CANDIDATES:
        raise typer.BadParameter("El modelo debe ser all, poisson o hgb_poisson")

    cutoff = pd.Timestamp(TRAIN_END)
    features = pd.read_parquet(
        features_path,
        columns=KEYS + FEATURE_COLUMNS + TARGETS,
        filters=[
            ("fecha_hora", ">=", pd.Timestamp(YEAR_START)),
            ("fecha_hora", "<", pd.Timestamp(YEAR_END)),
        ],
    )
    train = features[features["fecha_hora"] < cutoff]
    valid = features[features["fecha_hora"] >= cutoff]

    stations = train[["estacion_id", *STATION_NUMERIC, "zona"]].drop_duplicates()
    demand = pd.read_parquet(
        demand_path,
        columns=KEYS + TARGETS,
        filters=[("fecha_hora", ">=", pd.Timestamp(YEAR_START)), ("fecha_hora", "<", cutoff)],
    )
    tuning = FeatureBuilder(stations, train_end=TUNING_START).fit_transform(demand)
    tuning_train = tuning[tuning["particion"] == "entrenamiento"]
    tuning_valid = tuning[tuning["particion"] == "validacion"]
    del features, demand, tuning
    if (
        train.empty
        or tuning_train.empty
        or not peak_mask(valid).any()
        or not peak_mask(tuning_valid).any()
    ):
        raise typer.BadParameter("Los datos deben cubrir las particiones y sus horas pico")

    mlflow.set_experiment(experiment)
    data_tags = {
        "dataset_sha256": dataset_sha256,
        "demanda_sha256": file_sha256(demand_path),
    }
    for name in CANDIDATES if model == "all" else [model]:
        train_candidate(name, train, valid, tuning_train, tuning_valid, data_tags)


if __name__ == "__main__":
    app()
