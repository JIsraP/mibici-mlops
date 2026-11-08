"""Comparacion de lineas base y modelos registrados en MLflow.

Compara las corridas vigentes evaluadas con las mismas variables de validacion y marca
los modelos que cumplen la meta de mejora sin empeorar el MAE de llegadas en horas pico.
Si una corrida guarda `predicciones_validacion.parquet`, tambien desglosa su error por hora y
por segmento de estacion.

Uso:
    python -m mibici_mlops.modeling.compare   # escribe reports/comparacion_modelos.csv y figuras
"""

from pathlib import Path
from typing import Annotated

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import typer

from mibici_mlops.config import (
    EXPERIMENT_MODELS,
    FIGURES_DIR,
    MLFLOW_TRACKING_URI,
    OFFICIAL_FEATURES_PATH,
    REPORTS_DIR,
    SELECTION_METRIC,
    TARGET_IMPROVEMENT_PCT,
)
from mibici_mlops.features import TARGETS
from mibici_mlops.modeling.evaluate import METRICS, error_by, improvement_pct, station_segments
from mibici_mlops.plots import INK_2, SERIES, _save
from mibici_mlops.tracking import file_sha256

app = typer.Typer()

METRIC_COLUMNS = [f"{t}_{m}" for t in TARGETS for m in METRICS]
SECONDARY_METRIC = "salidas_mae"
PREDICTIONS_ARTIFACT = "predicciones_validacion.parquet"


def load_runs(
    experiment: str = EXPERIMENT_MODELS, dataset_sha256: str | None = None
) -> pd.DataFrame:
    runs = mlflow.search_runs(experiment_names=[experiment])
    if runs.empty:
        return runs
    if "tags.estado" in runs:
        runs = runs[runs["tags.estado"].fillna("") != "descartada"]
    runs = runs[runs["status"] == "FINISHED"]
    if dataset_sha256 is None:
        return runs

    empty_hash = pd.Series(index=runs.index, dtype="object")
    legacy_hash = runs.get("tags.dataset_sha256", empty_hash)
    if {"params.variante_datos", "params.variante_verdad"}.issubset(runs.columns):
        data_variant = runs["params.variante_datos"]
        truth_variant = runs["params.variante_verdad"]
        mixed = data_variant.notna() & truth_variant.notna() & data_variant.ne(truth_variant)
        legacy_hash = legacy_hash.mask(mixed)
    validation_hash = runs.get("tags.validacion_sha256", empty_hash).fillna(legacy_hash)
    return runs[validation_hash.eq(dataset_sha256)]


def comparison_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Una fila por corrida con sus metricas y su mejora frente a la mejor linea base."""
    cols = {f"metrics.{m}": m for m in METRIC_COLUMNS}
    table = runs.reindex(columns=["run_id", "tags.tipo", "params.modelo", *cols]).rename(
        columns={"tags.tipo": "tipo", "params.modelo": "modelo", **cols}
    )
    table = table[table["tipo"].isin(["linea_base", "modelo"])]
    table = table.dropna(subset=["modelo", SELECTION_METRIC, SECONDARY_METRIC])
    table = table.sort_values([SELECTION_METRIC, SECONDARY_METRIC]).reset_index(drop=True)
    baselines = table[table["tipo"] == "linea_base"]
    table["apto_registro"] = False
    if not baselines.empty:
        baseline = baselines.iloc[0]
        improvement = improvement_pct(table[SELECTION_METRIC], baseline[SELECTION_METRIC])
        table["mejora_vs_base_pct"] = improvement.round(1)
        table["cumple_meta"] = improvement.ge(TARGET_IMPROVEMENT_PCT)
        table["apto_registro"] = (
            table["tipo"].eq("modelo")
            & table["cumple_meta"]
            & table["llegadas_mae_pico"].le(baseline["llegadas_mae_pico"])
        )
    return table


def select_best(table: pd.DataFrame) -> pd.Series:
    """Prefiere candidatos al registro y ordena por MAE de salidas en pico y general."""
    models = table[table["tipo"] == "modelo"]
    pool = models if not models.empty else table
    if "apto_registro" in pool and pool["apto_registro"].any():
        pool = pool[pool["apto_registro"]]
    return pool.sort_values([SELECTION_METRIC, SECONDARY_METRIC]).iloc[0]


def comparison_plot(table: pd.DataFrame, out_dir: Path) -> Path:
    # Una barra por modelo: su mejor corrida (el ajuste de hiperparametros genera varias)
    t = table.drop_duplicates("modelo").sort_values(SELECTION_METRIC, ascending=False)
    colors = [SERIES[1] if k == "linea_base" else SERIES[0] for k in t["tipo"]]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(t) + 1.6))
    ax.barh(t["modelo"], t[SELECTION_METRIC], color=colors, height=0.6)
    for y, v in enumerate(t[SELECTION_METRIC]):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("MAE de salidas en horas pico laborales (viajes por estación y hora)")
    ax.set_title("Modelos frente a líneas base (naranja)")
    ax.grid(axis="y", visible=False)
    return _save(fig, "13_comparacion_modelos", out_dir)


def run_predictions(run_id: str) -> pd.DataFrame | None:
    names = {f.path for f in mlflow.MlflowClient().list_artifacts(run_id)}
    if PREDICTIONS_ARTIFACT not in names:  # la corrida no guardo predicciones
        return None
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=PREDICTIONS_ARTIFACT)
    return pd.read_parquet(path)


def diagnose(
    run_id: str, name: str, features_path: Path = OFFICIAL_FEATURES_PATH
) -> pd.DataFrame | None:
    """Error por hora y por segmento de estacion de una corrida con predicciones guardadas."""
    preds = run_predictions(run_id)
    if preds is None:
        return None
    features = pd.read_parquet(features_path)
    valid = features[features["particion"] == "validacion"]
    frame = valid.merge(preds, on=["estacion_id", "fecha_hora"], how="inner")
    segments = station_segments(features[features["particion"] == "entrenamiento"])
    by_hour = error_by(frame, frame["pred_salidas"], "salidas", "hora").assign(modelo=name)
    by_segment = error_by(frame, frame["pred_salidas"], "salidas", segments).assign(modelo=name)
    return pd.concat([by_hour.assign(corte="hora"), by_segment.assign(corte="segmento")])


def hourly_error_plot(diag: pd.DataFrame, out_dir: Path) -> Path:
    hourly = diag[diag["corte"] == "hora"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for color, (name, g) in zip(SERIES * 3, hourly.groupby("modelo")):
        ax.plot(g["hora"], g["mae"], color=color, lw=2, label=name)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("MAE de salidas")
    ax.set_title("Error por hora en validación (noviembre y diciembre)")
    ax.legend(loc="upper left")
    return _save(fig, "14_error_por_hora", out_dir)


@app.command()
def main(
    experiment: str = EXPERIMENT_MODELS,
    features_path: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = OFFICIAL_FEATURES_PATH,
):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    runs = load_runs(experiment, file_sha256(features_path))
    table = comparison_table(runs)
    if table.empty:
        logger.error(f"No hay corridas comparables para {features_path.name} en {experiment}")
        raise typer.Exit(1)
    table.to_csv(REPORTS_DIR / "comparacion_modelos.csv", index=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    comparison_plot(table, FIGURES_DIR)

    # Diagnostico de la mejor linea base y del mejor modelo, si guardaron predicciones
    picks = [select_best(table)]
    base = table[table["tipo"] == "linea_base"]
    if not base.empty and picks[0]["tipo"] != "linea_base":
        picks.append(base.iloc[0])
    diags = [
        d for p in picks if (d := diagnose(p["run_id"], p["modelo"], features_path)) is not None
    ]
    if diags:
        diag = pd.concat(diags)
        diag.to_csv(REPORTS_DIR / "diagnostico_error.csv", index=False)
        hourly_error_plot(diag, FIGURES_DIR)

    print(table.to_string(index=False))
    best = picks[0]
    if best["apto_registro"]:
        logger.success(f"Modelo elegible para registro: {best['modelo']} ({best['run_id']})")
    else:
        logger.warning("La comparación no tiene un modelo elegible para registro")


if __name__ == "__main__":
    app()
