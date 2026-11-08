"""Comparacion de lineas base y modelos registrados en MLflow.

Lee todas las corridas vigentes del experimento, arma la tabla comparativa con la mejora de
cada corrida frente a la mejor linea base y elige el modelo con `config.SELECTION_METRIC`.
Si una corrida guarda `predicciones_validacion.parquet`, tambien desglosa su error por hora y
por segmento de estacion.

Uso:
    python -m mibici_mlops.modeling.compare   # escribe reports/comparacion_modelos.csv y figuras
"""

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import typer

from mibici_mlops.config import (
    EXPERIMENT_MODELS,
    FIGURES_DIR,
    MLFLOW_TRACKING_URI,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    SELECTION_METRIC,
    TARGET_IMPROVEMENT_PCT,
)
from mibici_mlops.features import TARGETS
from mibici_mlops.modeling.evaluate import METRICS, error_by, improvement_pct, station_segments
from mibici_mlops.plots import INK_2, SERIES, _save

app = typer.Typer()

METRIC_COLUMNS = [f"{t}_{m}" for t in TARGETS for m in METRICS]
SECONDARY_METRIC = "salidas_mae"
PREDICTIONS_ARTIFACT = "predicciones_validacion.parquet"


def load_runs(experiment: str = EXPERIMENT_MODELS) -> pd.DataFrame:
    runs = mlflow.search_runs(experiment_names=[experiment])
    if runs.empty:
        return runs
    if "tags.estado" in runs:
        runs = runs[runs["tags.estado"].fillna("") != "descartada"]
    return runs[runs["status"] == "FINISHED"]


def comparison_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Una fila por corrida con sus metricas y su mejora frente a la mejor linea base."""
    cols = {f"metrics.{m}": m for m in METRIC_COLUMNS if f"metrics.{m}" in runs}
    table = runs[["run_id", "tags.tipo", "params.modelo", *cols]].rename(
        columns={"tags.tipo": "tipo", "params.modelo": "modelo", **cols}
    )
    table = table.dropna(subset=[SELECTION_METRIC])
    baselines = table[table["tipo"] == "linea_base"]
    if not baselines.empty:
        best_base = baselines[SELECTION_METRIC].min()
        table["mejora_vs_base_pct"] = improvement_pct(table[SELECTION_METRIC], best_base).round(1)
        table["cumple_meta"] = table["mejora_vs_base_pct"] >= TARGET_IMPROVEMENT_PCT
    return table.sort_values([SELECTION_METRIC, SECONDARY_METRIC]).reset_index(drop=True)


def select_best(table: pd.DataFrame) -> pd.Series:
    """Menor error en horas pico; en empate, menor MAE general. Solo compite `tipo = modelo`."""
    models = table[table["tipo"] == "modelo"]
    pool = models if not models.empty else table
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


def diagnose(run_id: str, name: str, variant: str = "oficial") -> pd.DataFrame | None:
    """Error por hora y por segmento de estacion de una corrida con predicciones guardadas."""
    preds = run_predictions(run_id)
    if preds is None:
        return None
    features = pd.read_parquet(PROCESSED_DATA_DIR / f"variables_{variant}.parquet")
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
def main(experiment: str = EXPERIMENT_MODELS):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    runs = load_runs(experiment)
    if runs.empty:
        logger.error(f"No hay corridas en {experiment}")
        raise typer.Exit(1)
    table = comparison_table(runs)
    table.to_csv(REPORTS_DIR / "comparacion_modelos.csv", index=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    comparison_plot(table, FIGURES_DIR)

    # Diagnostico de la mejor linea base y del mejor modelo, si guardaron predicciones
    picks = [select_best(table)]
    base = table[table["tipo"] == "linea_base"]
    if not base.empty and picks[0]["tipo"] != "linea_base":
        picks.append(base.iloc[0])
    diags = [d for p in picks if (d := diagnose(p["run_id"], p["modelo"])) is not None]
    if diags:
        diag = pd.concat(diags)
        diag.to_csv(REPORTS_DIR / "diagnostico_error.csv", index=False)
        hourly_error_plot(diag, FIGURES_DIR)

    print(table.to_string(index=False))
    best = select_best(table)
    logger.success(f"Mejor corrida por {SELECTION_METRIC}: {best['modelo']} ({best['run_id']})")


if __name__ == "__main__":
    app()
