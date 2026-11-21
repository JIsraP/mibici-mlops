"""Monitoreo de data drift y de perdida de desempeno del modelo de demanda.

Se usan dos conjuntos de monitoreo, ambos comparados contra el set de validacion
(noviembre y diciembre de 2025), que es la linea base del modelo registrado:

1. Drift real: viajes de enero a octubre de 2026, reservados desde la Fase 1. Se procesan con
   el mismo contrato que 2025 (solo se excluye la duracion no positiva) y con los perfiles PCA
   y el catalogo de estaciones del entrenamiento, como los veria el modelo en produccion.
2. Drift sintetico: escenarios controlados sobre el set de validacion (crecimiento de la
   demanda, falla de una variable y caida estacional), mas un barrido de intensidad.

Deteccion:
- PSI por variable de entrada y PSI de las predicciones (ninguno necesita etiquetas).
- Degradacion del MAE en horas pico frente a la validacion y ventaja frente a la linea base
  `promedio_historico` evaluada en los mismos datos.

Uso:
    python -m mibici_mlops.drift   # reports/drift_*.csv, figuras y corridas en MLflow
"""

from collections.abc import Callable
from pathlib import Path
import tempfile

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import typer

from mibici_mlops.config import (
    BIAS_ALERT_PCT,
    BIAS_WARN_PCT,
    DEGRADATION_ALERT_PCT,
    DEGRADATION_WARN_PCT,
    EXPERIMENT_DRIFT,
    FIGURES_DIR,
    HOLIDAYS_2025,
    HOLIDAYS_2026,
    MLFLOW_TRACKING_URI,
    MONITOR_PATTERN,
    MONITOR_RAW_DIR,
    NEW_STATIONS_ALERT_PCT,
    OFFICIAL_FEATURES_PATH,
    PROCESSED_DATA_DIR,
    PSI_ALERT,
    PSI_WARN,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
    SELECTION_METRIC,
    TRIP_COLUMNS,
)
from mibici_mlops.dataset import detect_encoding
from mibici_mlops.features import (
    BINARY,
    CATEGORICAL,
    FEATURE_COLUMNS,
    STATION_NUMERIC,
    TARGETS,
    FeatureBuilder,
)
from mibici_mlops.model_input import parse_dates
from mibici_mlops.modeling.baselines import historical_mean
from mibici_mlops.modeling.evaluate import improvement_pct, score_frame
from mibici_mlops.modeling.predict import KEYS, predict_frame
from mibici_mlops.plots import INK_2, MESES, SERIES, _save

app = typer.Typer(pretty_exceptions_show_locals=False)

DISCRETE = set(CATEGORICAL + BINARY)
LAG_COLUMNS = [c for c in FEATURE_COLUMNS if "_lag_" in c or "_media_" in c]
# El calendario cambia por construccion (un mes sin festivos no es drift): se reporta su PSI,
# pero no entra a la regla de alerta de entradas
CALENDAR = ["hora", "hora_sin", "hora_cos", "dia_semana", "es_fin_semana", "es_festivo"]
MONITORED = [c for c in FEATURE_COLUMNS if c not in CALENDAR]
STATUS_ORDER = ["estable", "advertencia", "alerta"]
ACTIONS = {
    "pipeline": "Revisar el feature pipeline (calidad y frescura de datos) y despues reentrenar",
    "reentrenar": "Reentrenar con datos recientes: cambio la relacion entre entradas y demanda",
    "vigilar_entradas": "Vigilar: hay drift en entradas sin perdida de desempeno todavia",
    "vigilar": "Vigilar y revisar en la siguiente ventana de monitoreo",
    "catalogo": "Actualizar el catalogo de estaciones y reentrenar: hay estaciones sin cobertura",
    "ninguna": "Sin accion",
}


# --- Deteccion -------------------------------------------------------------------------------


def _psi_from_shares(p: np.ndarray, q: np.ndarray, eps: float = 1e-4) -> float:
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def categorical_psi(reference, current) -> float:
    ref = pd.Series(reference).value_counts(normalize=True)
    cur = pd.Series(current).value_counts(normalize=True)
    cats = ref.index.union(cur.index)
    return _psi_from_shares(
        ref.reindex(cats, fill_value=0).to_numpy(), cur.reindex(cats, fill_value=0).to_numpy()
    )


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index con cortes por cuantiles de la referencia.

    Las variables de conteo tienen muchos ceros; los cuantiles repetidos se funden y, si
    quedan menos de dos intervalos, la variable se trata como categorica.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("PSI necesita valores en la referencia y en el periodo actual")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return categorical_psi(ref, cur)
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    return _psi_from_shares(p, q)


def psi_status(value: float) -> str:
    if value >= PSI_ALERT:
        return "alerta"
    if value >= PSI_WARN:
        return "advertencia"
    return "estable"


def degradation_status(pct: float) -> str:
    if pct >= DEGRADATION_ALERT_PCT:
        return "alerta"
    if pct >= DEGRADATION_WARN_PCT:
        return "advertencia"
    return "estable"


def bias_status(pct: float) -> str:
    """Sesgo en cualquier direccion: sobrestimar tambien cuesta (camiones mal enviados)."""
    value = abs(pct)
    if value >= BIAS_ALERT_PCT:
        return "alerta"
    if value >= BIAS_WARN_PCT:
        return "advertencia"
    return "estable"


def worst(*statuses: str) -> str:
    return max(statuses, key=STATUS_ORDER.index)


def bias_pct(actual, predicted) -> float:
    """Diferencia porcentual entre lo pronosticado y lo observado en el total de la ventana."""
    total = float(np.sum(actual))
    return 100 * (float(np.sum(predicted)) - total) / total


def change_pct(reference: float, current: float) -> float:
    """Cambio porcentual del error frente a la referencia (positivo = el error crecio)."""
    return 100 * (current - reference) / reference


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, columns) -> pd.DataFrame:
    rows = []
    for col in columns:
        fn = categorical_psi if col in DISCRETE else psi
        value = fn(reference[col], current[col])
        rows.append({"variable": col, "psi": value, "estado": psi_status(value)})
    return pd.DataFrame(rows)


def decide_action(performance: str, inputs: str, new_stations: bool = False) -> str:
    """Regla de alerta: el desempeno manda; el estado de las entradas orienta el diagnostico.

    Si una parte relevante de la demanda ocurre en estaciones que el modelo no conoce, la
    accion minima es actualizar el catalogo, aunque el error en las conocidas siga estable.
    """
    if performance == "alerta":
        return "pipeline" if inputs == "alerta" else "reentrenar"
    if new_stations:
        return "catalogo"
    if performance == "advertencia":
        return "vigilar"
    if inputs != "estable":
        return "vigilar_entradas"
    return "ninguna"


# --- Datos de monitoreo reales ---------------------------------------------------------------


def load_monitor_trips(
    raw_dir: Path = MONITOR_RAW_DIR, pattern: str = MONITOR_PATTERN
) -> tuple[pd.DataFrame, dict]:
    """Lee los CSV crudos del periodo de monitoreo y aplica el contrato de 2025."""
    files = sorted(Path(raw_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No hay archivos {pattern} en {raw_dir}")
    frames = []
    for path in files:
        df = pd.read_csv(path, encoding=detect_encoding(path), dtype=str, keep_default_na=False)
        df.columns = TRIP_COLUMNS
        frames.append(df)
    trips = pd.concat(frames, ignore_index=True)
    return prepare_trips(trips, len(files))


def prepare_trips(trips: pd.DataFrame, n_files: int = 0) -> tuple[pd.DataFrame, dict]:
    raw_rows = len(trips)
    trips = trips.drop_duplicates()
    duplicates = raw_rows - len(trips)
    trips = trips.assign(inicio=parse_dates(trips["inicio"]), fin=parse_dates(trips["fin"]))
    for col in ["origen_id", "destino_id"]:
        trips[col] = pd.to_numeric(trips[col], errors="coerce")
    valid = trips[["inicio", "fin", "origen_id", "destino_id"]].notna().all(axis=1)
    duration = (trips["fin"] - trips["inicio"]).dt.total_seconds()
    keep = valid & (duration > 0)
    out = trips.loc[keep, ["inicio", "fin", "origen_id", "destino_id"]].astype(
        {"origen_id": "int64", "destino_id": "int64"}
    )
    counts = {
        "archivos": n_files,
        "viajes_crudos": raw_rows,
        "duplicados_exactos": duplicates,
        "fechas_o_ids_invalidos": int((~valid).sum()),
        "duracion_no_positiva": int((valid & (duration <= 0)).sum()),
        "viajes_monitoreo": int(keep.sum()),
    }
    return out.reset_index(drop=True), counts


def aggregate_demand(trips: pd.DataFrame, start, end) -> pd.DataFrame:
    """Salidas y llegadas por estacion y hora en [start, end), horas sin viajes en cero.

    Misma regla que `features.build_demand`, pero con el periodo como parametro.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    dep = trips[(trips["inicio"] >= start) & (trips["inicio"] < end)]
    arr = trips[(trips["fin"] >= start) & (trips["fin"] < end)]
    out = dep.groupby(["origen_id", dep["inicio"].dt.floor("h")]).size()
    inn = arr.groupby(["destino_id", arr["fin"].dt.floor("h")]).size()
    out.index.names = inn.index.names = ["estacion_id", "fecha_hora"]
    stations = np.union1d(dep["origen_id"].unique(), arr["destino_id"].unique())
    hours = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h")
    grid = pd.MultiIndex.from_product([stations, hours], names=["estacion_id", "fecha_hora"])
    return pd.DataFrame(
        {"salidas": out.reindex(grid, fill_value=0), "llegadas": inn.reindex(grid, fill_value=0)}
    ).reset_index()


def build_monitor_features(
    demand: pd.DataFrame,
    history: pd.DataFrame,
    stations: pd.DataFrame,
    profiles: pd.DataFrame,
    start,
    holidays=HOLIDAYS_2025 + HOLIDAYS_2026,
) -> tuple[pd.DataFrame, dict]:
    """Variables del periodo de monitoreo con los artefactos del entrenamiento.

    `history` aporta la ultima semana previa para los rezagos; `profiles` son los
    componentes PCA calculados con 2025. Las estaciones sin perfil o sin catalogo son
    estaciones nuevas para el modelo: se cuentan y se excluyen, porque no tiene como
    describirlas.
    """
    start = pd.Timestamp(start)
    lookback = start - pd.Timedelta(days=8)
    hist = history[(history["fecha_hora"] >= lookback) & (history["fecha_hora"] < start)]
    df = pd.concat([hist[KEYS + TARGETS], demand[KEYS + TARGETS]], ignore_index=True)
    df = FeatureBuilder.add_calendar(df)
    df["es_festivo"] = df["fecha_hora"].dt.normalize().isin(pd.to_datetime(holidays)).astype(int)
    df = FeatureBuilder.add_lags(df)
    df = df[df["fecha_hora"] >= start]

    df = df.merge(
        stations[["estacion_id", *STATION_NUMERIC, "zona"]], on="estacion_id", how="left"
    )
    df = df.merge(profiles, left_on="estacion_id", right_index=True, how="left")
    new_station = df[[*STATION_NUMERIC, "zona", "perfil_pc1", "perfil_pc2"]].isna().any(axis=1)
    counts = {
        "estaciones_monitoreo": int(df["estacion_id"].nunique()),
        "estaciones_nuevas": int(df.loc[new_station, "estacion_id"].nunique()),
        "filas_estaciones_nuevas": int(new_station.sum()),
    }
    # Participacion de las estaciones nuevas en la demanda de cada mes
    month = df["fecha_hora"].dt.to_period("M")
    demand_new = df["salidas"].where(new_station, 0).groupby(month).sum()
    counts["pct_salidas_estaciones_nuevas"] = (
        (100 * demand_new / df["salidas"].groupby(month).sum()).round(2).to_dict()
    )
    df = df[~new_station].dropna(subset=LAG_COLUMNS)
    return df.reset_index(drop=True), counts


# --- Escenarios sinteticos -------------------------------------------------------------------


def demand_growth(frame: pd.DataFrame, factor: float = 1.3) -> pd.DataFrame:
    """Mas uso en todo el sistema: suben juntos los rezagos y la demanda real."""
    out = frame.copy()
    cols = LAG_COLUMNS + TARGETS
    out[cols] = out[cols] * factor
    return out


def missing_lag(frame: pd.DataFrame) -> pd.DataFrame:
    """Falla de una fuente: el rezago de 24 h llega en cero (feature faltante silenciosa)."""
    out = frame.copy()
    out[[f"{t}_lag_24" for t in TARGETS]] = 0
    return out


def seasonal_drop(frame: pd.DataFrame, factor: float = 0.6) -> pd.DataFrame:
    """Periodo vacacional: la demanda real cae, pero las entradas siguen como antes."""
    out = frame.copy()
    out[TARGETS] = out[TARGETS] * factor
    return out


SCENARIOS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "crecimiento_demanda_30": demand_growth,
    "falla_rezago_24h": missing_lag,
    "caida_estacional_40": seasonal_drop,
}


# --- Evaluacion ------------------------------------------------------------------------------


def evaluate_window(
    name: str,
    model,
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    reference_preds: pd.DataFrame,
    reference_metrics: dict,
    train_targets: pd.DataFrame,
    new_stations_pct: float = 0.0,
) -> tuple[dict, pd.DataFrame]:
    """Metricas, drift y accion para una ventana de monitoreo."""
    preds = predict_frame(model, frame)
    metrics = score_frame(frame, {t: preds[f"pred_{t}"].to_numpy() for t in TARGETS})
    base = {t: historical_mean(train_targets, frame, t).to_numpy() for t in TARGETS}
    base_metrics = score_frame(frame, base)

    drift = feature_drift(reference, frame, FEATURE_COLUMNS)
    monitored = drift[drift["variable"].isin(MONITORED)]
    pred_psi = psi(reference_preds["pred_salidas"], preds["pred_salidas"])
    target_psi = psi(reference["salidas"], frame["salidas"])
    top = monitored.sort_values("psi", ascending=False).iloc[0]

    degradation = change_pct(reference_metrics[SELECTION_METRIC], metrics[SELECTION_METRIC])
    bias = bias_pct(frame["salidas"], preds["pred_salidas"])
    perf_status = worst(degradation_status(degradation), bias_status(bias))
    input_status = psi_status(float(top["psi"]))
    new_stations = new_stations_pct >= NEW_STATIONS_ALERT_PCT
    action = decide_action(perf_status, input_status, new_stations)
    row = {
        "ventana": name,
        "filas": len(frame),
        "demanda_media_salidas": float(frame["salidas"].mean()),
        SELECTION_METRIC: metrics[SELECTION_METRIC],
        "salidas_mae": metrics["salidas_mae"],
        "llegadas_mae_pico": metrics["llegadas_mae_pico"],
        f"base_{SELECTION_METRIC}": base_metrics[SELECTION_METRIC],
        "degradacion_pct": degradation,
        "sesgo_pct": bias,
        "ventaja_vs_base_pct": improvement_pct(
            metrics[SELECTION_METRIC], base_metrics[SELECTION_METRIC]
        ),
        "psi_max": float(top["psi"]),
        "variable_psi_max": top["variable"],
        "psi_max_calendario": float(drift.loc[~drift["variable"].isin(MONITORED), "psi"].max()),
        "pct_salidas_estaciones_nuevas": new_stations_pct,
        "psi_prediccion": pred_psi,
        "psi_objetivo": target_psi,
        "estado_desempeno": perf_status,
        "estado_entradas": input_status,
        "accion": action,
    }
    return row, drift.assign(ventana=name)


def log_window(row: dict, drift: pd.DataFrame, tipo: str, extra: dict | None = None) -> None:
    with mlflow.start_run(run_name=row["ventana"]):
        mlflow.set_tags(
            {
                "tipo": tipo,
                "estado_desempeno": row["estado_desempeno"],
                "estado_entradas": row["estado_entradas"],
                "accion": ACTIONS[row["accion"]],
            }
        )
        mlflow.log_params(
            {
                "ventana": row["ventana"],
                "umbral_psi": f"{PSI_WARN}/{PSI_ALERT}",
                "umbral_degradacion_pct": f"{DEGRADATION_WARN_PCT}/{DEGRADATION_ALERT_PCT}",
                **(extra or {}),
            }
        )
        mlflow.log_metrics(
            {k: float(v) for k, v in row.items() if isinstance(v, (int, float, np.floating))}
        )
        mlflow.log_metrics({f"psi_{r.variable}": float(r.psi) for r in drift.itertuples()})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "psi_variables.csv"
            drift.to_csv(path, index=False)
            mlflow.log_artifact(str(path))


# --- Figuras ---------------------------------------------------------------------------------


def _threshold_bands(ax, reference: float):
    ax.axhline(reference, color=INK_2, lw=1, ls="--", label="Validación 2025 (referencia)")
    ax.axhspan(
        reference * (1 + DEGRADATION_WARN_PCT / 100),
        reference * (1 + DEGRADATION_ALERT_PCT / 100),
        color="#f2c94c",
        alpha=0.25,
        lw=0,
        label=f"Advertencia (+{DEGRADATION_WARN_PCT:.0f} %)",
    )
    ax.axhspan(
        reference * (1 + DEGRADATION_ALERT_PCT / 100),
        ax.get_ylim()[1],
        color="#e34948",
        alpha=0.12,
        lw=0,
        label=f"Alerta (+{DEGRADATION_ALERT_PCT:.0f} %)",
    )


def plot_real(summary: pd.DataFrame, reference: float, out_dir: Path) -> Path:
    real = summary[summary["tipo"] == "real"]
    labels = [MESES[int(v[-2:]) - 1] for v in real["ventana"]]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(real))
    ax.plot(x, real[SELECTION_METRIC], "o-", color=SERIES[0], lw=2, label="Modelo hgb_poisson")
    ax.plot(
        x,
        real[f"base_{SELECTION_METRIC}"],
        "s--",
        color=SERIES[1],
        lw=1.5,
        label="Promedio histórico",
    )
    top = max(real[f"base_{SELECTION_METRIC}"].max(), real[SELECTION_METRIC].max()) * 1.08
    ax.set_ylim(min(reference, real[SELECTION_METRIC].min()) * 0.85, top)
    _threshold_bands(ax, reference)
    ax.set_xticks(x, labels)
    ax.set_ylabel("MAE de salidas en hora pico")
    ax.set_title("Drift real: desempeño mensual en 2026 frente a la validación de 2025")
    ax.legend(loc="upper left", fontsize=8, ncols=2)
    return _save(fig, "drift_real_mensual", out_dir)


def plot_psi_heatmap(psi_table: pd.DataFrame, out_dir: Path) -> Path:
    pivot = psi_table.pivot(index="variable", columns="ventana", values="psi")
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
    pivot = pivot.head(14)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    im = ax.imshow(
        np.clip(pivot.to_numpy(), 0, 0.5), cmap="YlOrRd", vmin=0, vmax=0.5, aspect="auto"
    )
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]), pivot.index, fontsize=8)
    ax.grid(False)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iat[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if v > 0.3 else "#333333",
            )
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label(f"PSI (advertencia {PSI_WARN}, alerta {PSI_ALERT})")
    ax.set_title("PSI por variable y ventana de monitoreo (14 variables con más drift)")
    return _save(fig, "drift_psi_variables", out_dir)


def plot_synthetic(summary: pd.DataFrame, out_dir: Path) -> Path:
    syn = summary[summary["tipo"] == "sintetico"].reset_index(drop=True)
    colors = {"estable": SERIES[2], "advertencia": "#f2a33a", "alerta": "#e34948"}
    panels = [
        (
            "degradacion_pct",
            "Cambio del MAE pico (%)",
            degradation_status,
            DEGRADATION_WARN_PCT,
            DEGRADATION_ALERT_PCT,
        ),
        ("sesgo_pct", "Sesgo del pronóstico (%)", bias_status, BIAS_WARN_PCT, BIAS_ALERT_PCT),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    y = np.arange(len(syn))
    for ax, (col, label, status_fn, warn, alert) in zip(axes, panels):
        values = syn[col]
        ax.barh(y, values, color=[colors[status_fn(v)] for v in values])
        span = max(values.abs().max(), alert) * 1.25
        for yi, v in zip(y, values):
            ax.text(
                v + (0.02 if v >= 0 else -0.02) * span,
                yi,
                f"{v:+.1f} %",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8,
            )
        signs = (1, -1) if col == "sesgo_pct" else (1,)
        for sign in signs:
            ax.axvline(sign * warn, color="#f2a33a", ls="--", lw=1)
            ax.axvline(sign * alert, color="#e34948", ls="--", lw=1)
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.set_xlim(-span if col == "sesgo_pct" else min(0, values.min() * 1.3), span)
        ax.set_xlabel(label)
    axes[0].set_yticks(y, syn["ventana"])
    axes[0].set_title("Error en hora pico")
    axes[1].set_title("Sesgo")
    fig.suptitle(
        "Drift sintético: cada señal con su umbral (verde estable, ámbar advertencia, "
        "rojo alerta)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    return _save(fig, "drift_sintetico", out_dir)


def plot_sensitivity(sweep: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    panels = [
        (
            "degradacion_pct",
            "Cambio del MAE pico (%)",
            DEGRADATION_WARN_PCT,
            DEGRADATION_ALERT_PCT,
        ),
        ("sesgo_pct", "Sesgo del pronóstico (%)", BIAS_WARN_PCT, BIAS_ALERT_PCT),
    ]
    for ax, (col, label, warn, alert) in zip(axes, panels):
        for (escenario, grp), color in zip(sweep.groupby("escenario"), SERIES):
            ax.plot(grp["factor"], grp[col], "o-", color=color, lw=2, label=escenario)
        for sign in (1, -1) if col == "sesgo_pct" else (1,):
            ax.axhline(sign * warn, color="#f2a33a", ls="--", lw=1)
            ax.axhline(sign * alert, color="#e34948", ls="--", lw=1)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_xlabel("Factor aplicado a la demanda")
        ax.set_ylabel(label)
    axes[0].set_title("Error en hora pico")
    axes[1].set_title("Sesgo (umbrales en ambos sentidos)")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Sensibilidad de las dos señales de desempeño al cambio de demanda",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    return _save(fig, "drift_sensibilidad", out_dir)


# --- Orquestacion ----------------------------------------------------------------------------


@app.command()
def figures(log_mlflow: bool = True):
    """Regenera las figuras desde los CSV de reports/ sin volver a evaluar."""
    summary = pd.read_csv(REPORTS_DIR / "drift_resumen.csv")
    reference = mlflow_reference_metric() if log_mlflow else None
    paths = [
        plot_psi_heatmap(pd.read_csv(REPORTS_DIR / "drift_psi.csv"), FIGURES_DIR),
        plot_synthetic(summary, FIGURES_DIR),
        plot_sensitivity(pd.read_csv(REPORTS_DIR / "drift_sensibilidad.csv"), FIGURES_DIR),
    ]
    if reference is not None:
        paths.append(plot_real(summary, reference, FIGURES_DIR))
    logger.success(f"Figuras: {[p.name for p in paths]}")


def mlflow_reference_metric() -> float:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_DRIFT],
        filter_string="tags.tipo = 'referencia'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return float(runs.iloc[0][f"metrics.{SELECTION_METRIC}"])


@app.command()
def main(
    model_uri: str = f"models:/{REGISTERED_MODEL_NAME}/latest",
    features_path: Path = OFFICIAL_FEATURES_PATH,
    demand_path: Path = PROCESSED_DATA_DIR / "demanda_estacion_hora_oficial.parquet",
    raw_dir: Path = MONITOR_RAW_DIR,
    experiment: str = EXPERIMENT_DRIFT,
    log_mlflow: bool = True,
):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(model_uri)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cols = KEYS + FEATURE_COLUMNS + TARGETS + ["particion"]
    features = pd.read_parquet(features_path, columns=cols)
    train_targets = features.loc[
        features["particion"] == "entrenamiento", ["estacion_id", "dia_semana", "hora", *TARGETS]
    ]
    reference = features[features["particion"] == "validacion"].reset_index(drop=True)
    profiles = (
        pd.read_parquet(features_path, columns=["estacion_id", "perfil_pc1", "perfil_pc2"])
        .drop_duplicates("estacion_id")
        .set_index("estacion_id")
    )
    stations = features[["estacion_id", *STATION_NUMERIC, "zona"]].drop_duplicates("estacion_id")
    del features

    reference_preds = predict_frame(model, reference)
    reference_metrics = score_frame(
        reference, {t: reference_preds[f"pred_{t}"].to_numpy() for t in TARGETS}
    )
    logger.info(f"Referencia (validacion 2025): {reference_metrics}")

    if log_mlflow:
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name="referencia_validacion_2025"):
            mlflow.set_tags({"tipo": "referencia", "modelo": model_uri})
            mlflow.log_params({"modelo": model_uri, "filas": len(reference)})
            mlflow.log_metrics(reference_metrics)

    rows, psi_tables = [], []
    context = (reference, reference_preds, reference_metrics, train_targets)

    # 1) Escenarios sinteticos sobre la validacion
    for name, transform in SCENARIOS.items():
        row, drift = evaluate_window(name, model, transform(reference), *context)
        rows.append({"tipo": "sintetico", **row})
        psi_tables.append(drift)
        if log_mlflow:
            log_window(row, drift, "sintetico")
        logger.info(f"{name}: {row['degradacion_pct']:+.1f} % -> {row['accion']}")

    # Barrido de intensidad para ubicar donde se cruzan los umbrales
    sweep = []
    for factor in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        for label, fn in [
            ("crecimiento (entradas y demanda)", demand_growth),
            ("solo la demanda real cambia", seasonal_drop),
        ]:
            frame = fn(reference, factor)
            preds = predict_frame(model, frame)
            m = score_frame(frame, {t: preds[f"pred_{t}"].to_numpy() for t in TARGETS})
            change = change_pct(reference_metrics[SELECTION_METRIC], m[SELECTION_METRIC])
            bias = bias_pct(frame["salidas"], preds["pred_salidas"])
            sweep.append(
                {
                    "escenario": label,
                    "factor": factor,
                    "degradacion_pct": change,
                    "sesgo_pct": bias,
                }
            )
    sweep = pd.DataFrame(sweep)

    # 2) Drift real: enero a octubre de 2026, una ventana por mes
    trips, trip_counts = load_monitor_trips(raw_dir)
    start = trips["inicio"].min().floor("D").replace(day=1)
    end = trips["inicio"].max().floor("D") + pd.Timedelta(days=1)
    end = end if end.day == 1 else (end.replace(day=1) + pd.offsets.MonthBegin(1))
    demand = aggregate_demand(trips, start, end)
    history = pd.read_parquet(demand_path, columns=KEYS + TARGETS)
    monitor, station_counts = build_monitor_features(demand, history, stations, profiles, start)
    del trips, demand, history
    monthly_new = station_counts.pop("pct_salidas_estaciones_nuevas")
    logger.info(f"Monitoreo 2026: {trip_counts} {station_counts}")
    monitor.to_parquet(PROCESSED_DATA_DIR / "variables_monitoreo_2026.parquet", index=False)

    for period, frame in monitor.groupby(monitor["fecha_hora"].dt.to_period("M")):
        name = f"real_{period}"
        row, drift = evaluate_window(
            name, model, frame.reset_index(drop=True), *context, monthly_new.get(period, 0.0)
        )
        rows.append({"tipo": "real", **row})
        psi_tables.append(drift)
        if log_mlflow:
            log_window(row, drift, "real", {**trip_counts, **station_counts})
        logger.info(f"{name}: {row['degradacion_pct']:+.1f} % -> {row['accion']}")

    summary = pd.DataFrame(rows)
    summary["accion_texto"] = summary["accion"].map(ACTIONS)
    psi_table = pd.concat(psi_tables, ignore_index=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_DIR / "drift_resumen.csv", index=False)
    psi_table.to_csv(REPORTS_DIR / "drift_psi.csv", index=False)
    sweep.to_csv(REPORTS_DIR / "drift_sensibilidad.csv", index=False)
    pd.Series({**trip_counts, **station_counts}).to_csv(REPORTS_DIR / "drift_datos_2026.csv")

    figures = [
        plot_real(summary, reference_metrics[SELECTION_METRIC], FIGURES_DIR),
        plot_psi_heatmap(psi_table, FIGURES_DIR),
        plot_synthetic(summary, FIGURES_DIR),
        plot_sensitivity(sweep, FIGURES_DIR),
    ]
    if log_mlflow:
        with mlflow.start_run(run_name="resumen_drift"):
            mlflow.set_tags({"tipo": "resumen"})
            mlflow.log_params({"modelo": model_uri, **trip_counts, **station_counts})
            alerts = int((summary["estado_desempeno"] == "alerta").sum())
            mlflow.log_metrics({"ventanas": len(summary), "ventanas_en_alerta": alerts})
            for path in figures + [
                REPORTS_DIR / "drift_resumen.csv",
                REPORTS_DIR / "drift_psi.csv",
                REPORTS_DIR / "drift_sensibilidad.csv",
            ]:
                mlflow.log_artifact(str(path))
    logger.success(
        f"Drift evaluado en {len(summary)} ventanas; resumen en reports/drift_resumen.csv"
    )


if __name__ == "__main__":
    app()
