"""Validacion de la tabla de demanda por estacion y hora contra los viajes que la originan.

Uso:
    python -m mibici_mlops.validation --variant oficial   # escribe reports/validacion_demanda_*.csv
"""

from pathlib import Path
import tempfile

from loguru import logger
import mlflow
import pandas as pd
import typer

from mibici_mlops.config import (
    INTERIM_DATA_DIR,
    MLFLOW_TRACKING_URI,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from mibici_mlops.dataset import load_stations
from mibici_mlops.tracking import file_sha256

app = typer.Typer()

EXPERIMENT_VALIDATION = "MiBici_Fase1_Validacion_Datos"
YEAR_START, YEAR_END = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")


def check(name: str, passed: bool, detail: str) -> dict:
    return {"prueba": name, "resultado": "cumple" if passed else "no cumple", "detalle": detail}


def validate_demand(
    demand: pd.DataFrame, trips: pd.DataFrame, station_ids: set[int]
) -> list[dict]:
    hours = pd.date_range(YEAR_START, YEAR_END, freq="h", inclusive="left")
    n_stations = demand["estacion_id"].nunique()
    results = []

    per_station = demand.groupby("estacion_id").size()
    results.append(
        check(
            "horas completas por estacion",
            bool((per_station == len(hours)).all()),
            f"{n_stations} estaciones; esperadas {len(hours):,} horas cada una, "
            f"minimo {per_station.min():,} y maximo {per_station.max():,}",
        )
    )
    dup = int(demand.duplicated(["estacion_id", "fecha_hora"]).sum())
    results.append(check("sin estacion-hora repetida", dup == 0, f"{dup} repetidas"))
    out_range = int((~demand["fecha_hora"].between(hours[0], hours[-1])).sum())
    results.append(check("horas dentro de 2025", out_range == 0, f"{out_range} fuera de rango"))

    starts_2025 = trips["inicio"].between(YEAR_START, YEAR_END, inclusive="left")
    ends_2025 = trips["fin"].between(YEAR_START, YEAR_END, inclusive="left")
    results.append(
        check(
            "salidas cuadran con viajes",
            int(demand["salidas"].sum()) == int(starts_2025.sum()),
            f"salidas {int(demand['salidas'].sum()):,}; viajes que inician en 2025 "
            f"{int(starts_2025.sum()):,}",
        )
    )
    results.append(
        check(
            "llegadas cuadran con viajes",
            int(demand["llegadas"].sum()) == int(ends_2025.sum()),
            f"llegadas {int(demand['llegadas'].sum()):,}; viajes que terminan en 2025 "
            f"{int(ends_2025.sum()):,} ({int((~ends_2025).sum())} terminan en 2026)",
        )
    )

    month_trips = trips.loc[starts_2025].groupby(trips["inicio"].dt.month).size()
    month_demand = demand.groupby(demand["fecha_hora"].dt.month)["salidas"].sum()
    bad_months = int((month_trips.reindex(range(1, 13), fill_value=0) != month_demand).sum())
    results.append(
        check("salidas por mes cuadran", bad_months == 0, f"{bad_months} meses con diferencia")
    )

    negatives = int((demand[["salidas", "llegadas"]] < 0).any(axis=1).sum())
    results.append(check("conteos no negativos", negatives == 0, f"{negatives} filas negativas"))
    net_ok = bool((demand["neto"] == demand["llegadas"] - demand["salidas"]).all())
    results.append(check("neto = llegadas - salidas", net_ok, "revisado en todas las filas"))

    unknown = sorted(set(demand["estacion_id"]) - station_ids)
    results.append(
        check(
            "estaciones en el catalogo",
            not unknown,
            f"{len(unknown)} fuera del catalogo {unknown[:5]}",
        )
    )
    return results


@app.command()
def main(variant: str = "oficial"):
    trips_path = INTERIM_DATA_DIR / f"viajes_limpios_{variant}.parquet"
    demand_path = PROCESSED_DATA_DIR / f"demanda_estacion_hora_{variant}.parquet"
    trips = pd.read_parquet(trips_path, columns=["inicio", "fin"])
    demand = pd.read_parquet(demand_path)
    station_ids = set(load_stations()["estacion_id"])

    report = pd.DataFrame(validate_demand(demand, trips, station_ids))
    out = REPORTS_DIR / f"validacion_demanda_{variant}.csv"
    report.to_csv(out, index=False)
    failed = int((report["resultado"] == "no cumple").sum())

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_VALIDATION)
    with mlflow.start_run(run_name=f"validacion_demanda_{variant}"):
        mlflow.set_tags({"variante": variant, "demanda_sha256": file_sha256(demand_path)})
        mlflow.log_metrics({"pruebas": len(report), "pruebas_fallidas": failed})
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / out.name
            report.to_csv(copy, index=False)
            mlflow.log_artifact(str(copy))
    print(report.to_string(index=False))
    if failed:
        logger.warning(f"{failed} pruebas no se cumplen; revisar {out}")
    else:
        logger.success(f"Las {len(report)} pruebas se cumplen ({out})")


if __name__ == "__main__":
    app()
