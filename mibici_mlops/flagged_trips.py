"""EDA de los viajes marcados en la version oficial.

Distingue las banderas oficiales del data engineer de los casos que este analisis propone
revisar (no se excluye nada aqui). Sirve para decidir con evidencia si conviene agregar
banderas o reglas nuevas.

Uso:
    python -m mibici_mlops.flagged_trips   # escribe reports/eda_viajes_marcados*.csv y 2 figuras
"""

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import typer

from mibici_mlops.config import FIGURES_DIR, OFFICIAL_DATA_DIR, REPORTS_DIR
from mibici_mlops.dataset import load_stations
from mibici_mlops.model_input import load_official_trips
from mibici_mlops.plots import INK_2, SERIES, _save

app = typer.Typer()


def categorize(trips: pd.DataFrame) -> dict[str, tuple[str, pd.Series]]:
    minutes = trips["duracion_segundos"] / 60
    same = trips["origen_id"] == trips["destino_id"]
    return {
        "duracion_no_positiva": ("bandera oficial (se excluye)", trips["duracion_no_positiva"]),
        "duracion_mayor_24h": ("bandera oficial (se conserva)", trips["duracion_mayor_24h"]),
        "misma_estacion_menos_2_min": ("por revisar", same & (minutes > 0) & (minutes < 2)),
        "distinta_estacion_menos_1_min": ("por revisar", ~same & (minutes > 0) & (minutes < 1)),
        "entre_3_y_24_horas": ("por revisar", (minutes > 180) & (minutes <= 1440)),
    }


def summary_table(trips: pd.DataFrame, cats: dict) -> pd.DataFrame:
    rows = []
    for name, (kind, mask) in cats.items():
        sub = trips.loc[mask]
        rows.append(
            {
                "categoria": name,
                "tipo": kind,
                "viajes": int(mask.sum()),
                "pct_del_total": round(100 * mask.mean(), 3),
                "pct_misma_estacion": round(
                    100 * (sub["origen_id"] == sub["destino_id"]).mean(), 1
                )
                if len(sub)
                else None,
                "mediana_min": round(sub["duracion_segundos"].median() / 60, 2)
                if len(sub)
                else None,
                "estaciones": int(sub["origen_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def station_impact(trips: pd.DataFrame, mask: pd.Series, stations: pd.DataFrame) -> pd.DataFrame:
    """Que proporcion de las salidas de cada estacion son casos marcados."""
    total = trips.groupby("origen_id").size().rename("salidas")
    marked = trips.loc[mask].groupby("origen_id").size().rename("marcadas")
    out = pd.concat([total, marked], axis=1).fillna(0).astype(int)
    out["pct_marcadas"] = (100 * out["marcadas"] / out["salidas"]).round(2)
    out = out.rename_axis("estacion_id").reset_index()
    return out.merge(stations[["estacion_id", "nombre"]], how="left").sort_values(
        "marcadas", ascending=False
    )


def hourly_share_plot(trips: pd.DataFrame, mask: pd.Series, out_dir: Path) -> Path:
    hour = trips["inicio"].dt.hour
    all_share = hour.value_counts(normalize=True).reindex(range(24), fill_value=0) * 100
    marked_share = hour[mask].value_counts(normalize=True).reindex(range(24), fill_value=0) * 100
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(all_share.index, all_share.values, color=SERIES[0], lw=2, label="Todos los viajes")
    ax.plot(
        marked_share.index,
        marked_share.values,
        color=SERIES[1],
        lw=2,
        label="Misma estación en menos de 2 min",
    )
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hora de inicio")
    ax.set_ylabel("% de los viajes de cada grupo")
    ax.set_title("¿Cuándo ocurren los viajes de menos de 2 minutos a la misma estación?")
    ax.legend(loc="upper left")
    return _save(fig, "11_marcados_por_hora", out_dir)


def station_share_plot(impact: pd.DataFrame, out_dir: Path, n: int = 15) -> Path:
    top = impact[impact["salidas"] >= 1000].nlargest(n, "pct_marcadas")
    labels = top["nombre"].str.slice(0, 42)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels[::-1], top["pct_marcadas"][::-1], color=SERIES[1], height=0.7)
    for y, v in enumerate(top["pct_marcadas"][::-1]):
        ax.text(v + 0.3, y, f"{v:.1f} %", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("% de las salidas de la estación (estaciones con 1,000 salidas o más)")
    ax.set_title("Estaciones con más viajes de menos de 2 min a la misma estación")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK_2, labelsize=8)
    return _save(fig, "12_marcados_por_estacion", out_dir)


@app.command()
def main(directory: Path = OFFICIAL_DATA_DIR):
    trips, _ = load_official_trips(directory)
    cats = categorize(trips)
    table = summary_table(trips, cats)
    table.to_csv(REPORTS_DIR / "eda_viajes_marcados.csv", index=False)

    false_start = cats["misma_estacion_menos_2_min"][1]
    impact = station_impact(trips, false_start, load_stations())
    impact.to_csv(REPORTS_DIR / "eda_viajes_marcados_por_estacion.csv", index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    hourly_share_plot(trips, false_start, FIGURES_DIR)
    station_share_plot(impact, FIGURES_DIR)
    print(table.to_string(index=False))
    logger.success("EDA de viajes marcados en reports/ y reports/figures/")


if __name__ == "__main__":
    app()
