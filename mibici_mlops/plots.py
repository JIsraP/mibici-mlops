"""Figuras del analisis exploratorio.

Uso:
    python -m mibici_mlops.plots --variant oficial    # escribe reports/figures/*.png y reports/eda_*.csv
"""

from pathlib import Path

from loguru import logger
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from mibici_mlops.config import FIGURES_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from mibici_mlops.dataset import load_stations

app = typer.Typer()

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = LinearSegmentedColormap.from_list("azul", ["#cde2fb", "#5598e7", "#104281"])
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
    }
)


def _save(fig, name: str, out_dir: Path) -> Path:
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def trips_by_month(trips: pd.DataFrame, out_dir: Path) -> Path:
    counts = trips.groupby(trips["inicio"].dt.month).size() / 1000
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(counts.index, counts.values, color=SERIES[0], width=0.7)
    ax.set_xticks(range(1, 13), MESES)
    ax.set_ylabel("Miles de viajes")
    ax.set_title("Viajes por mes, 2025")
    ax.grid(axis="x", visible=False)
    for m, v in counts.items():
        ax.text(m, v + 4, f"{v:.0f}", ha="center", fontsize=8, color=INK_2)
    return _save(fig, "01_viajes_por_mes", out_dir)


def hourly_profile(demand: pd.DataFrame, out_dir: Path) -> Path:
    d = demand.assign(
        hora=demand["fecha_hora"].dt.hour,
        tipo=np.where(demand["fecha_hora"].dt.day_of_week >= 5, "Fin de semana", "Laboral"),
        dia=demand["fecha_hora"].dt.date,
    )
    per_day = d.groupby(["tipo", "dia", "hora"])["salidas"].sum().reset_index()
    prof = per_day.groupby(["tipo", "hora"])["salidas"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for color, col in zip(SERIES, ["Laboral", "Fin de semana"]):
        ax.plot(prof.index, prof[col], color=color, lw=2, label=col)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Viajes por hora (promedio diario)")
    ax.set_title("Perfil horario de salidas en toda la red")
    ax.legend(loc="upper left")
    return _save(fig, "02_perfil_horario", out_dir)


def weekday_hour_heatmap(demand: pd.DataFrame, out_dir: Path) -> Path:
    d = demand.assign(dia=demand["fecha_hora"].dt.day_of_week, hora=demand["fecha_hora"].dt.hour)
    n_days = d.groupby("dia")["fecha_hora"].apply(lambda s: s.dt.date.nunique())
    tab = d.groupby(["dia", "hora"])["salidas"].sum().unstack().div(n_days, axis=0)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    im = ax.imshow(tab.values, aspect="auto", cmap=SEQ)
    ax.set_yticks(range(7), DIAS)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hora del día")
    ax.grid(False)
    ax.set_title("Salidas promedio por día de la semana y hora")
    fig.colorbar(im, ax=ax, label="Viajes por hora", shrink=0.9)
    return _save(fig, "03_mapa_calor_dia_hora", out_dir)


def duration_hist(raw_duration: pd.Series, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(raw_duration.clip(upper=60), bins=120, color=SERIES[0])
    for x in (1, 2):
        ax.axvline(x, color=INK_2, lw=1, ls="--")
    ax.text(32, ax.get_ylim()[1] * 0.6, "líneas: umbrales de 1 y 2 min", fontsize=8, color=INK_2)
    ax.set_xlabel("Duración en minutos (valores mayores a 60 agrupados en 60)")
    ax.set_ylabel("Viajes")
    ax.set_title("Duración de los viajes antes de limpiar")
    return _save(fig, "04_duracion_viajes", out_dir)


def top_stations(trips: pd.DataFrame, stations: pd.DataFrame, out_dir: Path, n: int = 15):
    top = trips["origen_id"].value_counts().head(n).rename_axis("estacion_id").reset_index()
    top = top.merge(stations[["estacion_id", "nombre"]], how="left")
    top["nombre"] = top["nombre"].str.slice(0, 42)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["nombre"][::-1], top["count"][::-1] / 1000, color=SERIES[0], height=0.7)
    ax.set_xlabel("Miles de salidas en 2025")
    ax.set_title(f"Las {n} estaciones con más salidas")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK_2, labelsize=8)
    return _save(fig, "05_top_estaciones", out_dir)


def morning_net_map(demand: pd.DataFrame, stations: pd.DataFrame, out_dir: Path) -> Path:
    am = demand[
        demand["fecha_hora"].dt.hour.isin([7, 8, 9]) & (demand["fecha_hora"].dt.day_of_week < 5)
    ]
    n_days = am["fecha_hora"].dt.date.nunique()
    net = (am.groupby("estacion_id")["neto"].sum() / n_days).rename("neto").reset_index()
    net = net.merge(stations, how="left")
    lim = np.nanpercentile(net["neto"].abs(), 98)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    sc = ax.scatter(
        net["longitud"],
        net["latitud"],
        c=net["neto"],
        cmap=DIV,
        s=26,
        norm=TwoSlopeNorm(0, -lim, lim),
        edgecolors="#fcfcfb",
        linewidths=0.6,
    )
    fig.colorbar(
        sc, ax=ax, label="Llegadas menos salidas por día (azul acumula, rojo se vacía)", shrink=0.8
    )
    ax.set_aspect(1 / np.cos(np.radians(20.67)))
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("Desbalance por estación, días laborales de 7 a 10 h")
    return _save(fig, "06_mapa_desbalance_manana", out_dir)


def pca_profiles(features: pd.DataFrame, out_dir: Path) -> Path:
    """Colorea cada estacion por su indice de desbalance matutino para interpretar el PC1."""
    am = features[features["hora"].isin([7, 8, 9]) & (features["es_fin_semana"] == 0)]
    tot = am.groupby("estacion_id")[["salidas", "llegadas"]].sum()
    index = ((tot["llegadas"] - tot["salidas"]) / (tot["llegadas"] + tot["salidas"])).rename("i")
    st = features.drop_duplicates("estacion_id").merge(
        index, left_on="estacion_id", right_index=True
    )
    r = st["perfil_pc1"].corr(st["i"])
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        st["perfil_pc1"],
        st["perfil_pc2"],
        c=st["i"],
        cmap=DIV,
        norm=TwoSlopeNorm(0, -1, 1),
        s=26,
        edgecolors="#fcfcfb",
        linewidths=0.6,
    )
    fig.colorbar(sc, ax=ax, label="(llegadas - salidas) / total, de 7 a 10 h", shrink=0.85)
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.axvline(0, color="#c3c2b7", lw=0.8)
    ax.set_xlabel(f"Componente 1 del perfil horario (r = {r:.2f} con el desbalance matutino)")
    ax.set_ylabel("Componente 2")
    ax.set_title("Estaciones según su perfil horario (PCA)")
    return _save(fig, "07_pca_perfiles", out_dir)


def users_profile(trips: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [2, 1]})
    edad = trips["edad"].dropna()
    axes[0].hist(edad, bins=range(15, 96), color=SERIES[0])
    axes[0].set_xlabel("Edad en el año del viaje")
    axes[0].set_ylabel("Viajes")
    axes[0].set_title("Edad de quien viaja")
    gen = trips["genero"].value_counts(normalize=True).reindex(["M", "F", "Desconocido"]) * 100
    axes[1].bar(["Hombre", "Mujer", "Sin dato"], gen.values, color=SERIES[0], width=0.6)
    for i, v in enumerate(gen.values):
        axes[1].text(i, v + 1.5, f"{v:.1f} %", ha="center", fontsize=8, color=INK_2)
    axes[1].set_ylabel("% de viajes")
    axes[1].set_title("Género")
    axes[1].grid(axis="x", visible=False)
    return _save(fig, "08_usuarios", out_dir)


def target_distribution(demand: pd.DataFrame, out_dir: Path) -> Path:
    vc = demand["salidas"].clip(upper=15).value_counts(normalize=True).sort_index() * 100
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(vc.index, vc.values, color=SERIES[0], width=0.75)
    ax.set_xticks(range(16), [*map(str, range(15)), "15+"])
    ax.set_xlabel("Salidas en una estación durante una hora")
    ax.set_ylabel("% de estación-hora")
    ax.set_title(f"La variable objetivo es dispersa: {vc.iloc[0]:.0f} % de las horas sin salidas")
    ax.grid(axis="x", visible=False)
    return _save(fig, "09_distribucion_objetivo", out_dir)


def correlations(features: pd.DataFrame, out_dir: Path) -> Path:
    cols = [
        "salidas",
        "salidas_lag_24",
        "salidas_lag_168",
        "salidas_media_7d",
        "hora_sin",
        "hora_cos",
        "es_fin_semana",
        "capacidad",
        "perfil_pc1",
        "perfil_pc2",
    ]
    corr = features.loc[features["particion"] == "entrenamiento", cols].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(corr.values, cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlación de las variables con las salidas")
    return _save(fig, "10_correlaciones", out_dir)


def descriptive_tables(trips: pd.DataFrame, demand: pd.DataFrame, out_dir: Path) -> None:
    trips[["duracion_min", "edad"]].astype(float).describe().round(2).to_csv(
        out_dir / "eda_viajes_descriptivos.csv"
    )
    demand[["salidas", "llegadas", "neto"]].describe().round(3).to_csv(
        out_dir / "eda_demanda_descriptivos.csv"
    )


@app.command()
def main(variant: str = "oficial"):
    from mibici_mlops.dataset import load_raw_trips

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    trips = pd.read_parquet(INTERIM_DATA_DIR / f"viajes_limpios_{variant}.parquet")
    demand = pd.read_parquet(PROCESSED_DATA_DIR / f"demanda_estacion_hora_{variant}.parquet")
    features = pd.read_parquet(PROCESSED_DATA_DIR / f"variables_{variant}.parquet")
    stations = load_stations()
    raw, _ = load_raw_trips()
    raw_duration = (
        pd.to_datetime(raw["fin"]) - pd.to_datetime(raw["inicio"])
    ).dt.total_seconds() / 60

    trips_by_month(trips, FIGURES_DIR)
    hourly_profile(demand, FIGURES_DIR)
    weekday_hour_heatmap(demand, FIGURES_DIR)
    duration_hist(raw_duration, FIGURES_DIR)
    top_stations(trips, stations, FIGURES_DIR)
    morning_net_map(demand, stations, FIGURES_DIR)
    pca_profiles(features, FIGURES_DIR)
    users_profile(trips, FIGURES_DIR)
    target_distribution(demand, FIGURES_DIR)
    correlations(features, FIGURES_DIR)
    descriptive_tables(trips, demand, REPORTS_DIR)
    logger.success(f"Figuras en {FIGURES_DIR}")


if __name__ == "__main__":
    app()
