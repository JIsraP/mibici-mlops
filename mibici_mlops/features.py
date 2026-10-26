"""Agregacion por estacion y hora, variables del modelo y preprocesamiento.

Uso:
    python -m mibici_mlops.features --variant estricta   # registra v2 (demanda) y v3 (variables)
"""

from pathlib import Path
import tempfile

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import typer

from mibici_mlops.config import HOLIDAYS_2025, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_END
from mibici_mlops.dataset import load_stations
from mibici_mlops.tracking import DatasetVersioner, file_sha256

app = typer.Typer()

TARGETS = ["salidas", "llegadas"]
NUMERIC = [
    "hora",
    "hora_sin",
    "hora_cos",
    *[f"{t}_{lag}" for t in TARGETS for lag in ("lag_24", "lag_168", "media_7d")],
    "latitud",
    "longitud",
    "capacidad",
    "perfil_pc1",
    "perfil_pc2",
]
BINARY = ["es_fin_semana", "es_festivo"]
CATEGORICAL = ["dia_semana", "zona"]


def build_demand(trips: pd.DataFrame) -> pd.DataFrame:
    """Cuenta salidas y llegadas por estacion y hora, rellenando con 0 las horas sin viajes."""
    out = trips.groupby(["origen_id", trips["inicio"].dt.floor("h")]).size()
    inn = trips.groupby(["destino_id", trips["fin"].dt.floor("h")]).size()
    out.index.names = inn.index.names = ["estacion_id", "fecha_hora"]

    stations = np.union1d(trips["origen_id"].unique(), trips["destino_id"].unique())
    hours = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h")
    grid = pd.MultiIndex.from_product([stations, hours], names=["estacion_id", "fecha_hora"])

    demand = pd.DataFrame(
        {
            "salidas": out.reindex(grid, fill_value=0),
            "llegadas": inn.reindex(grid, fill_value=0),
        }
    ).reset_index()
    demand["neto"] = demand["llegadas"] - demand["salidas"]
    return demand


class FeatureBuilder:
    """Agrega calendario, rezagos y atributos de estacion a la tabla de demanda.

    El perfil horario de cada estacion se resume con PCA ajustado solo con meses de
    entrenamiento, para no filtrar informacion de validacion. El perfil no se estandariza:
    son proporciones en la misma escala, y estandarizar inflaba el ruido de las horas
    nocturnas (la varianza explicada por dos componentes bajaba de 44 % a 29 %).
    """

    def __init__(self, stations: pd.DataFrame, train_end: str = TRAIN_END, n_components: int = 2):
        self.stations = stations
        self.train_end = pd.Timestamp(train_end)
        self.pca = PCA(n_components)

    @staticmethod
    def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
        ts = df["fecha_hora"]
        df["hora"] = ts.dt.hour
        df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
        df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
        df["dia_semana"] = ts.dt.day_of_week
        df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)
        df["es_festivo"] = ts.dt.normalize().isin(pd.to_datetime(HOLIDAYS_2025)).astype(int)
        df["mes"] = ts.dt.month
        return df

    @staticmethod
    def add_lags(df: pd.DataFrame) -> pd.DataFrame:
        """Rezagos de la misma hora: ayer, hace una semana y promedio de los ultimos 7 dias."""
        df = df.sort_values(["estacion_id", "fecha_hora"])
        for t in TARGETS:
            g = df.groupby("estacion_id")[t]
            df[f"{t}_lag_24"] = g.shift(24)
            df[f"{t}_lag_168"] = g.shift(168)
            df[f"{t}_media_7d"] = sum(g.shift(24 * k) for k in range(1, 8)) / 7
        return df

    def station_profiles(self, demand: pd.DataFrame) -> pd.DataFrame:
        """Proporcion de salidas y llegadas por hora en dias laborales (forma, no volumen)."""
        train = demand[(demand["fecha_hora"] < self.train_end) & (demand["es_fin_semana"] == 0)]
        prof = train.pivot_table(
            index="estacion_id", columns="hora", values=TARGETS, aggfunc="sum"
        )
        shares = [prof[t].div(prof[t].sum(axis=1).replace(0, np.nan), axis=0) for t in TARGETS]
        return pd.concat(shares, axis=1, keys=TARGETS).fillna(0)

    def fit_transform(self, demand: pd.DataFrame) -> pd.DataFrame:
        df = self.add_calendar(demand.copy())
        profiles = self.station_profiles(df)
        comps = self.pca.fit_transform(profiles.values)
        pcs = pd.DataFrame(comps, index=profiles.index, columns=["perfil_pc1", "perfil_pc2"])

        df = self.add_lags(df)
        df = df.merge(
            self.stations[["estacion_id", "latitud", "longitud", "capacidad", "zona"]],
            on="estacion_id",
            how="left",
        )
        df = df.merge(pcs, left_on="estacion_id", right_index=True, how="left")
        df = df.dropna(subset=["salidas_lag_168", "llegadas_media_7d"])
        df["particion"] = np.where(
            df["fecha_hora"] < self.train_end, "entrenamiento", "validacion"
        )
        return df.reset_index(drop=True)

    @property
    def explained_variance(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_


def build_preprocessor() -> ColumnTransformer:
    """Escala variables numericas y codifica categoricas; se ajusta solo con entrenamiento."""
    return ColumnTransformer(
        [
            ("numericas", StandardScaler(), NUMERIC),
            ("binarias", "passthrough", BINARY),
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


@app.command()
def main(variant: str = "estricta"):
    trips = pd.read_parquet(INTERIM_DATA_DIR / f"viajes_limpios_{variant}.parquet")
    stations = load_stations()
    versioner = DatasetVersioner()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    demand = build_demand(trips)
    demand_path = PROCESSED_DATA_DIR / f"demanda_estacion_hora_{variant}.parquet"
    demand.to_parquet(demand_path, index=False)
    versioner.log_version(
        f"v2_demanda_{variant}",
        demand,
        file_sha256(demand_path),
        source=str(demand_path),
        description="Salidas y llegadas por estacion y hora, horas sin viajes en 0",
        params={"variante": variant, "estaciones": demand["estacion_id"].nunique()},
        metrics={
            "salidas_totales": int(demand["salidas"].sum()),
            "pct_horas_sin_salidas": 100 * float((demand["salidas"] == 0).mean()),
        },
        file=demand_path,
    )

    builder = FeatureBuilder(stations)
    features = builder.fit_transform(demand)
    features_path = PROCESSED_DATA_DIR / f"variables_{variant}.parquet"
    features.to_parquet(features_path, index=False)

    train = features[features["particion"] == "entrenamiento"]
    preprocessor = build_preprocessor().fit(train)
    n_out = preprocessor.transform(train.head(1)).shape[1]

    with tempfile.TemporaryDirectory() as tmp:
        prep_path = Path(tmp) / "preprocesador.joblib"
        joblib.dump({"preprocesador": preprocessor, "pca_perfil": builder.pca}, prep_path)
        run_id = versioner.log_version(
            f"v3_variables_{variant}",
            features,
            file_sha256(features_path),
            source=str(features_path),
            description="Variables de calendario, rezagos, estacion y perfil PCA; particion temporal",
            params={
                "variante": variant,
                "corte_entrenamiento": TRAIN_END,
                "variables_numericas": ",".join(NUMERIC),
                "variables_categoricas": ",".join(CATEGORICAL),
                "columnas_tras_preprocesar": n_out,
            },
            metrics={
                "filas_entrenamiento": len(train),
                "filas_validacion": int((features["particion"] == "validacion").sum()),
                "pca_varianza_pc1": float(builder.explained_variance[0]),
                "pca_varianza_pc2": float(builder.explained_variance[1]),
            },
            file=prep_path,
        )
    logger.success(f"Variables en {features_path} (run {run_id})")


if __name__ == "__main__":
    app()
