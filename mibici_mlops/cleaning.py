"""Reglas de limpieza de viajes. Cada regla deja constancia de cuantas filas toco."""

from dataclasses import asdict, dataclass

from loguru import logger
import pandas as pd


@dataclass(frozen=True)
class CleaningRules:
    """Umbrales de limpieza. La variante "estricta" es la que alimenta el modelo."""

    min_birth_year: int = 1930
    max_birth_year: int = 2010
    false_start_max_min: float | None = 2.0  # mismo origen y destino en menos de 2 min
    min_duration_min: float = 1.0
    max_duration_min: float = 180.0

    @classmethod
    def variant(cls, name: str) -> "CleaningRules":
        if name == "estricta":
            return cls()
        if name == "laxa":
            # Solo quita lo imposible: duracion nula o negativa y viajes de mas de un dia
            return cls(false_start_max_min=None, min_duration_min=0.0, max_duration_min=1440.0)
        raise ValueError(f"Variante desconocida: {name}")

    def as_params(self) -> dict:
        return {f"regla_{k}": v for k, v in asdict(self).items()}


class TripCleaner:
    """Convierte los viajes crudos (texto) en una tabla tipada y valida."""

    def __init__(self, rules: CleaningRules, valid_station_ids: set[int]):
        self.rules = rules
        self.valid_station_ids = valid_station_ids
        self.report: list[dict] = []

    def _log(self, rule: str, action: str, rows: int, df: pd.DataFrame) -> None:
        self.report.append(
            {"regla": rule, "accion": action, "filas_afectadas": int(rows), "filas_tras": len(df)}
        )
        logger.info(f"{rule}: {rows:,} filas ({action})")

    def _drop(self, df: pd.DataFrame, mask: pd.Series, rule: str) -> pd.DataFrame:
        out = df.loc[~mask]
        self._log(rule, "eliminar", mask.sum(), out)
        return out

    def clean(self, raw: pd.DataFrame) -> pd.DataFrame:
        self.report = []
        r = self.rules
        df = raw.copy()
        self._log("filas_crudas", "ninguna", 0, df)

        df = self._drop(df, df.duplicated("viaje_id"), "viaje_id_duplicado")

        df["inicio"] = pd.to_datetime(df["inicio"], errors="coerce")
        df["fin"] = pd.to_datetime(df["fin"], errors="coerce")
        df = self._drop(df, df["inicio"].isna() | df["fin"].isna(), "fecha_invalida")

        for col in ["viaje_id", "usuario_id", "origen_id", "destino_id"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        ids = ["viaje_id", "usuario_id", "origen_id", "destino_id"]
        df = self._drop(df, df[ids].isna().any(axis=1), "id_no_numerico")
        df[ids] = df[ids].astype("int64")

        unknown = ~df["origen_id"].isin(self.valid_station_ids) | ~df["destino_id"].isin(
            self.valid_station_ids
        )
        df = self._drop(df, unknown, "estacion_fuera_de_catalogo")

        # Genero y anio de nacimiento no invalidan el viaje: se marcan como desconocidos
        unknown_gender = ~df["genero"].isin(["M", "F"])
        df["genero"] = df["genero"].where(~unknown_gender, "Desconocido")
        self._log("genero_nulo", "imputar 'Desconocido'", unknown_gender.sum(), df)

        year = pd.to_numeric(df["anio_nacimiento"], errors="coerce")
        bad_year = year.isna() | (year < r.min_birth_year) | (year > r.max_birth_year)
        df["anio_nacimiento"] = year.where(~bad_year).astype("Int64")
        self._log("anio_nacimiento_invalido", "dejar nulo", bad_year.sum(), df)
        df["edad"] = (df["inicio"].dt.year - df["anio_nacimiento"]).astype("Int64")

        df["duracion_min"] = (df["fin"] - df["inicio"]).dt.total_seconds() / 60
        df["mismo_origen_destino"] = df["origen_id"] == df["destino_id"]

        df = self._drop(df, df["duracion_min"] <= 0, "duracion_no_positiva")
        if r.false_start_max_min is not None:
            false_start = df["mismo_origen_destino"] & (df["duracion_min"] < r.false_start_max_min)
            df = self._drop(df, false_start, "arranque_fallido")
        df = self._drop(df, df["duracion_min"] < r.min_duration_min, "duracion_muy_corta")
        df = self._drop(df, df["duracion_min"] > r.max_duration_min, "duracion_atipica")

        df["genero"] = df["genero"].astype("category")
        return df.reset_index(drop=True)

    def report_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.report)
