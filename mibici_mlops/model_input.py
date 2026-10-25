"""Entrada del modelo a partir de la version oficial con banderas del data engineer.

Contrato acordado con el data engineer: la version limpia conserva todos los viajes y agrega
`duracion_segundos`, `duracion_no_positiva` y `duracion_mayor_24h`. Para modelar 2025 solo se
excluyen los viajes con `duracion_no_positiva`; los de mas de 24 horas se quedan marcados.

Uso:
    python -m mibici_mlops.model_input   # escribe viajes_limpios_oficial.parquet y registra v1
"""

import hashlib
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mibici_mlops.config import INTERIM_DATA_DIR, OFFICIAL_DATA_DIR, OFFICIAL_PATTERN
from mibici_mlops.tracking import DatasetVersioner, file_sha256

app = typer.Typer()

# Columnas de la version oficial -> nombres que usa el resto del paquete
OFFICIAL_COLUMNS = {
    "Viaje_Id": "viaje_id",
    "Usuario_Id": "usuario_id",
    "Genero": "genero",
    "Año_de_nacimiento": "anio_nacimiento",
    "Inicio_del_viaje": "inicio",
    "Fin_del_viaje": "fin",
    "Origen_Id": "origen_id",
    "Destino_Id": "destino_id",
    "duracion_segundos": "duracion_segundos",
    "duracion_no_positiva": "duracion_no_positiva",
    "duracion_mayor_24h": "duracion_mayor_24h",
}
FLAGS = ["duracion_no_positiva", "duracion_mayor_24h"]
# Banderas informativas que el data engineer agrego despues; se leen si vienen en los archivos
OPTIONAL_FLAGS = ["viaje_fallido"]
EXCLUDE_FOR_MODEL = ["duracion_no_positiva"]


def parse_dates(values: pd.Series) -> pd.Series:
    """Mismos dos formatos que identifico el data engineer en el historico."""
    dates = pd.to_datetime(values, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    missing = dates.isna()
    if missing.any():
        dates[missing] = pd.to_datetime(values[missing], format="%d/%m/%Y %H:%M", errors="coerce")
    return dates


def present_flags(trips: pd.DataFrame) -> list[str]:
    return FLAGS + [f for f in OPTIONAL_FLAGS if f in trips.columns]


def to_bool(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().map({"true": True, "false": False})


def load_official_trips(directory: Path = OFFICIAL_DATA_DIR) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted(Path(directory).glob(OFFICIAL_PATTERN))
    if not files:
        raise FileNotFoundError(f"No hay archivos {OFFICIAL_PATTERN} en {directory}")
    trips = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files])
    missing = set(OFFICIAL_COLUMNS) - set(trips.columns)
    if missing:
        raise ValueError(f"La version oficial no cumple el contrato, faltan: {sorted(missing)}")

    optional = [f for f in OPTIONAL_FLAGS if f in trips.columns]
    trips = trips[list(OFFICIAL_COLUMNS) + optional].rename(columns=OFFICIAL_COLUMNS)
    trips = trips.reset_index(drop=True)
    trips["inicio"] = parse_dates(trips["inicio"])
    trips["fin"] = parse_dates(trips["fin"])
    for col in ["viaje_id", "usuario_id", "origen_id", "destino_id"]:
        trips[col] = pd.to_numeric(trips[col]).astype("int64")
    trips["anio_nacimiento"] = pd.to_numeric(trips["anio_nacimiento"], errors="coerce").astype(
        "Int64"
    )
    trips["genero"] = trips["genero"].replace({"": "Desconocido", "NULL": "Desconocido"})
    trips["duracion_segundos"] = pd.to_numeric(trips["duracion_segundos"], errors="coerce")
    for flag in present_flags(trips):
        trips[flag] = to_bool(trips[flag])
        if trips[flag].isna().any():
            raise ValueError(f"La bandera {flag} tiene valores que no son True o False")
    return trips, files


def model_trips(trips: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Quita solo lo que el contrato marca como invalido y cuenta cada bandera."""
    excluded = trips[EXCLUDE_FOR_MODEL].any(axis=1)
    out = trips.loc[~excluded].copy()
    out["duracion_min"] = out["duracion_segundos"] / 60
    out["mismo_origen_destino"] = out["origen_id"] == out["destino_id"]
    out["edad"] = (out["inicio"].dt.year - out["anio_nacimiento"]).astype("Int64")
    out["genero"] = out["genero"].astype("category")
    counts = {f"marcados_{f}": int(trips[f].sum()) for f in present_flags(trips)}
    counts["excluidos_para_modelo"] = int(excluded.sum())
    return out.reset_index(drop=True), counts


@app.command()
def main(directory: Path = OFFICIAL_DATA_DIR, output_dir: Path = INTERIM_DATA_DIR):
    trips, files = load_official_trips(directory)
    model, counts = model_trips(trips)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "viajes_limpios_oficial.parquet"
    model.to_parquet(output_path, index=False)

    source_sha = hashlib.sha256("".join(file_sha256(f) for f in files).encode()).hexdigest()
    DatasetVersioner().log_version(
        "v1_modelo_oficial",
        model,
        file_sha256(output_path),
        source=str(directory),
        description="Version oficial con banderas, sin viajes de duracion no positiva",
        params={
            "excluye": ",".join(EXCLUDE_FOR_MODEL),
            "conserva_marcados": "duracion_mayor_24h",
            "archivos_oficiales": len(files),
            "sha256_version_oficial": source_sha,
        },
        metrics={"filas_version_oficial": len(trips), **counts},
    )
    logger.success(f"{len(model):,} viajes para modelar en {output_path} ({counts})")


if __name__ == "__main__":
    app()
