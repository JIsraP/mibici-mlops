"""Descarga, lectura y limpieza de los viajes de MiBici.

Uso:
    python -m mibici.dataset download   # baja los CSV de references/urls_2025.txt a data/raw
    python -m mibici.dataset main       # limpia y registra las versiones v0 y v1 en MLflow
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

from loguru import logger
import pandas as pd
import requests
import typer

from mibici.cleaning import CleaningRules, TripCleaner
from mibici.config import (
    INTERIM_DATA_DIR,
    MIBICI_BASE_URL,
    RAW_DATA_DIR,
    STATIONS_FILE,
    TRIP_COLUMNS,
    TRIPS_PATTERN,
    URLS_FILE,
)
from mibici.tracking import DatasetVersioner

app = typer.Typer()


@dataclass
class RawFile:
    """Metadatos de un archivo crudo, necesarios para comprobar que todos usan los mismos datos."""

    name: str
    encoding: str
    rows: int
    sha256: str


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detect_encoding(path: Path) -> str:
    """Los CSV del portal mezclan UTF-8 y Latin-1 segun el mes."""
    try:
        path.read_bytes().decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def load_raw_trips(raw_dir: Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, list[RawFile]]:
    """Une los CSV mensuales sin modificar valores: todo se lee como texto."""
    frames, files = [], []
    for path in sorted(raw_dir.glob(TRIPS_PATTERN)):
        encoding = detect_encoding(path)
        df = pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False)
        df.columns = TRIP_COLUMNS
        frames.append(df)
        files.append(RawFile(path.name, encoding, len(df), sha256_of(path)))
        logger.info(f"{path.name}: {len(df):,} filas ({encoding})")
    if not frames:
        raise FileNotFoundError(f"No hay archivos {TRIPS_PATTERN} en {raw_dir}")
    return pd.concat(frames, ignore_index=True), files


def load_stations(path: Path = STATIONS_FILE) -> pd.DataFrame:
    st = pd.read_csv(path).rename(
        columns={"id": "estacion_id", "name": "nombre", "dpcapacity": "capacidad"}
    )
    st["zona"] = st["nombre"].str.extract(r"\((\w+)-")[0].fillna("SIN_ZONA")
    return st.rename(columns={"latitude": "latitud", "longitude": "longitud"})


@app.command()
def download(urls_file: Path = URLS_FILE, raw_dir: Path = RAW_DATA_DIR):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for url_path in urls_file.read_text().split():
        target = raw_dir / Path(url_path).name
        if target.exists() and target.stat().st_size > 0:
            continue
        logger.info(f"Descargando {target.name}")
        resp = requests.get(MIBICI_BASE_URL + url_path, timeout=300)
        resp.raise_for_status()
        target.write_bytes(resp.content)
    logger.success("Descarga completa")


@app.command()
def main(variant: str = "estricta", output_dir: Path = INTERIM_DATA_DIR):
    """Registra v0 (crudo) una sola vez por corrida y v1 (viajes limpios) para la variante dada."""
    rules = CleaningRules.variant(variant)
    raw, files = load_raw_trips()
    stations = load_stations()

    versioner = DatasetVersioner()
    versioner.log_raw(raw, files)

    cleaner = TripCleaner(rules, valid_station_ids=set(stations["estacion_id"]))
    clean = cleaner.clean(raw)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"viajes_limpios_{variant}.parquet"
    clean.to_parquet(output_path, index=False)
    versioner.log_clean(clean, output_path, rules, cleaner.report)
    logger.success(f"{len(clean):,} viajes limpios en {output_path}")


if __name__ == "__main__":
    app()
