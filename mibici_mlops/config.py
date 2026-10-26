import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REFERENCES_DIR = PROJ_ROOT / "references"

# Fuente de datos
MIBICI_BASE_URL = "https://mibici.net"
URLS_FILE = REFERENCES_DIR / "urls_2025.txt"
TRIPS_PATTERN = "datos_abiertos_2025_*.csv"
STATIONS_FILE = RAW_DATA_DIR / "nomenclatura_2026_09.csv"

# Nombres de columna normalizados (el encabezado original cambia de codificacion entre meses)
TRIP_COLUMNS = [
    "viaje_id",
    "usuario_id",
    "genero",
    "anio_nacimiento",
    "inicio",
    "fin",
    "origen_id",
    "destino_id",
]

# Version oficial del data engineer: CSV mensuales con banderas de duracion
OFFICIAL_DATA_DIR = Path(os.getenv("MIBICI_OFFICIAL_DIR", EXTERNAL_DATA_DIR / "version_oficial"))
OFFICIAL_PATTERN = "datos_abiertos_2025_*.csv"

# Particion temporal: entrenamiento enero-octubre, validacion noviembre-diciembre de 2025
TRAIN_END = "2025-11-01"

# Dias festivos oficiales de 2025 (Ley Federal del Trabajo, art. 74)
HOLIDAYS_2025 = [
    "2025-01-01",
    "2025-02-03",
    "2025-03-17",
    "2025-05-01",
    "2025-09-16",
    "2025-11-17",
    "2025-12-25",
]

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_DATA = "MiBici_Fase1_Versiones_Datos"
EXPERIMENT_BASELINE = "MiBici_Fase1_Experimentos_Datos"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
