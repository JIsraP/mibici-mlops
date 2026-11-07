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

# Periodo de modelado: un año calendario (los viajes de 2026 quedan reservados)
DATA_YEAR = int(os.getenv("MIBICI_YEAR", "2025"))
YEAR_START = f"{DATA_YEAR}-01-01"
YEAR_END = f"{DATA_YEAR + 1}-01-01"
OFFICIAL_PATTERN = f"datos_abiertos_{DATA_YEAR}_*.csv"

# Ventanas horarias que usa la operacion: picos de dias laborales y mañana laboral
PEAK_HOURS = [7, 8, 9, 18, 19, 20]
MORNING_HOURS = [7, 8, 9]

# Umbrales en minutos para describir viajes marcados (no excluyen viajes)
FALSE_START_MAX_MIN = 2.0
SHORT_TRIP_MAX_MIN = 1.0
LONG_TRIP_MIN = 180.0
MAX_TRIP_MIN = 1440.0
MIN_STATION_TRIPS_PLOT = 1000

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
EXPERIMENT_MODELS = "MiBici_Fase2_Modelos"

# Fase 2: metrica para elegir modelo y semilla comun para que las corridas sean reproducibles
SELECTION_METRIC = "salidas_mae_pico"
TARGET_IMPROVEMENT_PCT = 15.0  # meta inicial del ML Canvas frente a la linea base
RANDOM_SEED = int(os.getenv("MIBICI_SEED", "42"))

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
