
from pathlib import Path

import mlflow
import pandas as pd


# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")

DUPLICATES_FILE = INTERIM_DIR / "detalle_duplicados.csv"
SUMMARY_FILE = INTERIM_DIR / "resumen_limpieza.csv"

CHUNK_SIZE = 100_000
CURRENT_YEAR = 2026
MIN_BIRTH_YEAR = 1920

EXPERIMENT_NAME = "limpieza_mibici"
RUN_NAME = "limpieza_datos_mibici"


# --------------------------------------------------
# FUNCIONES AUXILIARES
# --------------------------------------------------

def normalize_columns(df):
    """Unifica el nombre de la columna del año de nacimiento."""
    new_columns = []

    for column in df.columns:
        column_clean = str(column).strip()

        if "nacimiento" in column_clean.lower():
            column_clean = "Año_de_nacimiento"

        new_columns.append(column_clean)

    df.columns = new_columns

    return df


def parse_dates(series):
    """Convierte las fechas usando los formatos identificados."""
    dates = pd.to_datetime(
        series,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    # Algunos archivos utilizan día/mes/año.
    missing = dates.isna()

    if missing.any():
        dates.loc[missing] = pd.to_datetime(
            series.loc[missing],
            format="%d/%m/%Y %H:%M",
            errors="coerce",
        )

    return dates


def detect_encoding(file_path):
    """Detecta la codificación del archivo."""
    try:
        pd.read_csv(
            file_path,
            encoding="utf-8",
            nrows=5,
        )
        return "utf-8"

    except UnicodeDecodeError:
        return "latin-1"


def load_duplicate_ids():
    """Carga los IDs de viajes duplicados detectados previamente."""
    duplicates_df = pd.read_csv(DUPLICATES_FILE)

    if "Viaje_Id" not in duplicates_df.columns:
        raise ValueError(
            "No se encontró la columna Viaje_Id "
            "en detalle_duplicados.csv."
        )

    duplicate_ids = set(
        pd.to_numeric(
            duplicates_df["Viaje_Id"],
            errors="coerce",
        )
        .dropna()
        .astype("int64")
    )

    return duplicate_ids


# --------------------------------------------------
# PROCESO PRINCIPAL DE LIMPIEZA
# --------------------------------------------------

def clean_data():
    """Limpia los archivos mensuales y devuelve las métricas."""

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"No existe el directorio de datos originales: {RAW_DIR}"
        )

    if not DUPLICATES_FILE.exists():
        raise FileNotFoundError(
            f"No existe el archivo de duplicados: {DUPLICATES_FILE}"
        )

    duplicate_ids = load_duplicate_ids()

    files = sorted(RAW_DIR.glob("datos_abiertos_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos mensuales en {RAW_DIR}"
        )

    # IMPORTANTE: eliminar salidas anteriores de estos archivos para no
    # anexar datos viejos ni conservar encabezados sin la nueva columna.
    for file_path in files:
        old_output = PROCESSED_DIR / file_path.name
        if old_output.exists():
            old_output.unlink()

    # Contadores generales.
    input_rows = 0
    output_rows = 0
    duplicate_rows_removed = 0
    missing_gender = 0
    missing_birth_year_initial = 0
    suspicious_birth_year = 0
    nonpositive_duration = 0
    duration_over_24h = 0
    same_station_count = 0
    failed_trip_count = 0
    unparseable_date_rows = 0
    processed_files = 0

    # Guarda los IDs duplicados que ya aparecieron.
    seen_duplicate_ids = set()

    for file_path in files:
        encoding = detect_encoding(file_path)

        # Leer cada archivo por bloques para ahorrar memoria.
        for chunk in pd.read_csv(
            file_path,
            encoding=encoding,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk = normalize_columns(chunk)

            # Convertir cadenas vacías o con espacios en valores faltantes.
            chunk = chunk.replace(
                r"^\s*$",
                pd.NA,
                regex=True,
            )

            required_columns = [
                "Viaje_Id",
                "Usuario_Id",
                "Genero",
                "Año_de_nacimiento",
                "Inicio_del_viaje",
                "Fin_del_viaje",
                "Origen_Id",
                "Destino_Id",
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in chunk.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Faltan columnas en {file_path.name}: "
                    f"{missing_columns}"
                )

            input_rows += len(chunk)

            # Validar y convertir los IDs de viaje.
            chunk["Viaje_Id"] = pd.to_numeric(
                chunk["Viaje_Id"],
                errors="coerce",
            )

            if chunk["Viaje_Id"].isna().any():
                raise ValueError(
                    f"Hay IDs de viaje vacíos o inválidos "
                    f"en {file_path.name}."
                )

            chunk["Viaje_Id"] = chunk["Viaje_Id"].astype("int64")

            # --------------------------------------------------
            # ELIMINACIÓN DE DUPLICADOS
            # --------------------------------------------------

            # Considerar solamente IDs identificados en el diagnóstico.
            duplicate_candidates = chunk["Viaje_Id"].isin(
                duplicate_ids
            )

            # IDs que ya aparecieron en bloques anteriores.
            already_seen = chunk["Viaje_Id"].isin(
                seen_duplicate_ids
            )

            # Repeticiones dentro del bloque actual.
            repeated_in_chunk = chunk["Viaje_Id"].duplicated(
                keep="first"
            )

            # Eliminar la segunda aparición de cada ID confirmado.
            duplicate_mask = duplicate_candidates & (
                already_seen | repeated_in_chunk
            )

            duplicate_rows_removed += int(
                duplicate_mask.sum()
            )

            # Registrar los IDs antes de retirar las filas.
            seen_duplicate_ids.update(
                chunk.loc[
                    duplicate_candidates,
                    "Viaje_Id",
                ].tolist()
            )

            chunk = chunk.loc[~duplicate_mask].copy()

            # --------------------------------------------------
            # VALORES FALTANTES
            # --------------------------------------------------

            missing_gender += int(
                chunk["Genero"].isna().sum()
            )

            chunk["Año_de_nacimiento"] = pd.to_numeric(
                chunk["Año_de_nacimiento"],
                errors="coerce",
            )

            missing_birth_year_initial += int(
                chunk["Año_de_nacimiento"].isna().sum()
            )

            # --------------------------------------------------
            # AÑOS DE NACIMIENTO SOSPECHOSOS
            # --------------------------------------------------

            suspicious_year_mask = (
                chunk["Año_de_nacimiento"].notna()
                & (
                    (
                        chunk["Año_de_nacimiento"]
                        < MIN_BIRTH_YEAR
                    )
                    | (
                        chunk["Año_de_nacimiento"]
                        > CURRENT_YEAR
                    )
                )
            )

            suspicious_birth_year += int(
                suspicious_year_mask.sum()
            )

            # No se elimina el viaje; solo se invalida el año.
            chunk.loc[
                suspicious_year_mask,
                "Año_de_nacimiento",
            ] = float("nan")

            # --------------------------------------------------
            # FECHAS Y DURACIONES
            # --------------------------------------------------

            start_dates = parse_dates(
                chunk["Inicio_del_viaje"]
            )

            end_dates = parse_dates(
                chunk["Fin_del_viaje"]
            )

            invalid_dates = (
                start_dates.isna()
                | end_dates.isna()
            )

            unparseable_date_rows += int(
                invalid_dates.sum()
            )

            duration = (
                end_dates - start_dates
            ).dt.total_seconds()

            chunk["duracion_segundos"] = duration

            chunk["duracion_no_positiva"] = (
                duration.notna()
                & (duration <= 0)
            )

            chunk["duracion_mayor_24h"] = (
                duration.notna()
                & (duration > 24 * 60 * 60)
            )

            nonpositive_duration += int(
                chunk["duracion_no_positiva"].sum()
            )

            duration_over_24h += int(
                chunk["duracion_mayor_24h"].sum()
            )
            
            # Normalizar IDs como texto para comparar 12 con "12" correctamente.
            origin_id = chunk["Origen_Id"].astype("string").str.strip()
            destination_id = chunk["Destino_Id"].astype("string").str.strip()
            same_station = (
                origin_id.notna()
                & destination_id.notna()
                & (origin_id == destination_id)
            )

            # Bandera informativa: no elimina estos viajes.
            chunk["viaje_fallido"] = (
                same_station
                & duration.notna()
                & (duration > 0)
                & (duration < 120)
            )

            same_station_count += int(same_station.sum())
            failed_trip_count += int(chunk["viaje_fallido"].sum())

            # --------------------------------------------------
            # GUARDAR DATOS PROCESADOS
            # --------------------------------------------------

            output_file = PROCESSED_DIR / file_path.name

            # El encabezado se escribe solamente en el primer bloque.
            write_header = not output_file.exists()

            chunk.to_csv(
                output_file,
                mode="a",
                index=False,
                header=write_header,
            )

            output_rows += len(chunk)

        processed_files += 1

    # --------------------------------------------------
    # VALIDACIONES AUTOMÁTICAS
    # --------------------------------------------------

    assert input_rows > 0, (
        "No se procesaron filas de entrada."
    )

    assert (
        input_rows - duplicate_rows_removed == output_rows
    ), (
        "La cantidad de filas no coincide después "
        "de eliminar duplicados."
    )

    assert processed_files == len(files), (
        "No se procesaron todos los archivos."
    )

    assert unparseable_date_rows == 0, (
        "Hay fechas que no se pudieron interpretar."
    )

    assert duplicate_ids.issubset(seen_duplicate_ids), (
        "Algunos IDs del diagnóstico de duplicados "
        "no aparecieron en los archivos originales."
    )

    # --------------------------------------------------
    # RESUMEN DE LIMPIEZA
    # --------------------------------------------------

    summary = pd.DataFrame(
        {
            "Indicador": [
                "Filas de entrada",
                "Filas de salida",
                "Filas duplicadas eliminadas",
                "Género faltante",
                "Año de nacimiento faltante inicialmente",
                "Años de nacimiento sospechosos",
                "Duraciones cero o negativas",
                "Duraciones mayores a 24 horas",
                "Viajes con mismo origen y destino",
                "Viajes marcados como viaje_fallido",
                "Filas con alguna fecha no interpretable",
                "Archivos procesados",
            ],
            "Cantidad": [
                input_rows,
                output_rows,
                duplicate_rows_removed,
                missing_gender,
                missing_birth_year_initial,
                suspicious_birth_year,
                nonpositive_duration,
                duration_over_24h,
                same_station_count,
                failed_trip_count,
                unparseable_date_rows,
                processed_files,
            ],
        }
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # Métricas numéricas para MLflow.
    metrics = {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "duplicate_rows_removed": duplicate_rows_removed,
        "missing_gender": missing_gender,
        "missing_birth_year_initial": missing_birth_year_initial,
        "suspicious_birth_year": suspicious_birth_year,
        "nonpositive_duration": nonpositive_duration,
        "duration_over_24h": duration_over_24h,
        "same_station_trips": same_station_count,
        "failed_trips": failed_trip_count,
        "unparseable_date_rows": unparseable_date_rows,
        "processed_files": processed_files,
        "duplicate_removal_rate": (
            duplicate_rows_removed / input_rows
        ),
    }

    print("\nLimpieza terminada.")
    print(summary.to_string(index=False))
    print(f"\nArchivos procesados: {processed_files}")
    print(f"Resultados guardados en: {PROCESSED_DIR}")

    return metrics


# --------------------------------------------------
# EJECUCIÓN Y REGISTRO EN MLFLOW
# --------------------------------------------------

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name=RUN_NAME
    ) as run:

        # Registrar las decisiones de limpieza.
        mlflow.log_params(
            {
                "chunk_size": CHUNK_SIZE,
                "current_year_reference": CURRENT_YEAR,
                "minimum_birth_year": MIN_BIRTH_YEAR,
                "input_directory": str(RAW_DIR),
                "output_directory": str(PROCESSED_DIR),
                "duplicate_handling": (
                    "Eliminar una segunda aparición de IDs "
                    "confirmados como duplicados"
                ),
                "birth_year_handling": (
                    "Convertir años menores a 1920 o mayores "
                    "a 2026 en valores faltantes"
                ),
                "duration_handling": (
                    "Conservar viajes sospechosos y agregar "
                    "indicadores"
                ),
                "failed_trip_rule": (
                    "Mismo origen y destino, duracion mayor que 0 "
                    "y menor que 120 segundos; conservar el viaje"
                ),
                "date_formats": (
                    "%Y-%m-%d %H:%M:%S; %d/%m/%Y %H:%M"
                ),
                "number_of_input_files": len(
                    list(
                        RAW_DIR.glob("datos_abiertos_*.csv")
                    )
                ),
            }
        )

        # Ejecutar la limpieza.
        metrics = clean_data()

        # Registrar las métricas.
        mlflow.log_metrics(metrics)

        # Registrar el resumen, no todos los CSV procesados.
        mlflow.log_artifact(str(SUMMARY_FILE))

        print("\nRegistro de MLflow completado.")
        print(f"Experimento: {EXPERIMENT_NAME}")
        print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()

