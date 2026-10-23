
from pathlib import Path

import pandas as pd

# Carpeta donde se guardaron los datos limpios
DATA_DIR = Path("data/processed")

# Resultados esperados después de la limpieza
EXPECTED_FILES = 142
EXPECTED_ROWS = 39_078_226

# Columnas necesarias para validar los datos
REQUIRED_COLUMNS = {
    "Viaje_Id",
    "Genero",
    "Año_de_nacimiento",
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "duracion_segundos",
    "duracion_no_positiva",
    "duracion_mayor_24h",
}


def parse_mibici_dates(series):
    """Interpreta los dos formatos de fecha identificados en MiBici."""

    # Formato principal: año-mes-día hora:minuto:segundo
    dates = pd.to_datetime(
        series,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    # Formato alternativo: día/mes/año hora:minuto
    mask = series.astype("string").str.contains("/", na=False)

    dates.loc[mask] = pd.to_datetime(
        series.loc[mask],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )

    return dates


def parse_boolean(series):
    """Convierte los indicadores guardados a valores booleanos."""

    values = series.astype("string").str.strip().str.lower()

    return values.map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    })


# Comprobar que exista la carpeta de datos
assert DATA_DIR.exists(), (
    f"No se encontró la carpeta {DATA_DIR}."
)

# Obtener los archivos procesados
files = sorted(DATA_DIR.glob("*.csv"))

assert len(files) == EXPECTED_FILES, (
    f"Se esperaban {EXPECTED_FILES} archivos, "
    f"pero se encontraron {len(files)}."
)

# Contadores de validación
total_rows = 0
missing_ids = 0
missing_gender = 0
missing_birth_year = 0

invalid_start_dates = 0
invalid_end_dates = 0
invalid_durations = 0
invalid_flags = 0
invalid_boolean_values = 0

# Revisar los archivos por lotes
for file in files:
    print(f"Revisando: {file.name}")

    for chunk in pd.read_csv(file, chunksize=100_000):

        # Comprobar que existan las columnas necesarias
        missing_columns = REQUIRED_COLUMNS - set(chunk.columns)

        assert not missing_columns, (
            f"{file.name}: faltan columnas {missing_columns}"
        )

        total_rows += len(chunk)

        # Revisar valores faltantes
        missing_ids += chunk["Viaje_Id"].isna().sum()
        missing_gender += chunk["Genero"].isna().sum()
        missing_birth_year += (
            chunk["Año_de_nacimiento"].isna().sum()
        )

        # Interpretar las fechas con los formatos conocidos
        start_dates = parse_mibici_dates(
            chunk["Inicio_del_viaje"]
        )

        end_dates = parse_mibici_dates(
            chunk["Fin_del_viaje"]
        )

        invalid_start_dates += start_dates.isna().sum()
        invalid_end_dates += end_dates.isna().sum()

        # Calcular nuevamente la duración en segundos
        calculated_duration = (
            end_dates - start_dates
        ).dt.total_seconds()

        saved_duration = pd.to_numeric(
            chunk["duracion_segundos"],
            errors="coerce",
        )

        # Detectar duraciones faltantes o no coincidentes
        invalid_duration = (
            calculated_duration.isna()
            | saved_duration.isna()
            | (
                calculated_duration.notna()
                & saved_duration.notna()
                & (calculated_duration != saved_duration)
            )
        )

        invalid_durations += invalid_duration.sum()

        # Calcular los indicadores esperados
        expected_non_positive = calculated_duration.le(0)
        expected_over_24h = calculated_duration.gt(86_400)

        # Interpretar los indicadores guardados
        saved_non_positive = parse_boolean(
            chunk["duracion_no_positiva"]
        )

        saved_over_24h = parse_boolean(
            chunk["duracion_mayor_24h"]
        )

        # Contar indicadores que no se pueden interpretar
        invalid_boolean_values += (
            saved_non_positive.isna().sum()
        )

        invalid_boolean_values += (
            saved_over_24h.isna().sum()
        )

        # Comparar los indicadores cuando las fechas son válidas
        valid_dates = (
            calculated_duration.notna()
            & saved_non_positive.notna()
            & saved_over_24h.notna()
        )

        invalid_flags += (
            (
                saved_non_positive != expected_non_positive
            )
            & valid_dates
        ).sum()

        invalid_flags += (
            (
                saved_over_24h != expected_over_24h
            )
            & valid_dates
        ).sum()


# Mostrar el resumen
print("\nVALIDACIÓN DE DATOS PROCESADOS")
print("-" * 45)
print(f"Archivos procesados: {len(files)}")
print(f"Total de registros: {total_rows:,}")
print(f"IDs faltantes: {missing_ids:,}")
print(f"Géneros faltantes: {missing_gender:,}")
print(f"Años de nacimiento faltantes: {missing_birth_year:,}")
print(f"Fechas de inicio inválidas: {invalid_start_dates:,}")
print(f"Fechas de fin inválidas: {invalid_end_dates:,}")
print(f"Duraciones inválidas o inconsistentes: {invalid_durations:,}")
print(f"Indicadores inconsistentes: {invalid_flags:,}")
print(f"Valores booleanos inválidos: {invalid_boolean_values:,}")

# Validaciones finales
assert total_rows == EXPECTED_ROWS, (
    f"Se esperaban {EXPECTED_ROWS:,} registros, "
    f"pero se encontraron {total_rows:,}."
)

assert missing_ids == 0, (
    "Hay viajes sin identificador."
)

assert invalid_start_dates == 0, (
    "Hay fechas de inicio que no se pudieron interpretar."
)

assert invalid_end_dates == 0, (
    "Hay fechas de fin que no se pudieron interpretar."
)

assert invalid_durations == 0, (
    "Hay duraciones faltantes o inconsistentes."
)

assert invalid_flags == 0, (
    "Hay indicadores de duración inconsistentes."
)

assert invalid_boolean_values == 0, (
    "Hay valores booleanos que no se pudieron interpretar."
)

print("\nTodas las validaciones se completaron correctamente.")

