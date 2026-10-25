
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")
INVENTORY_FILE = Path("data/interim/inventario_archivos.csv")
OUTPUT_DIR = Path("data/interim")
CHUNK_SIZE = 100_000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

inventory = pd.read_csv(INVENTORY_FILE)

missing_counts = {}
gender_counts = {}
birth_year_counts = {
    "no_numerico": 0,
    "antes_de_1920": 0,
    "despues_de_2026": 0,
}
date_counts = {
    "fechas_inicio_invalidas": 0,
    "fechas_fin_invalidas": 0,
    "duracion_cero_o_negativa": 0,
    "duracion_mayor_24_horas": 0,
}
schema_counts = {}
total_rows = 0


def parse_dates(values):
    """Convierte los formatos de fecha identificados."""
    values = values.astype("string").str.strip()

    dates = pd.to_datetime(
        values,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    missing = dates.isna()

    if missing.any():
        dates.loc[missing] = pd.to_datetime(
            values.loc[missing],
            format="%d/%m/%Y %H:%M",
            errors="coerce",
        )

    return dates


for _, row in inventory.iterrows():
    filename = row["archivo"]
    encoding = row["codificacion"]
    filepath = DATA_DIR / filename

    print(f"Analizando: {filename}")

    for chunk in pd.read_csv(
        filepath,
        encoding=encoding,
        dtype="string",
        chunksize=CHUNK_SIZE,
    ):
        # Unificar el nombre de la columna del año de nacimiento.
        chunk.columns = [
            "Año_de_nacimiento"
            if column in ["A?o_de_nacimiento", "Año_de_nacimiento", "A}äe_nacimiento"]
            else column
            for column in chunk.columns
        ]

        columns_key = " | ".join(sorted(chunk.columns))
        schema_counts[columns_key] = (
            schema_counts.get(columns_key, 0) + len(chunk)
        )

        total_rows += len(chunk)

        # Contar valores faltantes o vacíos.
        for column in chunk.columns:
            values = chunk[column].str.strip()
            missing = values.isna() | values.eq("")

            missing_counts[column] = (
                missing_counts.get(column, 0) + int(missing.sum())
            )

        # Revisar categorías de género.
        if "Genero" in chunk.columns:
            values = chunk["Genero"].str.strip().fillna("<VACÍO>")
            counts = values.value_counts()

            for value, count in counts.items():
                gender_counts[value] = (
                    gender_counts.get(value, 0) + int(count)
                )

        # Revisar años de nacimiento.
        if "Año_de_nacimiento" in chunk.columns:
            birth_year = pd.to_numeric(
                chunk["Año_de_nacimiento"].str.strip(),
                errors="coerce",
            )

            birth_year_counts["no_numerico"] += int(
                (
                    chunk["Año_de_nacimiento"].notna()
                    & birth_year.isna()
                ).sum()
            )

            birth_year_counts["antes_de_1920"] += int(
                (birth_year < 1920).sum()
            )

            birth_year_counts["despues_de_2026"] += int(
                (birth_year > 2026).sum()
            )

        # Revisar fechas y duración de los viajes.
        if "Inicio_del_viaje" in chunk.columns:
            start = parse_dates(chunk["Inicio_del_viaje"])
            end = parse_dates(chunk["Fin_del_viaje"])

            date_counts["fechas_inicio_invalidas"] += int(
                (
                    chunk["Inicio_del_viaje"].notna()
                    & start.isna()
                ).sum()
            )

            date_counts["fechas_fin_invalidas"] += int(
                (
                    chunk["Fin_del_viaje"].notna()
                    & end.isna()
                ).sum()
            )

            duration = (end - start).dt.total_seconds()

            date_counts["duracion_cero_o_negativa"] += int(
                (duration <= 0).sum()
            )

            date_counts["duracion_mayor_24_horas"] += int(
                (duration > 24 * 60 * 60).sum()
            )


# Guardar los resultados del diagnóstico.
missing_report = pd.DataFrame(
    [
        {
            "columna": column,
            "valores_faltantes": count,
            "porcentaje_faltante": round(
                count / total_rows * 100, 4
            ),
        }
        for column, count in missing_counts.items()
    ]
).sort_values("valores_faltantes", ascending=False)

missing_report.to_csv(
    OUTPUT_DIR / "diagnostico_faltantes.csv",
    index=False,
)

pd.DataFrame(
    [
        {"genero": value, "cantidad": count}
        for value, count in gender_counts.items()
    ]
).sort_values("cantidad", ascending=False).to_csv(
    OUTPUT_DIR / "diagnostico_genero.csv",
    index=False,
)

print("\n=== RESUMEN GENERAL ===")
print(f"Total de viajes revisados: {total_rows:,}")

print("\n=== COLUMNAS ENCONTRADAS ===")
for columns, count in schema_counts.items():
    print(f"{count:,} filas: {columns}")

print("\n=== VALORES FALTANTES ===")
print(missing_report.to_string(index=False))

print("\n=== DISTRIBUCIÓN DE GÉNERO ===")
print(pd.DataFrame(
    list(gender_counts.items()),
    columns=["Genero", "Cantidad"],
).sort_values("Cantidad", ascending=False).to_string(index=False))

print("\n=== AÑOS DE NACIMIENTO SOSPECHOSOS ===")
for name, count in birth_year_counts.items():
    print(f"{name}: {count:,}")

print("\n=== FECHAS Y DURACIONES ===")
for name, count in date_counts.items():
    print(f"{name}: {count:,}")

print("\nReportes guardados en:", OUTPUT_DIR)

