
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/interim/inventario_archivos.csv")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

csv_files = sorted(DATA_DIR.glob("datos_abiertos_*.csv"))

inventory = []


def convert_dates(values):
    """Convierte fechas de los distintos formatos encontrados."""
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


for file in csv_files:
    print(f"Revisando: {file.name}")

    try:
        # Detectar la codificación del archivo.
        encoding = "utf-8"

        try:
            pd.read_csv(file, nrows=5, encoding=encoding)
        except UnicodeDecodeError:
            encoding = "latin-1"

        # Obtener los nombres de las columnas.
        columns = pd.read_csv(
            file,
            nrows=0,
            encoding=encoding,
        ).columns.tolist()

        first_date = None
        last_date = None
        rows = 0

        # Leer el archivo por bloques para ahorrar memoria.
        for chunk in pd.read_csv(
            file,
            usecols=["Inicio_del_viaje"],
            chunksize=100_000,
            encoding=encoding,
            dtype={"Inicio_del_viaje": "string"},
        ):
            dates = convert_dates(
                chunk["Inicio_del_viaje"]
            ).dropna()

            rows += len(chunk)

            if not dates.empty:
                chunk_first = dates.min()
                chunk_last = dates.max()

                if first_date is None or chunk_first < first_date:
                    first_date = chunk_first

                if last_date is None or chunk_last > last_date:
                    last_date = chunk_last

        inventory.append({
            "archivo": file.name,
            "codificacion": encoding,
            "tamano_mb": round(
                file.stat().st_size / (1024 ** 2), 2
            ),
            "filas_datos": rows,
            "numero_columnas": len(columns),
            "columnas": " | ".join(columns),
            "primer_viaje": first_date,
            "ultimo_viaje": last_date,
            "estado": "OK",
        })

    except Exception as error:
        inventory.append({
            "archivo": file.name,
            "codificacion": None,
            "tamano_mb": round(
                file.stat().st_size / (1024 ** 2), 2
            ),
            "filas_datos": None,
            "numero_columnas": None,
            "columnas": None,
            "primer_viaje": None,
            "ultimo_viaje": None,
            "estado": str(error),
        })

result = pd.DataFrame(inventory)
result.to_csv(OUTPUT_FILE, index=False)

print("\nInventario terminado.")
print(f"Archivos revisados: {len(result)}")
print(f"Archivo generado: {OUTPUT_FILE}")

print("\nEstado de la revisión:")
print(result["estado"].value_counts())

print("\nCodificaciones utilizadas:")
print(result["codificacion"].value_counts(dropna=False))

print("\nPrimeros registros del inventario:")
print(
    result[
        ["archivo", "filas_datos", "primer_viaje", "ultimo_viaje", "estado"]
    ].head()
)

