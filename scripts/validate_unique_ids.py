
from pathlib import Path
import sqlite3

import pandas as pd

# Carpeta con los datos limpios
DATA_DIR = Path("data/processed")

# Base de datos temporal para revisar los identificadores
DB_PATH = Path("data/interim/validacion_ids.sqlite")

# Crear la carpeta de resultados si no existe
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Obtener los archivos limpios
files = sorted(DATA_DIR.glob("*.csv"))

assert len(files) == 142, (
    f"Se esperaban 142 archivos, pero se encontraron {len(files)}."
)

# Crear la base de datos de validación desde cero
if DB_PATH.exists():
    DB_PATH.unlink()

conn = sqlite3.connect(DB_PATH)

# La tabla guardará un solo registro por identificador
conn.execute("""
    CREATE TABLE viajes (
        Viaje_Id TEXT PRIMARY KEY
    )
""")

total_rows = 0
duplicate_ids = 0

try:
    for file in files:
        print(f"Revisando: {file.name}")

        for chunk in pd.read_csv(
            file,
            usecols=["Viaje_Id"],
            dtype={"Viaje_Id": "string"},
            chunksize=100_000,
        ):
            total_rows += len(chunk)

            # Insertar IDs únicos; los repetidos se ignoran
            ids = chunk["Viaje_Id"].dropna().drop_duplicates()

            before = conn.total_changes

            conn.executemany(
                """
                INSERT OR IGNORE INTO viajes (Viaje_Id)
                VALUES (?)
                """,
                [(str(value),) for value in ids],
            )

            inserted = conn.total_changes - before
            duplicate_ids += len(ids) - inserted

        conn.commit()

    unique_ids = conn.execute(
        "SELECT COUNT(*) FROM viajes"
    ).fetchone()[0]

    print("\nVALIDACIÓN DE IDENTIFICADORES")
    print("-" * 45)
    print(f"Registros revisados: {total_rows:,}")
    print(f"Identificadores únicos: {unique_ids:,}")
    print(f"Identificadores duplicados adicionales: {duplicate_ids:,}")

    assert total_rows == 39_078_226, (
        "El total de registros no coincide con el esperado."
    )

    assert duplicate_ids == 0, (
        "Todavía existen identificadores duplicados."
    )

    print("\nTodos los identificadores son únicos.")

finally:
    conn.close()

