
import csv
import sqlite3
from pathlib import Path

import pandas as pd


# Carpetas del proyecto
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/interim")

# Archivos de salida
DB_PATH = OUTPUT_DIR / "viaje_ids.sqlite"
REPORT_PATH = OUTPUT_DIR / "reporte_duplicados.csv"
SAMPLE_PATH = OUTPUT_DIR / "ejemplos_duplicados.csv"

CHUNK_SIZE = 100_000
MAX_EXAMPLES = 20


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(file for file in DATA_DIR.glob("*.csv") if file.name.startswith("datos_abiertos_"))

    if not files:
        print("No se encontraron archivos CSV en data/raw.")
        return

    # Crear una base de datos temporal para contar los identificadores.
    connection = sqlite3.connect(DB_PATH)

    connection.execute("PRAGMA journal_mode = DELETE")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS viaje_ids (
            viaje_id TEXT PRIMARY KEY,
            repeticiones INTEGER NOT NULL DEFAULT 1
        )
    """)

    # Permite volver a ejecutar el análisis desde cero.
    connection.execute("DELETE FROM viaje_ids")
    connection.commit()

    total_rows = 0
    filas_sin_id = 0
    ejemplos = []

    try:
        for file_number, file_path in enumerate(files, start=1):
            print(f"[{file_number}/{len(files)}] Revisando {file_path.name}")

            # Detectar automáticamente la codificación del archivo.
            try:
                with open(file_path, "r", encoding="utf-8-sig") as file:
                    first_line = file.readline()
            except UnicodeDecodeError:
                encoding = "latin-1"
            else:
                encoding = "utf-8-sig"

            # Leer el archivo por bloques para no cargarlo completo.
            for chunk in pd.read_csv(
                file_path,
                encoding=encoding,
                dtype=str,
                chunksize=CHUNK_SIZE,
                on_bad_lines="error",
            ):
                # Normalizar el nombre del identificador.
                id_column = next(
                    (
                        column
                        for column in chunk.columns
                        if column.strip().lstrip("\ufeff") == "Viaje_Id"
                    ),
                    None,
                )

                if id_column is None:
                    raise ValueError(
                        f"No se encontró Viaje_Id en {file_path.name}"
                    )

                ids = chunk[id_column].astype("string").str.strip()
                ids = ids.replace("", pd.NA)

                filas_sin_id += int(ids.isna().sum())

                valid_ids = ids.dropna()
                total_rows += len(valid_ids)

                # Contar repeticiones sin almacenar todos los IDs en RAM.
                counts = valid_ids.value_counts()

                connection.execute("BEGIN")

                for viaje_id, cantidad in counts.items():
                    cursor = connection.execute(
                        """
                        INSERT INTO viaje_ids (viaje_id, repeticiones)
                        VALUES (?, ?)
                        ON CONFLICT(viaje_id)
                        DO UPDATE SET
                            repeticiones = repeticiones + excluded.repeticiones
                        """,
                        (str(viaje_id), int(cantidad)),
                    )

                connection.commit()

        # Resumen global
        unique_ids = connection.execute(
            "SELECT COUNT(*) FROM viaje_ids"
        ).fetchone()[0]

        duplicated_ids = connection.execute(
            "SELECT COUNT(*) FROM viaje_ids WHERE repeticiones > 1"
        ).fetchone()[0]

        extra_rows = connection.execute(
            "SELECT COALESCE(SUM(repeticiones - 1), 0) "
            "FROM viaje_ids WHERE repeticiones > 1"
        ).fetchone()[0]

        report = [
            {"indicador": "Archivos revisados", "valor": len(files)},
            {"indicador": "Filas con Viaje_Id", "valor": total_rows},
            {"indicador": "Filas sin Viaje_Id", "valor": filas_sin_id},
            {"indicador": "Identificadores únicos", "valor": unique_ids},
            {
                "indicador": "IDs que aparecen más de una vez",
                "valor": duplicated_ids,
            },
            {
                "indicador": "Filas repetidas adicionales",
                "valor": extra_rows,
            },
        ]

        pd.DataFrame(report).to_csv(
            REPORT_PATH, index=False, encoding="utf-8-sig"
        )

        # Guardar una muestra de los identificadores repetidos.
        rows = connection.execute(
            """
            SELECT viaje_id, repeticiones
            FROM viaje_ids
            WHERE repeticiones > 1
            ORDER BY repeticiones DESC, viaje_id
            LIMIT ?
            """,
            (MAX_EXAMPLES,),
        ).fetchall()

        with open(SAMPLE_PATH, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["Viaje_Id", "repeticiones"])
            writer.writerows(rows)

        print("\n--- RESULTADOS ---")
        for item in report:
            print(f"{item['indicador']}: {item['valor']}")

        print(f"\nReporte guardado en: {REPORT_PATH}")
        print(f"Muestra guardada en: {SAMPLE_PATH}")
        print(f"Base de datos temporal: {DB_PATH}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()

