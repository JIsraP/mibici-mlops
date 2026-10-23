
import sqlite3
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/interim")

DB_PATH = OUTPUT_DIR / "viaje_ids.sqlite"
OUTPUT_PATH = OUTPUT_DIR / "detalle_duplicados.csv"

CHUNK_SIZE = 100_000


def normalize_columns(columns):
    normalized = []

    for column in columns:
        if column in [
            "A?o_de_nacimiento",
            "Año_de_nacimiento",
            "A}äe_nacimiento",
        ]:
            normalized.append("Año_de_nacimiento")
        else:
            normalized.append(column)

    return normalized


def read_matching_rows(file_path, duplicate_ids, encoding):
    matches = []

    for chunk in pd.read_csv(
        file_path,
        encoding=encoding,
        dtype=str,
        chunksize=CHUNK_SIZE,
        on_bad_lines="error",
    ):
        chunk.columns = normalize_columns(chunk.columns)

        if "Viaje_Id" not in chunk.columns:
            raise ValueError(
                f"No se encontró Viaje_Id en {file_path.name}"
            )

        ids = chunk["Viaje_Id"].astype("string").str.strip()
        selected = chunk.loc[ids.isin(duplicate_ids)].copy()

        if not selected.empty:
            selected.insert(0, "archivo_origen", file_path.name)
            matches.extend(selected.to_dict("records"))

    return matches


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Recuperar los IDs que aparecen más de una vez.
    with sqlite3.connect(DB_PATH) as connection:
        duplicate_ids = {
            row[0]
            for row in connection.execute(
                """
                SELECT viaje_id
                FROM viaje_ids
                WHERE repeticiones > 1
                """
            )
        }

    print(f"IDs repetidos por investigar: {len(duplicate_ids)}")

    all_matches = []
    files = sorted(DATA_DIR.glob("datos_abiertos_*.csv"))

    for number, file_path in enumerate(files, start=1):
        print(f"[{number}/{len(files)}] {file_path.name}")

        try:
            matches = read_matching_rows(
                file_path, duplicate_ids, "utf-8-sig"
            )
        except UnicodeDecodeError:
            # Si el archivo no es UTF-8, probar con Latin-1.
            matches = read_matching_rows(
                file_path, duplicate_ids, "latin-1"
            )

        all_matches.extend(matches)

    if not all_matches:
        print("No se encontraron filas para los IDs repetidos.")
        return

    result = pd.DataFrame(all_matches)

    # Contar cuántas filas originales encontramos por cada ID.
    counts = result.groupby("Viaje_Id")["Viaje_Id"].transform("size")
    result["apariciones_encontradas"] = counts

    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # Comparar las columnas del viaje, sin incluir el archivo de origen.
    columns_to_compare = [
        column
        for column in result.columns
        if column not in [
            "archivo_origen",
            "apariciones_encontradas",
        ]
    ]

    unique_rows = result[columns_to_compare].drop_duplicates()
    unique_row_count = len(unique_rows)
    total_rows = len(result)

    print("\n--- RESULTADOS DE LA COMPARACIÓN ---")
    print(f"Filas encontradas: {total_rows}")
    print(f"Filas distintas por contenido: {unique_row_count}")
    print(f"Filas idénticas adicionales: {total_rows - unique_row_count}")
    print(f"Archivo guardado en: {OUTPUT_PATH}")

    if total_rows != len(duplicate_ids) * 2:
        print(
            "\nAviso: no se encontraron exactamente dos filas por ID. "
            "Revisa el archivo de detalle antes de decidir cómo limpiar."
        )


if __name__ == "__main__":
    main()

