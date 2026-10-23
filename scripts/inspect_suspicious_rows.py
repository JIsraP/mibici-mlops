from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")
INVENTORY_FILE = Path("data/interim/inventario_archivos.csv")
OUTPUT_FILE = Path("data/interim/ejemplos_sospechosos.csv")

CHUNK_SIZE = 100_000
MAX_EXAMPLES = 10

inventory = pd.read_csv(INVENTORY_FILE)
examples = {
    "duracion_cero_o_negativa": [],
    "duracion_mayor_24_horas": [],
    "nacimiento_antes_de_1920": [],
}


def parse_dates(values):
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


for _, file_info in inventory.iterrows():
    filename = file_info["archivo"]
    filepath = DATA_DIR / filename
    encoding = file_info["codificacion"]

    print(f"Revisando: {filename}")

    for chunk in pd.read_csv(
        filepath,
        encoding=encoding,
        dtype="string",
        chunksize=CHUNK_SIZE,
    ):
        # Unificar el encabezado del año de nacimiento.
        chunk.columns = [
            "Año_de_nacimiento"
            if column in [
                "A?o_de_nacimiento",
                "Año_de_nacimiento",
                "A}äe_nacimiento",
            ]
            else column
            for column in chunk.columns
        ]

        start = parse_dates(chunk["Inicio_del_viaje"])
        end = parse_dates(chunk["Fin_del_viaje"])
        duration = (end - start).dt.total_seconds()

        year = pd.to_numeric(
            chunk["Año_de_nacimiento"].str.strip(),
            errors="coerce",
        )

        conditions = {
            "duracion_cero_o_negativa": duration.le(0),
            "duracion_mayor_24_horas": duration.gt(24 * 60 * 60),
            "nacimiento_antes_de_1920": year.lt(1920),
        }

        for category, condition in conditions.items():
            remaining = MAX_EXAMPLES - len(examples[category])

            if remaining <= 0:
                continue

            selected = chunk.loc[condition].head(remaining).copy()

            if selected.empty:
                continue

            selected["archivo_origen"] = filename
            selected["duracion_segundos"] = duration.loc[
                selected.index
            ]
            selected["anio_nacimiento_numerico"] = year.loc[
                selected.index
            ]

            examples[category].append(
                selected[
                    [
                        "archivo_origen",
                        "Viaje_Id",
                        "Usuario_Id",
                        "Año_de_nacimiento",
                        "Inicio_del_viaje",
                        "Fin_del_viaje",
                        "Origen_Id",
                        "Destino_Id",
                        "duracion_segundos",
                        "anio_nacimiento_numerico",
                    ]
                ].assign(tipo_problema=category)
            )

        if all(
            len(examples[category]) >= MAX_EXAMPLES
            for category in examples
        ):
            break

    if all(
        len(examples[category]) >= MAX_EXAMPLES
        for category in examples
    ):
        break


frames = []

for category_examples in examples.values():
    if category_examples:
        frames.extend(category_examples)

if frames:
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"\nEjemplos guardados en: {OUTPUT_FILE}")
    print(result.to_string(index=False))
else:
    print("No se encontraron ejemplos para inspeccionar.")
