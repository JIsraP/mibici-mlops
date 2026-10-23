
import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/interim/detalle_duplicados.csv")
OUTPUT_DIR = Path("data/interim")

def main():
    df = pd.read_csv(INPUT_PATH, dtype=str)

    # Revisar cuántas filas y archivos corresponden a cada ID.
    resumen = df.groupby("Viaje_Id").agg(
        numero_filas=("Viaje_Id", "size"),
        numero_archivos=("archivo_origen", "nunique"),
        archivos=("archivo_origen", lambda x: " | ".join(sorted(set(x)))),
    ).reset_index()

    resumen["ubicacion"] = resumen["numero_archivos"].apply(
        lambda n: "Varios archivos" if n > 1 else "Mismo archivo"
    )

    # Comprobar si cada ID aparece exactamente dos veces.
    resumen["aparece_dos_veces"] = resumen["numero_filas"] == 2

    resumen.to_csv(
        OUTPUT_DIR / "ubicacion_duplicados.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("--- UBICACIÓN DE LOS DUPLICADOS ---")
    print(f"IDs revisados: {len(resumen)}")
    print(f"IDs que aparecen dos veces: {resumen['aparece_dos_veces'].sum()}")
    print(
        "IDs dentro de un mismo archivo:",
        (resumen["ubicacion"] == "Mismo archivo").sum(),
    )
    print(
        "IDs repartidos entre varios archivos:",
        (resumen["ubicacion"] == "Varios archivos").sum(),
    )

    print("\nFrecuencia de filas por ID:")
    print(resumen["numero_filas"].value_counts().sort_index().to_string())

    print("\nArchivo generado: data/interim/ubicacion_duplicados.csv")


if __name__ == "__main__":
    main()

