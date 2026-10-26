from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")
resultados = {}

for archivo in sorted(PROCESSED_DIR.glob("datos_abiertos_*.csv")):
    for chunk in pd.read_csv(
        archivo,
        chunksize=100_000,
        low_memory=False
    ):
        # Obtener el año de inicio del viaje
        fechas = pd.to_datetime(
            chunk["Inicio_del_viaje"],
            errors="coerce",
            format="mixed"
        )

        # Comparar origen y destino
        misma_estacion = (
            chunk["Origen_Id"].astype("string").str.strip()
            == chunk["Destino_Id"].astype("string").str.strip()
        )

        # Contar por año
        años = fechas.dt.year

        conteo = misma_estacion.groupby(años).sum()

        for año, cantidad in conteo.items():
            if pd.notna(año):
                resultados[int(año)] = (
                    resultados.get(int(año), 0) + int(cantidad)
                )

# Mostrar resultados por año
tabla = pd.DataFrame(
    sorted(resultados.items()),
    columns=["Año", "Viajes_mismo_origen_destino"]
)

print(tabla.to_string(index=False))
print("\nTotal histórico:", tabla["Viajes_mismo_origen_destino"].sum())