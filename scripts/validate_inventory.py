
import pandas as pd
from pathlib import Path

# Leer el inventario
archivo_inventario = Path("data/interim/inventario_archivos.csv")
df = pd.read_csv(archivo_inventario)

# Obtener el año y mes a partir del nombre del archivo
df["periodo_archivo"] = df["archivo"].str.extract(
    r"datos_abiertos_(\d{4}_\d{2})"
)[0]

df["periodo_archivo"] = pd.to_datetime(
    df["periodo_archivo"], format="%Y_%m"
)

# Revisar si hay meses faltantes
meses_esperados = pd.date_range(
    start=df["periodo_archivo"].min(),
    end=df["periodo_archivo"].max(),
    freq="MS"
)

meses_encontrados = pd.DatetimeIndex(
    df["periodo_archivo"].dropna().unique()
).sort_values()

meses_faltantes = meses_esperados.difference(meses_encontrados)

print("=== COBERTURA DE ARCHIVOS ===")
print("Primer mes:", meses_encontrados.min().strftime("%Y-%m"))
print("Último mes:", meses_encontrados.max().strftime("%Y-%m"))
print("Meses encontrados:", len(meses_encontrados))
print("Meses faltantes:", len(meses_faltantes))

if len(meses_faltantes) > 0:
    print(meses_faltantes.strftime("%Y-%m").tolist())
else:
    print("No hay meses faltantes entre el inicio y el final.")

# Revisar fechas y cantidad de viajes
df["primer_viaje"] = pd.to_datetime(
    df["primer_viaje"], errors="coerce"
)
df["ultimo_viaje"] = pd.to_datetime(
    df["ultimo_viaje"], errors="coerce"
)

df["periodo_primer_viaje"] = (
    df["primer_viaje"].dt.to_period("M")
)
df["periodo_ultimo_viaje"] = (
    df["ultimo_viaje"].dt.to_period("M")
)
df["periodo_nombre"] = df["periodo_archivo"].dt.to_period("M")

fechas_fuera = df[
    (df["periodo_primer_viaje"] != df["periodo_nombre"])
    | (df["periodo_ultimo_viaje"] != df["periodo_nombre"])
]

print("\n=== FECHAS FUERA DEL MES DEL ARCHIVO ===")
print("Archivos con fechas fuera del mes:", len(fechas_fuera))

if not fechas_fuera.empty:
    print(
        fechas_fuera[
            ["archivo", "primer_viaje", "ultimo_viaje"]
        ].to_string(index=False)
    )

# Mostrar archivos con menor y mayor número de viajes
print("\n=== ARCHIVOS CON MENOS VIAJES ===")
print(
    df.nsmallest(5, "filas_datos")[
        ["archivo", "filas_datos"]
    ].to_string(index=False)
)

print("\n=== ARCHIVOS CON MÁS VIAJES ===")
print(
    df.nlargest(5, "filas_datos")[
        ["archivo", "filas_datos"]
    ].to_string(index=False)
)

