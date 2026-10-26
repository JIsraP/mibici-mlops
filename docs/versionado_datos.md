# Bitácora de versiones de datos

Cada versión del dataset es una corrida en el experimento de MLflow `MiBici_Fase1_Versiones_Datos`. La corrida guarda la etiqueta `version`, la huella SHA-256 del archivo, el número de filas y columnas, el esquema con nulos por columna y, según el caso, el reporte de limpieza, el archivo de datos o el preprocesador. Si se vuelve a registrar el mismo nombre de versión y la misma huella, se reutiliza el identificador de una corrida terminada correctamente que no esté marcada como descartada (`DatasetVersioner._existing_run`). Una ejecución fallida o descartada permite un nuevo intento; dos versiones distintas pueden compartir la misma huella sin bloquear su registro. Las nuevas corridas reciben la etiqueta `estado = vigente`.

Los archivos de datos no se suben a GitHub. Para reproducirlos basta con `make pipeline`: descarga los CSV listados en `references/urls_2025.txt` y regenera cada versión. Las huellas permiten comprobar que el resultado es idéntico. El registro conserva las corridas fallidas o descartadas como parte del historial; reutilizar una corrida válida no las elimina.

## Linaje

```
v0_crudo ─┬─ v1_limpio_estricta ── v2_demanda_estricta ── v3_variables_estricta ── lineas_base_estricta
          └─ v1_limpio_laxa ────── v2_demanda_laxa ────── v3_variables_laxa ────── lineas_base_laxa
```

## Versiones

| Versión | Archivo | Filas | SHA-256 (inicio) | Qué cambió respecto a la anterior |
|---|---|---|---|---|
| v0_crudo | 12 CSV en `data/raw/` | 4,532,032 | 3bbd269b92e5 | Unión de enero a diciembre de 2025 sin tocar valores; 3 meses (enero, febrero, julio) venían en Latin-1 y 9 en UTF-8. La huella combina las de los 12 archivos. |
| v1_limpio_estricta | `data/interim/viajes_limpios_estricta.parquet` | 4,399,833 | 275225e44f84 | Tipos de dato, género nulo como "Desconocido", años de nacimiento fuera de 1930 a 2010 como nulos, y se eliminan 132,199 viajes (2.92 %): duración cero, arranques fallidos, menos de 1 minuto y más de 180 minutos. |
| v1_limpio_laxa | `data/interim/viajes_limpios_laxa.parquet` | 4,529,246 | 0fe5b58608e0 | Mismas conversiones, pero solo se eliminan 2,786 viajes imposibles (duración cero o más de 24 horas). Sirve para medir el efecto de la limpieza. |
| v2_demanda_* | `data/processed/demanda_estacion_hora_*.parquet` | 3,249,960 | 9657866cd640 / 8b3b5008d621 | Salidas y llegadas por estación y hora para 371 estaciones y las 8,760 horas de 2025; las horas sin viajes se rellenan con 0. |
| v3_variables_* | `data/processed/variables_*.parquet` | 3,187,632 | bd2933f487bd / e4b6ab993647 | Calendario, rezagos de 24 y 168 horas, promedio de 7 días, atributos de estación y perfil horario resumido con PCA. Se pierde la primera semana de enero por los rezagos. Partición temporal: 2,644,488 filas de entrenamiento y 543,144 de validación. El preprocesador ajustado solo con entrenamiento se guarda como artefacto. |

## Cambios descartados

| Corrida | Motivo |
|---|---|
| v3_variables_estricta (184864af) y v3_variables_laxa (1d0de33d) | El PCA se ajustaba sobre perfiles estandarizados. Como el perfil ya está en proporciones, estandarizar daba el mismo peso a las horas de madrugada, casi sin viajes, y los dos componentes explicaban solo 29 % de la varianza. Sin estandarizar explican 44 %. Las corridas quedan con la etiqueta `estado = descartada`. |
| lineas_base_estricta (f71f6203) y lineas_base_laxa (7628299d) | Usaban las v3 descartadas. |

## Hallazgos que no cambiaron los datos, pero quedan anotados

- El año de nacimiento 1970 aparece en 721 usuarios, contra unos 170 en los años vecinos. Parece un valor por omisión del registro. Se conserva porque las variables de persona no entran al modelo de demanda.
- La estación 194 registra 202 salidas en una hora la noche del sábado 20 de septiembre, con salidas y llegadas casi iguales. Parece un evento masivo y no un error de captura. Se conserva y se señala para la detección de anomalías.
- Cuatro estaciones del catálogo tienen capacidad 0. Ninguna aparece en los viajes de 2025.
- El catálogo de septiembre de 2026 tiene 484 estaciones; en 2025 solo se usaron 371.

## Experimento con las versiones

En `MiBici_Fase1_Experimentos_Datos` se calculan dos líneas base de salidas con cada variante: el promedio histórico por estación, día y hora, y la misma hora de la semana anterior. Las dos se evalúan contra la misma verdad (validación de la variante estricta, noviembre y diciembre).

| Métrica (salidas por estación y hora) | Datos laxos | Datos estrictos |
|---|---|---|
| MAE, promedio histórico | 0.837 | 0.828 |
| MAE en horas pico, promedio histórico | 1.442 | 1.420 |
| MAE, misma hora de la semana anterior | 0.974 | 0.954 |

La limpieza estricta mejora poco el error (alrededor de 1 %), lo esperable cuando se quita 2.9 % de los viajes. El promedio histórico supera a la semana anterior en 13 % y es la referencia que el modelo de la Fase 2 debe mejorar.
