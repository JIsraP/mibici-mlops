# Monitoreo de data drift (Entrega final)

Documento del rol de data scientist. El código está en `mibici_mlops/drift.py`, las constantes en `config.py` y las pruebas en `tests/test_drift.py`.

## Cómo se ejecuta

```bash
python -m mibici_mlops.drift            # evalúa todo y registra en MLflow (MiBici_Final_Drift)
python -m mibici_mlops.drift figures    # regenera las figuras desde reports/drift_*.csv
pytest -q tests/test_drift.py           # 23 pruebas del módulo
```

Requisitos: el modelo registrado `models:/MiBici_Demanda/<versión>` (por omisión la última), `data/processed/variables_oficial.parquet`, `data/processed/demanda_estacion_hora_oficial.parquet` y los CSV de 2026 en `data/raw/2026/` (portal de datos abiertos de MiBici, enero a octubre).

## Datos de monitoreo

- Línea base: validación de noviembre y diciembre de 2025, con el modelo registrado.
- Drift real: enero a octubre de 2026 (3,997,597 viajes crudos; 758 duplicados exactos y 3,225 de duración no positiva quedan fuera). Se procesan con el mismo contrato que 2025, los perfiles PCA y el catálogo del entrenamiento, y los rezagos de la primera semana toman diciembre de 2025.
- Drift sintético sobre la validación: crecimiento de 30 % en rezagos y demanda, falla del rezago de 24 h (llega en cero) y caída estacional de 40 % solo en la demanda real. Además, un barrido de factores de 0.5 a 1.5.

## Señales y umbrales

| Señal | Necesita etiquetas | Advertencia | Alerta |
|---|---|---|---|
| PSI por variable de entrada (sin calendario) | No | 0.10 | 0.25 |
| Cambio del MAE de salidas en hora pico frente a la validación | Sí | +10 % | +20 % |
| Sesgo del total pronosticado frente al observado | Sí | ±10 % | ±20 % |
| Salidas en estaciones sin perfil ni catálogo | No | | 5 % del mes |

El calendario (hora, día, fin de semana, festivo) se reporta pero no dispara alertas: un mes sin festivos no es drift. El sesgo se agregó porque el MAE absoluto baja cuando la demanda cae y no detectaba la caída estacional.

## Regla de acción

| Desempeño | Entradas | Acción |
|---|---|---|
| Alerta | Alerta | Revisar el feature pipeline (calidad y frescura) y después reentrenar |
| Alerta | Estable o advertencia | Reentrenar con datos recientes |
| Estable o advertencia, con estaciones nuevas ≥ 5 % | | Actualizar el catálogo y reentrenar |
| Advertencia | | Vigilar en la siguiente ventana |
| Estable | Advertencia o alerta | Vigilar entradas (aún sin pérdida de desempeño) |

## Resultados

`reports/drift_resumen.csv` (una fila por ventana), `reports/drift_psi.csv`, `reports/drift_sensibilidad.csv`, `reports/drift_datos_2026.csv` y las figuras `reports/figures/drift_*.png`. En 2026, el modelo se mantiene estable de enero a mayo; junio y septiembre quedan en advertencia; de julio a septiembre las estaciones nuevas pasan de 5 % de las salidas, y octubre entra en alerta (+35 % de MAE en pico con 60 % más demanda por estación y hora que la validación).
