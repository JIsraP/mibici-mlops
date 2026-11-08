# Evaluación de modelos (Fase 2)

Documento del rol de data scientist. Define cómo se miden los modelos, cómo se elige el mejor y qué debe registrar cada corrida para que todas sean comparables. El código está en `mibici_mlops/modeling/evaluate.py` (métricas), `baselines.py` (líneas base) y `compare.py` (comparación desde MLflow).

## Qué se predice y contra qué se compara

- Objetivos: salidas y llegadas por estación y hora (conteos), con la tabla `variables_oficial.parquet`.
- Partición temporal: entrenamiento de enero a octubre de 2025 y validación de noviembre y diciembre (`config.TRAIN_END`). Nunca aleatoria, porque el modelo se usará para pronosticar el día siguiente.
- Referencia: la mejor línea base registrada en `MiBici_Fase2_Modelos`, el promedio histórico por estación, día de la semana y hora.

| Línea base | MAE salidas | MAE salidas en pico | RMSE salidas | Desviación de Poisson |
|---|---|---|---|---|
| Promedio histórico | 0.856 | 1.469 | 1.546 | 1.089 |
| Misma hora de la semana anterior | 0.990 | 1.608 | 1.909 | 5.871 |

## Métricas

| Métrica | Nombre en MLflow | Por qué |
|---|---|---|
| MAE | `{objetivo}_mae` | Error medio en viajes por estación y hora; es la unidad que entiende la operación. |
| MAE en horas pico | `{objetivo}_mae_pico` | Solo 7 a 9 y 18 a 20 h de días laborales (`config.PEAK_HOURS`), donde se decide la redistribución. |
| RMSE | `{objetivo}_rmse` | Penaliza más los errores grandes, que se concentran en las horas pico. |
| Desviación de Poisson | `{objetivo}_poisson` | Mide el ajuste de un modelo de conteos; 53 % de las horas no tienen salidas. |

Las predicciones negativas se recortan a cero antes de medir, porque no existen conteos negativos.

## Criterio para elegir el modelo

1. Métrica principal: `salidas_mae_pico` (`config.SELECTION_METRIC`). En empate decide `salidas_mae`.
2. Meta inicial del ML Canvas: reducirla al menos 15 % frente a la mejor línea base (`config.TARGET_IMPROVEMENT_PCT`). La tabla comparativa marca `cumple_meta`.
3. Revisión de llegadas: el modelo elegido no debe empeorar `llegadas_mae_pico` frente a la línea base.
4. Revisión por segmentos: el error se desglosa por hora y por estaciones de uso bajo, medio y alto. Con la línea base, el MAE de salidas es 1.38 en las estaciones de uso alto contra 0.43 en las de uso bajo, y las horas con más error son las 18, 17 y 8 h. Un modelo útil debe mejorar ahí, no solo en el promedio.

## Algoritmos propuestos y por qué

- Regresión de Poisson regularizada (`PoissonRegressor`): modelo lineal para conteos, interpretable y rápido; sirve como primer modelo frente a las líneas base.
- `HistGradientBoostingRegressor` con `loss="poisson"`: capta interacciones entre hora, día, rezagos y perfil de estación, maneja más de 2.6 millones de filas en minutos y la pérdida de Poisson respeta que la variable sea un conteo.
- Un modelo global para todas las estaciones, con la estación descrita por sus variables (ubicación, capacidad, zona y perfil horario), como se propuso en el ML Canvas.

El ajuste de hiperparámetros debe hacerse solo con los meses de entrenamiento, con validación temporal (`TimeSeriesSplit` o separando octubre como validación interna), para no usar noviembre y diciembre al elegir hiperparámetros.

## Lo que debe registrar cada corrida de modelo

Para que `compare.py` incluya la corrida sin cambios:

- Experimento: `MiBici_Fase2_Modelos` (`config.EXPERIMENT_MODELS`).
- Etiqueta `tipo = modelo` y parámetro `modelo` con un nombre corto (`poisson`, `hgb_poisson`).
- Métricas de `evaluate.score_frame(validacion, {"salidas": pred_salidas, "llegadas": pred_llegadas})`, que ya devuelve los nombres anteriores.
- Recomendado: artefacto `predicciones_validacion.parquet` con `estacion_id`, `fecha_hora`, `pred_salidas` y `pred_llegadas`, para el desglose por hora y segmento.
- Semilla: `config.RANDOM_SEED`.

Después de registrar las corridas: `make compare` (o `python -m mibici_mlops.modeling.compare`) actualiza `reports/comparacion_modelos.csv`, `reports/diagnostico_error.csv` y las figuras 13 y 14; `notebooks/4.0-frg-evaluacion-modelos.ipynb` las muestra.

## Texto sugerido para el bloque 9 del ML Canvas

Evaluación offline: partición temporal (enero a octubre para entrenar y ajustar, noviembre y diciembre para validar). Métricas MAE, RMSE, MAE en horas pico laborales y desviación de Poisson por objetivo. Se elige el modelo con menor MAE de salidas en horas pico y se exige mejorar al menos 15 % a la mejor línea base (promedio histórico, MAE en pico de 1.469), sin empeorar las llegadas. El error se revisa por hora y por estaciones de uso bajo, medio y alto.
