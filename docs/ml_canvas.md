# ML Canvas: predicción de demanda en MiBici

Plantilla editable para la Fase 1 del proyecto (TC5061). Sigue la estructura de diez bloques del Machine Learning Canvas de Louis Dorard. Cada bloque trae una propuesta inicial y, al final, una línea "Por decidir" con lo que el equipo debe acordar antes de cerrar la versión. Quien edite un bloque lo anota en el historial de cambios del final y lo sube en su propio commit.

| Campo | Valor |
|---|---|
| Equipo | Equipo 17 |
| Versión del canvas | 0.1 (borrador) |
| Dataset | Datos abiertos MiBici, https://mibici.net/en/open-data/ |
| Periodo de construcción | enero a diciembre de 2025 |
| Periodo reservado | enero a septiembre de 2026 (prueba fuera de tiempo y monitoreo en fases posteriores) |

## Vista rápida

| Bloque | Resumen |
|---|---|
| 1. Propuesta de valor | Menos estaciones vacías o llenas en horas pico |
| 2. Tarea de ML | Regresión: viajes que salen y llegan por estación y hora |
| 3. Decisiones | Plan nocturno de redistribución para el día siguiente |
| 4. Fuentes de datos | CSV mensuales de viajes y catálogo de estaciones |
| 5. Recolección de datos | Publicación mensual del portal, registrada en MLflow |
| 6. Variables | Calendario, rezagos de demanda, atributos de estación |
| 7. Construcción de modelos | Un modelo global, reentrenamiento mensual |
| 8. Predicciones | Lote diario con horizonte de 24 h |
| 9. Evaluación offline | MAE y RMSE contra el promedio histórico, partición temporal |
| 10. Monitoreo en vivo | Error real contra error esperado, deriva de datos |

## 1. Propuesta de valor

Usuario final: el equipo de operación de MiBici que mueve bicicletas en camionetas entre estaciones. Beneficiario indirecto: la persona usuaria que encuentra una estación vacía cuando quiere salir o llena cuando quiere dejar la bicicleta.

Problema: la demanda no es simétrica. En la mañana las bicicletas salen de zonas habitacionales y se acumulan en zonas de oficinas, y en la tarde ocurre lo contrario. Si la redistribución se hace por reacción, las estaciones ya están vacías o llenas cuando llega la camioneta.

Propuesta: anticipar, por estación y por hora, cuántos viajes saldrán y cuántos llegarán en las próximas 24 horas, para que el plan de redistribución se arme antes de que ocurra el desbalance.

Por decidir: si el entregable final es solo la predicción o también una lista priorizada de estaciones en riesgo.

## 2. Tarea de aprendizaje automático

Tipo: regresión supervisada sobre series de tiempo, una serie por estación.

Entrada: estación, fecha y hora, más el historial reciente de esa estación.

Salida: dos valores por estación y hora, viajes que salen (Origen_Id) y viajes que llegan (Destino_Id). La diferencia entre ambos es el cambio neto de inventario, que es lo que le importa a la operación.

Alternativa más sencilla si el tiempo no alcanza: clasificar cada estación y hora en riesgo de vaciarse, riesgo de llenarse o normal.

Por decidir: granularidad (hora o franjas de 30 minutos) y si se modelan salidas y llegadas por separado o solo el neto.

## 3. Decisiones

Cada noche se genera el pronóstico del día siguiente y con él se ordenan las estaciones por desbalance esperado. El supervisor de operación asigna rutas de camioneta empezando por las de mayor desbalance en la franja de las 7 a las 10 y de las 18 a las 21.

La predicción apoya la decisión, no la reemplaza: el supervisor puede ajustar por eventos que el modelo no conoce (cierres viales, partidos, marchas).

## 4. Fuentes de datos

- Viajes: archivos `datos_abiertos_AAAA_MM.csv`, uno por mes, de diciembre de 2014 a septiembre de 2026. Columnas: `Viaje_Id`, `Usuario_Id`, `Genero`, `Año_de_nacimiento`, `Inicio_del_viaje`, `Fin_del_viaje`, `Origen_Id`, `Destino_Id`. Un mes ronda los 33 MB.
- Estaciones: `nomenclatura_2026_09.csv`, catálogo con identificador, nombre y ubicación de cada estación.
- Externas opcionales: calendario de días festivos y periodos vacacionales; clima diario de Guadalajara si el equipo decide incluirlo.

Por decidir: si se incorpora clima en esta fase o se deja para la Fase 2.

## 5. Recolección de datos

El portal publica un archivo por mes. Los archivos crudos se guardan en `data/raw/` sin modificar y no se suben a GitHub (están en `.gitignore`). Cada descarga se registra en MLflow con nombre de archivo, número de filas y hash SHA-256, de modo que cualquier integrante pueda comprobar que trabaja con los mismos datos.

Cuando se publique un mes nuevo, se descarga, se registra y entra al siguiente reentrenamiento.

## 6. Variables

Se construyen a partir de la tabla agregada por estación y hora:

- Calendario: hora del día, día de la semana, fin de semana, festivo, mes, periodo vacacional.
- Rezagos de la propia serie: demanda de la misma hora el día anterior y la semana anterior; promedio móvil de los últimos 7 días.
- Atributos de la estación: zona o colonia, latitud y longitud, número de estaciones cercanas.
- Perfil de uso: proporción histórica de viajes de ida frente a vuelta en la estación.

Las columnas de persona (`Usuario_Id`, `Genero`, `Año_de_nacimiento`) no entran como variables del modelo de demanda; se usan solo en el EDA de forma agregada.

## 7. Construcción de modelos

- Línea base: promedio histórico de la misma estación, día de la semana y hora. Todo modelo debe superarla para justificarse.
- Candidatos: regresión lineal regularizada y un modelo de árboles con boosting (por ejemplo, `HistGradientBoostingRegressor` de scikit-learn).
- Un solo modelo global para todas las estaciones, con la estación como variable, en lugar de un modelo por estación.
- Reentrenamiento mensual, cuando entra un archivo nuevo. Cada entrenamiento queda como corrida en MLflow con parámetros, métricas y la versión de datos usada.

## 8. Predicciones

Lote nocturno: a las 23:00 se generan las predicciones de las 24 horas siguientes para todas las estaciones. No se requiere predicción en tiempo real para el caso de uso. El resultado es una tabla estación por hora que alimenta el plan de rutas.

## 9. Evaluación offline

- Partición temporal, nunca aleatoria: entrenar con enero a octubre de 2025, validar con noviembre y diciembre de 2025. Los meses de 2026 quedan intactos.
- Métricas: MAE y RMSE por estación y hora; MAE ponderado en horas pico.
- Criterio de éxito: reducir el MAE de la línea base al menos un 15 % en horas pico (meta inicial, a revisar con los resultados).
- Revisión por segmentos: estaciones de mucho y poco uso, días laborales y fines de semana.

## 10. Monitoreo en vivo

- Comparar cada día el error real contra el error esperado de la validación.
- Vigilar deriva en la distribución de viajes por hora y por estación (estaciones nuevas, estaciones dadas de baja, cambios de patrón por vacaciones).
- Disparar reentrenamiento antes de tiempo si el MAE de una semana supera en 25 % al de validación.
- En la Fase 1 esto queda solo diseñado; se implementa en fases posteriores con los datos de 2026.

## Roles responsables

| Rol | Integrante | Bloques que mantiene |
|---|---|---|
| Responsable de negocio y del canvas | [nombre] | 1, 3 |
| Data engineer | [nombre] | 4, 5 |
| Data scientist | [nombre] | 2, 6, 9 |
| ML engineer | [nombre] | 7, 8, 10 |

## Historial de cambios

| Versión | Fecha | Integrante | Cambio |
|---|---|---|---|
| 0.1 | 2026-10-23 | [nombre] | Borrador inicial con propuesta de los diez bloques |
