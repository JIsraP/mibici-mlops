# ML Canvas: predicción de demanda en MiBici

Borrador del ML Canvas de la Fase 1 del proyecto (TC5044), organizado en los diez bloques de Louis Dorard. Documenta el problema, la propuesta de valor y el alcance técnico. Las propuestas se apoyan en la implementación actual; los puntos marcados como "Por decidir" requieren acuerdo del equipo.

| Campo | Valor |
|---|---|
| Equipo | Equipo 17 |
| Versión del canvas | 0.2 (borrador) |
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

La solución busca anticipar, por estación y por hora, cuántos viajes saldrán y cuántos llegarán en las próximas 24 horas, para que el plan de redistribución se arme antes de que ocurra el desbalance.

Alcance propuesto: entregar una tabla de demanda prevista por estación y hora. En la Fase 1 se entrega la preparación de datos y su evaluación con líneas base. La priorización de estaciones y las rutas de redistribución quedan fuera de esta fase. Sin datos de inventario disponible no se puede afirmar que una estación vaya a quedar vacía o llena.

Por decidir: si la salida del proyecto se limita al pronóstico o incluye también una lista priorizada de estaciones en una fase posterior.

## 2. Tarea de aprendizaje automático

Tipo: regresión supervisada sobre series de tiempo, una serie por estación.

Entrada: estación, fecha y hora, más el historial reciente de esa estación.

Salida: dos valores por estación y hora, viajes que salen (Origen_Id) y viajes que llegan (Destino_Id). La diferencia entre ambos es el cambio neto de inventario, que es lo que le importa a la operación.

La implementación actual utiliza una granularidad de una hora y dos variables objetivo: salidas y llegadas. Se propone conservar esta estructura y usar el neto como indicador derivado, sin agregar una clasificación de riesgo en esta fase.

Por decidir: confirmar la granularidad horaria y las dos variables objetivo, o ajustar el alcance a franjas de 30 minutos o a la predicción del neto.

## 3. Decisiones

Cada noche se genera el pronóstico del día siguiente y con él se ordenan las estaciones por desbalance esperado. El supervisor de operación asigna rutas de camioneta empezando por las de mayor desbalance en la franja de las 7 a las 10 y de las 18 a las 21.

La predicción apoya la decisión, no la reemplaza: el supervisor puede ajustar por eventos que el modelo no conoce (cierres viales, partidos, marchas).

## 4. Fuentes de datos

- Viajes: archivos `datos_abiertos_AAAA_MM.csv`, uno por mes, de diciembre de 2014 a septiembre de 2026. Columnas: `Viaje_Id`, `Usuario_Id`, `Genero`, `Año_de_nacimiento`, `Inicio_del_viaje`, `Fin_del_viaje`, `Origen_Id`, `Destino_Id`. Un mes ronda los 33 MB.
- Estaciones: `nomenclatura_2026_09.csv`, catálogo con identificador, nombre y ubicación de cada estación.
- Calendario: días festivos de 2025 incorporados al código.

La implementación actual utiliza los viajes de 2025, el catálogo disponible de estaciones y el calendario de festivos. Se propone evaluar la incorporación de clima en una fase posterior. El catálogo de 2026 no reconstruye el inventario histórico de 2025.

Por decidir: confirmar el periodo de trabajo y si el clima se incorpora en esta fase o se deja para una etapa posterior.

## 5. Recolección de datos

El portal publica un archivo por mes. Los archivos crudos se guardan en `data/raw/` sin modificar y no se suben a GitHub (están en `.gitignore`). Cada descarga se registra en MLflow con nombre de archivo, número de filas y hash SHA-256, de modo que cualquier integrante pueda comprobar que trabaja con los mismos datos.

Cuando se publique un mes nuevo, se descarga, se registra y entra al siguiente reentrenamiento.

## 6. Variables

Se construyen a partir de la tabla agregada por estación y hora:

- Calendario: hora del día, día de la semana, fin de semana, festivo y mes. El código agrega estas columnas; el preprocesador selecciona las que utiliza.
- Rezagos de la propia serie: demanda de la misma hora el día anterior y la semana anterior; promedio móvil de los últimos 7 días.
- Atributos de la estación: zona, latitud, longitud y capacidad registrada en el catálogo.
- Perfil de uso: proporciones de salidas y llegadas por hora, resumidas con dos componentes de PCA ajustados solo con entrenamiento.

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
- Métricas: MAE y RMSE sobre los conteos por estación y hora; MAE en horas pico de días laborales (7, 8, 9, 18, 19 y 20 horas).
- Criterio de éxito: reducir el MAE de la línea base al menos un 15 % en horas pico (meta inicial, a revisar con los resultados).
- Revisión por segmentos: estaciones de mucho y poco uso, días laborales y fines de semana.

## 10. Monitoreo en vivo

- Comparar cada día el error real contra el error esperado de la validación.
- Vigilar deriva en la distribución de viajes por hora y por estación (estaciones nuevas, estaciones dadas de baja, cambios de patrón por vacaciones).
- Disparar reentrenamiento antes de tiempo si el MAE de una semana supera en 25 % al de validación.
- En la Fase 1 esto queda solo diseñado; se implementa en fases posteriores con los datos de 2026.

## Roles responsables

Los siguientes roles describen las responsabilidades del trabajo. Por decidir: asignar los responsables de cada rol entre los integrantes del equipo.

| Rol | Actividades | Bloques que mantiene |
|---|---|---|
| Responsable de negocio y del canvas | Definir el problema, el alcance y las decisiones que apoya el proyecto | 1, 3 |
| Data engineer | Descargar, limpiar y versionar los datos | 4, 5 |
| Data scientist | Explorar los datos, construir variables y evaluar resultados | 2, 6, 9 |
| ML engineer | Organizar la ejecución, el entrenamiento y el monitoreo | 7, 8, 10 |
| Validación y trazabilidad de datos | Verificar el registro de versiones, los reintentos y la consistencia documental | Transversal |

Interacción entre roles: el responsable de negocio define la necesidad; Data Engineering prepara y versiona los datos; Data Science explora y transforma las variables; ML Engineering organiza la ejecución. La validación comprueba que las ejecuciones fallidas o descartadas puedan repetirse y devuelve los hallazgos al responsable del módulo.

## Historial de cambios

| Versión | Fecha | Cambio |
|---|---|---|
| 0.1 | 2026-10-23 | Definición inicial de los diez bloques |
| 0.2 | 2026-10-23 | Propuestas de alcance y del rol de validación; revisión de variables y métricas frente al código |

## Pendientes para la entrega

- Acordar los puntos marcados como "Por decidir" y asignar los responsables de los roles.

- Verificar la evidencia de participación y los commits de todos los integrantes.
- Ejecutar el pipeline completo y conservar la evidencia de MLflow con los datos seleccionados.
- Preparar la presentación ejecutiva en PDF con las cinco etapas, resultados, roles y conclusiones.
- Incorporar al PDF la declaración de uso de IA de todos los integrantes. Para esta aportación se utilizó ChatGPT/Codex (GPT-6) como apoyo en revisión de código, depuración del registro de versiones, elaboración de pruebas y documentación.
