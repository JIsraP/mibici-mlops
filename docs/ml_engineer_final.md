# Secciones del ML Engineer para la entrega final
Ricardo Portilla Corona | A01797352 | Equipo 17

Estos apartados corresponden al rol de ML Engineer y conservan los títulos de la plantilla del
equipo. Los resultados de modelado proceden de la comparación registrada en la Fase 2.
La verificación del servicio usa un modelo pequeño entrenado para las pruebas y no sustituye
la validación operativa del artefacto final.

## Entrega 1. Análisis del problema, exploración de datos y estructura del proyecto
### ML Engineer - Ingeniería de Machine Learning
#### Actividades realizadas
Se revisó la organización del repositorio basado en Cookiecutter y se preparó la ejecución de
los componentes de procesamiento y experimentación como módulos de Python. Se identificaron
las rutas de datos, configuraciones y artefactos necesarias para integrar el trabajo de los roles.
Se estableció el uso de MLflow para relacionar las ejecuciones con sus parámetros, métricas y
archivos, manteniendo el código separado de los datos y del almacenamiento local de seguimiento.

#### Resultados obtenidos
El proyecto contó con una estructura modular, configuración centralizada y comandos para
ejecutar sus etapas. La organización facilitó integrar los datos preparados, registrar
experimentos y recuperar los resultados sin depender de pasos manuales dentro de un notebook.

#### Análisis reflexivo de los resultados
La estructura del proyecto facilitó la colaboración, pero también mostró que compartir el código
no comparte automáticamente los datos ni los artefactos locales de MLflow. Su localización y
versión deben documentarse para que otro integrante pueda repetir una ejecución.

#### Justificación técnica de los métodos utilizados
Se utilizaron módulos y comandos reproducibles para reducir diferencias entre ejecuciones.
Git permitió revisar los cambios por rama y MLflow aportó trazabilidad de las ejecuciones y
artefactos. La configuración centralizada evitó repetir rutas y parámetros en varios archivos.

## Entrega 2. Adquisición, limpieza, exploración y preparación de datos
### ML Engineer - Ingeniería de Machine Learning
#### Actividades realizadas
Se integraron los conjuntos preparados por el equipo con el flujo de experimentación y se
documentaron sus comandos de ejecución. Se utilizaron la tabla de demanda y el Parquet de
variables como contratos de entrada para los modelos y la evaluación.
Las variables temporales, rezagos y perfiles de estaciones con PCA fueron diseñados por el
data scientist; el trabajo de ingeniería de ML consistió en consumir e integrar esos componentes.

#### Resultados obtenidos
El entrenamiento pudo utilizar las variables oficiales con una partición temporal común y
compararse contra las líneas base bajo el mismo esquema de evaluación. El uso de hashes
permitió asociar los resultados con una versión identificable de los datos.

#### Análisis reflexivo de los resultados
La integración hizo necesario acordar nombres de columnas, tipos de datos y periodos de
evaluación. Mantener estos contratos comunes evitó comparar modelos evaluados sobre
verdades o versiones diferentes de los datos.

#### Justificación técnica de los métodos utilizados
Se reutilizaron los componentes del equipo para conservar la misma definición de variables
y métricas. Se eligió Parquet por su compatibilidad con pandas y por permitir seleccionar
columnas y periodos del conjunto preparado.

## Entrega 3. Integración del pipeline MLOps, validación y preparación para modelado
### ML Engineer - Ingeniería de Machine Learning
#### Actividades realizadas
Se implementaron los módulos de entrenamiento y predicción. El entrenamiento combina un
preprocesador ajustado solo con entrenamiento y un MultiOutputRegressor para estimar
salidas y llegadas por estación y hora.
Se evaluaron una regresión de Poisson regularizada y un HistGradientBoostingRegressor con
pérdida de Poisson. Los hiperparámetros se eligieron con enero-septiembre para entrenar y
octubre para validar internamente. El candidato seleccionado se entrenó con enero-octubre y
se evaluó con noviembre-diciembre, manteniendo separada la validación final.
Se integraron el registro de parámetros, métricas, predicciones y modelo en MLflow, la
comparación contra las líneas base y el registro del candidato elegible como MiBici_Demanda.

#### Resultados obtenidos
El modelo hgb_poisson obtuvo MAE de salidas de 0.7998, RMSE de 1.4200 y MAE de salidas
en horas pico de 1.2433. La línea base de promedio histórico obtuvo 1.4687 en esta ultima
métrica. La mejora fue de 15.3 %, cumpliendo la meta del canvas. El MAE de llegadas en
pico fue de 1.1646 frente a 1.3863 de la línea base.
La regresión de Poisson obtuvo MAE de salidas en pico de 1.7698 y no fue seleccionada.
Estas cifras proceden de reports/comparacion_modelos.csv y corresponden a la Fase 2.

#### Análisis reflexivo de los resultados
La ganancia del modelo seleccionado cumplió el objetivo con un margen pequeño. Por ello,
conservar el modelo, su preprocesamiento, los datos de evaluación y las métricas asociadas
resulta necesario para detectar si esa mejora se mantiene con datos posteriores.
La automatización del registro y comparación redujo decisiones manuales al elegir el modelo.

#### Justificación técnica de los métodos utilizados
El modelo de arboles con pérdida de Poisson permite representar relaciones no lineales e
interacciones en una demanda de conteos. El preprocesamiento se guarda dentro del pipeline
para mantener la misma transformación durante entrenamiento e inferencia.
La selección utiliza el criterio acordado con el data scientist: mejorar al menos 15 % el MAE de
salidas en pico sin empeorar las llegadas en pico. El artefacto se guarda en formato skops.

## Entrega final. Pruebas y servicio portable del modelo
### ML Engineer - Ingeniería de Machine Learning
#### Actividades realizadas
Se implementó FastAPI con POST /predict y validación mediante Pydantic. La peticion recibe
registros con estación, fecha y las 18 variables preparadas del modelo. La respuesta devuelve
salidas y llegadas estimadas. Los esquemas se publican en OpenAPI y Swagger.
El servicio carga una sola vez el pipeline completo desde una carpeta MLflow local. Esto
permite incorporar el artefacto en la imagen y atender peticiones sin un servidor MLflow externo.
Se agregó un exportador que exige un URI con versión numérica y conserva el artefacto completo,
sus dependencias, los hashes y cinco ejemplos de entrada con predicciones de referencia.
Se preparó un Dockerfile con Python 3.13.9, dependencias fijas, libgomp1 y Uvicorn. Se
documentaron los comandos de construcción, ejecución y publicación con el tag 1.0.0.
Se implementó un verificador HTTP que compara las salidas del servicio con la referencia
exportada, usando rtol de 1e-6 y atol de 1e-8 y conservando un resultado JSON.

#### Resultados obtenidos
Se agregaron 14 pruebas para entrenamiento y predicción, persistencia y carga en skops,
equivalencia entre API y predicción directa, rechazo de entradas inválidas, esquema Swagger,
exportación versionada y deteccion de diferencias frente a la referencia.
La suite de la copia de trabajo ejecuto 44 pruebas aprobadas y cero fallidas, incluyendo las
30 pruebas existentes del commit e853329. El módulo de drift del data scientist no forma parte
de esta copia, por lo que no se incluye en ese conteo.
Las pruebas del servicio utilizaron un modelo pequeño real entrenado con datos sintéticos.
La comprobación se realizo con Python 3.12.13, scikit-learn 1.8.0 y MLflow 3.17.0. El entorno
difiere de algunas versiones fijadas en main y requiere validación final con el entorno del equipo.

#### Análisis reflexivo de los resultados
Las pruebas verificaron que guardar y volver a cargar el pipeline conserva sus predicciones,
y que exponerlo por HTTP no cambia los resultados de la llamada directa.
La validación de entrada detecta registros incompletos, valores fuera de rango y columnas
adicionales antes de invocar al modelo. El ejemplo exportado relaciona la comparación
operativa con un artefacto y una versión concretos.
La ejecución del contenedor y la publicación de la imagen son pasos distintos a la validación
local del código y deben confirmarse con el modelo final del equipo.

#### Justificación técnica de los métodos utilizados
FastAPI y Pydantic permiten declarar un contrato de inferencia y generar la documentación.
Reutilizar predict_frame evita mantener una segunda implementacion de la predicción.
Docker empaqueta el código, el modelo y las dependencias; la exportación versionada permite
identificar el artefacto y el verificador ofrece un criterio cuantitativo de equivalencia.
Las pruebas usan conjuntos pequeños para ejecutarse rapidamente sin descargar los datos
históricos ni entrenar nuevamente el modelo completo.

## Insumos para las conclusiones generales
El registro versionado y la persistencia del preprocesamiento permiten relacionar una
predicción con el modelo que la genero. Las pruebas muestran la utilidad de verificar por
separado el cálculo del modelo, la serialización y el contrato HTTP.
Como mejora, conviene alinear las dependencias del entrenamiento y servicio con las que
declare el artefacto final y automatizar la comprobación de la imagen publicada.
La generación de variables de 2026 y el tratamiento de estaciones nuevas corresponden a
mejoras del feature pipeline que deben coordinarse con los roles de datos y ciencia de datos.

## Evidencia y cierre operativo
Verificado: 44 pruebas locales aprobadas; entrenamiento pequeño, serialización skops,
contrato API y comparación de predicciones.
Preparado: API, exportador, verificador HTTP, Dockerfile, dependencias y documentación.
Pendiente de comprobación: carga del artefacto final del equipo, construcción y ejecución de
Docker, comparación local-contenedor y publicación en la cuenta DockerHub del equipo.
La URI definitiva del modelo, la liga DockerHub y la salida del verificador se incorporarán
después de confirmar esos pasos. No se consideran resultados ya obtenidos.
