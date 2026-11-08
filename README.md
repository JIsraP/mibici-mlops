# MiBici MLOps

Proyecto de análisis, preparación y modelado de datos del sistema MiBici de Guadalajara, Jalisco, México, aplicando prácticas de MLOps para organizar el procesamiento, la limpieza, la validación y el seguimiento de experimentos.

El objetivo es construir una base de datos confiable y reproducible que permita analizar el comportamiento histórico del sistema y desarrollar modelos de aprendizaje automático para estudiar y predecir la demanda de viajes.

El proyecto contempla tanto el análisis histórico como una primera etapa de modelado enfocada en los datos de enero a diciembre de 2025.

Documentación relacionada:

- Requerimientos y propuesta de valor: [docs/ml_canvas.md](docs/ml_canvas.md)
- Bitácora de versiones de datos: [docs/versionado_datos.md](docs/versionado_datos.md)

---

## Equipo 17

| Integrante | Matrícula | Correo institucional |
|---|---|---|
| Francisco Gabriel Rodríguez González | A00702893 | A00702893@tec.mx |
| Joel Rodríguez Becerril | A01795945 | A01795945@tec.mx |
| José Israel Pérez Ontiveros | A01423294 | A01423294@tec.mx |
| Ricardo Portilla Corona | A01797352 | A01797352@tec.mx |

La asignación de roles y su interacción se describen en el [ML Canvas](docs/ml_canvas.md#roles-responsables).

---

## Objetivos del proyecto

- Organizar los archivos históricos de viajes de MiBici.
- Automatizar el procesamiento de los archivos CSV mensuales.
- Detectar y documentar problemas de calidad en los datos.
- Aplicar reglas de limpieza sin eliminar información potencialmente útil de manera innecesaria.
- Validar la consistencia de los datos procesados.
- Realizar un análisis exploratorio de los viajes y las estaciones.
- Utilizar MLflow para registrar parámetros, métricas y artefactos de los experimentos.
- Mantener un flujo de trabajo reproducible y versionar el código mediante Git.
- Construir la demanda por estación y hora de 2025.
- Generar variables predictoras y preparar los datos para el entrenamiento de modelos.
- Comparar distintas versiones de datos mediante experimentos y líneas base.

## Fuente de datos

Los datos se obtienen del portal oficial de datos abiertos de MiBici:

https://www.mibici.net/es/datos-abiertos/

El conjunto de datos está organizado en archivos CSV mensuales con información de los viajes registrados en el sistema.

**Periodo histórico analizado:** diciembre de 2014 a septiembre de 2026.

**Archivos mensuales procesados:** 142.

**Registros procesados después de eliminar duplicados exactos:** 39,078,226 viajes.

Las columnas utilizadas incluyen identificadores de viaje y usuario, género, año de nacimiento, fechas de inicio y finalización, estaciones de origen y destino, duración del viaje e indicadores de calidad.

**Periodo para modelar en la Fase 1:** enero a diciembre de 2025.

Los datos de enero a septiembre de 2026 quedan reservados para probar el modelo fuera de tiempo en fases posteriores. El análisis del histórico completo sirve como contexto y justifica esta elección.

## Tecnologías utilizadas

- **Python:** procesamiento, limpieza y análisis de datos.
- **pandas:** manipulación de datos y procesamiento por bloques.
- **NumPy:** operaciones numéricas y construcción de variables.
- **Scikit-learn:** PCA, escalamiento y codificación de variables.
- **Matplotlib y Seaborn:** visualización de resultados.
- **Jupyter Notebook:** documentación del análisis exploratorio.
- **MLflow:** seguimiento de experimentos, parámetros, métricas y artefactos.
- **PyArrow:** almacenamiento de datos en formato Parquet.
- **Joblib:** almacenamiento y recuperación de objetos de preprocesamiento.
- **pytest:** ejecución de pruebas automatizadas.
- **Ruff:** revisión estática y calidad del código.
- **Git y GitHub:** control de versiones y colaboración.

---

## Estructura del proyecto

El repositorio sigue una estructura basada en Cookiecutter Data Science, organizada para separar los datos, el código, los experimentos y la documentación.

```text
mibici_mlops/
├── data/
│   ├── raw/                 # Datos originales
│   ├── interim/             # Datos intermedios, viajes preparados
│   ├── processed/           # Datos procesados, demanda y variables
│   └── external/            # Datos externos
├── docs/                    # ML Canvas y bitácora de versiones
├── mibici_mlops/            # Código del paquete del proyecto
│   ├── config.py            # Rutas, fechas, festivos y configuración
│   ├── dataset.py           # Lectura de datos y catálogo de estaciones
│   ├── cleaning.py          # Reglas y procesos de limpieza
│   ├── model_input.py       # Entrada oficial de datos para modelar
│   ├── tracking.py          # Registro de versiones en MLflow
│   ├── features.py          # Demanda, variables y preprocesamiento
│   ├── validation.py        # Validación de demanda
│   ├── plots.py             # Figuras del EDA
│   ├── flagged_trips.py     # Análisis de viajes marcados
│   └── modeling/
│       └── train.py         # Líneas base y experimentos
├── models/                  # Modelos generados
├── notebooks/               # Notebooks de análisis
├── references/              # Material de referencia y URLs
├── reports/                 # Reportes y resultados
│   └── figures/             # Figuras del EDA
├── scripts/                 # Descarga, diagnóstico y procesamiento
├── tests/                   # Pruebas automatizadas
├── .gitignore
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

Los directorios `data/` contienen los datos locales y están excluidos del control de versiones para evitar subir archivos de gran tamaño a GitHub.

---

## Flujo de trabajo

El procesamiento de datos sigue estas etapas generales:

1. **Obtención de datos:** reunir los archivos mensuales descargados de la fuente oficial.
2. **Inspección inicial:** revisar estructura, columnas, tipos de datos y cobertura temporal.
3. **Diagnóstico de calidad:** identificar valores faltantes, registros duplicados, duraciones inconsistentes y otros problemas.
4. **Limpieza:** aplicar reglas definidas para tratar duplicados y valores sospechosos.
5. **Procesamiento:** guardar los archivos resultantes en los directorios correspondientes, trabajando por bloques cuando es necesario para controlar el uso de memoria.
6. **Validación:** comprobar la integridad de identificadores, fechas, duraciones e indicadores de calidad.
7. **Análisis exploratorio:** estudiar la evolución de los viajes, las estaciones y los problemas de calidad.
8. **Construcción de demanda:** agregar salidas y llegadas por estación y hora.
9. **Ingeniería de características:** generar variables de calendario, rezagos, atributos de estaciones y perfiles horarios.
10. **Seguimiento de experimentos:** registrar las versiones, resultados y artefactos en MLflow.

## Pipeline de la Fase 1 para modelar 2025

La primera etapa de modelado utiliza los archivos oficiales de 2025 y genera tres versiones principales de datos.

| Paso | Módulo | Versión en MLflow | Resultado |
|---|---|---|---|
| Entrada oficial de viajes | `model_input.py` | `v1_modelo_oficial` | Viajes preparados para modelar |
| Agregación de demanda | `features.py` | `v2_demanda_oficial` | Demanda por estación y hora |
| Construcción de variables | `features.py` | `v3_variables_oficial` | Variables predictoras y preprocesador |
| Validación de demanda | `validation.py` | `MiBici_Fase1_Validacion_Datos` | Reporte de nueve pruebas |
| Líneas base | `modeling/train.py` | `MiBici_Fase1_Experimentos_Datos` | Comparación de modelos |

### Ejecución de la Fase 1

Con el entorno virtual activado, los datos oficiales disponibles y el servidor de MLflow en funcionamiento, ejecutar desde la raíz del repositorio:

```bash
python -m mibici_mlops.model_input
```

```bash
python -m mibici_mlops.features --variant oficial
```

```bash
python -m mibici_mlops.validation --variant oficial
```

Para ejecutar posteriormente las líneas base:

```bash
python -m mibici_mlops.modeling.train
```

Los comandos de Makefile también están disponibles para automatizar las etapas correspondientes, según la configuración del proyecto.

---

## Limpieza y preparación de los datos

El procesamiento contempla los siguientes aspectos:

- Eliminación de duplicados exactos.
- Identificación de valores faltantes en género y año de nacimiento.
- Detección de años de nacimiento sospechosos.
- Conversión de valores de nacimiento inválidos a valores faltantes.
- Cálculo y revisión de la duración de los viajes.
- Identificación de duraciones iguales o menores que cero.
- Identificación de viajes con duración superior a 24 horas.
- Identificación de viajes fallidos.
- Conservación de indicadores que permiten distinguir los registros problemáticos.

En lugar de eliminar automáticamente todos los viajes con inconsistencias, se conservan indicadores para facilitar su análisis posterior. De esta forma, los registros se pueden estudiar sin perder información de manera innecesaria.

### Versión oficial de 2025 para modelar

La versión limpia oficial conserva los viajes de 2025 y agrega los siguientes campos:

- `duracion_segundos`
- `duracion_no_positiva`
- `duracion_mayor_24h`
- `viaje_fallido`

Se considera viaje fallido aquel que tiene la misma estación de origen y destino, con una duración mayor que cero y menor que 120 segundos.

Para modelar solo se excluyen los viajes con `duracion_no_positiva`. Los viajes con duración superior a 24 horas y los fallidos se conservan marcados para permitir comparaciones posteriores.

Las variantes estricta y laxa se mantienen en MLflow para comparar cuánto cambia el resultado con cada estrategia de limpieza.

### Resultados del procesamiento histórico

| Indicador | Resultado |
|---|---:|
| Archivos mensuales procesados | 142 |
| Registros antes de eliminar duplicados | 39,078,984 |
| Duplicados exactos eliminados | 758 |
| Registros procesados finales | 39,078,226 |
| Identificadores de viaje faltantes | 0 |
| Registros sin género | 144,508 |
| Registros sin año de nacimiento válido | 84,340 |
| Duraciones iguales o menores que cero | 55,038 |
| Duraciones mayores a 24 horas | 1,832 |

Los valores faltantes y las duraciones inconsistentes se documentan como características de calidad de los datos. Su presencia no implica necesariamente que todos los registros deban eliminarse.

---

## Construcción de la demanda y variables predictoras

Esta etapa utiliza la versión oficial de los viajes de 2025 para construir una tabla de demanda horaria y preparar las variables que se utilizarán en los modelos.

### v1. Preparación de viajes

El módulo `model_input.py` lee los archivos oficiales, normaliza sus columnas, conserva los indicadores de calidad y prepara los viajes para modelar.

Resultados obtenidos:

| Indicador | Resultado |
|---|---:|
| Viajes disponibles para modelar | 4,529,252 |
| Viajes marcados con duración no positiva | 2,780 |
| Viajes marcados con duración mayor a 24 horas | 6 |
| Viajes marcados como fallidos | 123,158 |
| Viajes excluidos para modelar | 2,780 |

Los viajes fallidos y los de duración superior a 24 horas se conservan marcados. Esto permite evaluar posteriormente su influencia en los resultados.

### v2. Demanda por estación y hora

El módulo `features.py` agrega los viajes por estación y hora, calculando:

- `salidas`: cantidad de viajes que comienzan en la estación durante esa hora.
- `llegadas`: cantidad de viajes que terminan en la estación durante esa hora.
- `neto`: diferencia entre llegadas y salidas.

También se genera una cuadrícula horaria completa para todas las estaciones, asignando cero a las horas sin viajes.

**Resultados obtenidos:**

| Indicador | Resultado |
|---|---:|
| Estaciones | 371 |
| Horas por estación | 8,760 |
| Registros de demanda | 3,249,960 |
| Salidas | 4,529,252 |
| Llegadas | 4,529,245 |
| Columnas | 5 |

La diferencia de siete registros entre salidas y llegadas se explica porque siete viajes comenzaron el 31 de diciembre de 2025 y terminaron después de la medianoche del 1 de enero de 2026.

Por consistencia con el periodo de análisis, las salidas se contabilizan cuando el viaje inicia en 2025 y las llegadas cuando el viaje termina en 2025.

El resultado se almacena en:

```text
data/processed/demanda_estacion_hora_oficial.parquet
```

### v3. Construcción de variables predictoras

A partir de la demanda horaria se generan variables para representar patrones temporales, comportamiento histórico y características de las estaciones.

Las variables incluyen:

**Calendario**
- Hora del día.
- Hora expresada mediante seno y coseno para representar su naturaleza cíclica.
- Día de la semana.
- Indicador de fin de semana.
- Indicador de día festivo.
- Mes.

**Rezagos y promedios históricos**
- Salidas y llegadas de hace 24 horas.
- Salidas y llegadas de hace 168 horas.
- Promedio de los siete valores históricos correspondientes a la misma hora de los siete días anteriores.

Los rezagos utilizan exclusivamente observaciones anteriores a la hora que se desea predecir, evitando utilizar información futura.

**Atributos de estaciones**
- Latitud.
- Longitud.
- Capacidad.
- Zona.

**Perfiles horarios**
- Dos componentes principales calculados mediante PCA a partir de los perfiles de salidas y llegadas de cada estación.
- El PCA se ajusta utilizando información del periodo de entrenamiento.

El preprocesamiento utiliza:

- `StandardScaler` para las variables numéricas.
- `OneHotEncoder` para las variables categóricas.
- `passthrough` para las variables binarias.

El preprocesador se ajusta exclusivamente con los datos de entrenamiento.

### Resultados de las variables predictoras

| Indicador | Resultado |
|---|---:|
| Registros generados | 3,187,632 |
| Columnas originales | 25 |
| Registros de entrenamiento | 2,644,488 |
| Registros de validación | 543,144 |
| Variables transformadas | 26 |
| Valores nulos | 0 |
| Valores infinitos | 0 |
| Duplicados estación-hora | 0 |

La separación temporal utiliza el corte configurado en `TRAIN_END = 2025-11-01`.

| Partición | Periodo |
|---|---|
| Entrenamiento | 8 de enero al 31 de octubre de 2025 |
| Validación | 1 de noviembre al 31 de diciembre de 2025 |

Los primeros siete días se utilizan para construir el historial requerido por los rezagos de 168 horas. Por esta razón, las primeras observaciones disponibles para el modelo comienzan el 8 de enero.

El archivo generado es:

```text
data/processed/variables_oficial.parquet
```

El preprocesador y el PCA se almacenan juntos en un archivo `preprocesador.joblib`.

---

## Validación de calidad de los datos

La validación de la tabla de demanda se realiza mediante `mibici_mlops/validation.py`.

El proceso compara los resultados agregados con los viajes que los originaron y comprueba la integridad temporal y estructural de la tabla.

Se realizan nueve pruebas:

| N.º | Prueba | Qué comprueba |
|---:|---|---|
| 1 | Horas completas por estación | Que cada estación tenga 8,760 horas. |
| 2 | Sin estación-hora repetida | Que no existan duplicados. |
| 3 | Horas dentro de 2025 | Que las fechas estén dentro del periodo. |
| 4 | Salidas cuadran con viajes | Que las salidas coincidan con los viajes iniciados en 2025. |
| 5 | Llegadas cuadran con viajes | Que las llegadas coincidan con los viajes terminados en 2025. |
| 6 | Salidas por mes cuadran | Que los totales mensuales coincidan. |
| 7 | Conteos no negativos | Que salidas y llegadas no tengan valores negativos. |
| 8 | Neto = llegadas - salidas | Que el balance esté correctamente calculado. |
| 9 | Estaciones en el catálogo | Que las estaciones utilizadas existan en el catálogo. |

**Resultados obtenidos en la validación oficial:**

| Indicador | Resultado |
|---|---:|
| Pruebas ejecutadas | 9 |
| Pruebas aprobadas | 9 |
| Pruebas fallidas | 0 |
| Estaciones revisadas | 371 |
| Horas esperadas por estación | 8,760 |
| Registros estación-hora duplicados | 0 |
| Registros fuera de rango | 0 |
| Estaciones sin correspondencia | 0 |

Las nueve pruebas se cumplieron.

El reporte se genera en:

```text
reports/validacion_demanda_oficial.csv
```

También se registra como artefacto de MLflow en el experimento:

```text
MiBici_Fase1_Validacion_Datos
```

Además de estas nueve pruebas, se realizaron comprobaciones sobre el archivo de variables predictoras, incluyendo duplicados, valores nulos, valores infinitos y distribución de las particiones.

---

## Análisis exploratorio de datos (EDA)

El análisis exploratorio se documenta principalmente en:

```text
notebooks/01_exploratory_data_analysis.ipynb
```

El notebook contiene revisiones y visualizaciones para comprender el comportamiento histórico del sistema y detectar patrones relevantes.

El análisis por estación y hora de 2025, la validación de la tabla de demanda y el análisis de los viajes marcados se documentan en:

- `notebooks/1.0-frg-exploracion-inicial.ipynb`
- `notebooks/2.0-frg-limpieza-eda-modulos.ipynb`
- `notebooks/3.0-frg-version-oficial-validacion-marcados.ipynb`

Las figuras se guardan en `reports/figures/`.

### Evolución de los viajes

El conjunto procesado contiene aproximadamente 39.08 millones de viajes.

Entre los resultados observados se encuentran:

- Crecimiento del número anual de viajes durante los primeros años del sistema.
- Un máximo de aproximadamente 4.64 millones de viajes en 2019.
- Una disminución aproximada del 38% en 2020 respecto a 2019.
- Recuperación posterior, con más de cuatro millones de viajes anuales entre 2022 y 2025.

La disminución de 2020 coincide temporalmente con el periodo de la pandemia de COVID-19, pero el análisis descriptivo no permite establecer una relación causal por sí solo.

Los años 2014 y 2026 tienen cobertura parcial y no deben compararse directamente con años completos sin considerar esta diferencia.

### Calidad de los datos a través del tiempo

Se compararon tanto los conteos como los porcentajes de registros problemáticos por año.

Los principales hallazgos incluyen:

- Aumento de los registros sin género en los años recientes.
- Disminución de los registros sin año de nacimiento válido después de aplicar las reglas de limpieza.
- Presencia de duraciones no positivas a partir de 2020, con los porcentajes más altos entre 2022 y 2024.
- Una cantidad reducida de viajes con duración superior a 24 horas.

Esta comparación permite identificar periodos que requieren mayor atención y evita depender únicamente de los conteos absolutos, que pueden variar según el volumen de viajes de cada año.

### Actividad de las estaciones

Se identificaron las estaciones con mayor cantidad acumulada de salidas y llegadas.

La estación López Cotilla / Penitenciaría, identificada con el ID 51, destacó por su actividad histórica, con aproximadamente 840 mil salidas y 1.08 millones de llegadas.

También se analizaron las diferencias entre los meses del calendario. Octubre presentó el promedio mensual más alto, con aproximadamente 306 mil viajes, mientras que diciembre presentó el más bajo, con aproximadamente 234 mil.

Estos resultados describen la actividad registrada, pero no permiten explicar por sí solos las causas de las diferencias observadas.

### Estaciones distintas con viajes registrados

Se contó el número de identificadores de estación distintos que aparecen en los registros de cada año.

El indicador muestra un crecimiento general durante el periodo analizado. Sin embargo, representa estaciones con al menos una salida o llegada registrada, no el número oficial de estaciones instaladas o disponibles.

Para reconstruir la evolución histórica de la red sería necesario disponer de catálogos de estaciones correspondientes a distintos periodos.

### Validación del catálogo de estaciones

En el análisis histórico de 2026 se compararon los identificadores presentes en los viajes de enero a septiembre con el catálogo disponible para septiembre de 2026.

| Indicador | Resultado |
|---|---:|
| Estaciones distintas en los viajes de 2026 | 471 |
| Estaciones en el catálogo | 484 |
| Identificadores presentes en ambos conjuntos | 469 |
| Identificadores en viajes, pero no en el catálogo | 2 |
| Identificadores del catálogo sin viajes en 2026 | 15 |

La coincidencia entre los identificadores de los viajes y el catálogo es aproximadamente del 99.6% respecto a las 471 estaciones identificadas en los viajes.

Los identificadores que no coinciden quedan documentados para su revisión. No se clasifican automáticamente como errores, ya que podrían existir diferencias entre versiones del catálogo o cambios históricos en las estaciones.

Para la versión oficial de 2025 utilizada en la Fase 1, se comprobó que las 371 estaciones presentes en los viajes tienen correspondencia con el catálogo de 484 estaciones.

---

## Seguimiento de experimentos con MLflow

MLflow se utiliza para registrar los resultados relevantes del procesamiento y facilitar su trazabilidad y reproducibilidad.

Los experimentos utilizados son:

- `limpieza_mibici`: seguimiento de las variantes de limpieza.
- `MiBici_Fase1_Versiones_Datos`: registro de las versiones de datos de la Fase 1.
- `MiBici_Fase1_Validacion_Datos`: registro de las validaciones.
- `MiBici_Fase1_Experimentos_Datos`: registro de líneas base y experimentos de modelado.

### Versiones registradas en la Fase 1

| Versión | Descripción |
|---|---|
| `v1_modelo_oficial` | Preparación de los viajes oficiales de 2025. |
| `v2_demanda_oficial` | Agregación de salidas y llegadas por estación y hora. |
| `v3_variables_oficial` | Variables predictoras, preprocesador y PCA. |

Cada registro puede incluir:

- Parámetros y reglas de procesamiento.
- Número de archivos y registros procesados.
- Métricas de calidad de los datos.
- Huellas SHA-256.
- Esquemas de los conjuntos de datos.
- Archivos Parquet y otros artefactos relevantes.

En la ejecución de v3 se almacenaron:

```text
datos/
├── esquema.csv
└── preprocesador.joblib
```

El archivo `preprocesador.joblib` contiene:

- El `ColumnTransformer` ajustado con los datos de entrenamiento.
- El objeto PCA con dos componentes principales.

Se comprobó que ambos objetos pueden descargarse desde MLflow y recuperarse mediante Joblib.

El código que genera los resultados se mantiene en Git, mientras que MLflow conserva la evidencia de las ejecuciones. Los artefactos deben corresponder a resultados realmente generados y no a valores estimados manualmente.

### Configuración local de MLflow

Para trabajar localmente se utiliza una base de datos SQLite y la interfaz de MLflow.

Desde la raíz del repositorio, activar el entorno virtual y ejecutar:

```bash
mlflow ui
```

La interfaz se encuentra disponible en:

```text
http://127.0.0.1:5000
```

La configuración del proyecto utiliza `MLFLOW_TRACKING_URI` para indicar dónde registrar y consultar los experimentos.

---

## Instalación y configuración

Se recomienda utilizar un entorno virtual de Python para instalar las dependencias sin afectar otros proyectos.

Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para verificar las dependencias:

```bash
python -m pip check
```

### Variables de entorno

El proyecto utiliza variables de entorno para configurar las rutas de datos y la conexión con MLflow.

Se recomienda crear un archivo `.env` local, que no debe subirse al repositorio.

Ejemplo de configuración utilizada para la Fase 1:

```env
MIBICI_OFFICIAL_DIR=data/processed
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

`MIBICI_OFFICIAL_DIR` indica dónde se encuentran los archivos mensuales oficiales de 2025.

`MLFLOW_TRACKING_URI` indica la dirección del servidor de seguimiento de experimentos.

La configuración de rutas se centraliza en `mibici_mlops/config.py`.

### Descarga de datos

- `python scripts/download_data.py` consulta la página oficial, identifica los CSV disponibles y los descarga en `data/raw/`.
- `make download` descarga los 12 meses de 2025 y el catálogo de estaciones utilizado por la Fase 1, a partir de `references/urls_2025.txt`.

Para abrir el proyecto en Jupyter, se puede utilizar el entorno configurado en VS Code y seleccionar el intérprete de `.venv`.

Si se desea ejecutar el notebook desde la terminal y Jupyter está instalado:

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

---

## Pruebas y calidad del código

El proyecto contempla pruebas automatizadas y revisiones estáticas del código.

Para ejecutar las pruebas:

```bash
pytest
```

Para revisar el código con Ruff:

```bash
ruff check .
```

Para comprobar la sintaxis de un módulo:

```bash
python -m compileall -q mibici_mlops/
```

Las pruebas y validaciones deben ejecutarse después de realizar cambios en los scripts para verificar que el procesamiento mantenga los resultados esperados.

Las pruebas de registro de versiones (`tests/test_tracking.py`) se ejecutan sin descargar MiBici ni iniciar un servidor de MLflow: usan una base SQLite y artefactos temporales.

---

## Control de versiones y manejo de datos

Git se utiliza para versionar el código fuente, los notebooks, las pruebas y la documentación.

Los archivos originales y procesados se mantienen localmente y están excluidos del repositorio mediante `.gitignore`. Lo mismo aplica al entorno virtual, los archivos temporales y el almacenamiento local de MLflow.

Esta separación permite mantener un repositorio ligero y evitar subir archivos de datos que pueden ocupar varios gigabytes.

### Forma de trabajo del equipo

- `main` solo recibe cambios por pull request revisado por otro integrante; nadie aprueba su propio trabajo.
- Cada integrante trabaja en su rama y modifica lo que corresponde a su rol.
- Las observaciones a otro rol se comunican a quien lo ejerce.
- La versión de cada dataset queda registrada en MLflow con su huella SHA-256.
- Los archivos `.env`, bases de datos locales y datos pesados no deben subirse a GitHub.

---

## Limitaciones actuales

- Los años 2014 y 2026 tienen cobertura temporal parcial.
- El catálogo disponible de estaciones no permite reconstruir por sí solo la red histórica.
- Los identificadores que no coinciden entre viajes y catálogo requieren revisión.
- Los valores faltantes y las duraciones inconsistentes necesitan interpretarse según su contexto.
- Los patrones observados en el EDA son descriptivos y no demuestran causalidad.
- Los viajes que terminan fuera del periodo de análisis se contabilizan de acuerdo con su fecha de llegada, por lo que los totales de salidas y llegadas pueden presentar pequeñas diferencias.
- La validación integral actual de nueve pruebas se concentra en la tabla de demanda. La validación de variables predictoras se ha realizado mediante comprobaciones adicionales, y puede ampliarse con pruebas automatizadas específicas.
- Los resultados de la Fase 1 corresponden al periodo de 2025 y no demuestran por sí solos el desempeño del modelo en periodos posteriores.
- La configuración de MLflow descrita corresponde al entorno local de desarrollo; para colaboración o despliegue compartido se requiere configurar un servidor y almacenamiento accesibles para el equipo.

---

## Próximas etapas

Las siguientes actividades previstas para continuar el proyecto son:

1. Completar las validaciones automatizadas de las variables predictoras.
2. Ejecutar y comparar las líneas base de modelado.
3. Evaluar el efecto de las distintas versiones y reglas de limpieza sobre el desempeño.
4. Analizar los resultados de entrenamiento y validación temporal.
5. Preparar pruebas fuera de tiempo utilizando datos posteriores a 2025.
6. Completar el reporte final y la presentación ejecutiva.
7. Documentar los resultados y las decisiones del equipo.

---

## Licencia y uso de los datos

Los datos utilizados provienen del portal de datos abiertos de MiBici. Para su reutilización, consulta las condiciones y los términos publicados por la fuente oficial.

El código de este repositorio se distribuye bajo la licencia indicada en el archivo `LICENSE`.

---

**Proyecto:** MiBici MLOps

**Repositorio:** https://github.com/JIsraP/mibici-mlops

**Fuente de datos:** https://www.mibici.net/es/datos-abiertos/