## Equipo 17

| Integrante | Matrícula | Correo institucional |
|---|---|---|
| Francisco Gabriel Rodríguez González | A00702893 | A00702893@tec.mx |
| Joel Rodríguez Becerril | A01795945 | A01795945@tec.mx |
| José Israel Pérez Ontiveros | A01423294 | A01423294@tec.mx |
| Ricardo Portilla Corona | A01797352 | A01797352@tec.mx |

La asignación de roles y su interacción se describen en el [ML Canvas](docs/ml_canvas.md#roles-responsables).


# MiBici MLOps
## Análisis y predicción de demanda del sistema de bicicletas públicas MiBici

Proyecto de análisis y predicción de la demanda del sistema de bicicletas públicas MiBici, ubicado en el Área Metropolitana de Guadalajara, Jalisco, México.

El objetivo es desarrollar un flujo de trabajo reproducible que permita analizar los patrones históricos de uso del sistema, preparar y validar los datos, construir modelos de aprendizaje automático y aplicar prácticas de MLOps para dar seguimiento a los experimentos y resultados.

El proyecto se desarrolla como parte del curso de Machine Learning Operations.

---

## Objetivos del proyecto

- Analizar el comportamiento histórico de los viajes del sistema MiBici.
- Identificar patrones temporales y espaciales de demanda.
- Evaluar la calidad de los datos y documentar las decisiones de limpieza.
- Preparar conjuntos de datos reproducibles para el desarrollo de modelos.
- Construir modelos para estimar la demanda por estación y hora.
- Evaluar los resultados mediante métricas y comparaciones con modelos de referencia.
- Incorporar herramientas y prácticas de MLOps para el seguimiento de experimentos, versiones y resultados.

## Descripción del problema

La disponibilidad de bicicletas y espacios para estacionarlas cambia a lo largo del día y entre estaciones. Esto puede provocar que algunas estaciones se encuentren vacías o llenas durante periodos de alta demanda.

El proyecto busca anticipar estos cambios mediante la predicción de viajes por estación y hora, con el propósito de generar información que pueda apoyar la planeación y redistribución de bicicletas.

## Datos utilizados

El proyecto utiliza los archivos históricos de viajes publicados en el portal oficial de datos abiertos de MiBici.

Fuente: [MiBici - Datos Abiertos](https://www.mibici.net/es/datos-abiertos/)

El análisis histórico considera archivos mensuales desde diciembre de 2014 hasta septiembre de 2026.

Durante el procesamiento histórico se trabajó con:

- 142 archivos CSV mensuales.
- 39,078,226 viajes únicos después de retirar duplicados exactos.
- Información de usuarios, fechas, estaciones de origen y destino, y duración de los viajes.
- Un catálogo de estaciones con identificadores, nombres, coordenadas y capacidad.

### Selección del periodo para modelado

Aunque se realizó un análisis exploratorio del histórico completo, se seleccionó 2025 como periodo de trabajo para el desarrollo inicial de los modelos.

Esta selección permite utilizar un año calendario completo, conservar la variación entre días y meses y trabajar con un volumen de datos más manejable.

El conjunto de 2025 contiene 4,532,032 viajes originales.

El análisis histórico se utiliza como contexto para comprender la evolución del sistema y justificar las decisiones de preparación y selección de datos.

## Proceso de preparación y calidad de datos

La preparación de datos contempla distintas validaciones y transformaciones para conservar la trazabilidad de los registros.

Entre las actividades realizadas se encuentran:

- Identificación y eliminación de duplicados confirmados.
- Normalización de nombres y tipos de datos.
- Conversión y validación de fechas.
- Cálculo de la duración de los viajes.
- Identificación de valores faltantes.
- Detección de años de nacimiento sospechosos.
- Identificación de duraciones cero o negativas.
- Identificación de viajes con duración mayor a 24 horas.
- Identificación de viajes con el mismo origen y destino en menos de dos minutos.

Los registros sospechosos se conservan cuando es posible, utilizando columnas indicadoras o banderas para facilitar su análisis posterior.

La identificación de un registro atípico no implica automáticamente su eliminación. Las reglas de exclusión para el modelado se documentan por separado.

## Análisis exploratorio de datos (EDA)

Se realizó un análisis exploratorio del histórico para comprender el comportamiento del sistema MiBici.

El análisis contempla:

- Evolución anual y mensual del volumen de viajes.
- Comportamiento de la demanda por día de la semana y hora.
- Identificación de horarios de mayor actividad.
- Distribución de viajes por estación.
- Análisis de estaciones con mayor movimiento.
- Revisión de valores atípicos y calidad de datos.
- Evolución del número de estaciones con registros de viajes.

Para el periodo 2025 se profundizó en los patrones de demanda y en la distribución de viajes por estación y hora, como base para la construcción de modelos.

## Preparación para el modelado

Para el desarrollo inicial se trabaja con la demanda agregada por estación y hora.

La preparación contempla la generación de registros estación-hora, incluyendo combinaciones sin viajes registrados, con el propósito de representar también los periodos de demanda cero.

Para 2025 se cuenta con una estructura de referencia de:

- 371 estaciones con actividad registrada.
- 8,760 horas en el año.
- 3,249,960 combinaciones estación-hora.
- Variables temporales y características derivadas para el modelado.

El objetivo es estimar la cantidad de llegadas y salidas por estación y hora.

## Modelado y evaluación

Se utilizan modelos de referencia (baseline) para establecer un punto inicial de comparación.

La evaluación contempla métricas como:

- MAE (Mean Absolute Error).
- RMSE (Root Mean Squared Error).
- Error en periodos de mayor demanda.
- Comparación del desempeño en diferentes periodos de validación.

También se comparan los resultados obtenidos con diferentes estrategias de limpieza para evaluar cómo las decisiones de preparación de datos afectan el desempeño de los modelos.

Los resultados y experimentos se documentan para facilitar su reproducción y comparación.

## MLOps y seguimiento de experimentos

El proyecto incorpora MLflow como herramienta de seguimiento de experimentos.

Su uso permite registrar:

- Parámetros de configuración.
- Métricas de evaluación.
- Resultados de los experimentos.
- Artefactos y resúmenes de procesamiento.
- Diferentes ejecuciones de los procesos de preparación y modelado.

También se utiliza Git y GitHub para el control de versiones del código, el trabajo mediante ramas y la revisión de cambios a través de Pull Requests.

## Estructura del proyecto

```text
mibici-mlops/
├── data/
│   ├── external/       # Datos de fuentes externas
│   ├── interim/        # Datos intermedios y diagnósticos
│   ├── processed/      # Datos procesados
│   └── raw/            # Datos originales
├── docs/               # Documentación del proyecto
├── models/             # Modelos y resultados relacionados
├── notebooks/          # Análisis y experimentación
├── references/         # Materiales de referencia
├── reports/
│   └── figures/        # Gráficas y figuras
├── mibici_mlops/       # Código fuente del proyecto
├── tests/              # Pruebas automatizadas
├── .gitignore
├── Makefile
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

Los archivos de datos, resultados intermedios y artefactos de experimentación pueden generarse localmente y no necesariamente forman parte del repositorio de GitHub.

## Entorno de desarrollo

El proyecto utiliza Python 3.11 y un entorno virtual para aislar sus dependencias.

### 1. Clonar el repositorio

```bash
git clone https://github.com/JIsraP/mibici-mlops.git
cd mibici-mlops
```

### 2. Crear el entorno virtual

```bash
python3 -m venv .venv
```

### 3. Activar el entorno virtual

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 4. Instalar las dependencias

```bash
python -m pip install -r requirements.txt
```

### 5. Ejecutar las pruebas

```bash
python -m pytest
```

### 6. Revisar el código con Ruff

```bash
ruff check .
```

## Descarga de datos

Los archivos originales de MiBici no se incluyen directamente en el repositorio debido a su tamaño.

Para obtenerlos, se puede utilizar el script de descarga disponible en el proyecto:

```bash
python scripts/download_data.py
```

El script consulta el portal oficial de datos abiertos de MiBici e identifica los archivos CSV disponibles para descargarlos en `data/raw/`.

Los datos procesados pueden compartirse mediante un almacenamiento externo cuando su tamaño no permita incluirlos en GitHub.

## Reproducibilidad y control de versiones

Para mantener la trazabilidad del proyecto se utilizan las siguientes prácticas:

- Git y GitHub para el control de versiones del código.
- Ramas independientes para el desarrollo de funcionalidades.
- Pull Requests para revisar e integrar cambios.
- MLflow para registrar experimentos, parámetros y métricas.
- Validaciones de datos mediante código y pruebas.
- Documentación de las reglas de limpieza y preparación.

## Estado del proyecto

El proyecto ha avanzado en las siguientes actividades:

- Organización del repositorio y entorno de desarrollo.
- Descarga y consolidación de información histórica.
- Diagnóstico y limpieza de datos.
- Análisis exploratorio histórico.
- Selección de 2025 para el desarrollo inicial de modelos.
- Preparación de datos agregados por estación y hora.
- Desarrollo y comparación de modelos de referencia.
- Seguimiento de experimentos mediante MLflow.

Las siguientes etapas contemplan continuar con la experimentación, mejorar el desempeño de los modelos y fortalecer la reproducibilidad del flujo de trabajo.

## Repositorio

[MiBici MLOps - GitHub](https://github.com/JIsraP/mibici-mlops)

## Licencia

Este proyecto se distribuye bajo la licencia indicada en el archivo `LICENSE` del repositorio.