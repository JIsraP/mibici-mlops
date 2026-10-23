# MiBici MLOps

Proyecto de análisis y predicción de la demanda del sistema de bicicletas públicas MiBici. El objetivo es construir un flujo de trabajo reproducible para preparar los datos, explorar sus características y, posteriormente, desarrollar y evaluar modelos de aprendizaje automático, utilizando herramientas de MLOps para dar seguimiento a los experimentos.

## Objetivos del proyecto

- Preparar un entorno de desarrollo reproducible.
- Obtener y organizar los datos abiertos de MiBici.
- Identificar y documentar problemas de calidad en los datos.
- Limpiar y preparar los registros para el análisis.
- Realizar análisis exploratorio para identificar patrones de uso y demanda.
- Desarrollar y evaluar modelos de aprendizaje automático.
- Registrar experimentos, parámetros y métricas mediante MLflow.
- Mantener el código y los procesos de trabajo versionados en GitHub.

## Estado actual del proyecto

Actualmente, el proyecto se encuentra en la etapa de preparación y limpieza de datos.

Se descargaron los archivos mensuales disponibles en la página oficial de MiBici, correspondientes al periodo de diciembre de 2014 a septiembre de 2026. En total, se identificaron 142 archivos mensuales de viajes y un archivo con información de las estaciones.

El diagnóstico inicial identificó valores faltantes, registros duplicados, años de nacimiento sospechosos y viajes con duraciones anómalas. También se detectaron diferencias en los formatos de fecha y en algunos nombres de columnas entre archivos.

Se implementó un proceso de limpieza por lotes para trabajar con el volumen de información sin tener que cargar todos los registros en memoria al mismo tiempo. Además, se integró MLflow para registrar los parámetros, las métricas y el resumen de las ejecuciones.

El siguiente paso es validar de manera independiente los datos procesados y realizar el análisis exploratorio antes de definir y entrenar un modelo.

## Fuente de los datos

Los datos se obtienen de la página oficial de [MiBici - Datos Abiertos](https://www.mibici.net/es/datos-abiertos/).

Los archivos de viajes contienen información como:

- Identificador del viaje y del usuario.
- Género y año de nacimiento del usuario, cuando están disponibles.
- Fecha y hora de inicio y fin del viaje.
- Identificadores de las estaciones de origen y destino.

También se utiliza un archivo de referencia con información de las estaciones, como su nombre, ubicación geográfica y capacidad.

Los datos originales y procesados no se incluyen en GitHub debido a su tamaño. Se almacenan localmente dentro de la carpeta `data/`.

## Estructura del proyecto

```text
mibici-mlops/
├── data/
│   ├── external/       # Datos externos
│   ├── interim/        # Inventarios, diagnósticos y resultados intermedios
│   ├── processed/      # Datos procesados para análisis posteriores
│   └── raw/            # Archivos originales descargados
├── docs/               # Documentación del proyecto
├── models/             # Modelos entrenados
├── notebooks/          # Análisis y experimentación
├── references/         # Material de referencia
├── reports/
│   └── figures/        # Gráficas y figuras para reportes
├── scripts/            # Scripts de descarga, diagnóstico y limpieza
├── mibici_mlops/       # Código fuente del proyecto
├── tests/              # Pruebas automatizadas
├── .gitignore
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.11.
- Git.
- Conexión a internet para descargar los archivos originales.
- Dependencias de Python definidas en `requirements.txt`.

Los comandos siguientes están pensados para ejecutarse desde la carpeta raíz del repositorio.

## Configuración del entorno

### 1. Clonar el repositorio

```bash
git clone https://github.com/JIsraP/mibici-mlops.git
cd mibici-mlops
```

### 2. Crear y activar el entorno virtual

En macOS o Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar las dependencias

```bash
python -m pip install -r requirements.txt
```

## Flujo de trabajo de datos

### 1. Descargar los datos

```bash
python scripts/download_data.py
```

El script consulta la página oficial de MiBici y descarga los archivos CSV disponibles en `data/raw/`. Los archivos que ya existen y contienen información se omiten para evitar descargarlos nuevamente.

### 2. Generar el inventario y validar la cobertura

```bash
python scripts/inspect_data.py
python scripts/validate_inventory.py
```

Estos scripts permiten revisar los archivos disponibles, sus tamaños, el número de registros y la continuidad de los meses incluidos en el conjunto de datos.

### 3. Diagnosticar la calidad de los datos

```bash
python scripts/diagnose_quality.py
python scripts/check_duplicates.py
python scripts/inspect_duplicates.py
python scripts/analyze_duplicate_locations.py
python scripts/inspect_suspicious_rows.py
```

Estos procesos ayudan a identificar valores faltantes, años de nacimiento sospechosos, duraciones anómalas y viajes duplicados. También permiten revisar ejemplos de los registros que requieren atención.

Los resultados de los diagnósticos se guardan en `data/interim/`.

### 4. Ejecutar la limpieza

```bash
python scripts/clean_data.py
```

El proceso de limpieza trabaja por lotes y genera los archivos procesados en `data/processed/`. También registra un resumen de la ejecución en `data/interim/resumen_limpieza.csv`.

Entre las acciones implementadas se encuentran:

- Normalización de nombres de columnas.
- Tratamiento de cadenas vacías como valores faltantes.
- Eliminación de registros duplicados previamente identificados.
- Conversión de años de nacimiento sospechosos a valores faltantes.
- Interpretación de los formatos de fecha detectados.
- Cálculo de la duración de cada viaje.
- Creación de indicadores para viajes con duración cero o negativa y viajes mayores a 24 horas.

Los registros con duraciones anómalas se conservan para su análisis posterior, en lugar de eliminarlos automáticamente. De esta manera, se evita descartar información sin haber evaluado antes su comportamiento.

**Importante:** los archivos procesados se generan localmente. Antes de volver a ejecutar el proceso de limpieza desde cero, elimina la carpeta `data/processed/` para evitar acumular registros si los archivos de salida ya existen.

## Seguimiento de experimentos con MLflow

MLflow se utiliza para registrar información de las ejecuciones del proceso de limpieza, incluyendo parámetros, métricas y resúmenes de resultados.

Para abrir la interfaz de seguimiento, activa el entorno virtual y ejecuta:

```bash
mlflow ui
```

Después, abre en el navegador:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

El experimento de limpieza se registra con el nombre `limpieza_mibici`.

Por el momento, el seguimiento de MLflow utiliza almacenamiento local. La base de datos, los registros de ejecuciones y los artefactos se excluyen de Git mediante `.gitignore`, al igual que los datos originales y procesados.

## Validación del código

Para ejecutar las pruebas automatizadas:

```bash
python -m pytest
```

Para revisar el código con Ruff:

```bash
ruff check .
```

## Control de versiones y reproducibilidad

El código y los scripts se versionan mediante Git y se alojan en GitHub. Los datos se mantienen fuera del repositorio para evitar subir archivos de gran tamaño.

Para reproducir el flujo de trabajo, es necesario configurar el entorno, instalar las dependencias y ejecutar los scripts en el orden descrito. Los resultados de las ejecuciones pueden variar si la fuente oficial modifica o actualiza los archivos disponibles.

## Próximas etapas

1. Validar los datos procesados y comprobar la consistencia de los resultados de limpieza.
2. Realizar el análisis exploratorio de datos (EDA).
3. Identificar variables relevantes y definir el enfoque de predicción.
4. Preparar los datos para el entrenamiento y la evaluación.
5. Desarrollar y comparar modelos de aprendizaje automático.
6. Ampliar el seguimiento de experimentos y documentar los resultados.
