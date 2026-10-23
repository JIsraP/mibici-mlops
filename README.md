
# MiBici MLOps

Proyecto de análisis y predicción de la demanda del sistema de bicicletas públicas MiBici. El objetivo es preparar un entorno reproducible para trabajar con datos, desarrollar modelos de aprendizaje automático y evaluar sus resultados.

## Objetivos

- Preparar un entorno de desarrollo reproducible.
- Organizar los datos y el código del proyecto.
- Explorar los datos para identificar patrones de demanda.
- Desarrollar y evaluar modelos de predicción.
- Incorporar prácticas de MLOps durante el desarrollo.

## Estructura del proyecto

```text
├── data/
│   ├── external/       # Datos de fuentes externas
│   ├── interim/        # Datos intermedios
│   ├── processed/     # Datos preparados para el análisis
│   └── raw/            # Datos originales
├── docs/               # Documentación
├── models/             # Modelos entrenados y resultados relacionados
├── notebooks/          # Notebooks de análisis y experimentación
├── references/         # Documentación y materiales de referencia
├── reports/
│   └── figures/        # Gráficas y figuras
├── mibici_mlops/       # Código fuente del proyecto
├── tests/              # Pruebas automatizadas
├── .gitignore          # Archivos que Git debe ignorar
├── Makefile            # Comandos de apoyo
├── pyproject.toml      # Configuración y metadatos del proyecto
├── requirements.txt    # Dependencias de Python
└── LICENSE             # Licencia
```

## Entorno de desarrollo

El proyecto utiliza Python 3.11 y un entorno virtual para aislar sus dependencias.

Para activar el entorno virtual en macOS:

```bash
source .venv/bin/activate
```

Para instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

Para ejecutar las pruebas:

```bash
python -m pytest
```

Para revisar el código con Ruff:

```bash
ruff check .
```

## Estado actual

Actualmente se está preparando la estructura inicial del proyecto y las herramientas de desarrollo. El análisis de datos y la construcción de modelos se realizarán en etapas posteriores.


## Descarga de datos

Los datos originales de MiBici no se incluyen en este repositorio debido a su tamaño. Para obtenerlos, sigue estos pasos desde la carpeta raíz del proyecto.

### 1. Activar el entorno virtual

Si todavía no tienes un entorno virtual creado, ejecuta:

```bash
python3 -m venv .venv
```

Actívalo con:

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

### 2. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar los datos

Ejecuta el siguiente comando:

```bash
python scripts/download_data.py
```

El script consulta la página oficial de [MiBici - Datos Abiertos](https://www.mibici.net/es/datos-abiertos/), identifica los archivos CSV disponibles y los descarga automáticamente en la carpeta `data/raw/`.

### Consideraciones

- Los datos no se almacenan en GitHub debido a su tamaño.
- Los archivos CSV descargados se guardan localmente en `data/raw/`.
- El script omite los archivos que ya existen y tienen contenido.
- Si algunos archivos no se descargan, puedes volver a ejecutar el script.
- La descarga depende de la disponibilidad del sitio oficial de MiBici y de la conexión a internet.

