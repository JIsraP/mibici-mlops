
# MiBici MLOps

Proyecto de análisis y predicción de la demanda del sistema de bicicletas públicas MiBici. El objetivo es preparar un entorno reproducible para trabajar con datos, desarrollar modelos de aprendizaje automático y evaluar sus resultados.

- Requerimientos y propuesta de valor: [docs/ml_canvas.md](docs/ml_canvas.md)
- Bitácora de versiones de datos: [docs/versionado_datos.md](docs/versionado_datos.md)
- Datos: [datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/). En la Fase 1 se usan los viajes de enero a diciembre de 2025 y el catálogo de estaciones de septiembre de 2026; enero a septiembre de 2026 queda reservado para probar el modelo en fases posteriores.

## Objetivos

- Preparar un entorno de desarrollo reproducible.
- Organizar los datos y el código del proyecto.
- Explorar los datos para identificar patrones de demanda.
- Desarrollar y evaluar modelos de predicción.
- Incorporar prácticas de MLOps durante el desarrollo.

## Entorno de desarrollo

El proyecto utiliza Python 3.11 o superior (la Fase 1 se probó con 3.13) y un entorno virtual para aislar sus dependencias.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
cp .env.example .env               # ajustar MLFLOW_TRACKING_URI si el servidor no es local
```

Para ejecutar las pruebas y revisar el código:

```bash
make test
make lint
```

## Descarga de datos

Los datos originales de MiBici no se incluyen en este repositorio debido a su tamaño. Hay dos formas de obtenerlos; las dos guardan los CSV en `data/raw/` y omiten los archivos que ya existen.

- `make download` baja solo los 12 meses de 2025 y el catálogo de estaciones que usa la Fase 1, a partir de la lista fija en `references/urls_2025.txt` (unos 400 MB).
- `python scripts/download_data.py` consulta la página oficial, identifica todos los CSV disponibles (diciembre de 2014 a la fecha) y los descarga. Sirve para trabajar con el histórico completo.

La descarga depende de la disponibilidad del sitio oficial de MiBici y de la conexión a internet; si algún archivo falla, basta con volver a ejecutar el comando.

## Tubería de la Fase 1

Con la MLflow UI corriendo (`mlflow ui`, http://localhost:5000):

```bash
make pipeline     # descarga, limpieza, variables, líneas base y figuras
```

| Paso | Comando | Versión en MLflow |
|---|---|---|
| Descarga | `make download` | |
| Limpieza (variantes estricta y laxa) | `make data` | v0_crudo, v1_limpio_* |
| Demanda y variables | `make features` | v2_demanda_*, v3_variables_* |
| Líneas base | `make baseline` | experimento `MiBici_Fase1_Experimentos_Datos` |
| Figuras del EDA | `make plots` | `reports/figures/` |

## Cómo trabajamos en el repositorio

- `main` solo recibe cambios por pull request revisado por otro integrante.
- Cada tarea va en su rama y cada quien hace commits con su propia cuenta.
- Los datos no se suben a git (`data/` está en `.gitignore`); su versión queda en MLflow con su huella SHA-256.

## Estructura del proyecto

```text
├── data/
│   ├── external/       # Datos de fuentes externas
│   ├── interim/        # Viajes limpios (v1)
│   ├── processed/      # Demanda por estación y hora (v2) y variables (v3)
│   └── raw/            # CSV originales del portal (v0)
├── docs/               # ML Canvas y bitácora de versiones de datos
├── models/             # Modelos entrenados y resultados relacionados
├── notebooks/          # 1.0 exploración inicial; 2.0 análisis usando los módulos
├── references/         # Lista de URL de descarga de 2025
├── reports/
│   └── figures/        # Figuras del EDA
├── scripts/            # Descarga del histórico completo
├── tests/              # Pruebas de limpieza y agregación (pytest)
├── mibici_mlops/       # Código fuente del proyecto
│   ├── config.py       # Rutas, fechas de corte, festivos y configuración de MLflow
│   ├── dataset.py      # Descarga, lectura con detección de codificación y limpieza
│   ├── cleaning.py     # CleaningRules y TripCleaner: reglas con reporte por regla
│   ├── tracking.py     # DatasetVersioner: registro de versiones en MLflow
│   ├── features.py     # Demanda por estación y hora, FeatureBuilder y preprocesador
│   ├── plots.py        # Figuras del EDA
│   └── modeling/
│       └── train.py    # Líneas base para comparar versiones de datos
├── .gitignore          # Archivos que Git debe ignorar
├── Makefile            # Comandos de la tubería y de apoyo
├── pyproject.toml      # Configuración y metadatos del proyecto
├── requirements.txt    # Dependencias de Python
└── LICENSE             # Licencia
```

## Estado actual

Fase 1 terminada: análisis de requerimientos, limpieza, EDA, preprocesamiento y versionado de datos. La construcción y evaluación de modelos corresponde a la Fase 2.
