# MiBici MLOps

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Predicción de demanda por estación y hora para MiBici, el sistema de bicicletas públicas del Área Metropolitana de Guadalajara. Proyecto del curso TC5061 Operaciones de aprendizaje automático, Fase 1.

- Requerimientos y propuesta de valor: [docs/ml_canvas.md](docs/ml_canvas.md)
- Bitácora de versiones de datos: [docs/versionado_datos.md](docs/versionado_datos.md)
- Datos: [datos abiertos de MiBici](https://mibici.net/en/open-data/), viajes de enero a diciembre de 2025 y catálogo de estaciones de septiembre de 2026.

## Cómo reproducir

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # ajustar MLFLOW_TRACKING_URI si el servidor no es local
mlflow ui                   # en otra terminal, http://localhost:5000
make pipeline               # descarga, limpieza, variables, líneas base y figuras
make test
```

| Paso | Comando | Versión en MLflow |
|---|---|---|
| Descarga | `make download` | |
| Limpieza (estricta y laxa) | `make data` | v0_crudo, v1_limpio_* |
| Demanda y variables | `make features` | v2_demanda_*, v3_variables_* |
| Líneas base | `make baseline` | experimento `MiBici_Fase1_Experimentos_Datos` |
| Figuras del EDA | `make plots` | `reports/figures/` |

## Cómo trabajamos en el repositorio

- `main` solo recibe cambios por pull request revisado por otro integrante.
- Cada tarea va en su rama (`limpieza`, `eda`, `canvas`, `refactor`) y cada quien hace commits con su propia cuenta.
- Los datos no se suben a git (`data/` está en `.gitignore`); su versión queda en MLflow con su huella SHA-256.

## Organización del proyecto

```
├── Makefile           <- Comandos de la tubería: make download, data, features, baseline, plots
├── README.md
├── data
│   ├── interim        <- Viajes limpios (v1)
│   ├── processed      <- Demanda por estación y hora (v2) y variables (v3)
│   └── raw            <- CSV originales del portal (v0)
├── docs               <- ML Canvas y bitácora de versiones de datos
├── notebooks          <- 1.0 exploración inicial; 2.0 análisis usando los módulos
├── references         <- Lista de URL de descarga
├── reports            <- Estadísticas descriptivas y figuras del EDA
├── requirements.txt
├── tests              <- Pruebas de limpieza y agregación (pytest)
└── mibici             <- Código fuente
    ├── config.py      <- Rutas, fechas de corte, festivos y configuración de MLflow
    ├── dataset.py     <- Descarga, lectura con detección de codificación y limpieza
    ├── cleaning.py    <- CleaningRules y TripCleaner: reglas con reporte por regla
    ├── tracking.py    <- DatasetVersioner: registro de versiones en MLflow
    ├── features.py    <- Demanda por estación y hora, FeatureBuilder y preprocesador
    ├── plots.py       <- Figuras del EDA
    └── modeling
        └── train.py   <- Líneas base para comparar versiones de datos
```
