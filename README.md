# MiBici MLOps

Proyecto de análisis y preparación de datos del sistema MiBici de Guadalajara, Jalisco, México, aplicando prácticas de MLOps para organizar el procesamiento, la limpieza, la validación y el seguimiento de los experimentos.

El objetivo es construir una base de datos confiable y reproducible que permita analizar el comportamiento histórico del sistema y sirva como punto de partida para desarrollar modelos de aprendizaje automático en etapas posteriores.

## Objetivos del proyecto

- Organizar los archivos históricos de viajes de MiBici.
- Automatizar el procesamiento de los archivos CSV mensuales.
- Detectar y documentar problemas de calidad en los datos.
- Aplicar reglas de limpieza sin eliminar información potencialmente útil de manera innecesaria.
- Validar la consistencia de los datos procesados.
- Realizar un análisis exploratorio de los viajes y las estaciones.
- Utilizar MLflow para registrar parámetros, métricas y artefactos de los experimentos.
- Mantener un flujo de trabajo reproducible y versionar el código mediante Git.

## Fuente de datos

Los datos se obtienen del portal oficial de datos abiertos de MiBici:

https://www.mibici.net/es/datos-abiertos/

El conjunto de datos está organizado en archivos CSV mensuales con información de los viajes registrados en el sistema.

**Periodo analizado:** diciembre de 2014 a septiembre de 2026.

**Archivos mensuales procesados:** 142.

**Registros procesados después de eliminar duplicados exactos:** 39,078,226 viajes.

Las columnas utilizadas incluyen identificadores de viaje y usuario, género, año de nacimiento, fechas de inicio y finalización, estaciones de origen y destino, duración del viaje e indicadores de calidad.

## Tecnologías utilizadas

- **Python:** procesamiento, limpieza y análisis de datos.
- **pandas:** manipulación de datos y procesamiento por bloques.
- **Matplotlib:** visualización de resultados.
- **Jupyter Notebook:** documentación del análisis exploratorio.
- **MLflow:** seguimiento de experimentos, parámetros, métricas y artefactos.
- **pytest:** ejecución de pruebas automatizadas.
- **Ruff:** revisión estática y calidad del código.
- **Git y GitHub:** control de versiones y colaboración.

## Estructura del proyecto

El repositorio sigue una estructura basada en Cookiecutter Data Science, organizada para separar los datos, el código, los experimentos y la documentación.

```text
mibici_mlops/
├── data/
│   ├── raw/                 # Datos originales
│   ├── interim/             # Datos intermedios
│   └── processed/           # Datos procesados y limpios
├── docs/                    # Documentación del proyecto
├── mibici_mlops/            # Código del paquete del proyecto
├── models/                  # Modelos generados
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── references/              # Material de referencia
├── reports/                 # Reportes y resultados
├── tests/                   # Pruebas automatizadas
├── .gitignore
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

Los scripts de procesamiento y validación se encuentran organizados dentro del código del proyecto, de acuerdo con la implementación actual.

Los directorios `data/` contienen los datos locales y están excluidos del control de versiones para evitar subir archivos de gran tamaño a GitHub.

## Flujo de trabajo

El procesamiento de datos sigue estas etapas generales:

1. **Obtención de datos:** reunir los archivos mensuales descargados de la fuente oficial.
2. **Inspección inicial:** revisar la estructura, las columnas, los tipos de datos y la cobertura temporal.
3. **Diagnóstico de calidad:** identificar valores faltantes, registros duplicados, duraciones inconsistentes y otros problemas.
4. **Limpieza:** aplicar reglas definidas para tratar duplicados y valores sospechosos.
5. **Procesamiento:** guardar los archivos resultantes en `data/processed/`, trabajando por bloques cuando es necesario para controlar el uso de memoria.
6. **Validación:** comprobar la integridad de los identificadores, las fechas, las duraciones y los indicadores de calidad.
7. **Análisis exploratorio:** estudiar la evolución de los viajes, las estaciones y los problemas de calidad.
8. **Seguimiento de experimentos:** registrar resultados relevantes en MLflow.

## Limpieza y preparación de los datos

El procesamiento contempla los siguientes aspectos:

- Eliminación de duplicados exactos.
- Identificación de valores faltantes en género y año de nacimiento.
- Detección de años de nacimiento sospechosos.
- Conversión de los valores de nacimiento inválidos a valores faltantes.
- Cálculo y revisión de la duración de los viajes.
- Identificación de duraciones iguales o menores que cero.
- Identificación de viajes con duración superior a 24 horas.
- Conservación de indicadores que permiten distinguir los registros problemáticos.

En lugar de eliminar automáticamente todos los viajes con inconsistencias, se conservan indicadores para facilitar su análisis posterior. De esta forma, los registros se pueden estudiar sin perder información de manera innecesaria.

### Resultados del procesamiento

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

Las validaciones automatizadas también revisan la consistencia de las fechas, las duraciones calculadas y los indicadores asociados. La validación completa de fechas debe confirmarse después de aplicar correctamente el tratamiento de los distintos formatos encontrados en los archivos originales.

## Análisis exploratorio de datos (EDA)

El análisis exploratorio se documenta en:

`notebooks/01_exploratory_data_analysis.ipynb`

El notebook contiene revisiones y visualizaciones para comprender el comportamiento histórico del sistema y detectar patrones relevantes.

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

Se compararon los identificadores presentes en los viajes de enero a septiembre de 2026 con el catálogo disponible para septiembre de 2026.

| Indicador | Resultado |
|---|---:|
| Estaciones distintas en los viajes de 2026 | 471 |
| Estaciones en el catálogo | 484 |
| Identificadores presentes en ambos conjuntos | 469 |
| Identificadores en viajes, pero no en el catálogo | 2 |
| Identificadores del catálogo sin viajes en 2026 | 15 |

La coincidencia entre los identificadores de los viajes y el catálogo es aproximadamente del 99.6% respecto a las 471 estaciones identificadas en los viajes.

Los identificadores que no coinciden quedan documentados para su revisión. No se clasifican automáticamente como errores, ya que podrían existir diferencias entre versiones del catálogo o cambios históricos en las estaciones.

## Seguimiento de experimentos con MLflow

MLflow se utiliza para registrar los resultados relevantes del procesamiento y facilitar su trazabilidad.

El experimento de limpieza se identifica como:

`limpieza_mibici`

Los registros de los experimentos pueden incluir:

- Parámetros y reglas de procesamiento.
- Número de archivos y registros procesados.
- Métricas de calidad de los datos.
- Resultados de las validaciones.
- Artefactos con tablas, gráficas y resúmenes relevantes.

El código que genera los resultados se mantiene en Git, mientras que MLflow conserva la evidencia de las ejecuciones. Los artefactos deben corresponder a resultados realmente generados y no a valores estimados manualmente.

La ubicación del almacenamiento de MLflow depende de la configuración local del proyecto.

## Instalación y configuración

Se recomienda utilizar un entorno virtual de Python para instalar las dependencias sin afectar otros proyectos.

Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para abrir el proyecto en Jupyter, se puede utilizar el entorno configurado en VS Code y seleccionar el intérprete de `.venv`.

Si se desea ejecutar el notebook desde la terminal y Jupyter está instalado, se puede utilizar:

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

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

Las pruebas y validaciones deben ejecutarse después de realizar cambios en los scripts para verificar que el procesamiento mantenga los resultados esperados.

## Control de versiones y manejo de datos

Git se utiliza para versionar el código fuente, los notebooks, las pruebas y la documentación.

Los archivos originales y procesados se mantienen localmente y están excluidos del repositorio mediante `.gitignore`. Lo mismo aplica al entorno virtual, los archivos temporales y el almacenamiento local de MLflow.

Esta separación permite mantener un repositorio ligero y evitar subir archivos de datos que pueden ocupar varios gigabytes.

## Limitaciones actuales

- Los años 2014 y 2026 tienen cobertura temporal parcial.
- El catálogo disponible de estaciones no permite reconstruir por sí solo la red histórica.
- Los identificadores que no coinciden entre viajes y catálogo requieren revisión.
- Los valores faltantes y las duraciones inconsistentes necesitan interpretarse según su contexto.
- La validación de fechas debe confirmarse considerando los distintos formatos presentes en los archivos.
- Los patrones observados en el EDA son descriptivos y no demuestran causalidad.

## Próximas etapas

Las siguientes actividades previstas para continuar el proyecto son:

1. Confirmar que todas las validaciones automatizadas se ejecuten correctamente.
2. Documentar y revisar las diferencias de calidad detectadas.
3. Profundizar en el análisis histórico de viajes y estaciones.
4. Completar el reporte final y la presentación ejecutiva.
5. Preparar los datos y las métricas necesarias para las siguientes etapas de experimentación con modelos de aprendizaje automático.

## Licencia y uso de los datos

Los datos utilizados provienen del portal de datos abiertos de MiBici. Para su reutilización, consulta las condiciones y los términos publicados por la fuente oficial.

El código de este repositorio se distribuye bajo la licencia indicada en el archivo `LICENSE`.

---

**Proyecto:** MiBici MLOps  
**Repositorio:** https://github.com/JIsraP/mibici-mlops  
**Fuente de datos:** https://www.mibici.net/es/datos-abiertos/