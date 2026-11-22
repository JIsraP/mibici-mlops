# Servicio del modelo para la entrega final

El servicio recibe variables ya calculadas por el pipeline de MiBici. Cada registro contiene
`estacion_id`, `fecha_hora` y las 18 columnas de `FEATURE_COLUMNS`. Devuelve las salidas
y llegadas estimadas. El preprocesamiento aprendido forma parte del modelo exportado.

## Preparar el modelo

Usar el mismo entorno y versiones con los que se genero el artefacto. Comparar el archivo
`requirements.txt` de MLflow con `requirements-serving.txt` antes de construir la imagen.
Las versiones de Python, NumPy, scikit-learn y skops deben ser compatibles con el artefacto.

Desde la maquina que tenga el Model Registry y el Parquet preparado:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python -m mibici_mlops.modeling.export models:/MiBici_Demanda/1 data/processed/variables_oficial.parquet
```

Reemplazar `1` por la version numerica real. La ruta y version autorizadas quedan registradas
en `models/serving/serving_metadata.json`; no se presupone que la version 1 sea la correcta.
El comando rechaza una carpeta de destino existente para evitar reemplazar otro modelo.

La carpeta exportada incluye el modelo MLflow completo, sus dependencias y tres archivos:

- `example_request.json`: cinco registros del dataset preparado.
- `example_response.json`: predicciones directas del mismo modelo.
- `serving_metadata.json`: URI versionado y hashes del modelo y datos de referencia.

Si el modelo procede de otro integrante, solicitar esta carpeta completa exportada.
El data engineer puede versionarla con `dvc add models/serving` y publicarla en el remoto
DVC acordado. El artefacto debe estar materializado antes del build; la imagen lo incluye.

## Ejecutar y probar

```bash
python -m uvicorn mibici_mlops.api:app --host 127.0.0.1 --port 8000
```

Swagger: http://127.0.0.1:8000/docs
OpenAPI: http://127.0.0.1:8000/openapi.json

```bash
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @models/serving/example_request.json
python -m mibici_mlops.modeling.verify_serving
python -m pytest -q
```

El verificador guarda `reports/reproducibilidad_serving.json`. Compara las dos predicciones,
la cantidad de registros y las llaves estacion-fecha con tolerancias `rtol=1e-6` y `atol=1e-8`.
Termina con error si la respuesta excede la tolerancia.

## Contrato de entrada y salida

El cuerpo es un objeto `{"records": [...]}`, con 1 a 1000 registros.
Swagger enumera todos los campos, tipos y restricciones.

| Grupo | Campos |
|---|---|
| Identificacion | estacion_id, fecha_hora |
| Calendario | hora, hora_sin, hora_cos, dia_semana, es_fin_semana, es_festivo |
| Historia de salidas | salidas_lag_24, salidas_lag_168, salidas_media_7d |
| Historia de llegadas | llegadas_lag_24, llegadas_lag_168, llegadas_media_7d |
| Estacion | latitud, longitud, capacidad, zona, perfil_pc1, perfil_pc2 |

La respuesta tiene forma `{"predictions": [...]}`. Cada prediccion incluye
`estacion_id`, `fecha_hora`, `pred_salidas` y `pred_llegadas`.
Las predicciones son valores reales no negativos, no conteos redondeados.
Una entrada incompleta, fuera de rango o con columnas adicionales devuelve HTTP 422.
El cliente debe preparar calendario, rezagos y perfiles con el mismo contrato del entrenamiento.
El servicio solo se considera disponible si pudo cargar el modelo al iniciar.

## Docker y DockerHub

```bash
docker build --platform linux/amd64 -t ml-service:1.0.0 .
docker run --rm --platform linux/amd64 -p 8000:8000 ml-service:1.0.0
```

Con el contenedor activo, repetir `verify_serving` y guardar su JSON junto con la captura de Swagger.
El contenedor carga el artefacto local y no necesita un servidor MLflow para atender peticiones.

Para publicar, reemplazar `CUENTA_EQUIPO` por el nombre acordado:

```bash
docker login
docker tag ml-service:1.0.0 CUENTA_EQUIPO/mibici-mlops:1.0.0
docker push CUENTA_EQUIPO/mibici-mlops:1.0.0
```

El enlace a incluir en el PDF sera `https://hub.docker.com/r/CUENTA_EQUIPO/mibici-mlops`
una vez que la publicacion se haya completado y verificado. El nombre de cuenta es una
instruccion pendiente de completar, no una publicacion existente.

## Evidencias y limites de la verificacion

La implementacion agrega pruebas de entrenamiento con un modelo pequeno real, persistencia
y carga en skops, contrato HTTP, Swagger, exportacion versionada y comparacion contra referencia.
La suite ejecutada en la copia de trabajo dio 44 pruebas aprobadas (14 nuevas y 30 existentes).

El entorno de comprobacion disponible usa Python 3.12 y versiones distintas de algunas
dependencias fijadas en main; se registran en `reports/validacion_ml_engineer.json`.
Esta corrida no sustituye la validacion del modelo final con las dependencias acordadas.

El Dockerfile y las instrucciones estan preparados. La ejecucion de Docker, la comparacion del
modelo final y la publicacion en DockerHub requieren el artefacto del equipo y acceso a Docker.
No se reportan como verificadas hasta ejecutarlas.
