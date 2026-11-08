# ML Canvas de MiBici

Actualización del Canvas con los resultados de la Fase 2. Equipo 17, 8 de noviembre de 2026.

El resultado histórico publicado del modelo de árboles reduce el error MAE en horas pico un 15.35 % en salidas y un 15.99 % en llegadas frente al promedio histórico. Cumple los dos criterios propuestos de evaluación. La versión evaluada ya está integrada en main. Falta reproducir sus resultados de forma independiente y verificar el registro y versión del modelo. El beneficio para el servicio y las personas aún debe demostrarse.

**Cómo leer los cambios:** El PDF muestra en azul los párrafos nuevos o modificados frente al Canvas original de `main` en [f5912eae](https://github.com/JIsraP/mibici-mlops/tree/f5912eae8a0f55d66c3238ae4b2a3621b67d0a7f); el texto conservado está en negro.

## 1. Propuesta de valor

Anticipar salidas y llegadas por estación y hora para que operación prepare el servicio con tiempo, considerando bicicletas y espacios disponibles.

Usuario directo: responsable de operación y redistribución.

Beneficiarios: personas que utilizan o intentan utilizar MiBici.

Buscamos contribuir a traslados más confiables. Para demostrar el beneficio hacen falta datos de disponibilidad y una evaluación de la experiencia de uso.

La evaluación histórica publicada reporta un modelo que cumple la meta de error. Esto apoya seguir evaluando la propuesta; su utilidad operativa aún debe comprobarse.

Negocio conecta las necesidades de las personas con los criterios de evaluación. El equipo revisa resultados y límites. También se consideran estaciones de bajo uso: pocos viajes no prueban poca necesidad.

## 2. Tarea de aprendizaje automático

Regresión supervisada: predecir salidas y llegadas por estación y hora.

Entradas: estación, fecha, hora e historial disponible al emitir el pronóstico.

Se estiman viajes registrados. La diferencia entre llegadas y salidas no indica bicicletas disponibles ni demanda no atendida.

## 3. Decisiones

Uso futuro: operación revisaría las salidas y llegadas previstas junto con bicicletas, espacios y recursos disponibles.

Consideraría horas pico, mantenimiento y eventos. La decisión seguiría en manos del responsable de operación.

Fase 2: acordar cómo evaluar el pronóstico. Las rutas y cantidades de traslado quedan fuera de esta fase.

## 4. Fuentes de datos

CSV mensuales de viajes MiBici. Base de trabajo: 2025. Propuesta: reservar enero a septiembre de 2026 para una prueba final.

Catálogo de estaciones y calendario de festivos. Comprobar que la versión del catálogo corresponda al periodo analizado y cubra las estaciones.

Los viajes no permiten conocer bicicletas o espacios disponibles, esperas ni abandonos. El clima queda como ampliación posterior por acordar.

## 5. Recolección de datos

Conservar los archivos originales e identificar su periodo y versión para que el equipo trabaje con los mismos datos.

Revisar cobertura, duplicados y duración. Excluir viajes con duración no positiva; conservar y marcar los mayores de 24 horas para su revisión.

Agrupar salidas y llegadas por estación y hora. Incorporar nuevos meses solo después de validar su calidad.

## 6. Variables

Hora, día de semana, fin de semana, mes y festivos.

Viajes de la misma hora del día y semana anteriores y promedio de esa hora en los siete días previos. Usar solo información disponible al predecir.

Ubicación, zona y capacidad de estación, si el catálogo es válido para el periodo. La capacidad no indica espacios libres.

Excluir identificador, género y año de nacimiento de las personas usuarias.

## 7. Construcción de modelos

Fase 2: comparar modelos de salidas y llegadas con los mismos datos y criterios, en un proceso que el equipo pueda repetir.

La línea base es una referencia sencilla para evaluar los modelos. Las referencias publicadas son el promedio histórico y la misma hora de la semana anterior.

La elección se apoya en el error de salidas y llegadas. El resultado histórico del HGB Poisson cumple ambos criterios; su uso operativo aún debe comprobarse.

Actualizar cada mes es una propuesta sujeta a datos nuevos validados. No hay un servicio operativo comprobado.

## 8. Predicciones

En Fase 2 se comparan predicciones con viajes registrados para comprobar el desempeño con datos históricos.

Salida: tabla con estación, fecha, hora, salidas y llegadas previstas.

Emitir a las 23:00 para las siguientes 24 horas es una propuesta futura. El uso diario requiere datos más recientes que los archivos mensuales.

Esta fase no exige poner el servicio en operación ni ofrecerlo en tiempo real.

## 9. Evaluación offline

Entrenamiento final: enero a octubre de 2025. Validación histórica: noviembre y diciembre. La prueba de enero a septiembre de 2026 sigue como propuesta.

Modelo HGB Poisson publicado: error absoluto medio en horas pico (MAE) de 1.24 salidas y 1.16 llegadas, frente a 1.47 y 1.39 del promedio histórico. Viajes por estación y hora; cifras redondeadas.

Reducción del MAE pico: 15.35 % en salidas y 15.99 % en llegadas. Cumple la meta histórica de reducir 15 % salidas sin empeorar llegadas. Pico: 7, 8, 9, 18, 19 y 20 h, lunes a viernes. No demuestra mejora operativa.

## 10. Monitoreo en vivo

Etapa futura: revisar el error, los datos faltantes, las estaciones nuevas y los cambios de uso. Se propone revisar el modelo si el error semanal supera en 25 % al de validación; el umbral debe validarse.

Para comprobar menos esperas o abandonos hacen falta disponibilidad y experiencias de usuarios. Un menor error predictivo no demuestra por sí solo una mejora social.

## Respaldo de la propuesta

### Qué sabemos y qué falta demostrar

Los viajes registrados permiten estudiar el uso por estación y hora. No incluyen todas las personas que intentaron usar el servicio ni muestran el inventario real. El balance es llegadas menos salidas; faltan las bicicletas iniciales, los movimientos de redistribución y las bicicletas fuera de servicio para reconstruir disponibilidad.

Las preguntas frecuentes de MiBici contemplan buscar otra estación o esperar si no hay puertos disponibles [1]. Esto respalda la necesidad de estudiar el problema, pero no cuantifica su frecuencia. El análisis exploratorio muestra diferencias por estación y horario [2]; no permite atribuirlas por sí solas a falta de bicicletas, condiciones socioeconómicas o trayectos entre zonas habitacionales y de oficinas.

El pronóstico podría apoyar una revisión humana de estaciones y horarios junto con disponibilidad y recursos. No asigna rutas ni cantidades de traslado. Para demostrar menos búsquedas, esperas o abandonos se deben combinar datos de disponibilidad y evidencia de la experiencia de uso. Las reducciones de error aquí reportadas no son porcentajes de mejora del servicio, disponibilidad ni personas beneficiadas.

### Datos y preparación

- Fuente: CSV mensuales de MiBici. Base de trabajo de la evaluación: viajes de 2025 y calendario de festivos. Conservar originales en `data/raw/`, fuera de GitHub, e identificar archivo, periodo, filas y huella SHA-256.
- Revisar cobertura, duplicados, faltantes y consistencia temporal. Excluir duraciones no positivas y conservar con marca los viajes mayores de 24 horas.
- Usar la misma hora del día y semana anteriores (24 y 168 horas), más el promedio de esa hora en los siete días previos. Solo se puede usar historia disponible al emitir cada pronóstico, también durante un posible horizonte futuro de 24 horas.
- La documentación previa considera el catálogo de septiembre de 2026. Los cambios de ingeniería de ML, integrados mediante PR10, incorporan la opción del catálogo de octubre de 2026, donde falta la estación 240, y una imputación opcional. Su presencia en el código no demuestra que se haya utilizado en todas las ejecuciones. Registrar la versión y el tratamiento efectivamente usados, y revisar cobertura y vigencia histórica.
- La capacidad del catálogo no equivale a espacios libres. El clima no tiene una fuente ni un uso acordados para esta actualización.
- El identificador, género y año de nacimiento de las personas no se incluyen como variables del modelo. Las transformaciones aprendidas, incluido un posible resumen del perfil horario con PCA, deben ajustarse solo con entrenamiento.

### Diseño de evaluación

El código integrado usa enero a septiembre de 2025 para ajustar modelos y octubre para seleccionar sus parámetros. Después vuelve a entrenar con enero a octubre y evalúa con noviembre y diciembre de 2025, con inicio de validación el 1 de noviembre. Se deben mantener separados los periodos de ajuste y de evaluación final.

Reservar enero a septiembre de 2026 para una prueba fuera de tiempo sigue como propuesta; no se presenta como una prueba ejecutada. La reproducción independiente de los resultados publicados aún debe confirmarse.

### Resultados históricos publicados

La línea base es una referencia sencilla contra la cual evaluar los modelos. Se comparan el promedio histórico por estación, día de semana y hora, la misma hora de la semana anterior, una regresión Poisson regularizada y un modelo de árboles HGB con pérdida Poisson.

Fuente de los resultados: [comparación publicada](https://github.com/JIsraP/mibici-mlops/blob/f4231ecb27d74ad53634c835794c1b620eb0c352/reports/comparacion_modelos.csv), commit `f4231ecb27d74ad53634c835794c1b620eb0c352` de `main`, integrado mediante PR10. MAE y RMSE se expresan en viajes por estación y hora; un valor menor indica menos error. Las tablas siguientes redondean a cinco decimales. La lámina redondea errores a dos decimales y porcentajes a dos, pero el cumplimiento de criterios se evalúa con los valores completos.

| Referencia | Objetivo | MAE | RMSE | MAE pico |
|---|---|---:|---:|---:|
| Promedio histórico | Salidas | 0.85614 | 1.54562 | 1.46875 |
| Promedio histórico | Llegadas | 0.81776 | 1.48786 | 1.38628 |
| Misma hora de la semana anterior | Salidas | 0.98996 | 1.90877 | 1.60830 |
| Misma hora de la semana anterior | Llegadas | 0.94237 | 1.82921 | 1.49692 |

| Resultado del modelo HGB Poisson | Promedio histórico MAE pico | Modelo MAE pico | Reducción del error |
|---|---:|---:|---:|
| Salidas | 1.46875 | 1.24332 | 15.35 % |
| Llegadas | 1.38628 | 1.16461 | 15.99 % |

Valores completos de MAE pico: salidas 1.2433171820564155 frente a 1.4687480367821895; llegadas 1.1646059691845259 frente a 1.3862827653315408. La reducción relativa se calcula como `(referencia - modelo) / referencia × 100`: 15.348504 % en salidas y 15.990734 % en llegadas, redondeadas a seis decimales.

La meta propuesta exige reducir al menos 15 % el MAE pico de salidas sin empeorar el MAE pico de llegadas. El umbral de salidas es 1.248435831264861 viajes por estación y hora. El resultado publicado del HGB cumple ambos criterios. La regresión Poisson regularizada no cumple la meta. Esto constituye evidencia de evaluación histórica; no acredita despliegue ni mejora operativa real.

### Selección y registro

En la versión `f4231ecb`, comparación y registro comparten tabla y criterios. Las corridas comparables se filtran por la huella del conjunto de validación (`validacion_sha256`), con compatibilidad con `dataset_sha256` para corridas antiguas sin mezclar conjuntos. Se verifica el 15 % sin redondear y la condición de no empeorar el MAE pico de llegadas. La selección prioriza candidatos elegibles que cumplen ambas condiciones.

El CSV marca el HGB con `apto_registro=True`. Significa que cumple los requisitos para registrarlo según la comparación publicada; **no demuestra que exista una versión efectiva en Model Registry**. El servidor, el registro y la versión deben verificarse independientemente.

Identificadores publicados en el CSV:

- HGB Poisson: `53a5aaec4e66405ca818cdb47ef4aa3f`.
- Promedio histórico: `f41a09a0f88e4889bb99ae4e10574a02`.
- Semana anterior: `ae9e147420d645d3b5d0a20d21d03708`.
- Poisson regularizado: `9333cfdd26ee45c7a6a4effcd28e3e52`.

El resultado global no asegura una mejora uniforme en cada estación y horario. Revisar desgloses de salidas y llegadas, estaciones de alto y bajo uso y fines de semana antes de definir un uso operativo. No se presenta el diagnóstico anterior de la línea base como un diagnóstico del modelo nuevo.

### Alcance técnico de Fase 2

La consigna [3] contempla estructura Cookiecutter, refactorización, preparación y estimador en un Pipeline de scikit-learn, comparación de modelos e hiperparámetros y uso de MLflow Tracking y Model Registry. No exige despliegue ni fija algoritmos o un número específico de modelos.

El entrenamiento publicado integra un Pipeline de scikit-learn. El README integrado documenta una secuencia de preparación de referencias, entrenamiento, comparación y registro (`make baseline && make train && make compare && make register`). Hay pruebas de comparabilidad y elegibilidad integradas. La reproducción completa y su verificación independiente siguen pendientes.

Los entregables de Fase 2 incluyen PDF ejecutivo, repositorio con participación de todos y video de hasta cinco minutos. La emisión a las 23:00, el horizonte de 24 horas, la actualización mensual y la alerta de 25 % de error son diseños futuros por validar, no compromisos operativos ni logros.

## Roles responsables

La responsabilidad de negocio está confirmada. Las otras asignaciones se conservan como antecedentes o propuestas por ratificar para Fase 2.

| Rol | Referencia de responsabilidad | Aportación e interacción propuesta |
|---|---|---|
| Negocio y Canvas | Rol confirmado | Conectar necesidades de las personas, alcance y criterios de valor; revisar resultados y límites con el equipo |
| Data engineer | José Israel Pérez Ontiveros, asignación previa | Compartir versiones de datos y reglas de calidad; acordar cobertura y periodos |
| Data scientist | Francisco Gabriel Rodríguez González, asignación previa | Definir métricas y comparación de salidas y llegadas; compartir resultados con ingeniería y negocio |
| ML engineer | Ricardo Portilla Corona, propuesta previa por ratificar | Integrar preparación, entrenamiento, predicción y registro; entregar resultados que el equipo pueda reproducir |

El equipo debe ratificar responsables e intercambios, revisar la reproducción de los modelos y confirmar su registro y versión. Las propuestas de roles no atribuyen ejecución ni compromisos que no estén documentados.

## Fuentes

- [Canvas original](https://github.com/JIsraP/mibici-mlops/blob/f5912eae8a0f55d66c3238ae4b2a3621b67d0a7f/docs/ml_canvas.md), base del resaltado de cambios.
- [Resultados de la Fase 2](https://github.com/JIsraP/mibici-mlops/blob/f4231ecb27d74ad53634c835794c1b620eb0c352/reports/comparacion_modelos.csv), integrados en `main` mediante [PR10](https://github.com/JIsraP/mibici-mlops/pull/10), commit `f4231ecb27d74ad53634c835794c1b620eb0c352`.
- [Entrenamiento](https://github.com/JIsraP/mibici-mlops/blob/f4231ecb27d74ad53634c835794c1b620eb0c352/mibici_mlops/modeling/train.py), [criterio de selección](https://github.com/JIsraP/mibici-mlops/blob/f4231ecb27d74ad53634c835794c1b620eb0c352/mibici_mlops/modeling/compare.py) y [README](https://github.com/JIsraP/mibici-mlops/blob/f4231ecb27d74ad53634c835794c1b620eb0c352/README.md) de la misma versión.
- [Consigna de Fase 2][3], con acceso institucional.

## Actualización

Actualización del Canvas con los resultados de la Fase 2.

## ML Canvas visual

[PDF del Canvas actualizado](Canvas_MiBici_Fase_2.pdf)

[1]: https://www.mibici.net/es/preguntas-frecuentes/
[2]: https://github.com/JIsraP/mibici-mlops/blob/f5912eae8a0f55d66c3238ae4b2a3621b67d0a7f/notebooks/1.0-frg-exploracion-inicial.ipynb
[3]: https://experiencia21.tec.mx/courses/708335/assignments/23606020
