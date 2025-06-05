# Proyecto MLOps - Clasificación de Mensajes en Batch

La solución refleja un enfoque práctico -mantenible, reproducible y escalable- para implementar sistemas de ML en producción, aplicando buenas prácticas de MLOps desde la experimentación hasta el despliegue

El proyecto consiste en un sistema batch que automatiza la clasificación de mensajes enviados por usuarios a un canal de atención, simulados mediante reseñas del **Yelp Open Dataset**. Se diseñó una arquitectura modular basada en el enfoque **FTI (Feature - Training - Inference)**, con énfasis en mantenibilidad, reproducibilidad y escalabilidad

La implementación considera el ciclo completo: desde la ingesta de datos y procesamiento, hasta el entrenamiento, predicción y activación de reentrenamiento bajo condiciones controladas

Este repositorio contiene todo el código, configuraciones y artefactos necesarios para replicar la solución

## **Objetivo**

El objetivo técnico fue desarrollar un sistema de clasificación de mensajes en batch, alineado con principios de MLOps y capaz de escalar hacia entornos productivos

***La prioridad no estuvo en optimizar cada línea de código***, sino en definir una **arquitectura clara**, una **lógica desacoplada entre etapas** y un flujo robusto de punta a punta. Se diseñó una solución que refleja cómo se debe estructurar un sistema de ML real, más allá de un simple modelo funcional

El proyecto priorizó:

- **Diseñar pipelines desacoplados**, compatibles con ejecución secuencial o por orquestador
- **Definir pasos explícitos por etapa** (extracción, validación, agregación, etc.), facilitando la trazabilidad del flujo
- **Aplicar validaciones** de datos y modelos
- **Controlar dependencias y calidad del código** mediante herramientas como `uv`, `hydra`, `ruff`, `mypy`, `bandit`, `pre-commit`, `pytest` y `coverage.py`
- **Simular condiciones realistas de producción**, incluyendo detección de drift, uso de feature store y reentrenamiento automático

Este enfoque permitió construir un sistema sólido y extensible, listo para ser escalado o refactorizado sin comprometer su arquitectura base

## Arquitectura del sistema

La solución sigue la arquitectura FTI (Feature → Training → Inference), separando claramente la lógica de transformación, entrenamiento e inferencia. Cada etapa fue implementada como un pipeline independiente, permitiendo trazabilidad, reutilización de componentes y mantenibilidad

El flujo completo también contempla la lógica de reentrenamiento ante degradación del modelo

---

### 🟧 Feature Pipeline

Toma los datos crudos desde Yelp y los transforma en un conjunto de features limpios, validados y listos para modelar

Pasos implementados:
- `extract`: lectura de nuevos datos
- `remove duplicates`: eliminación de duplicados
- `validate`: validación de estructura y tipos
- `clean text`: preprocesamiento básico del texto
- `create new features`: variables derivadas
- `embedding`: transformación del texto en vectores numéricos

[📎 Ver Feature Pipeline Scikit-learn](https://htmlpreview.github.io/?https://github.com/juan-gomezj4/ml-message-classifier/blob/main/data/08_reporting/feature_pipeline.html)

---

### 🟩 Training Pipeline

Utiliza los datos procesados para generar un modelo entrenado. Mantiene consistencia entre entrenamiento y validación, asegurando que las transformaciones aplicadas en producción sean equivalentes

Pasos implementados:
- `imputer`: imputación de valores nulos
- `encoding`: codificación de variables
- `scaler`: escalado de variables numéricas
- `dimensionality reducer`: reducción opcional de dimensionalidad
- `training`: entrenamiento del modelo
- `validate`: evaluación con métricas definidas

[📎 Ver Training Pipeline Scikit-learn](https://htmlpreview.github.io/?https://github.com/juan-gomezj4/ml-message-classifier/blob/main/data/08_reporting/training_pipeline.html)

---

### 🟦 Inference Pipeline

Aplica el modelo entrenado sobre nuevos datos manteniendo la misma lógica de transformación que en el entrenamiento

Pasos implementados:
- Repite el mismo flujo de Feature Pipeline para nuevos datos
- Carga el `Pipeline Trainer` con los artefactos del modelo
- Aplica `predict` y entrega resultados listos para gestión operativa



---

### 🔁 Lógica de reentrenamiento (CT)

El sistema activa el reentrenamiento automático cuando se detecta sesgo en las predicciones (por ejemplo, si cualquier clase predicha representa menos del 10% o más del 90% del total). En ese caso:

- Se extraen nuevos datos

- Se ejecuta nuevamente el Feature Pipeline

- Se actualiza el modelo usando el mismo algoritmo y configuración actual (no se ajustan hiperparámetros)

- El Pipeline Trainer es reescrito con los nuevos datos

Este proceso permite mantener el modelo actualizado ante cambios en la distribución de los datos sin modificar su estructura base

## Diccionario de datos

Para esta solución se utilizó el Yelp Open Dataset como simulación de mensajes enviados por usuarios a un canal de atención. A partir de este conjunto se estructuraron las tablas base que alimentan el pipeline

### 🧑‍💼 Tabla: `user`

| Campo                  | Tipo     | Descripción                                                                 |
|------------------------|----------|-----------------------------------------------------------------------------|
| `user_id`              | string   | ID único del usuario                                                        |
| `name`                 | string   | Nombre del usuario                                                          |
| `review_count`         | int      | Número total de reseñas escritas                                           |
| `yelping_since`        | string   | Fecha en que se unió a Yelp (AAAA-MM-DD)                                   |
| `useful`               | int      | Votos 'useful' recibidos                                                    |
| `funny`                | int      | Votos 'funny' recibidos                                                     |
| `cool`                 | int      | Votos 'cool' recibidos                                                      |
| `elite`                | string   | Años como usuario elite (separados por coma o vacío)                        |
| `friends`              | string   | Lista de IDs de amigos (separados por coma o vacío)                         |
| `fans`                 | int      | Número de seguidores                                                        |
| `average_stars`        | float    | Promedio de estrellas que ha dado                                           |

---

### 🏢 Tabla: `business`

| Campo           | Tipo     | Descripción                                               |
|-----------------|----------|-----------------------------------------------------------|
| `business_id`   | string   | ID único del negocio                                      |
| `name`          | string   | Nombre del negocio                                        |
| `address`       | string   | Dirección                                                 |
| `city`          | string   | Ciudad                                                    |
| `state`         | string   | Estado o región                                           |
| `postal_code`   | string   | Código postal                                             |
| `latitude`      | float    | Latitud geográfica                                        |
| `longitude`     | float    | Longitud geográfica                                       |
| `stars`         | float    | Promedio de estrellas recibido                           |
| `review_count`  | int      | Número total de reseñas recibidas                        |
| `is_open`       | int      | Indicador de si el negocio está abierto (1) o cerrado (0)|
| `attributes`    | string   | Información adicional sobre el negocio (JSON u objeto)   |
| `categories`    | string   | Categorías a las que pertenece el negocio                |
| `hours`         | string   | Horario de atención (JSON u objeto)                      |

---

### 📝 Tabla: `review`

| Campo         | Tipo        | Descripción                                                        |
|---------------|-------------|--------------------------------------------------------------------|
| `review_id`   | string      | ID único de la reseña                                              |
| `user_id`     | string      | ID del usuario que hizo la reseña (relación con `user.user_id`)    |
| `business_id` | string      | ID del negocio reseñado (relación con `business.business_id`)      |
| `stars`       | int         | Calificación en estrellas dada por el usuario                      |
| `useful`      | int         | Votos 'useful' que recibió la reseña                               |
| `funny`       | int         | Votos 'funny' que recibió la reseña                                |
| `cool`        | int         | Votos 'cool' que recibió la reseña                                 |
| `text`        | string      | Texto completo de la reseña                                        |
| `date`        | datetime    | Fecha en que se escribió la reseña                                 |


## Decisiones técnicas

Durante el desarrollo del sistema se tomaron decisiones centradas en una solución reproducible, mantenible, escalable  y criterios reales de ingeniería, más allá de la construcción de un modelo puntual. A continuación, se describen los principales puntos técnicos:

---

### 🔨 Modularidad y diseño de pipelines

- Se definió una arquitectura **FTI (Feature - Training - Inference)** que permite separar responsabilidades y facilita testing, mantenimiento y reusabilidad.
- Cada pipeline está desacoplado y puede ejecutarse en forma independiente o como parte de un flujo orquestado

---

### 🧠 Feature Store

- Se implementó una estructura que permite reutilizar las features generadas entre entrenamiento e inferencia
- Esto asegura coherencia en la transformación y reduce tiempos de cómputo en producción

---

### 📦 Entorno y gestión de dependencias

- Se utilizó `uv` para garantizar entornos reproducibles, ligeros y aislados.
- La configuración del proyecto se gestiona con `hydra`, evitando hardcoding y facilitando la parametrización de pipelines

---

### 🧪 Validación y calidad del código

- Se integraron `pre-commit` hooks para asegurar validaciones constantes sobre el código
- Uso de `ruff` (linter + formatter), `mypy` (tipado estático) y `bandit` (seguridad)
- Validaciones de datos estructurales
- Testing funcional cubierto con `pytest`, con tracking de cobertura (`coverage.py`)

---

### 🧬 Selección de variables

Para reducir la dimensionalidad y evitar ruido, se implementó un pipeline específico que incluye:

- Eliminación de variables constantes
- Eliminación de variables altamente correlacionadas
- Selección de variables basadas en importancia usando un Random Forest

Esto permitió construir un conjunto de features compacto y relevante, manteniendo la interpretabilidad del modelo

---

### 🎯 Construcción de la variable objetivo y métrica seleccionada

Se trató como un problema de clasificación multiclase, donde la variable objetivo fue construida a partir de las estrellas de calificación de la reseña (`stars`) del dataset de Yelp:

- `0` → calificaciones menores o iguales a 2
- `1` → calificaciones de 3
- `2` → calificaciones de 4 o 5

Dado que no se aplicó balanceo de clases y se buscaba una métrica robusta ante desbalance, se utilizó **F1-score ponderado (`f1_weighted`)** para la selección del modelo final

---

### ⚙️ Modelado y experimentación

- No se aplicó un benchmark exhaustivo, pero se probaron múltiples algoritmos incluyendo AutoML.
- El mejor modelo fue seleccionado por desempeño base y luego afinado con `GridSearchCV`
- El foco no estuvo en optimización algorítmica, sino en asegurar un sistema replicable y extensible

---

### 🔄 Coherencia entre entrenamiento e inferencia

Se garantizó la coherencia entre etapas identificando claramente las responsabilidades de cada una, definiendo su orden en un diagrama general de solución, y encapsulando la lógica dentro de `PipelineTrainer`. Esto asegura que el flujo de transformación en inferencia sea equivalente al usado durante el entrenamiento

---

### ✅ Validación del flujo end-to-end

Aunque se trata de un proyecto de prueba técnica, se implementaron estrategias reales para asegurar trazabilidad:

- Cada paso del Feature Pipeline guarda una versión intermedia de los datos para su inspección
- Se incluyeron `loggers` en todo el flujo para facilitar seguimiento y debugging
- Se probaron corridas con subconjuntos pequeños (1.000–2.000 registros) para validar la lógica general antes de escalar
- El desarrollo partió de una versión básica en notebooks, sin clases, que permitió validar la idea general antes de modularizar

---

### ☁️ Preparación para producción

- Aunque el flujo fue probado localmente, se definieron componentes para despliegue en AWS (Glue, Step Functions, S3, Lambda).
- Se estableció una estrategia de monitoreo basada en métricas clave del flujo y del modelo.
- Se propuso la integración con SNS para alertas y Lambda para ejecución automática ante errores o drift.

## Estructura del proyecto

La organización del repositorio sigue principios de separación de responsabilidades, modularidad y trazabilidad. A continuación se describe la estructura principal:
```
├── conf/                           # Configuración del proyecto por etapa (feature, training, inference)
│   ├── data_feature/
│   ├── model_inference/
│   └── model_training/
│
├── data/                           # Capas de datos según el flujo FTI
│   ├── 01_raw/                     # Archivos originales del dataset (Yelp)
│   ├── 02_intermediate/           # Datos validados
│   ├── 03_primary/                # Datos agregados y comprimidos
│   ├── 04_feature/                # Dataset final con features (MIT)
│   ├── 05_model_input/            # Dataset listo para modelado
│   ├── 06_models/                 # Parámetros y metadata del mejor modelo
│   ├── 07_model_output/           # Resultados de predicción
│   ├── 08_reporting/              # Métricas y visualizaciones
│   └── 09_inference/              # Datos de entrada/salida para inferencia
│
├── docs/                          # Documentación en formato Markdown
│
├── models/                        # Artefactos del modelo serializado (.pkl)
│
├── notebooks/                     # Exploración, experimentación y pruebas manuales
│
├── src/                           # Código fuente modular del proyecto
│   ├── data/                      # Módulos para extracción, validación, agregación, etc.
│   ├── inference/                 # Lógica de inferencia y carga del modelo
│   ├── model/                     # Entrenamiento, transformación y validación del modelo
│   ├── pipelines/                # Definición de los pipelines FTI
│   │   ├── feature_pipeline/
│   │   ├── training_pipeline/
│   │   └── inference_pipeline/
│
├── tests/                         # Estructura básica para pruebas por componente
│
├── start_batch_nlp.py            # Script principal para ejecutar el sistema
├── Makefile                      # Comandos automatizados (ej. test, lint, run)
├── pyproject.toml                # Configuración general del proyecto (mypy, ruff, etc.)
├── uv.lock                       # Lockfile para gestión de dependencias con uv
├── codecov.yml                   # Configuración de cobertura de código
└── README.md                     # Documento principal del proyecto
```

## 🧪 CI / 🚀 CD  

El proyecto implementa un flujo completo y funcional de MLOps:

### ✅ CI – Continuous Integration

- Validación automática de estilo, tipos y errores con `pre-commit`, `ruff` y `mypy`
- Pruebas unitarias con `pytest` y cobertura de código, integradas en GitHub Actions.
- Reporte de coverage enviado a [Codecov](https://about.codecov.io/)

### 🚀 CD – Continuous Delivery

- Entrenamiento del modelo desde datos preprocesados (`data/04_feature/review_user_business_mit_sample.parquet`).
- Pipeline modular usando `sklearn.Pipeline`, compuesto por:
  - `MDTYelpData`: transformaciones dependientes del modelo.
  - `TrainModelTransformer`: entrenamiento con `XGBClassifier`.
- Integración en el CI/CD con el comando `make train`, ejecutado automáticamente tras pasar los tests.


## Siguientes pasos técnicos

Este MVP deja sentada la arquitectura base del sistema, pero se identificaron varias oportunidades de mejora para la siguiente iteración:

- 🧼 **Refactor del código**  
  Reorganizar y simplificar transformadores, funciones y clases para mayor legibilidad y mantenimiento

- 🧪 **Revisión de tipado estático**  
  Ajustar las anotaciones de tipo (`mypy`) en transformadores, funciones y clases para mejorar validación en desarrollo

- ⚖️ **Balanceo de clases**  
  Evaluar técnicas como reponderación o submuestreo para mejorar el rendimiento del modelo en clases minoritarias

- 🧠 **Interpretabilidad del modelo**  
  Incluir análisis de importancia de variables (`feature_importance_`) y herramientas como SHAP para explicar predicciones

- 📈 **Optimización del rendimiento**  
  La métrica `f1_macro` pasó de **0.63 a 0.70** tras ajustes simples; nuevas mejoras podrían llevarla más lejos con balanceo e interpretabilidad

Estas mejoras son compatibles con la arquitectura actual y se pueden incorporar sin romper el diseño del sistema
