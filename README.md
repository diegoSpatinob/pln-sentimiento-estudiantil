# Clasificación de sentimiento en comentarios estudiantiles

Proyecto grupal de **Ciencia de Datos** orientado a la clasificación binaria de comentarios universitarios escritos en español mediante técnicas de procesamiento de lenguaje natural (PLN).

Se compararon dos enfoques:

- **TF-IDF + Regresión Logística** como modelo base.
- **BETO (BERT en español)** ajustado mediante fine-tuning.

La métrica principal de comparación fue **F1-macro**, complementada con accuracy, precision macro, recall macro y métricas por clase.

## Integrantes

- **Diego Patiño:** preparación de datos, análisis exploratorio, limpieza, particiones y modelo base.
- **María José Vire:** desarrollo, entrenamiento y evaluación del modelo Transformer, además de la comparación final.
- **Asignatura:** Ciencia de Datos.
- **Docente:** Ing. Jorge Maldonado.

## Objetivo

Clasificar cada comentario como:

- **Negativo (0)**
- **Positivo (1)**

El predictor utilizado fue `comentario_limpio` y la variable objetivo fue `sentimiento_id`.

Para garantizar una comparación consistente, ambos enfoques utilizaron las mismas particiones de entrenamiento, validación y prueba.

## Dataset

Se utilizó el **Student Feedback Sentiment Analysis Dataset**, versión **1.0.1**, de Anabel Pineda-Briseño y Jimy Oblitas, publicado en Zenodo, registro **18797217**.

DOI: `10.5281/zenodo.18797217`

El dataset original contiene **23.168 registros**. Después de aplicar los criterios de limpieza y consistencia definidos en el proyecto, se conservaron **23.123 observaciones**.

La partición estratificada con semilla 42 quedó distribuida de la siguiente manera:

| Partición | Registros |
| --- | ---: |
| Entrenamiento | 16.186 |
| Validación | 3.468 |
| Prueba | 3.469 |

Los archivos procesados no se redistribuyen directamente en el repositorio.

## Estructura del proyecto

```text
pln-sentimiento-estudiantil/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   ├── README.md
│   ├── 01_datos_y_modelo_base.ipynb
│   ├── 02_transformer.ipynb
│   └── 03_BETO_entrenamiento_evaluacion.ipynb
├── docs/
│   └── referencias.md
├── scripts/
│   ├── fase0.py
│   ├── transformer.py
│   └── final_evaluation.py
├── tests/
└── results/
    └── final/

## Declaración de uso de IA generativa

Durante el desarrollo del proyecto se utilizó **ChatGPT** como herramienta de apoyo en distintas actividades de documentación, análisis y desarrollo.

| Herramienta | Uso en el proyecto | Verificación realizada |
| --- | --- | --- |
| ChatGPT | Apoyo en la redacción y revisión de documentación, estructuración de explicaciones, análisis e interpretación de resultados, preparación del material educativo y revisión de fragmentos de código. | Las respuestas y sugerencias fueron contrastadas con los notebooks ejecutados, las salidas reales del pipeline, los archivos versionados del repositorio y la documentación técnica correspondiente. |

La IA generativa se utilizó como herramienta de apoyo y no sustituyó la ejecución ni la validación experimental del proyecto. El código, las particiones de datos, las métricas y los resultados reportados fueron revisados por los integrantes del equipo a partir de las ejecuciones y artefactos reales del proyecto.

Las decisiones metodológicas finales, la selección de modelos y la interpretación de los resultados fueron realizadas y validadas por los integrantes del equipo.
