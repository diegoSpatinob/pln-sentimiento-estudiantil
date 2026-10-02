# Clasificación de sentimiento en comentarios estudiantiles

Proyecto grupal de **Ciencia de Datos** sobre clasificación binaria de comentarios universitarios escritos en español mediante procesamiento de lenguaje natural (PLN). Se compararán un **Transformer preentrenado ajustado mediante fine-tuning** y un modelo base de **TF-IDF + Regresión Logística**.

> **Estado actual:** la documentación y el notebook de datos/modelo base de Diego (etapas 1–5) ya están publicados desde la versión ejecutada en Google Colab. El Transformer, la comparación y la evaluación final todavía están pendientes. Las métricas expuestas aquí provienen de **validación**, no de prueba.

## Integrantes

- **Diego Patiño:** datos, EDA, limpieza, particiones y modelo base.
- **María José Vire:** desarrollo del Transformer y colaboración posterior en la comparación.
- **Asignatura:** Ciencia de Datos. **Docente:** Ing. Jorge Maldonado.

## Objetivo y metodología

Clasificar comentarios como **Negativo (0)** o **Positivo (1)**. Se emplean las mismas particiones para la comparación futura y se establece **F1-macro** como métrica principal, con accuracy y métricas por clase como complementarias.

## Dataset

*Student Feedback Sentiment Analysis Dataset*, versión **1.0.1**, de Anabel Pineda-Briseño y Jimy Oblitas, publicado en [Zenodo (registro 18797217)](https://zenodo.org/records/18797217), DOI [10.5281/zenodo.18797217](https://doi.org/10.5281/zenodo.18797217). El notebook indica licencia **CC BY-ND 4.0**.

No se redistribuyen archivos del dataset ni copias procesadas: el notebook consulta la API de Zenodo y descarga el ZIP original en el entorno de ejecución. Véase [data/README.md](data/README.md).

## Estructura actual

```text
pln-sentimiento-estudiantil/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   ├── README.md
│   └── 01_datos_y_modelo_base.ipynb
├── docs/
│   └── referencias.md
└── results/
    └── README.md
```

El archivo [`notebooks/01_datos_y_modelo_base.ipynb`](notebooks/01_datos_y_modelo_base.ipynb) ya está publicado conservando íntegramente el notebook original y sus salidas. Más adelante se incorporará el trabajo real del Transformer, sin crear archivos vacíos ni resultados anticipados.

## Reproducir el proyecto

**Google Colab (recomendado):** abrir `notebooks/01_datos_y_modelo_base.ipynb` desde **Archivo → Abrir cuaderno → GitHub**, buscando este repositorio. Para reproducir las dependencias documentadas, antes de ejecutar el notebook se puede usar una celda nueva:

```python
!git clone https://github.com/diegoSpatinob/pln-sentimiento-estudiantil.git
%pip install -r /content/pln-sentimiento-estudiantil/requirements.txt
```

Si el directorio ya fue clonado en la misma sesión, no repetir el comando de clonación. Ejecutar el notebook de principio a fin y en orden con acceso a Internet para descargar los datos desde Zenodo. El notebook guarda archivos temporales bajo `/content/data/`, `/content/models/` y `/content/results/`; una nueva sesión de Colab no recupera automáticamente el estado en memoria.

**Entorno local:** utilizar Python 3.13, instalar dependencias con `python -m pip install -r requirements.txt` y abrir Jupyter desde la raíz del repositorio. Los directorios de ejecución se crean automáticamente. Las versiones registradas en la ejecución original se encuentran en `requirements.txt`; la versión exacta de `requests` no se registró en aquella celda y se especifica un rango compatible.

## Avance del modelo base

El notebook desarrollado hasta la etapa 5 parte de **23.168** registros y conserva **23.123** después de los criterios documentados de limpieza. Distribución estratificada con semilla 42: entrenamiento **16.186**, validación **3.468**, prueba **3.469**.

Se ensayaron seis configuraciones de TF-IDF + Regresión Logística sobre la partición de validación. Entre las configuraciones evaluadas se seleccionó **C1**: unigramas, vocabulario efectivo de **5.653** características y Regresión Logística `C=1.0` con regularización L2.

| Métrica del modelo base (validación) | Valor |
| --- | ---: |
| **F1-macro** | **0,8757** |
| Accuracy | 0,8789 |
| F1 Negativo (0) | 0,8556 |
| F1 Positivo (1) | 0,8957 |

**El conjunto de prueba permanece reservado.** El equipo acordará un mismo protocolo final antes de evaluar ambos modelos. Una única partición de validación y seis configuraciones no demuestran optimalidad global ni generalización definitiva.

## Trabajo colaborativo

Las contribuciones deben corresponder a cambios reales de cada integrante y quedar registradas mediante commits propios. El notebook de María José se integrará cuando exista una ejecución real verificable del Transformer. El contenido educativo y el video se incorporarán según el cronograma de la actividad.

## Declaración de uso de IA generativa

Se utilizó **ChatGPT** para apoyar la estructuración del notebook, la explicación de conceptos, la preparación/revisión de código Python y la redacción de documentación del repositorio. La verificación realizada hasta este avance comprende revisión humana, resultados guardados de ejecuciones en Google Colab y comprobaciones programáticas incluidas en el notebook. **La ejecución independiente desde el repositorio nuevo todavía está pendiente** y deberá verificarse antes de declarar cumplimiento pleno de reproducibilidad. El grupo mantiene la responsabilidad por los cálculos, fuentes y contenido entregado.

## Referencias

Fuentes iniciales y documentación: [docs/referencias.md](docs/referencias.md). La investigación de la herramienta principal y las referencias completas requeridas para el material educativo siguen en elaboración.
