# Clasificación de sentimiento en comentarios estudiantiles

Proyecto grupal de **Ciencia de Datos** sobre clasificación binaria de comentarios universitarios escritos en español mediante procesamiento de lenguaje natural (PLN). Se compararán un **Transformer preentrenado ajustado mediante fine-tuning** y un modelo base de **TF-IDF + Regresión Logística**.

> **Estado actual:** datos y modelo base (etapas 1–5) cuentan con una Fase 0 de persistencia y verificación. El flujo se reprodujo en un proceso limpio con Python 3.12.3 y las versiones declaradas del baseline, conservando las salidas históricas de Colab. El Transformer, la comparación y la evaluación final todavía están pendientes. Las métricas expuestas aquí provienen de **validación**, no de prueba.

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
├── scripts/
│   └── fase0.py
└── results/
    └── README.md
```

El archivo [`notebooks/01_datos_y_modelo_base.ipynb`](notebooks/01_datos_y_modelo_base.ipynb) conserva las salidas históricas. La Fase 0 corrige una línea no comentada en 5.3.1, resuelve la raíz local y añade una celda final de exportación verificada; no modifica las decisiones metodológicas ni los resultados existentes. Más adelante se incorporará el trabajo real del Transformer.

## Reproducir el proyecto

**Google Colab (recomendado):** abrir `notebooks/01_datos_y_modelo_base.ipynb` desde **Archivo → Abrir cuaderno → GitHub**, buscando este repositorio. Para reproducir las dependencias documentadas, antes de ejecutar el notebook se puede usar una celda nueva:

```python
!git clone https://github.com/diegoSpatinob/pln-sentimiento-estudiantil.git
%pip install -r /content/pln-sentimiento-estudiantil/requirements.txt
```

Si el directorio ya fue clonado en la misma sesión, no repetir el comando de clonación. Se requiere una copia del repositorio que incluya `scripts/fase0.py`. Ejecutar el notebook de principio a fin y en orden con acceso a Internet para descargar los datos desde Zenodo. La celda final guarda las particiones bajo `/content/data/processed/`, C1 bajo `/content/models/` y resultados de validación bajo `/content/results/`. Descargar estos artefactos antes de finalizar la sesión: una sesión nueva no recupera archivos ni objetos anteriores.

**Entorno local:** la ejecución original registra Python 3.13.15; la Fase 0 también se verificó con Python 3.12.3. Instalar dependencias con `python -m pip install -r requirements.txt`. Desde la raíz, `python scripts/fase0.py` ejecuta todas las celdas de código en orden en un proceso limpio y genera los artefactos sin sobrescribir las salidas históricas. Usa Matplotlib Agg: no renderiza las expresiones finales de las celdas ni genera un notebook ejecutado. También se puede abrir el notebook en Jupyter, instalado aparte, y ejecutarlo en orden. La raíz local se resuelve incluso si el kernel inicia en `notebooks/`.

`python scripts/fase0.py --verify` carga las particiones, verifica sus SHA-256, tamaños, clases y solapamientos, carga C1 y reproduce sus resultados exclusivamente sobre validación. Una discrepancia detiene el proceso; no se regeneran particiones diferentes. El manifiesto registra el entorno completo de generación. El ZIP requiere Internet y disponibilidad de Zenodo; `--verify` trabaja sin red.

Los archivos generados, excluidos de Git, son `data/processed/{train,validation,test}.csv`, `data/processed/metadata.json`, `models/baseline_C1.joblib`, `models/baseline_C1_metadata.json` y `results/baseline_C1_validation.json`. La interfaz de lectura para otra persona se documenta en [data/README.md](data/README.md). Debe recibir los CSV y su manifiesto juntos.

La infraestructura BETO está preparada en Fase 3A para ejecutar posteriormente
T1/T2/T3 en Colab. El smoke real de Fase 2B está verificado y sigue siendo NO OFICIAL;
todavía no hay resultados oficiales Transformer. Véase el
[protocolo y auditoría de Fase 3A](docs/fase3a_transformer.md) para comandos,
checkpoints, selección única y artefactos. Test continúa reservado.

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

Se utilizó **ChatGPT** para apoyar la estructuración del notebook, la explicación de conceptos, la preparación/revisión de código Python y la redacción de documentación del repositorio. La verificación comprende revisión humana, salidas históricas de Colab y ejecución secuencial independiente en la Fase 0, con comprobaciones de corpus, particiones, serialización y resultados de validación. Las salidas históricas conservan referencias temporales inconsistentes entre Markdown y tablas; no se sustituyeron con tiempos de esta ejecución. La reproducción en el Python 3.13.15 original y la evaluación final siguen pendientes. El grupo mantiene la responsabilidad por los cálculos, fuentes y contenido entregado.

## Referencias

Fuentes iniciales y documentación: [docs/referencias.md](docs/referencias.md). La investigación de la herramienta principal y las referencias completas requeridas para el material educativo siguen en elaboración.
