# Datos del proyecto

El notebook utiliza **Student Feedback Sentiment Analysis Dataset** (`student-feedback-peru`), versión **1.0.1**, de Anabel Pineda-Briseño y Jimy Oblitas, publicado en Zenodo.

- Registro: https://zenodo.org/records/18797217
- DOI: https://doi.org/10.5281/zenodo.18797217
- Licencia indicada en el notebook: **Creative Commons Attribution-NoDerivatives 4.0 International (CC BY-ND 4.0)**.
- Archivo ZIP identificado durante la ejecución: `student-feedback-peru-v1.0.1.zip`.

El dataset **no se versiona en Git**. El notebook consulta la API `https://zenodo.org/api/records/18797217`, identifica el ZIP original, lo descarga y carga su archivo tabular. La Fase 0 genera copias procesadas locales después de verificar el experimento existente.

No subir aquí datos de personas, copias modificadas o particiones derivadas sin verificar los términos de distribución. Las rutas `data/raw/` y `data/processed/` están excluidas de Git por `.gitignore`.

La ejecución limpia de Fase 0 reprodujo los 23.123 registros finales, con las mismas reglas del notebook. Los conjuntos persistidos contienen 16.186 registros de entrenamiento (6.870 negativos / 9.316 positivos), 3.468 de validación (1.472 / 1.996) y 3.469 de prueba (1.473 / 1.996), sin índices ni textos normalizados compartidos. Estos archivos locales no se almacenan automáticamente en GitHub.

## Interfaz común persistida

`processed/train.csv`, `processed/validation.csv` y `processed/test.csv` usan CSV UTF-8: es legible, portable y no requiere motores adicionales. Conservan el orden de cada partición y las columnas `indice_original`, `comentario`, `Valor`, `nombreAspecto`, `comentario_limpio` y `sentimiento_id`. `indice_original` identifica la fila original; no es un predictor. Los únicos predictores son los textos de `comentario_limpio`; la etiqueta es `sentimiento_id` (`Negativo=0`, `Positivo=1`). `nombreAspecto` permanece como contexto.

`processed/metadata.json` registra origen, versión 1.0.1, hashes del ZIP y del CSV fuente, tamaños, clases, semilla 42, proporciones, esquema, SHA-256 de cada partición, fecha UTC y versiones del entorno. Debe conservarse y transferirse junto con los tres CSV. La primera exportación establece sus checksums; una ejecución posterior no sobrescribe particiones distintas. Las salidas históricas no contenían un manifiesto de índices con el cual contrastar la asignación fila a fila anterior.

Desde la raíz del repositorio, la interfaz verificada es:

```python
from scripts.fase0 import cargar_particiones

particiones = cargar_particiones()
train = particiones["train"]
validation = particiones["validation"]
test = particiones["test"]
```

La función comprueba checksums, tamaños, distribución, normalización y solapamientos antes de devolver los DataFrames; mantiene `indice_original` como columna y como índice. Para Colab se pasa `cargar_particiones("/content")` si los artefactos están allí. Si se usa `pd.read_csv` directamente, especificar `encoding="utf-8"`, `keep_default_na=False` y tipos enteros para `indice_original` y `sentimiento_id`: así se conservan textos que pandas podría interpretar como valores ausentes.

Test se usa únicamente para controles de integridad y distribución en esta fase, sin entrenamiento, selección ni métricas predictivas.

La descarga requiere conexión a Internet y disponibilidad de Zenodo.
