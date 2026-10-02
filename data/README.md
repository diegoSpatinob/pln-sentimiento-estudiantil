# Datos del proyecto

El notebook utiliza **Student Feedback Sentiment Analysis Dataset** (`student-feedback-peru`), versión **1.0.1**, de Anabel Pineda-Briseño y Jimy Oblitas, publicado en Zenodo.

- Registro: https://zenodo.org/records/18797217
- DOI: https://doi.org/10.5281/zenodo.18797217
- Licencia indicada en el notebook: **Creative Commons Attribution-NoDerivatives 4.0 International (CC BY-ND 4.0)**.
- Archivo ZIP identificado durante la ejecución: `student-feedback-peru-v1.0.1.zip`.

El dataset **no está incluido en este repositorio**. El notebook consulta la API `https://zenodo.org/api/records/18797217`, identifica el ZIP original, lo descarga y carga su archivo tabular.

No subir aquí datos de personas, copias modificadas o particiones derivadas sin verificar los términos de distribución. Las rutas `data/raw/` y `data/processed/` están excluidas de Git por `.gitignore`.

En la ejecución documentada, se partió de 23.168 registros y quedaron 23.123 tras excluir registros anómalos y duplicados de acuerdo con los criterios explicados en el notebook. Los conjuntos son 16.186 entrenamiento, 3.468 validación y 3.469 prueba. Son resultados de esa ejecución, no archivos ya almacenados en GitHub.

La descarga requiere conexión a Internet y disponibilidad de Zenodo.
