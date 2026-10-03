# Notebooks

## Datos y modelo base (Diego Patiño)

Archivo integrado: `01_datos_y_modelo_base.ipynb`.

Incluye las etapas 1–5: preparación del entorno, carga/validación, EDA, limpieza y particiones, entrenamiento y selección del baseline **TF-IDF + Regresión Logística**. El archivo procede del notebook de Google Colab trabajado por Diego y debe conservar sus celdas, salidas y trazabilidad experimental.

La Fase 0 comenta la línea que impedía ejecutar 5.3.1, resuelve la raíz de salida local y añade una celda final de persistencia. Se conservan todas las salidas históricas y el análisis científico. La exportación requiere `scripts/fase0.py` y se detiene ante diferencias respecto al corpus, las particiones o los resultados C1 documentados. `python scripts/fase0.py`, desde la raíz, permite ejecutar sus celdas de código en orden sin modificar las salidas del archivo.

**Para ejecutarlo desde Colab:** abrir el notebook de este directorio, instalar primero las dependencias indicadas en `../requirements.txt` si el entorno no las satisface y ejecutar las celdas en orden. Su descarga de datos parte del registro original de Zenodo; no requiere una copia de datos en GitHub.

## Transformer (María José Vire)

**Pendiente de integración:** se incorporará su implementación real y los resultados obtenidos al completar la primera ejecución. No se crea un notebook ficticio ni se atribuyen resultados todavía no entregados.

## Evaluación comparativa

Pendiente: se acordará un protocolo común antes de utilizar el conjunto de prueba reservado. F1-macro será la métrica principal, complementada con accuracy y resultados por clase.
