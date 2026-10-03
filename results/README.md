# Resultados

La Fase 0 exporta localmente `baseline_C1_validation.json`, además de conservar las salidas históricas del notebook. Incluye configuración seleccionada, métricas generales, precision/recall/F1 y soporte por clase, y matriz de confusión con orden `[0, 1]`, filas reales y columnas predichas. Los archivos generados continúan excluidos de Git.

**Validación del modelo base (C1):** F1-macro = 0,8757; accuracy = 0,8789. Son resultados documentados del conjunto de validación y **no** del conjunto de prueba.

Los modelos serializados, datasets derivados y artefactos locales no se publican automáticamente. La comparación con el Transformer y la evaluación final aún están pendientes.

C1 se guarda en `../models/baseline_C1.joblib` con configuración, entorno, procedencia del ajuste y SHA-256 en `baseline_C1_metadata.json`. Permanece ajustado únicamente con los 16.186 registros de entrenamiento. `python scripts/fase0.py --verify` comprueba carga y resultados de validación, sin predecir test. La compatibilidad del modelo se verificó dentro del entorno registrado; no se garantiza entre versiones distintas de Python o scikit-learn.
