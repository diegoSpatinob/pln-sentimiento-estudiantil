"""Persistencia y lectura verificable del experimento existente; sin nuevo modelado."""

import argparse
import ast
import hashlib
import io
import json
import os
import platform
import warnings
import zipfile
from datetime import datetime, timezone
from importlib.metadata import distributions, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/01_datos_y_modelo_base.ipynb"
COLUMNS = ["indice_original", "comentario", "Valor", "nombreAspecto",
           "comentario_limpio", "sentimiento_id"]
DTYPES = {column: "int64" if column in {"indice_original", "sentimiento_id"}
          else str for column in COLUMNS}
EXPECTED = {"train": (16186, 6870, 9316),
            "validation": (3468, 1472, 1996), "test": (3469, 1473, 1996)}
LABEL_MAP = {"Negativo": 0, "Positivo": 1}
TFIDF = {"ngram_range": (1, 1), "min_df": 2, "max_features": 20000,
         "lowercase": True, "strip_accents": None, "stop_words": None,
         "token_pattern": r"(?u)\b\w+\b", "use_idf": True, "smooth_idf": True,
         "sublinear_tf": False, "norm": "l2"}
LOGREG = {"C": 1.0, "penalty": "l2", "solver": "liblinear", "max_iter": 1000,
          "class_weight": None, "random_state": 42}
METRICS = ["f1_macro", "accuracy", "precision_macro", "recall_macro"]


def require(condition, message):
    """Los controles no desaparecen al ejecutar Python con -O."""
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def check_partitions(partitions):
    require(set(partitions) == set(EXPECTED), "Faltan particiones comunes.")
    for name, frame in partitions.items():
        total, negative, positive = EXPECTED[name]
        require(list(frame.columns) == COLUMNS, f"Esquema distinto: {name}.")
        require(len(frame) == total, f"Tamaño distinto: {name}.")
        counts = frame["sentimiento_id"].value_counts().to_dict()
        require(counts == {0: negative, 1: positive}, f"Clases distintas: {name}.")
        require(frame.notna().all().all(), f"Nulos en {name}.")
        require(frame["indice_original"].dtype == np.dtype("int64"),
                f"Índices no enteros en {name}.")
        require(frame["indice_original"].is_unique, f"Índices repetidos en {name}.")
        require(frame["comentario_limpio"].is_unique, f"Textos repetidos en {name}.")
        require(frame["sentimiento_id"].equals(frame["Valor"].map(LABEL_MAP)),
                f"Codificación distinta en {name}.")
        normalized = frame["comentario"].str.replace(r"\s+", " ", regex=True).str.strip()
        require(normalized.equals(frame["comentario_limpio"]),
                f"Normalización distinta en {name}.")
        require(frame["comentario_limpio"].str.strip().ne("").all(),
                f"Comentarios vacíos en {name}.")
    names = list(EXPECTED)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            for column in ["indice_original", "comentario_limpio"]:
                require(set(partitions[first][column]).isdisjoint(partitions[second][column]),
                        f"Solapamiento {column}: {first}/{second}.")
    combined = pd.concat(partitions.values(), ignore_index=True)
    require(len(combined) == 23123, "El corpus final no contiene 23.123 registros.")
    require(combined["indice_original"].between(0, 23167).all(),
            "Índices fuera del dataset original.")


def check_model(model):
    require(list(model.named_steps) == ["tfidf", "logreg"], "Pipeline distinto de C1.")
    for step, expected in [("tfidf", TFIDF), ("logreg", LOGREG)]:
        params = model.named_steps[step].get_params()
        for key, value in expected.items():
            require(params[key] == value, f"Parámetro distinto: {step}.{key}.")
    tfidf, classifier = model.named_steps["tfidf"], model.named_steps["logreg"]
    require(len(tfidf.vocabulary_) == len(tfidf.idf_) == 5653,
            "Vocabulario distinto de 5.653 características.")
    require(classifier.coef_.shape == (1, 5653), "Coeficientes de dimensión distinta.")
    require(np.array_equal(classifier.classes_, [0, 1]), "Clases distintas de [0, 1].")


def load_model(source):
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = joblib.load(source)
    check_model(model)
    return model


def cargar_particiones(base_dir=ROOT):
    """Comprueba hashes, esquema, etiquetas y solapamientos antes de devolver datos.

    Cada DataFrame conserva indice_original como columna y como índice. No evalúa test.
    """
    base_dir = Path(base_dir)
    metadata = json.loads((base_dir / "data/processed/metadata.json").read_text("utf-8"))
    require(metadata["random_state"] == 42 and metadata["label_map"] == LABEL_MAP,
            "Metadatos metodológicos distintos.")
    require(metadata["text_column"] == "comentario_limpio"
            and metadata["target_column"] == "sentimiento_id"
            and metadata["index_column"] == "indice_original"
            and metadata["primary_metric"] == "f1_macro"
            and metadata["predictors"] == ["comentario_limpio"]
            and metadata["columns"] == COLUMNS
            and metadata["total_records"] == 23123
            and metadata["split_proportions"] == {"train": 0.7, "validation": 0.15, "test": 0.15},
            "Interfaz de datos distinta.")
    partitions = {}
    for name, (total, negative, positive) in EXPECTED.items():
        entry = metadata["partitions"][name]
        relative_path = f"data/processed/{name}.csv"
        require(entry["path"] == relative_path and entry["records"] == total
                and entry["class_counts"] == {"0": negative, "1": positive},
                f"Metadatos distintos: {name}.")
        data = (base_dir / relative_path).read_bytes()
        require(sha256(data) == entry["sha256"], f"Checksum incorrecto: {name}.")
        partitions[name] = pd.read_csv(io.BytesIO(data), encoding="utf-8", dtype=DTYPES,
                                       keep_default_na=False)
    check_partitions(partitions)
    return {name: frame.set_index("indice_original", drop=False)
            for name, frame in partitions.items()}


def persistir_estado(state):
    """Exporta únicamente los objetos finales del notebook tras verificar referencias."""
    require(state["RANDOM_STATE"] == 42 and state["label_map_44"] == LABEL_MAP,
            "Semilla o etiquetas distintas; no se exporta.")
    require((state["TRAIN_SIZE"], state["VALIDATION_SIZE"], state["TEST_SIZE"])
            == (0.7, 0.15, 0.15), "Proporciones distintas; no se exporta.")
    require(len(state["df_clean"]) == 23123, "Corpus distinto; no se exporta.")
    partitions = {}
    for name, variable in [("train", "df_train_45"), ("validation", "df_val_45"),
                           ("test", "df_test_45")]:
        frame = state[variable].copy(deep=True)
        frame.insert(0, "indice_original", frame.index)
        partitions[name] = frame[COLUMNS]
    check_partitions(partitions)
    reconstructed = pd.concat([state[v] for v in ["df_train_45", "df_val_45", "df_test_45"]])
    require(reconstructed.sort_index().equals(state["df_clean"].sort_index()),
            "Las particiones no reconstruyen df_clean.")
    require(state["selected_id_551"] == "C1", "Selección distinta; no se exporta.")
    model = state["selected_model_551"]
    require(model is state["candidate_models_542"]["C1"], "Objeto seleccionado distinto.")
    check_model(model)
    expected_confusion = np.array([[1244, 228], [192, 1804]])
    require(np.array_equal(state["selected_confusion_552"], expected_confusion),
            "Matriz de validación distinta; no se exporta.")
    expected_metrics = {"f1_macro": 0.8756503649146483, "accuracy": 0.8788927335640139,
                        "precision_macro": 0.8770452701072533, "recall_macro": 0.8744581554413174}
    metrics = {key: float(state["selected_val_metrics_545"][key]) for key in METRICS}
    require(all(np.isclose(metrics[k], v, rtol=0, atol=1e-12)
                for k, v in expected_metrics.items()),
            "Métricas de validación distintas; no se exporta.")

    # Serialización y carga en memoria antes de escribir cualquier artefacto.
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    model_bytes = buffer.getvalue()
    reloaded = load_model(io.BytesIO(model_bytes))
    require(np.array_equal(reloaded.predict(state["X_val_531"]),
                           state["selected_validation_predictions_545"].to_numpy()),
            "La serialización alteró las predicciones de validación.")

    base_dir = Path(state["BASE_DIR"])
    payloads, entries = {}, {}
    for name, frame in partitions.items():
        data = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        restored = pd.read_csv(io.BytesIO(data), encoding="utf-8", dtype=DTYPES,
                               keep_default_na=False)
        require(restored.equals(frame.reset_index(drop=True)),
                f"El formato CSV alteró datos de {name}.")
        relative_path = f"data/processed/{name}.csv"
        payloads[relative_path] = data
        total, negative, positive = EXPECTED[name]
        entries[name] = {"path": relative_path, "sha256": sha256(data), "records": total,
                         "class_counts": {"0": negative, "1": positive}}
    manifest_path = base_dir / "data/processed/metadata.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text("utf-8"))
        require(previous["partitions"] == entries,
                "Las particiones difieren del manifiesto existente; no se sobrescriben.")
    for name, entry in entries.items():
        existing = base_dir / entry["path"]
        if existing.exists():
            require(sha256(existing.read_bytes()) == entry["sha256"],
                    f"El archivo existente {name} difiere; no se sobrescribe.")

    generated_at = datetime.now(timezone.utc).isoformat()
    environment = {dist.metadata["Name"]: dist.version for dist in distributions()}
    configuration = {"config_id": "C1", "tfidf": TFIDF, "logistic_regression": LOGREG,
                     "vocabulary_size": 5653, "fit_partition": "train",
                     "training_records": 16186, "primary_metric": "f1_macro"}
    per_class = state["selected_per_class_552"].to_dict(orient="records")
    results = {"evaluation_partition": "validation", "records": 3468,
               "configuration": configuration, "metrics": metrics, "per_class": per_class,
               "confusion_matrix": {"labels": [0, 1], "rows": "real", "columns": "predicho",
                                    "values": expected_confusion.tolist()},
               "test_predictive_evaluation": False}
    zip_path = state["dataset_zip_path"]
    with zipfile.ZipFile(zip_path) as archive:
        source_csv_sha = sha256(archive.read(state["csv_file"]))
    model_path = "models/baseline_C1.joblib"
    model_metadata = {**configuration, "path": model_path, "sha256": sha256(model_bytes),
                      "generated_at_utc": generated_at, "python": platform.python_version(),
                      "scikit_learn": version("scikit-learn"), "joblib": version("joblib"),
                      "environment": environment,
                      "validation_results": "results/baseline_C1_validation.json"}
    metadata = {"schema_version": 1, "dataset": {"zenodo_record": "18797217",
                "doi": "10.5281/zenodo.18797217", "version": "1.0.1",
                "zip_filename": zip_path.name, "zip_sha256": sha256(zip_path.read_bytes()),
                "csv_member": state["csv_file"], "csv_sha256": source_csv_sha},
                "total_records": 23123, "random_state": 42, "label_map": LABEL_MAP,
                "split_proportions": {"train": 0.7, "validation": 0.15, "test": 0.15},
                "primary_metric": "f1_macro", "text_column": "comentario_limpio",
                "target_column": "sentimiento_id", "index_column": "indice_original",
                "predictors": ["comentario_limpio"], "columns": COLUMNS,
                "csv": {"encoding": "utf-8", "separator": ",", "keep_default_na": False},
                "partitions": entries, "generated_at_utc": generated_at,
                "python": platform.python_version(), "environment": environment,
                "source_notebook_code_sha256": sha256(json_bytes([
                    c["source"] for c in json.loads(NOTEBOOK.read_text("utf-8"))["cells"]
                    if c["cell_type"] == "code"]))}
    payloads[model_path] = model_bytes
    payloads["models/baseline_C1_metadata.json"] = json_bytes(model_metadata)
    payloads["results/baseline_C1_validation.json"] = json_bytes(results)
    # Todos los controles anteriores se completan antes de persistir las particiones.
    for relative_path, data in payloads.items():
        path = base_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest_path.write_bytes(json_bytes(metadata))
    validar_artefactos(base_dir)
    print(f"Fase 0: particiones, C1 y resultados verificados en {base_dir}.")


def validar_artefactos(base_dir=ROOT):
    base_dir = Path(base_dir)
    partitions = cargar_particiones(base_dir)
    metadata = json.loads((base_dir / "models/baseline_C1_metadata.json").read_text("utf-8"))
    model_bytes = (base_dir / "models/baseline_C1.joblib").read_bytes()
    require(sha256(model_bytes) == metadata["sha256"], "Checksum incorrecto del modelo.")
    require(metadata["fit_partition"] == "train" and metadata["training_records"] == 16186
            and metadata["config_id"] == "C1", "Procedencia distinta del modelo.")
    model = load_model(io.BytesIO(model_bytes))
    results = json.loads((base_dir / "results/baseline_C1_validation.json").read_text("utf-8"))
    require(results["evaluation_partition"] == "validation"
            and results["test_predictive_evaluation"] is False, "Evaluación distinta.")
    # Reproduce las predicciones de validación tras cargar; jamás predice sobre test.
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 f1_score, precision_score, recall_score)
    validation = partitions["validation"]
    labels = validation["sentimiento_id"]
    predictions = model.predict(validation["comentario_limpio"])
    computed = {"accuracy": accuracy_score(labels, predictions)}
    for name, metric in [("f1_macro", f1_score), ("precision_macro", precision_score),
                         ("recall_macro", recall_score)]:
        computed[name] = metric(labels, predictions, labels=[0, 1], average="macro", zero_division=0)
    require(all(np.isclose(computed[k], results["metrics"][k], rtol=0, atol=1e-12)
                for k in METRICS), "Métricas persistidas distintas de C1 cargado.")
    require(confusion_matrix(labels, predictions, labels=[0, 1]).tolist()
            == results["confusion_matrix"]["values"], "Matriz persistida distinta.")
    report = classification_report(labels, predictions, labels=[0, 1],
                                   target_names=["Negativo (0)", "Positivo (1)"],
                                   output_dict=True, zero_division=0)
    require(results["records"] == 3468 and len(results["per_class"]) == 2,
            "Soporte persistido distinto.")
    for row, label in zip(results["per_class"], ["Negativo (0)", "Positivo (1)"]):
        require(row["class_label"] == label, "Orden de clases persistido distinto.")
        for key, report_key in [("precision", "precision"), ("recall", "recall"),
                                ("f1", "f1-score"), ("support", "support")]:
            require(np.isclose(row[key], report[label][report_key], rtol=0, atol=1e-12),
                    f"Métrica por clase distinta: {label}/{key}.")
    return {name: len(frame) for name, frame in partitions.items()}


def ejecutar_notebook():
    """Ejecuta todas las celdas de código en orden, en un único espacio limpio.

    Conserva el archivo y sus salidas históricas. Los gráficos usan Agg y las llamadas
    explícitas a display usan IPython; las expresiones finales no se renderizan aquí.
    """
    import matplotlib
    matplotlib.use("Agg")
    from IPython.display import display
    notebook = json.loads(NOTEBOOK.read_text("utf-8"))
    code_cells = [(i, "".join(c["source"])) for i, c in enumerate(notebook["cells"])
                  if c["cell_type"] == "code"]
    # Revisar todas las celdas antes de efectuar descargas o entrenamiento.
    for i, code in code_cells:
        ast.parse(code, filename=f"{NOTEBOOK.name}:celda_{i}")
    os.chdir(ROOT)
    state = {"__name__": "__main__", "display": display}
    for i, code in code_cells:
        print(f"Ejecutando celda {i}", flush=True)
        exec(compile(code, f"{NOTEBOOK.name}:celda_{i}", "exec"), state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Cargar y verificar artefactos existentes.")
    parser.add_argument("--base-dir", type=Path, default=ROOT, help="Directorio de artefactos para --verify.")
    args = parser.parse_args()
    if args.verify:
        print(validar_artefactos(args.base_dir))
    else:
        ejecutar_notebook()
