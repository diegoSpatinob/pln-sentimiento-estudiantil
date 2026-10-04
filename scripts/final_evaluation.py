"""Fase 4A: inferencia final congelada. Importar o --help nunca predice.

Entradas independientes de development/validation; no se usa Trainer, fit,
model_init, resume ni selectores. Las operaciones predictivas requieren
--confirm-final-test y reservan un único intento por modelo/directorio.
"""

import argparse
import hashlib
import io
import json
import platform
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from scripts import fase0, transformer as development


ROOT = fase0.ROOT
EXPERIMENT_COMMIT = "90afa082979bbe396f5a6a4906cadee9c351dd4f"
WINNER = {
    "experiment_id": "T2", "learning_rate": 2e-5,
    "run_id": "RUN_265ff1dcbdb94a6d95f49abc552ee758",
    "checkpoint": "checkpoint-3036", "epoch": 3, "global_step": 3036,
    "model_id": "dccuchile/bert-base-spanish-wwm-cased",
    "model_revision": "c4d86612f51b4f46759c8390d1798c2febe71b93",
    "git_commit": EXPERIMENT_COMMIT,
}
CONFIG_SHA256 = "2bb202463f79de75f2f3afcd4b9d5300b16c9780a67a8dfcf7eb51050e85f29d"
DATA_METADATA_SHA256 = "92a24524587855f76d0fbcb19a4e5513efe329f708d762c96d61ec879eeefcf4"
PARTITION_HASHES = {
    "train": "01507f87ed4832452b40846146d1b61874799377a27cc5a8357486f71b3c2d98",
    "validation": "5756e5b98f53c7af0dfeb214ebba5e9b6961ff8708b69b73fe69ad6c8d4d093e",
    "test": "f1e7c5da7760aaf3ee0f89c3d86d1307db5a347cb1b543f551276b12e9c7851b",
}
BASELINE_SHA256 = "7654a99a32068fee9442d647adcd3b3b6969f1bb8feafe034147d71138f622c0"
BASELINE_METADATA_SHA256 = "6b0796976cd0272f7636acbd02c7f12d84414d327a76f10126b55829023295d9"
METRICS = ("f1_macro", "accuracy", "precision_macro", "recall_macro")
OUTPUT_NAMES = {"transformer_T2": "transformer_T2_test.json", "baseline_C1": "baseline_C1_test.json"}
require = fase0.require


def file_hash(path):
    """Hash incremental: no mantiene otra copia de los pesos en memoria."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_final_mode(confirmed, split="test"):
    require(confirmed is True, "La predicción final requiere --confirm-final-test.")
    require(split == "test", "Modo final rechaza train/validation; solo test está autorizado.")


@dataclass(frozen=True)
class FinalTestPartition:
    split: str
    texts: tuple
    labels: tuple
    indices: tuple
    sha256: str

    def __post_init__(self):
        require(self.split == "test", "Partición final debe ser test; train/validation rechazados.")
        require(len(self.texts) == len(self.labels) == len(self.indices) > 0, "Tamaños finales inconsistentes.")
        require(set(self.labels).issubset({0, 1}) and len(set(self.indices)) == len(self.indices),
                "Etiquetas o índices finales inválidos.")
        require(all(isinstance(text, str) and text.strip() for text in self.texts), "Textos finales inválidos.")


def guard_final_partition(partition):
    require(isinstance(partition, FinalTestPartition) and partition.split == "test"
            and partition.sha256 == PARTITION_HASHES["test"], "Solo test congelado en la interfaz final.")
    return partition


def frozen_data_metadata(base_dir=ROOT):
    path = Path(base_dir) / "data/processed/metadata.json"
    require(file_hash(path) == DATA_METADATA_SHA256, "Manifiesto de Fase 0 distinto al congelado.")
    metadata = read_json(path)
    require(metadata["text_column"] == "comentario_limpio"
            and metadata["target_column"] == "sentimiento_id"
            and metadata["label_map"] == fase0.LABEL_MAP and metadata["random_state"] == 42,
            "Interfaz metodológica distinta.")
    for split, expected_hash in PARTITION_HASHES.items():
        entry = metadata["partitions"][split]
        require(entry["path"] == f"data/processed/{split}.csv" and entry["sha256"] == expected_hash
                and entry["records"] == fase0.EXPECTED[split][0], "Particiones distintas a Fase 0.")
    return metadata


def load_final_test(base_dir=ROOT, split="test"):
    """Lee solamente test.csv; reutiliza esquema/tipos Fase 0 sin reconstruir splits."""
    require(split == "test", "Modo final rechaza train/validation.")
    metadata = frozen_data_metadata(base_dir)
    data = (Path(base_dir) / metadata["partitions"]["test"]["path"]).read_bytes()
    require(fase0.sha256(data) == PARTITION_HASHES["test"], "Checksum incorrecto del test congelado.")
    frame = pd.read_csv(io.BytesIO(data), encoding="utf-8", dtype=fase0.DTYPES, keep_default_na=False)
    require(list(frame.columns) == fase0.COLUMNS and len(frame) == fase0.EXPECTED["test"][0],
            "Esquema/tamaño de test distinto.")
    require(frame["sentimiento_id"].value_counts().to_dict() == {0: 1473, 1: 1996}
            and frame["sentimiento_id"].equals(frame["Valor"].map(fase0.LABEL_MAP)),
            "Clases de test distintas.")
    return FinalTestPartition("test", tuple(frame["comentario_limpio"]),
                              tuple(int(v) for v in frame["sentimiento_id"]),
                              tuple(int(v) for v in frame["indice_original"]), PARTITION_HASHES["test"])


def classification_metrics(labels, predictions):
    """Una misma implementación para C1/T2; argmax se realiza antes, sin umbrales."""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    require(labels.ndim == predictions.ndim == 1 and labels.size == predictions.size > 0
            and set(labels.tolist()).issubset({0, 1}) and set(predictions.tolist()).issubset({0, 1}),
            "Se requieren etiquetas/predicciones binarias no vacías.")
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], average=None, zero_division=0)
    return {
        "metrics": {"f1_macro": float(f1.mean()), "accuracy": float(accuracy_score(labels, predictions)),
                    "precision_macro": float(precision.mean()), "recall_macro": float(recall.mean())},
        "per_class": [{"class_id": i, "class_label": ("Negativo", "Positivo")[i],
                       "precision": float(precision[i]), "recall": float(recall[i]),
                       "f1": float(f1[i]), "support": int(support[i])} for i in (0, 1)],
        "confusion_matrix": {"labels": [0, 1], "rows": "real", "columns": "predicho",
                             "values": confusion_matrix(labels, predictions, labels=[0, 1]).tolist()},
    }


def verify_transformer_checkpoint(checkpoint_dir, base_dir=ROOT):
    """Verifica exclusivamente el ganador declarado; nunca compara épocas/métricas."""
    checkpoint = Path(checkpoint_dir).resolve()
    require(checkpoint.name == WINNER["checkpoint"] and checkpoint.parent.name == WINNER["run_id"],
            "Checkpoint distinto a T2/RUN congelado/checkpoint-3036; no cargar la raíz del RUN.")
    config_path = Path(base_dir) / "configs/transformer.json"
    require(file_hash(config_path) == CONFIG_SHA256, "Configuración distinta al commit experimental.")
    config = development.load_config(config_path)
    manifest_path = checkpoint.parent / "run_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest["experiment_id"] == WINNER["experiment_id"] and manifest["run_id"] == WINNER["run_id"]
            and manifest["learning_rate"] == WINNER["learning_rate"]
            and manifest["git_commit"] == EXPERIMENT_COMMIT and manifest["working_tree_clean"] is True
            and manifest["git"]["commit"] == EXPERIMENT_COMMIT and manifest["git"]["dirty"] is False
            and manifest["official"] is True and manifest["status"] == "COMPLETED"
            and manifest["test_used"] is False and manifest["resume_from_checkpoint"] is False,
            "Manifiesto ajeno al ganador congelado.")
    require(manifest["configuration"] == config and manifest["configuration_sha256"] == CONFIG_SHA256
            and manifest["initial_checkpoint"] == {"model_id": WINNER["model_id"],
                                                   "revision": WINNER["model_revision"]},
            "Modelo/revisión/configuración del ganador incorrectos.")
    frozen_data_metadata(base_dir)
    require(set(manifest["data"]) == {"train", "validation"}, "Procedencia de desarrollo distinta.")
    for split in ("train", "validation"):
        require(manifest["data"][split]["sha256"] == PARTITION_HASHES[split]
                and manifest["data"][split]["records"] == fase0.EXPECTED[split][0], "Hashes de desarrollo distintos.")
    state = read_json(checkpoint / "trainer_state.json")
    require(state["global_step"] == WINNER["global_step"] and state["epoch"] == WINNER["epoch"],
            "El checkpoint no corresponde al paso 3036/época 3.")
    model_config = read_json(checkpoint / "config.json")
    require(model_config["model_type"] == "bert"
            and model_config["id2label"] == {"0": "Negativo", "1": "Positivo"}
            and model_config["label2id"] == fase0.LABEL_MAP,
            "Configuración del clasificador distinta a BETO binario.")
    if model_config.get("_name_or_path"):
        require(model_config["_name_or_path"] == WINNER["model_id"], "Modelo original distinto.")
    files = ("model.safetensors", "config.json", "trainer_state.json", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")
    require(all((checkpoint / name).is_file() for name in files), "Checkpoint/tokenizer incompletos.")
    return {**WINNER, "checkpoint_path": str(checkpoint), "configuration_sha256": CONFIG_SHA256,
            "run_manifest_sha256": file_hash(manifest_path),
            "files_sha256": {name: file_hash(checkpoint / name) for name in files},
            "selection_source": "winner_frozen_before_test_by_user_and_validation"}


def load_transformer_for_inference(checkpoint_dir, device="cuda"):
    """Una carga local del checkpoint; rechaza pesos faltantes o incompatibles."""
    require(device in {"cpu", "cuda"}, "Dispositivo final debe ser cpu o cuda.")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model, info = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_dir), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, output_loading_info=True,
    )
    require(not any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
            "Carga incompleta: no se permite inicializar pesos nuevos.")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), local_files_only=True,
                                               trust_remote_code=False, use_fast=True)
    require(model.config.num_labels == 2, "El modelo final debe tener dos clases.")
    model.requires_grad_(False)
    model.to(device).eval()
    return model, tokenizer


def predict_transformer_test(model, tokenizer, partition, confirm_final_test=False, device="cuda"):
    """Una pasada: batch 16/max_length 128; CUDA fp16, CPU sin autocast."""
    require_final_mode(confirm_final_test)
    require(device in {"cpu", "cuda"}, "Dispositivo final debe ser cpu o cuda.")
    import torch

    guard_final_partition(partition)
    predictions = []
    model.eval()
    precision_context = (torch.autocast(device_type="cuda", dtype=torch.float16)
                         if device == "cuda" else nullcontext())
    with torch.inference_mode(), precision_context:
        for start in range(0, len(partition.texts), 16):
            inputs = tokenizer(list(partition.texts[start:start + 16]), max_length=128,
                               truncation=True, padding=True, return_tensors="pt").to(device)
            require(set(inputs).issubset({"input_ids", "attention_mask", "token_type_ids"}),
                    "Campos ajenos al predictor textual.")
            logits = model(**inputs).logits.detach().float().cpu().numpy()
            require(logits.shape == (min(16, len(partition.texts) - start), 2)
                    and np.isfinite(logits).all(), "Logits finales inválidos.")
            predictions.extend(logits.argmax(axis=-1).tolist())
    return np.asarray(predictions)


def load_frozen_baseline(base_dir=ROOT):
    metadata_path = Path(base_dir) / "models/baseline_C1_metadata.json"
    require(file_hash(metadata_path) == BASELINE_METADATA_SHA256, "Metadatos C1 distintos al congelado.")
    metadata = read_json(metadata_path)
    require(metadata["config_id"] == "C1" and metadata["path"] == "models/baseline_C1.joblib"
            and metadata["fit_partition"] == "train" and metadata["training_records"] == 16186
            and metadata["sha256"] == BASELINE_SHA256
            and metadata["tfidf"] == json.loads(fase0.json_bytes(fase0.TFIDF))
            and metadata["logistic_regression"] == fase0.LOGREG, "Metodología C1 distinta.")
    require(version("scikit-learn") == metadata["scikit_learn"] and version("joblib") == metadata["joblib"],
            "Usar el entorno de Fase 0 para cargar C1.")
    path = Path(base_dir) / metadata["path"]
    require(file_hash(path) == BASELINE_SHA256, "Checksum incorrecto del baseline C1.")
    model = fase0.load_model(path)
    return model, {"config_id": "C1", "path": metadata["path"], "sha256": BASELINE_SHA256,
                   "metadata_sha256": BASELINE_METADATA_SHA256, "fit_partition": "train"}


def output_path(model_key, base_dir=ROOT):
    return Path(base_dir) / "results/final" / OUTPUT_NAMES[model_key]


def ensure_new_evaluation(path):
    require(not Path(path).exists() and not Path(str(path) + ".started.json").exists(),
            "Evaluación ya realizada o iniciada; no repetir ni sobrescribir test automáticamente.")


def evaluate_once(model_key, partition, predict, model_metadata, runtime, base_dir=ROOT,
                  confirm_final_test=False):
    """Reserva exclusiva antes de predecir; conserva evidencia también ante fallos."""
    require_final_mode(confirm_final_test)
    guard_final_partition(partition)
    path = output_path(model_key, base_dir)
    ensure_new_evaluation(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = Path(str(path) + ".started.json")
    attempt = {"phase": "final_test_evaluation", "model_key": model_key, "status": "STARTED"}
    with marker.open("xb") as stream:
        stream.write(fase0.json_bytes(attempt))
    started = time.perf_counter()
    try:
        predictions = predict()  # Única llamada predictiva, sin ciclos de evaluación/selección.
        computed = classification_metrics(partition.labels, predictions)
        result = {"schema_version": 1, "phase": "final_test_evaluation", "evaluation_partition": "test",
                  "model_key": model_key, "test_used": True, "selection_frozen": True,
                  "training_performed": False, "reselection_performed": False, "predictive_evaluations": 1,
                  "predictors": ["comentario_limpio"], "target_column": "sentimiento_id",
                  "label_mapping": fase0.LABEL_MAP, "model": model_metadata,
                  "data": {"path": "data/processed/test.csv", "sha256": partition.sha256,
                           "metadata_sha256": DATA_METADATA_SHA256, "records": len(partition.labels)},
                  "runtime": runtime, "runtime_seconds": time.perf_counter() - started,
                  "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "evaluation_git": development.git_metadata(base_dir),
                  "evaluation_script_sha256": file_hash(Path(__file__)), **computed}
        with path.open("xb") as stream:
            stream.write(fase0.json_bytes(result))
        development._write_run_json(marker, {**attempt, "status": "COMPLETED"})
        return result
    except BaseException as error:
        development._write_run_json(marker, {**attempt, "status": "FAILED_OR_INTERRUPTED",
                                            "failure_type": type(error).__name__})
        raise


def run_transformer(checkpoint_dir, confirm_final_test=False, base_dir=ROOT, split="test", device="cuda"):
    require_final_mode(confirm_final_test, split)
    require(device in {"cpu", "cuda"}, "Dispositivo final debe ser cpu o cuda.")
    ensure_new_evaluation(output_path("transformer_T2", base_dir))
    provenance = verify_transformer_checkpoint(checkpoint_dir, base_dir)
    runtime = development.detect_runtime(device)
    precision = "fp16" if device == "cuda" else "fp32"
    require(runtime["torch_available"] and runtime["device"] == device
            and runtime["precision_support"][precision] and version("transformers") == "4.57.1",
            "PyTorch/Transformers 4.57.1 y precisión del dispositivo solicitados requeridos.")
    if device == "cuda":
        require(len(runtime["gpu"]) == 1, "CUDA final requiere una única GPU.")
    model, tokenizer = load_transformer_for_inference(provenance["checkpoint_path"], device=device)
    partition = load_final_test(base_dir, split)
    runtime = {**runtime, "precision": precision, "batch_size": 16, "max_length": 128}
    return evaluate_once("transformer_T2", partition,
                         lambda: predict_transformer_test(model, tokenizer, partition,
                                                          confirm_final_test=True, device=device),
                         provenance, runtime, base_dir, confirm_final_test=True)


def run_baseline(confirm_final_test=False, base_dir=ROOT, split="test"):
    require_final_mode(confirm_final_test, split)
    ensure_new_evaluation(output_path("baseline_C1", base_dir))
    model, provenance = load_frozen_baseline(base_dir)
    partition = load_final_test(base_dir, split)
    runtime = {"python": platform.python_version(), "scikit_learn": version("scikit-learn"),
               "joblib": version("joblib"), "numpy": version("numpy"), "device": "cpu"}
    return evaluate_once("baseline_C1", partition, lambda: model.predict(list(partition.texts)),
                         provenance, runtime, base_dir, confirm_final_test=True)


def validate_final_result(result, expected_model):
    require(result["phase"] == "final_test_evaluation" and result["evaluation_partition"] == "test"
            and result["model_key"] == expected_model and result["test_used"] is True
            and result["selection_frozen"] is True and result["training_performed"] is False
            and result["reselection_performed"] is False and result["predictive_evaluations"] == 1
            and result["predictors"] == ["comentario_limpio"] and result["label_mapping"] == fase0.LABEL_MAP,
            "Resultado ajeno a la evaluación final congelada.")
    require(result["data"] == {"path": "data/processed/test.csv", "sha256": PARTITION_HASHES["test"],
                               "metadata_sha256": DATA_METADATA_SHA256, "records": 3469},
            "Test distinto; no se pueden comparar resultados.")
    if expected_model == "transformer_T2":
        require(all(result["model"][key] == value for key, value in WINNER.items())
                and result["model"]["configuration_sha256"] == CONFIG_SHA256, "Transformer distinto al ganador.")
    else:
        require(result["model"]["config_id"] == "C1" and result["model"]["sha256"] == BASELINE_SHA256
                and result["model"]["metadata_sha256"] == BASELINE_METADATA_SHA256, "Baseline distinto a C1.")
    matrix = result["confusion_matrix"]
    values = np.asarray(matrix["values"])
    require(matrix["labels"] == [0, 1] and matrix["rows"] == "real" and matrix["columns"] == "predicho"
            and values.shape == (2, 2) and np.issubdtype(values.dtype, np.integer)
            and np.all(values >= 0) and values.sum() == 3469 and values.sum(axis=1).tolist() == [1473, 1996],
            "Matriz de confusión/orden/soportes incorrectos.")
    require(set(result["metrics"]) == set(METRICS)
            and all(np.isfinite(v) and 0 <= v <= 1 for v in result["metrics"].values())
            and [r["class_id"] for r in result["per_class"]] == [0, 1]
            and [r["support"] for r in result["per_class"]] == [1473, 1996], "Métricas finales inválidas.")
    diagonal = values.diagonal()
    column_support = values.sum(axis=0)
    precision = np.divide(diagonal, column_support, out=np.zeros(2), where=column_support != 0)
    recall = diagonal / values.sum(axis=1)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(2), where=precision + recall != 0)
    expected = {"f1_macro": f1.mean(), "accuracy": diagonal.sum() / values.sum(),
                "precision_macro": precision.mean(), "recall_macro": recall.mean()}
    require(all(np.isclose(result["metrics"][key], value, rtol=0, atol=1e-12)
                for key, value in expected.items()), "Métricas incompatibles con la matriz de confusión.")
    for i, row in enumerate(result["per_class"]):
        require(row["class_label"] == ("Negativo", "Positivo")[i]
                and all(np.isclose(row[key], value, rtol=0, atol=1e-12)
                        for key, value in (("precision", precision[i]), ("recall", recall[i]), ("f1", f1[i]))),
                "Métricas por clase incompatibles con la matriz.")
    return result


def compare_results(transformer_result=None, baseline_result=None, base_dir=ROOT):
    """Comparación descriptiva: lee dos JSON, sin datos, modelos ni selección."""
    path = Path(base_dir) / "results/final/comparison_test.json"
    require(not path.exists(), "Comparación final existente; no se sobrescribe.")
    transformer_path = Path(transformer_result) if transformer_result else output_path("transformer_T2", base_dir)
    baseline_path = Path(baseline_result) if baseline_result else output_path("baseline_C1", base_dir)
    transformer = validate_final_result(read_json(transformer_path), "transformer_T2")
    baseline = validate_final_result(read_json(baseline_path), "baseline_C1")
    require(transformer["data"] == baseline["data"], "Los modelos no usaron el mismo test congelado.")
    delta = {metric: transformer["metrics"][metric] - baseline["metrics"][metric] for metric in METRICS}
    better = ("transformer_T2" if delta["f1_macro"] > 0 else
              "baseline_C1" if delta["f1_macro"] < 0 else "tie")
    result = {"phase": "final_test_evaluation", "evaluation_partition": "test", "test_used": True,
              "comparison_is_descriptive_only": True, "reselection_performed": False,
              "models": {"transformer_T2": transformer, "baseline_C1": baseline},
              "delta_transformer_minus_baseline": delta, "better_f1_macro_on_test": better,
              "source_sha256": {"transformer": file_hash(transformer_path), "baseline": file_hash(baseline_path)}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(fase0.json_bytes(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="operation", required=True)
    for operation in ("transformer", "baseline", "compare"):
        command = commands.add_parser(operation, allow_abbrev=False)
        command.add_argument("--base-dir", type=Path, default=ROOT)
        if operation != "compare":
            command.add_argument("--confirm-final-test", action="store_true")
            command.add_argument("--split", choices=["test"], default="test")
        if operation == "transformer":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
        if operation == "compare":
            command.add_argument("--transformer-result", type=Path)
            command.add_argument("--baseline-result", type=Path)
    args = parser.parse_args(argv)
    if args.operation != "compare" and not args.confirm_final_test:
        parser.error("La predicción sobre test requiere --confirm-final-test.")
    if args.operation == "transformer":
        result = run_transformer(args.checkpoint, args.confirm_final_test, args.base_dir, args.split, args.device)
    elif args.operation == "baseline":
        result = run_baseline(args.confirm_final_test, args.base_dir, args.split)
    else:
        result = compare_results(args.transformer_result, args.baseline_result, args.base_dir)
    print(fase0.json_bytes(result).decode(), end="")


if __name__ == "__main__":
    main()
