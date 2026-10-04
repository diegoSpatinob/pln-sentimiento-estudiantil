"""Infraestructura BETO: preanálisis (1C) y preparación del fine-tuning (2A).

Desde la raíz, reproducir sin cambiar la configuración:
    python -m scripts.transformer
Registrar explícitamente la longitud derivada de train:
    python -m scripts.transformer --record-max-length
El modo predeterminado solo descarga el tokenizer. --smoke-test requiere una
invocación explícita y dependencias de entrenamiento. No hay CLI para T1/T2/T3.
Importar este módulo nunca inicializa ni entrena un modelo.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import re
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np

from scripts.fase0 import cargar_particiones


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/transformer.json"
ANALYSIS_PATH = ROOT / "results/transformer/token_length_analysis.json"
MODEL_ID = "dccuchile/bert-base-spanish-wwm-cased"
EXPECTED_SIZES = {"train": 16186, "validation": 3468, "test": 3469}
CANDIDATES = [64, 128, 256, 512]
SELECTION_RULE = {
    "partition": "validation",
    "order": ["f1_macro_desc", "accuracy_desc", "learning_rate_asc", "epoch_asc"],
    "f1_macro_tie_decimals": 4,
}
FIXED = {
    "model_id": MODEL_ID, "task": "binary_sequence_classification", "num_labels": 2,
    "text_column": "comentario_limpio", "target_column": "sentimiento_id",
    "label_mapping": {"Negativo": 0, "Positivo": 1}, "seed": 42,
    "primary_metric": "f1_macro", "max_epochs": 3, "weight_decay": 0.01,
    "learning_rates": [1e-5, 2e-5, 3e-5], "effective_batch_size": 16,
    "evaluation_strategy": "epoch", "selection_rule": SELECTION_RULE,
    "test_policy": "reserved_for_common_final_phase_no_development_evaluation",
    "max_length_rule": {"candidates": CANDIDATES, "minimum_coverage": 0.99, "split": "train"},
    "padding_strategy": "dynamic_per_batch_DataCollatorWithPadding",
    "checkpoints_dir": "models/transformer/", "results_dir": "results/transformer/",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config):
    """Falla si falta una decisión requerida o contradice el protocolo acordado."""
    required = set(FIXED) | {"model_revision", "max_length", "max_length_status",
                            "physical_batch_size", "gradient_accumulation_steps",
                            "precision", "training_device"}
    missing = sorted(required - config.keys())
    _require(not missing, f"Faltan campos metodológicos requeridos: {missing}")
    for key, expected in FIXED.items():
        _require(config[key] == expected, f"Configuración metodológica distinta: {key}")
    _require(isinstance(config["model_revision"], str)
             and re.fullmatch(r"[0-9a-f]{40}", config["model_revision"]),
             "model_revision debe ser un commit completo de Hugging Face.")
    length = config["max_length"]
    _require(length is None or (type(length) is int and length in CANDIDATES),
             "max_length debe ser null o un candidato acordado.")
    expected_status = ("pending_train_tokenizer_analysis" if length is None
                       else "selected_from_train_99_percent_coverage")
    _require(config["max_length_status"] == expected_status,
             "Estado de max_length inconsistente.")
    for key in ("physical_batch_size", "gradient_accumulation_steps"):
        value = config[key]
        _require(value is None or (type(value) is int and value > 0), f"{key} inválido.")
    return config


def load_config(path=CONFIG_PATH):
    """Carga JSON local; no escribe archivos ni ejecuta el análisis."""
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def set_seed(seed=42):
    """Reproducibilidad básica, sin prometer determinismo total ni requerir GPU."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        return {"seed": seed, "python_random": True, "numpy": True, "torch": False}
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {"seed": seed, "python_random": True, "numpy": True, "torch": True}


def load_verified_partitions(base_dir=ROOT):
    """Usa exclusivamente Fase 0. No cambia limpieza, splits ni DataFrames.

    La lectura de test solo realiza controles de integridad de la interfaz común.
    indice_original se conserva como índice/columna únicamente para trazabilidad.
    """
    partitions = cargar_particiones(base_dir=base_dir)
    _require(set(partitions) == set(EXPECTED_SIZES), "Faltan particiones comunes.")
    for name, expected in EXPECTED_SIZES.items():
        frame = partitions[name]
        _require(len(frame) == expected, f"Tamaño incorrecto de {name}: {len(frame)}")
        _require({"comentario_limpio", "sentimiento_id"}.issubset(frame.columns),
                 f"Columnas obligatorias ausentes en {name}.")
        _require(set(frame["sentimiento_id"].unique()) == {0, 1},
                 f"Etiquetas incorrectas en {name}.")
    return partitions


def load_tokenizer(config, base_dir=ROOT, local_files_only=False):
    """Carga solo AutoTokenizer, con revisión fija y sin código remoto."""
    validate_config(config)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"], use_fast=True,
        trust_remote_code=False, local_files_only=local_files_only,
        cache_dir=str(Path(base_dir) / "results/transformer/tokenizer_cache"),
    )


def tokenizer_metadata(tokenizer, config):
    """Metadatos descriptivos; no carga pesos de ningún modelo."""
    return {
        "model_id": config["model_id"], "tokenizer_id": tokenizer.name_or_path,
        "revision": config["model_revision"],
        "class": type(tokenizer).__name__, "is_fast": tokenizer.is_fast,
        "model_max_length": int(tokenizer.model_max_length),
        "vocab_size": int(tokenizer.vocab_size), "vocab_with_added_tokens": len(tokenizer),
        "special_tokens": tokenizer.special_tokens_map,
        "special_token_ids": dict(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)),
        "special_tokens_per_single_sequence": tokenizer.num_special_tokens_to_add(pair=False),
        "versions": {package: version(package) for package in
                     ("transformers", "tokenizers", "huggingface-hub", "numpy")},
    }


def select_max_length(lengths):
    """Menor candidato con >=99 %; 512 si ninguno alcanza el criterio.

    La comparación de enteros evita errores de redondeo en el umbral del 99 %.
    Una secuencia exactamente igual al límite está cubierta.
    """
    values = np.asarray(lengths)
    _require(values.ndim == 1 and values.size > 0
             and np.issubdtype(values.dtype, np.integer) and np.all(values > 0),
             "Se requieren longitudes enteras positivas no vacías.")
    count = int(values.size)
    coverage = []
    selected = None
    for limit in CANDIDATES:
        exceeds = int(np.count_nonzero(values > limit))
        covered = count - exceeds
        coverage.append({"candidate": limit, "exceeds_count": exceeds,
                         "exceeds_percent": 100 * exceeds / count,
                         "covered_percent": 100 * covered / count})
        if selected is None and covered * 100 >= count * 99:
            selected = limit
    met = selected is not None
    selected = selected if met else 512
    row = next(row for row in coverage if row["candidate"] == selected)
    return {"coverage": coverage, "max_length": selected,
            "coverage_target_met": met, "selected_truncation_percent": row["exceeds_percent"]}


def _config_fingerprint(config):
    # Excluye únicamente la decisión derivada, para reproducir antes/después de registrarla.
    protocol = {key: value for key, value in config.items()
                if key not in {"max_length", "max_length_status"}}
    return hashlib.sha256(_json_bytes(protocol)).hexdigest()


def analyze_train_lengths(tokenizer, config, base_dir=ROOT, batch_size=128):
    """Carga datos oficiales y tokeniza EXCLUSIVAMENTE train, sin padding/truncamiento.

    No acepta una partición externa: evita sustituir train por validation o test.
    Los lotes solo limitan memoria de tokenización; no son batches de entrenamiento.
    No devuelve textos, etiquetas, predicciones ni longitudes por observación.
    """
    validate_config(config)
    _require(type(batch_size) is int and batch_size > 0, "batch_size debe ser positivo.")
    _require(tokenizer.name_or_path == config["model_id"], "Tokenizer ajeno al modelo acordado.")
    train = load_verified_partitions(base_dir)["train"]
    texts = train[config["text_column"]]
    lengths = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts.iloc[start:start + batch_size].tolist(), add_special_tokens=True,
            truncation=False, padding=False, return_attention_mask=False,
            return_token_type_ids=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    _require(len(lengths) == EXPECTED_SIZES["train"], "Número de secuencias inesperado.")
    values = np.asarray(lengths, dtype=np.int64)
    percentiles = np.percentile(values, [50, 90, 95, 99], method="linear")
    statistics = {"min": int(values.min()), "mean": float(values.mean()),
                  "median": float(percentiles[0]), "p90": float(percentiles[1]),
                  "p95": float(percentiles[2]), "p99": float(percentiles[3]),
                  "max": int(values.max())}
    return {
        "model_id": config["model_id"], "tokenizer": tokenizer_metadata(tokenizer, config),
        "split": "train", "n_observations": len(lengths), "statistics": statistics,
        **select_max_length(values),
        "criterion": config["max_length_rule"], "add_special_tokens": True,
        "truncation": False, "padding": False, "percentile_method": "linear",
        "data_reference": {
            "train_path": "data/processed/train.csv",
            "train_sha256": _file_hash(Path(base_dir) / "data/processed/train.csv"),
            "metadata_path": "data/processed/metadata.json",
            "metadata_sha256": _file_hash(Path(base_dir) / "data/processed/metadata.json"),
        },
        "configuration_reference": {
            "path": "configs/transformer.json", "protocol_sha256": _config_fingerprint(config),
            "excluded_derived_fields": ["max_length", "max_length_status"],
        },
    }


def record_max_length(analysis, config_path=CONFIG_PATH, base_dir=ROOT):
    """ÚNICA actualización de configuración, que requiere llamada explícita.

    Comprueba modelo, protocolo y hashes de datos para no registrar un análisis ajeno.
    """
    config = load_config(config_path)
    _require(analysis["split"] == "train"
             and analysis["n_observations"] == EXPECTED_SIZES["train"]
             and analysis["criterion"] == config["max_length_rule"]
             and analysis["model_id"] == config["model_id"]
             and analysis["tokenizer"]["revision"] == config["model_revision"]
             and analysis["configuration_reference"]["protocol_sha256"] == _config_fingerprint(config),
             "El análisis no corresponde a train o a la configuración actual.")
    rows = analysis["coverage"]
    count = analysis["n_observations"]
    _require([row["candidate"] for row in rows] == CANDIDATES,
             "La cobertura no contiene los candidatos acordados.")
    previous = count
    selected = None
    for row in rows:
        exceeds = row["exceeds_count"]
        _require(type(exceeds) is int and 0 <= exceeds <= previous,
                 "Recuentos de cobertura inconsistentes.")
        _require(np.isclose(row["exceeds_percent"], 100 * exceeds / count, rtol=0, atol=1e-12)
                 and np.isclose(row["covered_percent"], 100 * (count - exceeds) / count,
                                rtol=0, atol=1e-12), "Porcentajes de cobertura inconsistentes.")
        if selected is None and (count - exceeds) * 100 >= count * 99:
            selected = row["candidate"]
        previous = exceeds
    met = selected is not None
    selected = selected if met else 512
    selected_row = next(row for row in rows if row["candidate"] == selected)
    _require(analysis["max_length"] == selected and analysis["coverage_target_met"] is met
             and analysis["selected_truncation_percent"] == selected_row["exceeds_percent"],
             "La selección no cumple la regla de cobertura del 99 % de train.")
    for key, relative in (("train_sha256", "data/processed/train.csv"),
                          ("metadata_sha256", "data/processed/metadata.json")):
        _require(analysis["data_reference"][key] == _file_hash(Path(base_dir) / relative),
                 "Los datos cambiaron desde el análisis.")
    config["max_length"] = analysis["max_length"]
    config["max_length_status"] = "selected_from_train_99_percent_coverage"
    validate_config(config)
    Path(config_path).write_bytes(_json_bytes(config))
    return config


def save_analysis(analysis, path=ANALYSIS_PATH):
    """Persiste únicamente el resumen descriptivo, bajo results/transformer/."""
    path = Path(path).resolve()
    _require(path.is_relative_to((ROOT / "results/transformer").resolve()),
             "El resumen debe permanecer bajo results/transformer/.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(analysis))
    return path


def load_training_frames(base_dir=ROOT):
    """Solo devuelve train/validation; test se verifica por Fase 0 y se descarta."""
    partitions = load_verified_partitions(base_dir)
    return {name: partitions[name] for name in ("train", "validation")}


def _smoke_frame(frame, seed):
    """16 registros por clase en train o 8 en validation; muestreo determinista."""
    per_class = 16 if len(frame) == EXPECTED_SIZES["train"] else 8
    rng = np.random.default_rng(seed)
    indices = []
    for label in (0, 1):
        eligible = frame.loc[frame["sentimiento_id"].eq(label), "indice_original"].to_numpy()
        indices.extend(rng.choice(eligible, size=per_class, replace=False).tolist())
    rng.shuffle(indices)
    return frame.loc[indices]


@dataclass(frozen=True)
class DevelopmentDataset:
    """Dataset de estilo map para DataLoader; no necesita torch al prepararse.

    Los registros solo contienen entradas del tokenizer y labels. Los índices y
    la procedencia se guardan fuera de __getitem__, nunca se entregan al modelo.
    Las tuplas internas y las copias devueltas evitan mutaciones accidentales.
    """

    split: str
    indice_original: tuple
    features: tuple
    labels: tuple
    source_sha256: str
    model_id: str
    model_revision: str
    max_length: int
    smoke: bool

    def __post_init__(self):
        _require(self.split in {"train", "validation"}, "test no es un dataset de desarrollo.")
        _require(len(self.features) == len(self.labels) == len(self.indice_original)
                 and len(self.labels) > 0, "Dataset vacío o de tamaños inconsistentes.")
        _require(len(set(self.indice_original)) == len(self.indice_original), "Índices repetidos.")
        _require(set(self.labels).issubset({0, 1}), "Etiquetas inválidas.")
        _require(self.max_length == 128, "El entrenamiento requiere max_length=128.")
        for record in self.features:
            fields = dict(record)
            _require({"input_ids", "attention_mask"}.issubset(fields)
                     and set(fields).issubset({"input_ids", "attention_mask", "token_type_ids"}),
                     "Campos ajenos a las entradas predictivas del tokenizer.")
            length = len(fields["input_ids"])
            _require(0 < length <= 128 and all(len(value) == length for value in fields.values()),
                     "Longitudes tokenizadas inconsistentes.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {name: list(value) for name, value in self.features[index]}
        item["labels"] = self.labels[index]
        return item


def prepare_training_datasets(tokenizer, config, base_dir=ROOT, smoke=False, batch_size=128):
    """Convierte solamente train/validation oficiales; no acepta un split externo.

    Tokenización con truncamiento a 128 y sin padding fijo. El padding posterior
    lo realiza DataCollatorWithPadding por batch. No modifica los DataFrames.
    """
    validate_config(config)
    _require(config["max_length"] == 128, "Fase 1C debe haber registrado max_length=128.")
    _require(tokenizer.name_or_path == config["model_id"], "Tokenizer no corresponde a BETO.")
    _require(type(smoke) is bool and type(batch_size) is int and batch_size > 0,
             "Modo o batch de tokenización inválido.")
    datasets = {}
    for name, original in load_training_frames(base_dir).items():
        frame = _smoke_frame(original, config["seed"]) if smoke else original
        features = []
        texts = frame[config["text_column"]]
        for start in range(0, len(frame), batch_size):
            encoded = tokenizer(texts.iloc[start:start + batch_size].tolist(),
                                max_length=128, truncation=True, padding=False,
                                add_special_tokens=True, return_attention_mask=True)
            _require(set(encoded).issubset({"input_ids", "attention_mask", "token_type_ids"}),
                     "Tokenizer devolvió campos no autorizados.")
            for i in range(len(encoded["input_ids"])):
                features.append(tuple((key, tuple(values[i])) for key, values in encoded.items()))
        datasets[name] = DevelopmentDataset(
            split=name, indice_original=tuple(int(i) for i in frame["indice_original"]),
            features=tuple(features), labels=tuple(int(i) for i in frame[config["target_column"]]),
            source_sha256=_file_hash(Path(base_dir) / f"data/processed/{name}.csv"),
            model_id=config["model_id"], model_revision=config["model_revision"],
            max_length=128, smoke=smoke,
        )
    return datasets


def guard_development_dataset(dataset, expected_split, base_dir=ROOT):
    """Rechaza test, datasets externos y registros atribuidos al split equivocado."""
    _require(expected_split in {"train", "validation"}, "test está reservado para la fase final.")
    _require(isinstance(dataset, DevelopmentDataset) and dataset.split == expected_split,
             f"Solo se permite el dataset verificado de {expected_split}; test está prohibido.")
    frames = load_training_frames(base_dir)
    reference = frames[expected_split]
    if dataset.smoke:
        reference = _smoke_frame(reference, 42)
    _require(dataset.indice_original == tuple(int(i) for i in reference["indice_original"])
             and dataset.labels == tuple(int(i) for i in reference["sentimiento_id"]),
             "Procedencia o etiquetas distintas de la partición autorizada.")
    config = load_config(Path(base_dir) / "configs/transformer.json")
    _require(dataset.source_sha256 == _file_hash(Path(base_dir) / f"data/processed/{expected_split}.csv")
             and dataset.model_id == config["model_id"]
             and dataset.model_revision == config["model_revision"] and dataset.max_length == 128,
             "Dataset preparado con datos o tokenizer distintos.")
    return dataset


def compute_metrics(eval_prediction, include_per_class=True):
    """Métricas binarias, compatibles con EvalPrediction; admite pruebas sintéticas."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    if hasattr(eval_prediction, "predictions"):
        logits, labels = eval_prediction.predictions, eval_prediction.label_ids
    else:
        logits, labels = eval_prediction
    logits = np.asarray(logits[0] if isinstance(logits, tuple) else logits)
    labels = np.asarray(labels)
    _require(logits.shape == (labels.size, 2) and labels.ndim == 1 and labels.size > 0
             and np.isfinite(logits).all() and set(labels.tolist()).issubset({0, 1}),
             "Se requieren logits finitos (n,2) y etiquetas binarias (n,).")
    predicted = logits.argmax(axis=-1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=[0, 1], average=None, zero_division=0)
    metrics = {"f1_macro": float(f1.mean()), "accuracy": float(accuracy_score(labels, predicted)),
               "precision_macro": float(precision.mean()), "recall_macro": float(recall.mean())}
    if include_per_class:
        for i in (0, 1):
            metrics.update({f"precision_class_{i}": float(precision[i]),
                            f"recall_class_{i}": float(recall[i]), f"f1_class_{i}": float(f1[i]),
                            f"support_class_{i}": int(support[i])})
    return metrics


def detect_runtime(device="auto"):
    """Detecta CPU/CUDA y registra versiones/capacidades sin entrenar ni escribir."""
    _require(device in {"auto", "cpu", "cuda"}, "Dispositivo debe ser auto, cpu o cuda.")
    info = {"python": platform.python_version(), "transformers": version("transformers"),
            "torch_available": False, "torch": None, "cuda_runtime": None,
            "cuda_available": False, "gpu": [], "device": None,
            "precision_support": {"fp32": False, "fp16": False, "bf16": False}}
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        return info
    available = torch.cuda.is_available() and torch.version.cuda is not None
    _require(device != "cuda" or available, "CUDA solicitado pero no disponible.")
    chosen = "cuda" if device == "cuda" or (device == "auto" and available) else "cpu"
    info.update(torch_available=True, torch=torch.__version__, cuda_runtime=torch.version.cuda,
                cuda_available=available, device=chosen)
    if available:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpu"].append({"index": i, "name": props.name,
                                "memory_bytes": props.total_memory,
                                "compute_capability": [props.major, props.minor]})
    info["precision_support"]["fp32"] = True
    if chosen == "cuda":
        capability = torch.cuda.get_device_capability(torch.cuda.current_device())
        info["precision_support"].update(fp16=capability >= (5, 3),
                                         bf16=torch.cuda.is_bf16_supported(including_emulation=False))
    return info


def _require_training_backend():
    """Dependencias diferidas: el preanálisis sigue funcionando sin PyTorch."""
    from importlib.metadata import PackageNotFoundError
    from packaging.version import Version

    try:
        torch_version, accelerate_version = version("torch"), version("accelerate")
    except PackageNotFoundError as error:
        raise RuntimeError("Entrenamiento requiere PyTorch y Accelerate; instalar en Colab "
                           "según requirements-transformer.txt y su entorno CPU/GPU.") from error
    _require(Version("2.6") <= Version(torch_version) < Version("3"),
             "BETO original en formato .bin requiere torch>=2.6,<3 con Transformers 4.57.1.")
    _require(Version(accelerate_version) >= Version("0.26"), "Accelerate demasiado antiguo.")
    _require(version("transformers") == "4.57.1", "Validar de nuevo la API si cambia Transformers.")


def initialize_original_model(config, base_dir=ROOT, local_files_only=False):
    """Cada inicialización parte de BETO original; nunca de T1/T2/T3 ni del smoke."""
    validate_config(config)
    registered = load_config(Path(base_dir) / "configs/transformer.json")
    _require(config["model_revision"] == registered["model_revision"],
             "La revisión debe coincidir con el checkpoint original registrado.")
    _require_training_backend()
    set_seed(config["seed"])
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        config["model_id"], revision=config["model_revision"], num_labels=2,
        id2label={value: key for key, value in config["label_mapping"].items()},
        label2id=config["label_mapping"].copy(), trust_remote_code=False,
        local_files_only=local_files_only,
        cache_dir=str(Path(base_dir) / "models/transformer/pretrained_cache"),
    )


def training_argument_values(config, experiment_id, learning_rate, physical_batch_size,
                             precision, runtime, base_dir=ROOT, smoke=False):
    """Plan de argumentos sin inicializar Trainer/modelo ni seleccionar hiperparámetros.

    Batch y precisión deben indicarse explícitamente después de inspeccionar Colab.
    Se admite un solo proceso/dispositivo para garantizar batch efectivo 16.
    """
    validate_config(config)
    _require(config["max_length"] == 128, "max_length de entrenamiento debe ser 128.")
    _require(learning_rate in config["learning_rates"], "Learning rate fuera del protocolo.")
    _require(type(physical_batch_size) is int and physical_batch_size in {8, 16},
             "Batch físico debe ser 8 o 16.")
    _require(precision in {"fp32", "fp16", "bf16"}, "Indicar explícitamente la precisión.")
    _require(runtime["torch_available"] and runtime["device"] in {"cpu", "cuda"},
             "PyTorch/dispositivo no disponible.")
    _require(runtime["precision_support"][precision], "Precisión no soportada por el dispositivo.")
    _require(int(os.environ.get("WORLD_SIZE", "1")) == 1
             and (runtime["device"] != "cuda" or len(runtime["gpu"]) == 1),
             "Usar un único proceso/GPU visible para conservar batch efectivo 16.")
    if smoke:
        _require(re.fullmatch(r"SMOKE_[A-Za-z0-9_-]+", experiment_id), "ID smoke debe empezar por SMOKE_.")
        relative = Path("models/transformer/smoke") / experiment_id
    else:
        _require(experiment_id in {"T1", "T2", "T3"}, "ID oficial debe ser T1, T2 o T3.")
        relative = Path("models/transformer") / experiment_id
    return {
        "output_dir": str(Path(base_dir) / relative), "run_name": experiment_id,
        "learning_rate": learning_rate, "weight_decay": config["weight_decay"],
        "num_train_epochs": 1 if smoke else config["max_epochs"], "max_steps": 2 if smoke else -1,
        "per_device_train_batch_size": physical_batch_size,
        "per_device_eval_batch_size": physical_batch_size,
        "gradient_accumulation_steps": 16 // physical_batch_size,
        "eval_strategy": "epoch", "save_strategy": "epoch", "logging_strategy": "epoch",
        "load_best_model_at_end": True, "metric_for_best_model": "f1_macro",
        "greater_is_better": True, "save_total_limit": 2, "save_safetensors": True,
        "seed": config["seed"], "data_seed": config["seed"],
        "fp16": precision == "fp16", "bf16": precision == "bf16",
        "use_cpu": runtime["device"] == "cpu", "optim": "adamw_torch",
        "lr_scheduler_type": "linear", "warmup_steps": 0,
        "dataloader_num_workers": 0, "dataloader_pin_memory": runtime["device"] == "cuda",
        "remove_unused_columns": False, "label_names": ["labels"],
        "report_to": "none", "push_to_hub": False,
    }


def build_training_arguments(*args, **kwargs):
    """Construye TrainingArguments con la API comprobada de Transformers 4.57.1."""
    values = training_argument_values(*args, **kwargs)
    _require_training_backend()
    from transformers import TrainingArguments

    arguments = TrainingArguments(**values)
    _require(arguments.world_size == 1 and arguments.n_gpu <= 1,
             "El batch efectivo 16 requiere un único proceso/dispositivo.")
    return arguments


def development_trainer_class(base_dir=ROOT):
    """Trainer protegido; incluso sus métodos públicos rechazan test y reanudación."""
    _require_training_backend()
    from transformers import Trainer

    class DevelopmentTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            guard_development_dataset(kwargs.get("train_dataset"), "train", base_dir)
            guard_development_dataset(kwargs.get("eval_dataset"), "validation", base_dir)
            self._started = False
            self._best_validation_key = None
            super().__init__(*args, **kwargs)

        def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
            _require(resume_from_checkpoint is None or resume_from_checkpoint is False,
                     "No se reutilizan checkpoints entre experimentos.")
            _require(not self._started and trial is None, "Crear un Trainer nuevo para cada ejecución.")
            guard_development_dataset(self.train_dataset, "train", base_dir)
            guard_development_dataset(self.eval_dataset, "validation", base_dir)
            _require(not any(Path(self.args.output_dir).glob("checkpoint-*")),
                     "El directorio ya contiene checkpoints; no se reutiliza.")
            self._started = True
            return super().train(resume_from_checkpoint=False, trial=None,
                                 ignore_keys_for_eval=ignore_keys_for_eval, **kwargs)

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            dataset = self.eval_dataset if eval_dataset is None else eval_dataset
            guard_development_dataset(dataset, "validation", base_dir)
            return super().evaluate(dataset, ignore_keys, metric_key_prefix)

        def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="validation"):
            # test_dataset es el nombre de la API HF; aquí solo se admite validation.
            guard_development_dataset(test_dataset, "validation", base_dir)
            return super().predict(test_dataset, ignore_keys, metric_key_prefix)

        def get_train_dataloader(self):
            guard_development_dataset(self.train_dataset, "train", base_dir)
            return super().get_train_dataloader()

        def get_eval_dataloader(self, eval_dataset=None):
            dataset = self.eval_dataset if eval_dataset is None else eval_dataset
            guard_development_dataset(dataset, "validation", base_dir)
            return super().get_eval_dataloader(dataset)

        def get_test_dataloader(self, test_dataset):
            guard_development_dataset(test_dataset, "validation", base_dir)
            return super().get_test_dataloader(test_dataset)

        def hyperparameter_search(self, *args, **kwargs):
            raise ValueError("La selección entre T1/T2/T3 corresponde a una fase posterior.")

        def _determine_best_metric(self, metrics, trial):
            # API interna comprobada en 4.57.1. Dentro del run LR es constante;
            # empates en F1 a 4 decimales se resuelven por accuracy y época temprana.
            f1, accuracy = float(metrics["eval_f1_macro"]), float(metrics["eval_accuracy"])
            _require(np.isfinite([f1, accuracy]).all(), "Métricas no finitas.")
            key = (round(f1, 4), accuracy)
            if self._best_validation_key is None or key > self._best_validation_key:
                self._best_validation_key = key
                self.state.best_metric = f1
                self.state.best_global_step = self.state.global_step
                return True
            return False

    return DevelopmentTrainer


def build_trainer(config, tokenizer, datasets, experiment_id, learning_rate,
                  physical_batch_size, precision, device="auto", base_dir=ROOT,
                  smoke=False, local_files_only=False):
    """Preparación explícita para ejecución posterior; esta función no llama train().

    Trainer inicializa el modelo al construirse: invocar solo en el entorno futuro.
    Los directorios deben ser nuevos. No acepta modelos/checkpoints alternativos.
    """
    # Fija una copia del protocolo para model_init y el manifiesto: un cambio en
    # el diccionario del llamador no puede alterar inicializaciones posteriores.
    config = json.loads(_json_bytes(config))
    _require(set(datasets) == {"train", "validation"}, "Solo train/validation están autorizados.")
    for name, dataset in datasets.items():
        guard_development_dataset(dataset, name, base_dir)
        _require(dataset.smoke is smoke, "No mezclar datasets smoke y oficiales.")
    runtime = detect_runtime(device)
    values = training_argument_values(config, experiment_id, learning_rate, physical_batch_size,
                                      precision, runtime, base_dir, smoke)
    output = Path(values["output_dir"])
    _require(not output.exists(), "Directorio de experimento ya existente; no se sobrescribe ni reanuda.")
    arguments = build_training_arguments(config, experiment_id, learning_rate, physical_batch_size,
                                         precision, runtime, base_dir, smoke)
    trainer_type = development_trainer_class(base_dir)
    from transformers import DataCollatorWithPadding

    output.mkdir(parents=True, exist_ok=False)
    trainer = trainer_type(
        args=arguments, model_init=lambda: initialize_original_model(config, base_dir, local_files_only),
        train_dataset=datasets["train"], eval_dataset=datasets["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt"),
        processing_class=tokenizer, compute_metrics=compute_metrics,
    )
    manifest = {
        "status": "PREPARED_NO_TRAINING_RESULTS", "official": not smoke,
        "purpose": "SMOKE_TEST_NO_OFICIAL" if smoke else "FUTURE_EXPERIMENT",
        "experiment_id": experiment_id, "configuration": config, "runtime": runtime,
        "accelerate": version("accelerate"), "training_arguments": arguments.to_dict(),
        "initial_checkpoint": {"model_id": config["model_id"], "revision": config["model_revision"]},
        "data": {name: {"records": len(dataset), "sha256": dataset.source_sha256,
                         "indice_original": list(dataset.indice_original)}
                 for name, dataset in datasets.items()},
        "test_policy": config["test_policy"],
    }
    (output / "run_manifest.json").write_bytes(_json_bytes(manifest))
    return trainer


def verify_smoke_save_reload(trainer, reloaded, batch, diagnostics_path):
    """Verifica serialización; no cambia la precisión del entrenamiento/evaluación.

    Invocar al final del smoke con su batch de validation. Accelerate envuelve
    forward con autocast aunque eval() esté activo; algunos contenedores de
    salida conservan logits BF16/fp16 en lugar de recibir la conversión a fp32.
    Se mide esa comparación previa y después se retira el wrapper mediante la
    API pública: ambos forwards finales usan fp32, eval y los mismos tensores.
    La retirada es definitiva en este Trainer, que no se reutiliza tras el smoke.
    Solo se persisten diferencias agregadas, nunca logits ni predicciones.
    """
    import torch

    diagnostics_path = Path(diagnostics_path)
    diagnostics = {"purpose": "SMOKE_TEST_NO_OFICIAL", "save_reload_verified": False,
                   "verification_precision": "fp32", "device": str(trainer.args.device),
                   "rtol": 1e-4, "atol": 1e-4}
    diagnostics_path.write_bytes(_json_bytes(diagnostics))
    trainer.model.to(trainer.args.device).eval()
    reloaded.to(trainer.args.device).eval()
    batch = {key: value.to(trainer.args.device) for key, value in batch.items()}

    original_state, reloaded_state = trainer.model.state_dict(), reloaded.state_dict()
    different_keys = sorted(set(original_state) ^ set(reloaded_state))
    unequal_tensors = [key for key in sorted(set(original_state) & set(reloaded_state))
                       if original_state[key].shape != reloaded_state[key].shape
                       or original_state[key].dtype != reloaded_state[key].dtype
                       or not torch.equal(original_state[key].detach().cpu(),
                                          reloaded_state[key].detach().cpu())]
    floating_dtypes = sorted({str(value.dtype) for state in (original_state, reloaded_state)
                             for value in state.values() if value.is_floating_point()})
    diagnostics.update(state_dict_verified=not different_keys and not unequal_tensors,
                       state_dict_tensor_count=len(original_state),
                       different_state_keys=different_keys, unequal_state_tensors=unequal_tensors,
                       floating_state_dtypes=floating_dtypes)
    diagnostics_path.write_bytes(_json_bytes(diagnostics))

    def compare_logits(original, restored):
        # allclose exige dtypes compatibles. Comparar copias fp32 sin alterar
        # los outputs AMP originales; conservar sus dtypes en el diagnóstico.
        original_fp32 = original.detach().to(dtype=torch.float32, copy=True)
        restored_fp32 = restored.detach().to(dtype=torch.float32, copy=True)
        finite = bool(torch.isfinite(original_fp32).all().item()
                      and torch.isfinite(restored_fp32).all().item())
        difference = (original_fp32 - restored_fp32).abs() if finite else None
        return {"max_abs_diff": difference.max().item() if finite else None,
                "mean_abs_diff": difference.mean().item() if finite else None,
                "finite": finite, "original_dtype": str(original.dtype),
                "reloaded_dtype": str(restored.dtype),
                "comparison_dtype": "torch.float32",
                "allclose": bool(torch.allclose(original_fp32, restored_fp32, rtol=1e-4, atol=1e-4))}

    # Diagnóstico del fallo anterior: el wrapper interno puede activar autocast
    # incluso dentro de este contexto externo con autocast desactivado.
    with torch.no_grad(), torch.autocast(device_type=trainer.args.device.type, enabled=False):
        wrapped_output = trainer.model(**batch)
        reloaded_output = reloaded(**batch)
    diagnostics["before_removing_amp_wrapper"] = compare_logits(wrapped_output.logits,
                                                                reloaded_output.logits)
    original = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False,
                                               keep_torch_compile=False)
    original.to(trainer.args.device).eval()
    with torch.no_grad(), torch.autocast(device_type=trainer.args.device.type, enabled=False):
        original_output = original(**batch)
        reloaded_output = reloaded(**batch)
    diagnostics["equivalent_fp32_forwards"] = compare_logits(original_output.logits,
                                                             reloaded_output.logits)
    diagnostics["finite_losses"] = bool(torch.isfinite(original_output.loss).item()
                                         and torch.isfinite(reloaded_output.loss).item())
    diagnostics["save_reload_verified"] = bool(
        diagnostics["state_dict_verified"] and floating_dtypes == ["torch.float32"]
        and diagnostics["finite_losses"] and diagnostics["equivalent_fp32_forwards"]["finite"]
        and diagnostics["equivalent_fp32_forwards"]["original_dtype"]
            == diagnostics["equivalent_fp32_forwards"]["reloaded_dtype"] == "torch.float32"
        and diagnostics["equivalent_fp32_forwards"]["allclose"])
    # Se conserva evidencia también cuando la comprobación falla.
    diagnostics_path.write_bytes(_json_bytes(diagnostics))
    print("SMOKE_SAVE_RELOAD_DIAGNOSTICS " + json.dumps(diagnostics, allow_nan=False), flush=True)
    _require(diagnostics["save_reload_verified"],
             f"La recarga técnica no conserva el estado/forward de validation; ver {diagnostics_path}.")
    return diagnostics


def run_smoke_test(learning_rate, physical_batch_size, precision, device="auto",
                   base_dir=ROOT, run_id=None, local_files_only=False):
    """Único ejecutor de esta fase: 32 train/16 validation, 2 pasos, NO OFICIAL.

    Ejecuta forward/backward, validación, checkpoint y recarga local de su propio
    resultado técnico. No altera configuración ni realiza selección T1/T2/T3.
    """
    _require_training_backend()
    config = load_config(Path(base_dir) / "configs/transformer.json")
    runtime = detect_runtime(device)
    run_id = run_id or f"SMOKE_{uuid4().hex[:12]}"
    training_argument_values(config, run_id, learning_rate, physical_batch_size,
                             precision, runtime, base_dir, smoke=True)
    result_dir = Path(base_dir) / "results/transformer/smoke" / run_id
    _require(not result_dir.exists(), "Resultados de smoke ya existentes.")
    set_seed(config["seed"])
    tokenizer = load_tokenizer(config, base_dir, local_files_only)
    datasets = prepare_training_datasets(tokenizer, config, base_dir, smoke=True)
    trainer = build_trainer(config, tokenizer, datasets, run_id, learning_rate,
                            physical_batch_size, precision, device, base_dir,
                            smoke=True, local_files_only=local_files_only)
    import torch

    initial_head = {key: value.detach().cpu().clone()
                    for key, value in trainer.model.classifier.state_dict().items()}
    outcome = trainer.train(resume_from_checkpoint=False)
    _require(any(not torch.equal(value, trainer.model.classifier.state_dict()[key].detach().cpu())
                 for key, value in initial_head.items()),
             "El smoke no verificó una actualización efectiva de la cabeza de clasificación.")
    metrics = trainer.evaluate()
    checkpoint = trainer.state.best_model_checkpoint
    _require(checkpoint is not None and Path(checkpoint).is_relative_to(Path(trainer.args.output_dir)),
             "No se guardó un mejor checkpoint dentro de este smoke.")
    _require(outcome.global_step == 2, "El smoke debe ejecutar exactamente dos pasos de optimización.")
    saved = Path(trainer.args.output_dir) / "smoke_saved_model"
    trainer.save_model(str(saved))
    from transformers import AutoModelForSequenceClassification

    # Única excepción a la inicialización original: probar la recarga del smoke
    # guardado. Este modelo se descarta y jamás inicializa un experimento oficial.
    reloaded = AutoModelForSequenceClassification.from_pretrained(str(saved), local_files_only=True)
    sample = [datasets["validation"][i] for i in range(min(2, len(datasets["validation"])))]
    reload_diagnostics = verify_smoke_save_reload(
        trainer, reloaded, trainer.data_collator(sample),
        Path(trainer.args.output_dir) / "save_reload_diagnostics_NO_OFICIAL.json")
    result = {"purpose": "SMOKE_TEST_NO_OFICIAL", "official": False,
              "not_for_comparison_or_hyperparameter_selection": True,
              "run_id": run_id, "train_records": 32, "validation_records": 16,
              "optimizer_steps": outcome.global_step, "validation_metrics_non_official": metrics,
              "checkpoint": str(checkpoint), "parameter_update_verified": True,
              "save_reload_verified": reload_diagnostics["save_reload_verified"],
              "save_reload_diagnostics": reload_diagnostics,
              "test_used": False, "runtime": runtime}
    result_dir.mkdir(parents=True, exist_ok=False)
    (result_dir / "smoke_test_NO_OFICIAL.json").write_bytes(_json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-max-length", action="store_true",
                        help="Actualiza explícitamente configs/transformer.json después del análisis.")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Usa únicamente el tokenizer ya descargado en la caché local.")
    parser.add_argument("--smoke-test", action="store_true", help="Ejecuta el smoke técnico NO OFICIAL.")
    parser.add_argument("--learning-rate", type=float, choices=[1e-5, 2e-5, 3e-5])
    parser.add_argument("--physical-batch-size", type=int, choices=[8, 16])
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.smoke_test:
        if args.record_max_length:
            parser.error("El smoke no modifica configs/transformer.json.")
        if any(value is None for value in (args.learning_rate, args.physical_batch_size, args.precision)):
            parser.error("Smoke requiere --learning-rate, --physical-batch-size y --precision explícitos.")
        result = run_smoke_test(args.learning_rate, args.physical_batch_size, args.precision,
                                args.device, local_files_only=args.local_files_only)
        print(_json_bytes(result).decode("utf-8"), end="")
        return
    if any(value is not None for value in (args.learning_rate, args.physical_batch_size, args.precision)):
        parser.error("Los argumentos de entrenamiento requieren --smoke-test.")
    config = load_config()
    set_seed(config["seed"])
    tokenizer = load_tokenizer(config, local_files_only=args.local_files_only)
    analysis = analyze_train_lengths(tokenizer, config)
    if args.record_max_length:
        record_max_length(analysis)
    save_analysis(analysis)
    print(_json_bytes(analysis).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
