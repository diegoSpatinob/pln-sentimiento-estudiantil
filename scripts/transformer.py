"""Fase 1C: configuración, datos verificados y longitudes de train; sin entrenamiento.

Desde la raíz, reproducir sin cambiar la configuración:
    python -m scripts.transformer
Registrar explícitamente la longitud derivada de train:
    python -m scripts.transformer --record-max-length
Solo se descargan archivos del tokenizer, en una caché ignorada por Git.
"""

import argparse
import hashlib
import json
import random
import re
from importlib.metadata import version
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-max-length", action="store_true",
                        help="Actualiza explícitamente configs/transformer.json después del análisis.")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Usa únicamente el tokenizer ya descargado en la caché local.")
    args = parser.parse_args()
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
