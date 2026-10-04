"""Fase 4A: solo fixtures sintéticos en /tmp, sin test.csv real ni pesos BETO."""

import ast
import copy
import io
import json
import sys
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from scripts import final_evaluation as f


def synthetic_partition(count=3469):
    labels = ((0,) * 1473 + (1,) * 1996) if count == 3469 else tuple(i % 2 for i in range(count))
    return f.FinalTestPartition("test", tuple(f"UNIT_TEST_TEXT_{i}" for i in range(count)),
                                labels, tuple(range(count)), f.PARTITION_HASHES["test"])


def baseline_provenance():
    return {"config_id": "C1", "sha256": f.BASELINE_SHA256, "metadata_sha256": f.BASELINE_METADATA_SHA256}


def transformer_provenance():
    return {**f.WINNER, "configuration_sha256": f.CONFIG_SHA256}


class MetricsAndGuardsTests(unittest.TestCase):
    def test_known_confusion_same_metrics_for_both_models(self):
        result = f.classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(result["confusion_matrix"], {"labels": [0, 1], "rows": "real",
                                                    "columns": "predicho", "values": [[1, 1], [0, 2]]})
        self.assertAlmostEqual(result["metrics"]["f1_macro"], 11 / 15)
        self.assertAlmostEqual(result["metrics"]["accuracy"], 3 / 4)
        self.assertAlmostEqual(result["metrics"]["precision_macro"], 5 / 6)
        self.assertAlmostEqual(result["metrics"]["recall_macro"], 3 / 4)
        self.assertEqual([row["class_id"] for row in result["per_class"]], [0, 1])
        self.assertEqual([row["support"] for row in result["per_class"]], [2, 2])

    def test_missing_predicted_class_uses_zero_division_zero(self):
        result = f.classification_metrics([0, 1], [1, 1])
        self.assertEqual(result["per_class"][0]["precision"], 0)
        self.assertAlmostEqual(result["metrics"]["f1_macro"], 1 / 3)

    def test_invalid_predictions_rejected(self):
        for labels, predictions in (([], []), ([0], [2]), ([0], [np.nan]), ([0], [0, 1])):
            with self.assertRaises(ValueError):
                f.classification_metrics(labels, predictions)

    def test_final_interface_rejects_train_validation_before_reading_files(self):
        with patch.object(f, "frozen_data_metadata") as reader:
            for split in ("train", "validation"):
                with self.assertRaises(ValueError):
                    f.load_final_test(split=split)
                with self.assertRaises(ValueError):
                    f.run_baseline(True, split=split)
                with self.assertRaises(ValueError):
                    f.run_transformer("UNIT_TEST_PATH", True, split=split)
            reader.assert_not_called()
        for split in ("train", "validation"):
            with self.assertRaises(ValueError):
                f.FinalTestPartition(split, ("UNIT_TEST",), (0,), (0,), f.PARTITION_HASHES["test"])

    def test_no_confirmation_no_read_model_or_predict(self):
        with patch.object(f, "load_frozen_baseline") as baseline, \
             patch.object(f, "verify_transformer_checkpoint") as transformer:
            with self.assertRaises(ValueError):
                f.run_baseline()
            with self.assertRaises(ValueError):
                f.run_transformer("UNIT_TEST_PATH")
            baseline.assert_not_called()
            transformer.assert_not_called()

    def test_development_guard_still_rejects_test(self):
        with self.assertRaises(ValueError):
            f.development.guard_development_dataset(synthetic_partition(2), "test")

    def test_final_module_has_no_training_or_selection_calls(self):
        tree = ast.parse(Path(f.__file__).read_text())
        forbidden = {"train", "fit", "fit_transform", "partial_fit", "backward", "initialize_original_model",
                     "select_validation_candidate", "validation_selection_key", "load_selected_checkpoint",
                     "validate_official_result", "Trainer", "build_trainer", "set_seed"}
        calls = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Attribute, ast.Name))]
        self.assertFalse(forbidden.intersection(calls))


class FrozenArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        (self.base / "data/processed").mkdir(parents=True)
        (self.base / "models").mkdir()

    def test_only_synthetic_test_csv_read_without_train_or_validation(self):
        partition = synthetic_partition()
        frame = pd.DataFrame({
            "indice_original": partition.indices,
            "comentario": ["UNUSED_RAW_TEXT"] * 3469,
            "Valor": [("Negativo", "Positivo")[v] for v in partition.labels],
            "nombreAspecto": ["UNUSED_ASPECT"] * 3469,
            "comentario_limpio": partition.texts,
            "sentimiento_id": partition.labels,
        })[f.fase0.COLUMNS]
        test_path = self.base / "data/processed/test.csv"
        test_path.write_text(frame.to_csv(index=False), encoding="utf-8")
        digest = f.file_hash(test_path)
        metadata = f.read_json(f.ROOT / "data/processed/metadata.json")
        metadata["partitions"]["test"]["sha256"] = digest
        metadata_path = self.base / "data/processed/metadata.json"
        metadata_path.write_bytes(f.fase0.json_bytes(metadata))
        with patch.dict(f.PARTITION_HASHES, {"test": digest}), \
             patch.object(f, "DATA_METADATA_SHA256", f.file_hash(metadata_path)):
            loaded = f.load_final_test(self.base)
        self.assertEqual(loaded.texts, partition.texts)
        self.assertEqual(loaded.labels, partition.labels)
        self.assertEqual(loaded.indices, partition.indices)
        self.assertEqual(loaded.sha256, digest)
        self.assertFalse((self.base / "data/processed/train.csv").exists())
        self.assertFalse((self.base / "data/processed/validation.csv").exists())

    def test_wrong_test_checksum_rejected_before_parsing(self):
        (self.base / "data/processed/test.csv").write_text("UNIT_TEST_TAMPERED")
        metadata = {"partitions": {"test": {"path": "data/processed/test.csv"}}}
        with patch.object(f, "frozen_data_metadata", return_value=metadata), \
             patch.object(f.pd, "read_csv") as parse:
            with self.assertRaisesRegex(ValueError, "Checksum"):
                f.load_final_test(self.base)
            parse.assert_not_called()

    def test_original_baseline_metadata_verified_without_loading_real_model(self):
        metadata_path = self.base / "models/baseline_C1_metadata.json"
        metadata_path.write_bytes((f.ROOT / "models/baseline_C1_metadata.json").read_bytes())
        model_path = self.base / "models/baseline_C1.joblib"
        model_path.write_bytes(b"UNIT_TEST_MODEL_DOUBLE")
        original_hash = f.file_hash
        def hash_double(path):
            return f.BASELINE_SHA256 if Path(path) == model_path else original_hash(path)
        model = Mock()
        with patch.object(f, "file_hash", side_effect=hash_double), \
             patch.object(f.fase0, "load_model", return_value=model) as load:
            loaded, provenance = f.load_frozen_baseline(self.base)
        self.assertIs(loaded, model)
        load.assert_called_once_with(model_path)
        model.fit.assert_not_called()
        model.predict.assert_not_called()
        self.assertEqual(provenance["sha256"], f.BASELINE_SHA256)

    def test_baseline_metadata_weights_and_environment_mismatch_rejected(self):
        metadata_path = self.base / "models/baseline_C1_metadata.json"
        original = (f.ROOT / "models/baseline_C1_metadata.json").read_bytes()
        metadata_path.write_bytes(original)
        (self.base / "models/baseline_C1.joblib").write_bytes(b"UNIT_TEST_WRONG_MODEL")
        with patch.object(f.fase0, "load_model") as load:
            with self.assertRaisesRegex(ValueError, "Checksum"):
                f.load_frozen_baseline(self.base)
            with patch.object(f, "version", return_value="UNIT_TEST_WRONG_VERSION"):
                with self.assertRaisesRegex(ValueError, "entorno"):
                    f.load_frozen_baseline(self.base)
            metadata_path.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "Metadatos"):
                f.load_frozen_baseline(self.base)
            load.assert_not_called()


class CheckpointVerificationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        (self.base / "configs").mkdir()
        (self.base / "configs/transformer.json").write_bytes(f.development.CONFIG_PATH.read_bytes())
        config = f.development.load_config(self.base / "configs/transformer.json")
        self.checkpoint = self.base / "models/transformer/T2" / f.WINNER["run_id"] / f.WINNER["checkpoint"]
        self.checkpoint.mkdir(parents=True)
        self.manifest_path = self.checkpoint.parent / "run_manifest.json"
        self.manifest = {"experiment_id": "T2", "run_id": f.WINNER["run_id"], "learning_rate": 2e-5,
                         "git_commit": f.EXPERIMENT_COMMIT, "working_tree_clean": True,
                         "git": {"commit": f.EXPERIMENT_COMMIT, "dirty": False},
                         "official": True, "status": "COMPLETED", "test_used": False,
                         "resume_from_checkpoint": False, "configuration": config,
                         "configuration_sha256": f.CONFIG_SHA256,
                         "initial_checkpoint": {"model_id": f.WINNER["model_id"],
                                                "revision": f.WINNER["model_revision"]},
                         "data": {split: {"sha256": f.PARTITION_HASHES[split],
                                          "records": f.fase0.EXPECTED[split][0]}
                                  for split in ("train", "validation")}}
        self.manifest_path.write_bytes(f.fase0.json_bytes(self.manifest))
        (self.checkpoint / "trainer_state.json").write_text('{"global_step":3036,"epoch":3}')
        (self.checkpoint / "config.json").write_bytes(f.fase0.json_bytes({
            "model_type": "bert", "id2label": {"0": "Negativo", "1": "Positivo"},
            "label2id": f.fase0.LABEL_MAP, "_name_or_path": f.WINNER["model_id"]}))
        for name in ("model.safetensors", "tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json", "vocab.txt"):
            (self.checkpoint / name).write_bytes(b"UNIT_TEST_NOT_MODEL_WEIGHTS_OR_TOKENIZER")
        mocked = patch.object(f, "frozen_data_metadata", return_value={})
        mocked.start()
        self.addCleanup(mocked.stop)

    def test_exact_winner_verified_without_loading_any_model(self):
        with patch.object(f, "load_transformer_for_inference") as loader:
            provenance = f.verify_transformer_checkpoint(self.checkpoint, self.base)
            loader.assert_not_called()
        for key, value in f.WINNER.items():
            self.assertEqual(provenance[key], value)
        self.assertEqual(provenance["files_sha256"]["model.safetensors"], f.file_hash(self.checkpoint / "model.safetensors"))

    def test_checkpoint_earlier_epoch_other_run_and_run_root_rejected(self):
        for path in (self.checkpoint.with_name("checkpoint-2024"), self.checkpoint.parent,
                     self.checkpoint.parent.with_name("RUN_OTHER") / "checkpoint-3036"):
            with self.assertRaises(ValueError):
                f.verify_transformer_checkpoint(path, self.base)

    def test_manifest_experiment_run_lr_commit_revision_hashes_rejected(self):
        for key, value in (("experiment_id", "T1"), ("run_id", "RUN_OTHER"), ("learning_rate", 1e-5),
                           ("git_commit", "a" * 40), ("test_used", True), ("configuration_sha256", "b" * 64)):
            with self.subTest(key=key):
                changed = {**self.manifest, key: value}
                self.manifest_path.write_bytes(f.fase0.json_bytes(changed))
                with self.assertRaises(ValueError):
                    f.verify_transformer_checkpoint(self.checkpoint, self.base)
        for field in ("revision", "model_id"):
            changed = copy.deepcopy(self.manifest)
            changed["initial_checkpoint"][field] = "UNIT_TEST_WRONG"
            self.manifest_path.write_bytes(f.fase0.json_bytes(changed))
            with self.assertRaises(ValueError):
                f.verify_transformer_checkpoint(self.checkpoint, self.base)
        changed = copy.deepcopy(self.manifest)
        changed["data"]["train"]["sha256"] = "c" * 64
        self.manifest_path.write_bytes(f.fase0.json_bytes(changed))
        with self.assertRaises(ValueError):
            f.verify_transformer_checkpoint(self.checkpoint, self.base)

    def test_epoch_step_and_incomplete_checkpoint_rejected(self):
        for epoch, step in ((2, 3036), (3, 2024)):
            (self.checkpoint / "trainer_state.json").write_bytes(f.fase0.json_bytes({"epoch": epoch, "global_step": step}))
            with self.assertRaises(ValueError):
                f.verify_transformer_checkpoint(self.checkpoint, self.base)
        (self.checkpoint / "trainer_state.json").write_text('{"global_step":3036,"epoch":3}')
        (self.checkpoint / "model.safetensors").rename(self.checkpoint / "UNIT_TEST_MISSING")
        with self.assertRaises(ValueError):
            f.verify_transformer_checkpoint(self.checkpoint, self.base)

    def test_configuration_hash_change_rejected(self):
        (self.base / "configs/transformer.json").write_text("{}")
        with self.assertRaises(ValueError):
            f.verify_transformer_checkpoint(self.checkpoint, self.base)


class InferenceTests(unittest.TestCase):
    def test_only_selected_local_weights_loaded_and_gradients_disabled(self):
        model = Mock(config=SimpleNamespace(num_labels=2))
        model.to.return_value = model
        loader = Mock(return_value=(model, {}))
        tokenizer = Mock()
        module = SimpleNamespace(AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=loader),
                                 AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer))
        with patch.dict(sys.modules, {"transformers": module}):
            f.load_transformer_for_inference("UNIT_TEST_SELECTED_CHECKPOINT")
        loader.assert_called_once_with("UNIT_TEST_SELECTED_CHECKPOINT", local_files_only=True,
                                       trust_remote_code=False, use_safetensors=True, output_loading_info=True)
        model.requires_grad_.assert_called_once_with(False)
        model.to.assert_called_once_with("cuda")
        model.eval.assert_called_once()
        model.train.assert_not_called()

    def test_cpu_load_moves_only_selected_model_to_cpu(self):
        model = Mock(config=SimpleNamespace(num_labels=2))
        model.to.return_value = model
        loader = Mock(return_value=(model, {}))
        module = SimpleNamespace(AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=loader),
                                 AutoTokenizer=SimpleNamespace(from_pretrained=Mock()))
        with patch.dict(sys.modules, {"transformers": module}):
            f.load_transformer_for_inference("UNIT_TEST_SELECTED_CHECKPOINT", device="cpu")
        loader.assert_called_once_with("UNIT_TEST_SELECTED_CHECKPOINT", local_files_only=True,
                                       trust_remote_code=False, use_safetensors=True, output_loading_info=True)
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once()
        model.requires_grad_.assert_called_once_with(False)
        model.train.assert_not_called()

    def test_incomplete_weights_cannot_initialize_new_head(self):
        loader = Mock(return_value=(Mock(), {"missing_keys": ["classifier.weight"]}))
        tokenizer = Mock()
        module = SimpleNamespace(AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=loader),
                                 AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer))
        with patch.dict(sys.modules, {"transformers": module}), self.assertRaises(ValueError):
            f.load_transformer_for_inference("UNIT_TEST_SELECTED_CHECKPOINT")
        tokenizer.assert_not_called()

    def test_one_pass_no_gradients_text_only_and_dynamic_batches(self):
        self.check_one_pass("cuda")

    def test_cpu_one_pass_inference_mode_cpu_tensors_without_autocast(self):
        self.check_one_pass("cpu")

    def check_one_pass(self, device):
        partition = synthetic_partition(17)
        state = {"inference_active": False, "seen": [], "forwards": 0}

        @contextmanager
        def inference():
            state["inference_active"] = True
            yield
            state["inference_active"] = False

        @contextmanager
        def autocast(**kwargs):
            self.assertEqual(kwargs, {"device_type": "cuda", "dtype": "UNIT_TEST_FP16"})
            yield

        class Batch(dict):
            def to(batch, destination):
                self.assertEqual(destination, device)
                return batch

        def tokenize(texts, **kwargs):
            state["seen"].extend(texts)
            self.assertEqual(kwargs, {"max_length": 128, "truncation": True,
                                      "padding": True, "return_tensors": "pt"})
            return Batch(input_ids=np.zeros((len(texts), 2)), attention_mask=np.ones((len(texts), 2)))

        def forward(**inputs):
            self.assertTrue(state["inference_active"])
            self.assertEqual(set(inputs), {"input_ids", "attention_mask"})
            state["forwards"] += 1
            logits = Mock()
            logits.detach.return_value.float.return_value.cpu.return_value.numpy.return_value = np.tile([1., 0.], (len(inputs["input_ids"]), 1))
            return SimpleNamespace(logits=logits)

        model = Mock(side_effect=forward)
        cpu_autocast = Mock(side_effect=AssertionError("CPU must not use autocast"))
        torch = (SimpleNamespace(inference_mode=inference, autocast=autocast, float16="UNIT_TEST_FP16")
                 if device == "cuda" else SimpleNamespace(inference_mode=inference, autocast=cpu_autocast))
        with patch.dict(sys.modules, {"torch": torch}):
            predictions = f.predict_transformer_test(model, tokenize, partition, True, device=device)
        cpu_autocast.assert_not_called()
        self.assertEqual(state["seen"], list(partition.texts))
        self.assertEqual(state["forwards"], 2)
        self.assertEqual(predictions.tolist(), [0] * 17)
        model.train.assert_not_called()


class FinalOperationsTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.partition = synthetic_partition()
        mocked = patch.object(f.development, "git_metadata", return_value={"commit": "UNIT_TEST_COMMIT", "dirty": False})
        mocked.start()
        self.addCleanup(mocked.stop)

    def evaluate(self, key, predictions):
        predictor = Mock(return_value=predictions)
        result = f.evaluate_once(key, self.partition, predictor,
                                 transformer_provenance() if key == "transformer_T2" else baseline_provenance(),
                                 {"device": "UNIT_TEST"}, self.base, confirm_final_test=True)
        predictor.assert_called_once_with()
        return result

    def test_baseline_predict_once_text_only_never_fit(self):
        model = Mock()
        model.predict.return_value = self.partition.labels
        with patch.object(f, "load_frozen_baseline", return_value=(model, baseline_provenance())), \
             patch.object(f, "load_final_test", return_value=self.partition) as load:
            result = f.run_baseline(True, self.base)
        load.assert_called_once_with(self.base, "test")
        model.predict.assert_called_once_with(list(self.partition.texts))
        model.fit.assert_not_called()
        model.fit_transform.assert_not_called()
        self.assertEqual(result["predictive_evaluations"], 1)
        self.assertFalse(result["training_performed"])
        self.assertEqual(result["confusion_matrix"]["labels"], [0, 1])

    def test_transformer_predict_once_and_shared_metrics(self):
        runtime = {"torch_available": True, "device": "cuda", "gpu": [{}], "precision_support": {"fp16": True}}
        model, tokenizer = Mock(), Mock()
        with patch.object(f, "verify_transformer_checkpoint", return_value={**transformer_provenance(), "checkpoint_path": "UNIT_TEST_CP"}), \
             patch.object(f.development, "detect_runtime", return_value=runtime), \
             patch.object(f, "version", return_value="4.57.1"), \
             patch.object(f, "load_transformer_for_inference", return_value=(model, tokenizer)) as load_model, \
             patch.object(f, "load_final_test", return_value=self.partition) as load_test, \
             patch.object(f, "predict_transformer_test", return_value=self.partition.labels) as predict:
            result = f.run_transformer("UNIT_TEST_EXPLICIT_CP", True, self.base)
        load_model.assert_called_once_with("UNIT_TEST_CP", device="cuda")
        load_test.assert_called_once_with(self.base, "test")
        predict.assert_called_once_with(model, tokenizer, self.partition, confirm_final_test=True, device="cuda")
        self.assertEqual(result["metrics"], f.classification_metrics(self.partition.labels, self.partition.labels)["metrics"])
        self.assertEqual(result["runtime"]["device"], "cuda")
        self.assertEqual(result["runtime"]["precision"], "fp16")

    def test_cpu_final_operation_records_device_fp32_and_frozen_winner(self):
        runtime = {"torch_available": True, "device": "cpu", "gpu": [], "precision_support": {"fp32": True}}
        model, tokenizer = Mock(), Mock()
        with patch.object(f, "verify_transformer_checkpoint", return_value={**transformer_provenance(), "checkpoint_path": "UNIT_TEST_CP"}), \
             patch.object(f.development, "detect_runtime", return_value=runtime) as detect, \
             patch.object(f, "version", return_value="4.57.1"), \
             patch.object(f, "load_transformer_for_inference", return_value=(model, tokenizer)) as load_model, \
             patch.object(f, "load_final_test", return_value=self.partition) as load_test, \
             patch.object(f, "predict_transformer_test", return_value=self.partition.labels) as predict:
            result = f.run_transformer("UNIT_TEST_EXPLICIT_CP", True, self.base, device="cpu")
        detect.assert_called_once_with("cpu")
        load_model.assert_called_once_with("UNIT_TEST_CP", device="cpu")
        load_test.assert_called_once_with(self.base, "test")
        predict.assert_called_once_with(model, tokenizer, self.partition, confirm_final_test=True, device="cpu")
        saved = f.read_json(f.output_path("transformer_T2", self.base))
        self.assertEqual(saved, result)
        self.assertEqual(saved["runtime"]["device"], "cpu")
        self.assertEqual(saved["runtime"]["precision"], "fp32")
        for key, value in f.WINNER.items():
            self.assertEqual(saved["model"][key], value)
        self.assertFalse(saved["training_performed"])
        self.assertFalse(saved["reselection_performed"])

    def test_rerun_cannot_predict_again_or_overwrite(self):
        self.evaluate("baseline_C1", self.partition.labels)
        with patch.object(f, "load_frozen_baseline") as loader:
            with self.assertRaises(ValueError):
                f.run_baseline(True, self.base)
            loader.assert_not_called()

    def test_failed_attempt_blocks_automatic_retry(self):
        predictor = Mock(side_effect=RuntimeError("UNIT_TEST_FAILED_FORWARD"))
        with self.assertRaises(RuntimeError):
            f.evaluate_once("baseline_C1", self.partition, predictor, baseline_provenance(), {}, self.base, True)
        marker = Path(str(f.output_path("baseline_C1", self.base)) + ".started.json")
        self.assertEqual(f.read_json(marker)["status"], "FAILED_OR_INTERRUPTED")
        self.assertFalse(f.output_path("baseline_C1", self.base).exists())
        with self.assertRaises(ValueError):
            f.evaluate_once("baseline_C1", self.partition, predictor, baseline_provenance(), {}, self.base, True)
        self.assertEqual(predictor.call_count, 1)

    def test_compare_correct_deltas_and_descriptive_winner_without_prediction(self):
        transformer = self.evaluate("transformer_T2", self.partition.labels)
        predictions = list(self.partition.labels)
        predictions[:100] = [1] * 100
        baseline = self.evaluate("baseline_C1", predictions)
        with patch.object(f, "load_final_test") as data, patch.object(f, "load_frozen_baseline") as model, \
             patch.object(f, "predict_transformer_test") as predict:
            result = f.compare_results(base_dir=self.base)
            data.assert_not_called()
            model.assert_not_called()
            predict.assert_not_called()
        self.assertEqual(result["better_f1_macro_on_test"], "transformer_T2")
        for metric in f.METRICS:
            self.assertAlmostEqual(result["delta_transformer_minus_baseline"][metric],
                                   transformer["metrics"][metric] - baseline["metrics"][metric])
        self.assertFalse(result["reselection_performed"])
        self.assertTrue((self.base / "results/final/comparison_test.json").exists())
        with self.assertRaises(ValueError):
            f.compare_results(base_dir=self.base)

    def test_compare_reports_tie_and_baseline_better(self):
        for better in ("tie", "baseline_C1"):
            with self.subTest(better=better), TemporaryDirectory(dir="/tmp") as temporary:
                self.base = Path(temporary)
                predictions = list(self.partition.labels)
                if better == "baseline_C1":
                    predictions[:100] = [1] * 100
                self.evaluate("transformer_T2", predictions)
                self.evaluate("baseline_C1", self.partition.labels)
                self.assertEqual(f.compare_results(base_dir=self.base)["better_f1_macro_on_test"], better)

    def test_compare_rejects_wrong_test_validation_and_altered_metrics(self):
        transformer = self.evaluate("transformer_T2", self.partition.labels)
        self.evaluate("baseline_C1", self.partition.labels)
        path = f.output_path("transformer_T2", self.base)
        for kind in ("test_hash", "validation", "metric", "checkpoint", "labels"):
            changed = copy.deepcopy(transformer)
            if kind == "test_hash":
                changed["data"]["sha256"] = "a" * 64
            elif kind == "validation":
                changed["evaluation_partition"] = "validation"
            elif kind == "metric":
                changed["metrics"]["f1_macro"] = 0.5
            elif kind == "checkpoint":
                changed["model"]["checkpoint"] = "checkpoint-2024"
            else:
                changed["confusion_matrix"]["labels"] = [1, 0]
            path.write_bytes(f.fase0.json_bytes(changed))
            with self.assertRaises(ValueError):
                f.compare_results(base_dir=self.base)
            self.assertFalse((self.base / "results/final/comparison_test.json").exists())


class CLITests(unittest.TestCase):
    def test_prediction_cli_requires_confirmation_and_rejects_development_splits(self):
        with patch.object(f, "run_transformer") as transformer, patch.object(f, "run_baseline") as baseline, \
             redirect_stderr(io.StringIO()):
            for argv in (["baseline"], ["transformer", "--checkpoint", "UNIT_TEST_CP"],
                         ["transformer", "--checkpoint", "UNIT_TEST_CP", "--device", "cpu"],
                         ["transformer", "--checkpoint", "UNIT_TEST_CP", "--confirm-final-test", "--device", "auto"],
                         ["baseline", "--confirm-final-test", "--split", "train"],
                         ["transformer", "--checkpoint", "UNIT_TEST_CP", "--confirm-final-test", "--split", "validation"],
                         ["baseline", "--confirm-final-t"], ["baseline", "--confirm-final-test", "--learning-rate", "1e-5"]):
                with self.assertRaises(SystemExit):
                    f.main(argv)
            transformer.assert_not_called()
            baseline.assert_not_called()

    def test_three_operations_route_only_to_doubles(self):
        with patch.object(f, "run_transformer", return_value={}) as transformer, \
             patch.object(f, "run_baseline", return_value={}) as baseline, \
             patch.object(f, "compare_results", return_value={}) as compare, redirect_stdout(io.StringIO()):
            f.main(["transformer", "--checkpoint", "UNIT_TEST_CP", "--confirm-final-test"])
            transformer.assert_called_once_with(Path("UNIT_TEST_CP"), True, f.ROOT, "test", "cuda")
            transformer.reset_mock()
            f.main(["transformer", "--checkpoint", "UNIT_TEST_CP", "--confirm-final-test", "--device", "cpu"])
            transformer.assert_called_once_with(Path("UNIT_TEST_CP"), True, f.ROOT, "test", "cpu")
            f.main(["baseline", "--confirm-final-test"])
            baseline.assert_called_once_with(True, f.ROOT, "test")
            f.main(["compare"])
            compare.assert_called_once_with(None, None, f.ROOT)


if __name__ == "__main__":
    unittest.main()
