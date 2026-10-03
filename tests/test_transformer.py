"""Controles Fase 2A sin PyTorch, descargas, entrenamiento ni logits del corpus.

Ejecutar desde la raíz: python -m unittest discover -s tests -v
Las comprobaciones de datos locales se omiten si las particiones no están disponibles.
Los dobles de Trainer/modelo solo prueban contratos, no equivalen al smoke real.
"""

import ast
import hashlib
import json
import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import transformer as t


def cpu_runtime():
    return {"torch_available": True, "device": "cpu", "gpu": [],
            "precision_support": {"fp32": True, "fp16": False, "bf16": False}}


class ConfigurationAndMetricsTests(unittest.TestCase):
    def setUp(self):
        self.config = t.load_config()

    def test_metrics_known_confusion_and_tuple_logits(self):
        labels = np.array([0, 0, 1, 1])
        logits = np.array([[2., 0.], [0., 2.], [0., 2.], [0., 2.]])
        metrics = t.compute_metrics((logits, labels))
        self.assertAlmostEqual(metrics["f1_macro"], 11 / 15)
        self.assertAlmostEqual(metrics["accuracy"], 3 / 4)
        self.assertAlmostEqual(metrics["precision_macro"], 5 / 6)
        self.assertAlmostEqual(metrics["recall_macro"], 3 / 4)
        self.assertAlmostEqual(metrics["f1_class_0"], 2 / 3)
        self.assertAlmostEqual(metrics["f1_class_1"], 4 / 5)
        self.assertEqual(metrics["support_class_0"], 2)
        self.assertEqual(metrics["support_class_1"], 2)
        prediction = SimpleNamespace(predictions=(logits, np.zeros(4)), label_ids=labels)
        self.assertEqual(metrics, t.compute_metrics(prediction))
        self.assertEqual(len(t.compute_metrics((logits, labels), include_per_class=False)), 4)

    def test_missing_predicted_class_and_invalid_inputs(self):
        metrics = t.compute_metrics((np.array([[0., 1.], [0., 1.]]), np.array([0, 1])))
        self.assertAlmostEqual(metrics["f1_macro"], 1 / 3)
        self.assertEqual(metrics["precision_class_0"], 0)
        for logits, labels in ((np.zeros((0, 2)), np.array([])),
                               (np.zeros((2, 3)), np.array([0, 1])),
                               (np.array([[np.nan, 0]]), np.array([0])),
                               (np.zeros((1, 2)), np.array([2]))):
            with self.assertRaises(ValueError):
                t.compute_metrics((logits, labels))

    def test_batch_checkpoint_api_and_distinct_directories(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            plans = []
            for experiment, learning_rate in zip(("T1", "T2", "T3"), self.config["learning_rates"]):
                plans.append(t.training_argument_values(self.config, experiment, learning_rate,
                                                        16, "fp32", cpu_runtime()))
            self.assertEqual(len({p["output_dir"] for p in plans}), 3)
            for physical, accumulation in ((16, 1), (8, 2)):
                plan = t.training_argument_values(self.config, "T1", 1e-5, physical,
                                                 "fp32", cpu_runtime())
                self.assertEqual(plan["gradient_accumulation_steps"], accumulation)
                self.assertEqual(physical * accumulation, 16)
                self.assertEqual(plan["eval_strategy"], "epoch")
                self.assertEqual(plan["save_strategy"], "epoch")
                self.assertTrue(plan["load_best_model_at_end"])
                self.assertEqual(plan["metric_for_best_model"], "f1_macro")
                self.assertTrue(plan["greater_is_better"])
                self.assertEqual(plan["save_total_limit"], 2)
                self.assertNotIn("evaluation_strategy", plan)
            smoke = t.training_argument_values(self.config, "SMOKE_contract", 1e-5,
                                              8, "fp32", cpu_runtime(), smoke=True)
            self.assertEqual(smoke["max_steps"], 2)
            self.assertEqual(smoke["num_train_epochs"], 1)
            self.assertIn("/smoke/", smoke["output_dir"])

    def test_precision_and_world_size_are_checked(self):
        for precision in ("fp16", "bf16", None):
            with self.assertRaises(ValueError):
                t.training_argument_values(self.config, "T1", 1e-5, 16, precision, cpu_runtime())
        with self.assertRaises(ValueError):
            t.training_argument_values(self.config, "T1", 1e-5, 4, "fp32", cpu_runtime())
        with patch.dict(os.environ, {"WORLD_SIZE": "2"}), self.assertRaises(ValueError):
            t.training_argument_values(self.config, "T1", 1e-5, 16, "fp32", cpu_runtime())
        gpu = {"torch_available": True, "device": "cuda", "gpu": [{}, {}],
               "precision_support": {"fp32": True, "fp16": True, "bf16": True}}
        with self.assertRaises(ValueError):
            t.training_argument_values(self.config, "T1", 1e-5, 16, "fp16", gpu)

    def test_original_initialization_cannot_accept_an_experiment_checkpoint(self):
        calls = []
        model_class = SimpleNamespace(from_pretrained=lambda *args, **kwargs: calls.append((args, kwargs)))
        with patch.object(t, "_require_training_backend"), patch.object(t, "set_seed"), \
             patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoModelForSequenceClassification=model_class)}):
            t.initialize_original_model(self.config)
            t.initialize_original_model(self.config)
        self.assertEqual(len(calls), 2)
        for positional, keywords in calls:
            self.assertEqual(positional, (self.config["model_id"],))
            self.assertEqual(keywords["revision"], self.config["model_revision"])
            self.assertEqual(keywords["num_labels"], 2)
            self.assertEqual(keywords["id2label"], {0: "Negativo", 1: "Positivo"})
            self.assertEqual(keywords["label2id"], self.config["label_mapping"])
        altered = dict(self.config, model_id="models/transformer/T1/checkpoint-1")
        with self.assertRaises(ValueError):
            t.initialize_original_model(altered)
        with self.assertRaises(ValueError):
            t.initialize_original_model(dict(self.config, model_revision="0" * 40))

    def test_runtime_cpu_cuda_and_native_bf16_detection(self):
        cuda = SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1, current_device=lambda: 0,
            get_device_properties=lambda _: SimpleNamespace(name="Synthetic GPU", total_memory=1024,
                                                              major=7, minor=5),
            get_device_capability=lambda _: (7, 5),
            is_bf16_supported=lambda including_emulation: including_emulation,
        )
        torch = SimpleNamespace(__version__="2.6.0", version=SimpleNamespace(cuda="12.4"), cuda=cuda)
        with patch.dict(sys.modules, {"torch": torch}):
            gpu = t.detect_runtime("cuda")
            cpu = t.detect_runtime("cpu")
        self.assertEqual(gpu["device"], "cuda")
        self.assertEqual(gpu["cuda_runtime"], "12.4")
        self.assertEqual(gpu["gpu"][0]["memory_bytes"], 1024)
        self.assertTrue(gpu["precision_support"]["fp16"])
        self.assertFalse(gpu["precision_support"]["bf16"])
        self.assertEqual(cpu["precision_support"], {"fp32": True, "fp16": False, "bf16": False})
        cuda.is_available = lambda: False
        with patch.dict(sys.modules, {"torch": torch}), self.assertRaises(ValueError):
            t.detect_runtime("cuda")

    def test_smoke_stops_before_artifacts_when_backend_missing(self):
        original_config = t.CONFIG_PATH.read_bytes()
        with TemporaryDirectory(dir="/tmp") as tmp, \
             patch.object(t, "_require_training_backend", side_effect=RuntimeError("Missing torch")), \
             patch.object(t, "load_tokenizer") as loader:
            with self.assertRaises(RuntimeError):
                t.run_smoke_test(1e-5, 8, "fp32", base_dir=tmp)
            loader.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])
        self.assertEqual(t.CONFIG_PATH.read_bytes(), original_config)

    def test_arguments_match_installed_source(self):
        import transformers

        root = Path(transformers.__file__).parent
        tree = ast.parse((root / "training_args.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainingArguments")
        fields = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            plan = t.training_argument_values(self.config, "T1", 1e-5, 16, "fp32", cpu_runtime())
        self.assertTrue(set(plan).issubset(fields), set(plan) - fields)
        tree = ast.parse((root / "trainer.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
        initializer = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        self.assertIn("processing_class", [n.arg for n in initializer.args.args])
        hook = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_determine_best_metric")
        self.assertEqual([n.arg for n in hook.args.args], ["self", "metrics", "trial"])


@unittest.skipUnless((t.ROOT / "data/processed/metadata.json").exists(), "Datos Fase 0 no disponibles")
class DataPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = t.load_config()
        cls.frames = t.load_training_frames()
        cls.snapshot = {name: frame.copy(deep=True) for name, frame in cls.frames.items()}
        cls.hashes = {str(path): t._file_hash(path) for path in (t.ROOT / "data/processed").iterdir() if path.is_file()}
        cls.seen = []

        class TokenizerSpy:
            name_or_path = t.MODEL_ID

            def __call__(self, texts, **kwargs):
                if kwargs != dict(max_length=128, truncation=True, padding=False,
                                  add_special_tokens=True, return_attention_mask=True):
                    raise AssertionError(kwargs)
                cls.seen.extend(texts)
                return {"input_ids": [[4, 10, 5] for _ in texts],
                        "attention_mask": [[1, 1, 1] for _ in texts]}

        cls.spy = TokenizerSpy()
        cls.datasets = t.prepare_training_datasets(cls.spy, cls.config, smoke=True)

    def test_only_train_validation_tokenized_and_no_mutation(self):
        self.assertEqual(set(self.frames), {"train", "validation"})
        expected = []
        for name in ("train", "validation"):
            frame = t._smoke_frame(self.frames[name], 42)
            expected.extend(frame["comentario_limpio"].tolist())
        self.assertEqual(self.seen, expected)
        self.assertEqual({name: len(data) for name, data in self.datasets.items()},
                         {"train": 32, "validation": 16})
        for name, frame in self.frames.items():
            self.assertTrue(frame.equals(self.snapshot[name]))
        for filename, digest in self.hashes.items():
            self.assertEqual(t._file_hash(filename), digest)

    def test_reproducible_indices_and_traceability_outside_model_inputs(self):
        previous_seen = list(self.seen)
        again = t.prepare_training_datasets(self.spy, self.config, smoke=True)
        self.seen[:] = previous_seen
        for name, dataset in self.datasets.items():
            self.assertEqual(dataset.indice_original, again[name].indice_original)
            self.assertEqual(set(dataset[0]), {"input_ids", "attention_mask", "labels"})
            item = dataset[0]
            item["input_ids"].append(999)
            self.assertEqual(len(dataset[0]["input_ids"]), 3)
            self.assertIs(t.guard_development_dataset(dataset, name), dataset)

    def test_test_external_and_renamed_datasets_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.datasets["validation"], split="test")
        with self.assertRaises(ValueError):
            t.guard_development_dataset(self.datasets["validation"], "test")
        with self.assertRaises(ValueError):
            t.guard_development_dataset(self.frames["validation"], "validation")
        with self.assertRaises(ValueError):
            t.guard_development_dataset(replace(self.datasets["validation"], split="train"), "train")

    def test_trainer_public_entrypoints_protect_test_and_resume(self):
        class TrainerDouble:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self.state = SimpleNamespace(best_metric=None, best_global_step=None, global_step=1)

            def get_eval_dataloader(self, dataset):
                return dataset

        with patch.object(t, "_require_training_backend"), \
             patch.dict(sys.modules, {"transformers": SimpleNamespace(Trainer=TrainerDouble)}):
            trainer_type = t.development_trainer_class()
        with TemporaryDirectory(dir="/tmp") as tmp:
            trainer = trainer_type(train_dataset=self.datasets["train"],
                                   eval_dataset=self.datasets["validation"],
                                   args=SimpleNamespace(output_dir=tmp))
            for entrypoint in (trainer.evaluate, trainer.predict, trainer.get_eval_dataloader,
                               trainer.get_test_dataloader):
                with self.assertRaises(ValueError):
                    entrypoint(SimpleNamespace(split="test"))
            with self.assertRaises(ValueError):
                trainer.train(resume_from_checkpoint="models/transformer/T1/checkpoint-1")
            trainer.train_dataset = self.datasets["validation"]
            with self.assertRaises(ValueError):
                trainer.get_train_dataloader()
            with self.assertRaises(ValueError):
                trainer.hyperparameter_search()
            metrics = {"eval_f1_macro": 0.80001, "eval_accuracy": 0.8}
            self.assertTrue(trainer._determine_best_metric(metrics, None))
            trainer.state.global_step = 2
            self.assertTrue(trainer._determine_best_metric(
                {"eval_f1_macro": 0.80002, "eval_accuracy": 0.9}, None))
            trainer.state.global_step = 3
            self.assertFalse(trainer._determine_best_metric(
                {"eval_f1_macro": 0.80003, "eval_accuracy": 0.9}, None))
            self.assertEqual(trainer.state.best_global_step, 2)


if __name__ == "__main__":
    unittest.main()
