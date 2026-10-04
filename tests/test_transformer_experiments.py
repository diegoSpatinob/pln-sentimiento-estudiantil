"""Fase 3A: contratos y fixtures sintéticos en /tmp; ningún modelo ni entrenamiento."""

import copy
import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from scripts import transformer as t


def runtime():
    return {"torch_available": True, "torch": "UNIT_TEST_TORCH", "device": "cuda",
            "gpu": [{"name": "UNIT_TEST_GPU"}], "cuda_runtime": "UNIT_TEST_CUDA",
            "cuda_available": True, "transformers": "4.57.1",
            "precision_support": {"fp32": True, "fp16": True, "bf16": False}}


def metrics(f1=0.8, accuracy=0.8):
    return {"f1_macro": f1, "accuracy": accuracy, "precision_macro": 0.8, "recall_macro": 0.8,
            **{f"{metric}_class_{label}": 0.8 for label in (0, 1)
               for metric in ("precision", "recall", "f1")},
            "support_class_0": 1, "support_class_1": 1}


def candidate(f1=0.8, accuracy=0.8, learning_rate=1e-5, epoch=1):
    return {"metrics": metrics(f1, accuracy), "learning_rate": learning_rate, "epoch": epoch}


class SelectionTests(unittest.TestCase):
    def assert_winner(self, winner, loser):
        for order in ([loser, winner], [winner, loser]):
            self.assertIs(t.select_validation_candidate(order), winner)

    def test_f1_is_primary_outside_four_decimal_tie(self):
        self.assert_winner(candidate(0.80006, 0.1), candidate(0.80004, 0.99))

    def test_four_decimal_tie_uses_accuracy_even_with_lower_raw_f1(self):
        self.assert_winner(candidate(0.80001, 0.9), candidate(0.80004, 0.8))

    def test_accuracy_is_not_rounded(self):
        self.assert_winner(candidate(0.8, 0.80002), candidate(0.8, 0.80001))

    def test_accuracy_tie_uses_lower_learning_rate(self):
        self.assert_winner(candidate(learning_rate=1e-5, epoch=3),
                           candidate(learning_rate=2e-5, epoch=1))

    def test_learning_rate_tie_uses_earlier_epoch(self):
        self.assert_winner(candidate(epoch=1), candidate(epoch=3))

    def test_three_experiments_global_selection_uses_same_function(self):
        rows = [dict(candidate(learning_rate=lr), experiment_id=e)
                for e, lr in t.EXPERIMENT_LEARNING_RATES.items()]
        self.assertEqual(t.select_validation_candidate(reversed(rows))["experiment_id"], "T1")
        rows[1]["metrics"]["accuracy"] = 0.9
        self.assertEqual(t.select_validation_candidate(rows)["experiment_id"], "T2")
        rows[2]["metrics"]["f1_macro"] = 0.85
        self.assertEqual(t.select_validation_candidate(rows)["experiment_id"], "T3")

    def test_rounding_boundaries_and_all_exact_ties(self):
        for f1 in (0.80005, 0.80015, 0.0, 1.0):
            self.assertEqual(t.validation_selection_key(candidate(f1))[0], round(f1, 4))
        row = candidate()
        self.assertIs(t.select_validation_candidate([row, copy.deepcopy(row)]), row)

    def test_invalid_candidates_rejected(self):
        with self.assertRaises(ValueError):
            t.select_validation_candidate([])
        for kwargs in ({"f1": float("nan")}, {"accuracy": float("inf")}, {"epoch": 0},
                       {"learning_rate": 0}, {"f1": 1.1}):
            with self.assertRaises(ValueError):
                t.select_validation_candidate([candidate(**kwargs)])


class OfficialProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = t.load_config()

    def plan(self, experiment="T1", lr=1e-5, batch=16, precision="fp16", device=None, run_id=None):
        return t.training_argument_values(self.config, experiment, lr, batch, precision,
                                          device or runtime(), run_id=run_id or "RUN_" + "a" * 32)

    def test_exact_ids_rates_and_no_fourth_rate(self):
        self.assertEqual(t.EXPERIMENT_LEARNING_RATES, {"T1": 1e-5, "T2": 2e-5, "T3": 3e-5})
        for experiment, expected in t.EXPERIMENT_LEARNING_RATES.items():
            self.assertEqual(self.plan(experiment, expected)["learning_rate"], expected)
            for lr in (1e-5, 2e-5, 3e-5, 4e-5):
                if lr != expected:
                    with self.assertRaises(ValueError):
                        self.plan(experiment, lr)
        with self.assertRaises(ValueError):
            self.plan("T4", 1e-5)

    def test_official_batch_precision_cuda_single_process(self):
        plan = self.plan()
        self.assertEqual(plan["per_device_train_batch_size"] * plan["gradient_accumulation_steps"], 16)
        self.assertTrue(plan["fp16"])
        self.assertFalse(plan["use_cpu"])
        for kwargs in ({"batch": 8}, {"precision": "fp32"},
                       {"device": {**runtime(), "device": "cpu"}}):
            with self.assertRaises(ValueError):
                self.plan(**kwargs)
        with patch.dict(os.environ, {"WORLD_SIZE": "2"}), self.assertRaises(ValueError):
            self.plan()

    def test_three_epochs_preserved_and_no_trainer_selection(self):
        plan = self.plan()
        self.assertEqual(plan["num_train_epochs"], 3)
        self.assertEqual(plan["max_steps"], -1)
        self.assertEqual(plan["eval_strategy"], "epoch")
        self.assertEqual(plan["save_strategy"], "epoch")
        self.assertEqual(plan["save_total_limit"], 3)
        self.assertFalse(plan["load_best_model_at_end"])
        self.assertIsNone(plan["metric_for_best_model"])

    def test_configuration_fixes_methodology(self):
        for field, invalid in {"seed": 1, "max_epochs": 4, "physical_batch_size": 8,
                               "gradient_accumulation_steps": 2, "precision": "fp32",
                               "training_device": "cpu", "class_weight": "balanced",
                               "oversampling": True, "early_stopping": True}.items():
            with self.assertRaises(ValueError):
                t.validate_config(dict(self.config, **{field: invalid}))
        self.config["max_length"] = 256
        with self.assertRaises(ValueError):
            self.plan()

    def test_cli_rejects_overrides_before_any_executor(self):
        with patch.object(t, "run_official_experiment") as run, redirect_stderr(io.StringIO()):
            for arguments in (["--experiment", "T1", "--learning-rate", "2e-5"],
                              ["--experiment", "T1", "--learning-rate", "1e-5"],
                              ["--experiment", "T2", "--precision", "fp32"],
                              ["--experiment", "T3", "--device", "cpu"],
                              ["--experiment", "T1", "--physical-batch-size", "8"],
                              ["--experiment", "T4"],
                              ["--experiment", "T1", "--smoke-test"],
                              ["--experiment", "T1", "--record-max-length"]):
                with self.assertRaises(SystemExit):
                    t.main(arguments)
            run.assert_not_called()

    def test_cli_explicit_id_routes_to_executor_double_only(self):
        with patch.object(t, "run_official_experiment", return_value={}) as run, \
             redirect_stdout(io.StringIO()):
            for experiment in t.EXPERIMENT_LEARNING_RATES:
                t.main(["--experiment", experiment])
                run.assert_called_with(experiment, local_files_only=False)


class CleanWorkingTreeTests(unittest.TestCase):
    """Git real en /tmp sin crear commits; solo HEAD/diff tienen un doble."""

    def setUp(self):
        temporary = TemporaryDirectory(dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        subprocess.run(["git", "init", "--quiet", str(self.base)], check=True, capture_output=True)
        # Reutilizar exactamente las reglas del proyecto sin agregar un archivo
        # .gitignore no rastreado al repositorio temporal sin commits.
        (self.base / ".git/info/exclude").write_text((t.ROOT / ".gitignore").read_text())
        real_check_output = subprocess.check_output

        def git_output(command, **kwargs):
            if command == ["git", "rev-parse", "HEAD"]:
                return ("a" * 40 + "\n").encode()
            if command == ["git", "diff", "HEAD", "--binary"]:
                return b""
            return real_check_output(command, **kwargs)

        mocked = patch.object(t.subprocess, "check_output", side_effect=git_output)
        self.commands = mocked.start()
        self.addCleanup(mocked.stop)

    def test_clean_repository_accepted(self):
        metadata = t.require_clean_working_tree(self.base)
        self.assertEqual(metadata["commit"], "a" * 40)
        self.assertFalse(metadata["dirty"])
        self.commands.assert_any_call(["git", "status", "--porcelain", "--untracked-files=all"],
                                      cwd=self.base)

    def test_ignored_data_models_results_do_not_block(self):
        for relative in ("data/processed/UNIT_TEST.csv", "models/UNIT_TEST.bin", "results/UNIT_TEST.json"):
            path = self.base / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"UNIT_TEST_ONLY")
        self.assertFalse(t.require_clean_working_tree(self.base)["dirty"])

    def test_untracked_file_rejected_even_if_git_config_hides_untracked(self):
        subprocess.run(["git", "-C", str(self.base), "config", "status.showUntrackedFiles", "no"],
                       check=True)
        (self.base / "UNIT_TEST_UNTRACKED.py").write_text("# unit test\n")
        with self.assertRaisesRegex(ValueError, "git status --porcelain"):
            t.require_clean_working_tree(self.base)

    def test_staged_and_modified_tracked_file_rejected(self):
        path = self.base / "UNIT_TEST_TRACKED.py"
        path.write_text("# staged unit test\n")
        subprocess.run(["git", "-C", str(self.base), "add", path.name], check=True)
        path.write_text("# modified unit test\n")
        with self.assertRaisesRegex(ValueError, "git status --porcelain"):
            t.require_clean_working_tree(self.base)


class InfrastructureTests(unittest.TestCase):
    """Dobles de APIs: archivos UNIT_TEST en TemporaryDirectory, sin pesos reales."""

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.base = Path(stack.enter_context(TemporaryDirectory(dir="/tmp")))
        (self.base / "configs").mkdir()
        (self.base / "configs/transformer.json").write_bytes(t.CONFIG_PATH.read_bytes())
        (self.base / "data/processed").mkdir(parents=True)
        for split in ("train", "validation"):
            (self.base / f"data/processed/{split}.csv").write_bytes(b"UNIT_TEST_FIXTURE_ONLY")
        self.config = t.load_config(self.base / "configs/transformer.json")
        self.datasets = {split: t.DevelopmentDataset(
            split, (1,), ((("input_ids", (2, 3)), ("attention_mask", (1, 1))),), (0,),
            t._file_hash(self.base / f"data/processed/{split}.csv"), t.MODEL_ID,
            self.config["model_revision"], 128, False) for split in ("train", "validation")}

        class ArgumentDouble:
            def __init__(self, **values):
                self.__dict__.update(values)
                self.report_to = [] if values["report_to"] == "none" else values["report_to"]
                self.world_size, self.n_gpu = 1, 1

            def to_dict(self):
                return vars(self).copy()

        class TrainerDouble:
            interrupt_after = None

            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self.model = self.model_init()
                self.callbacks = []
                self.state = SimpleNamespace(epoch=0, global_step=0, best_metric=None,
                                             best_global_step=None)

            def add_callback(self, callback):
                self.callbacks.append(callback)

            def train(self, resume_from_checkpoint, **kwargs):
                # Simula SOLO eventos de control; ningún forward/backward/optimizador.
                self.received_resume = resume_from_checkpoint
                self.model = self.model_init()
                for epoch in (1, 2, 3):
                    self.state.epoch, self.state.global_step = epoch, epoch * 10
                    values = {"eval_" + k: v for k, v in metrics().items()}
                    values["eval_f1_macro"] = (0.80004, 0.80001, 0.80002)[epoch - 1]
                    values["eval_accuracy"] = (0.8, 0.9, 0.9)[epoch - 1]
                    for callback in self.callbacks:
                        callback.on_evaluate(self.args, self.state, None, metrics=values)
                    checkpoint = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
                    checkpoint.mkdir()
                    (checkpoint / "model.safetensors").write_bytes(b"UNIT_TEST_NOT_MODEL_WEIGHTS")
                    (checkpoint / "trainer_state.json").write_text("{}")
                    for callback in self.callbacks:
                        callback.on_save(self.args, self.state, None)
                    if self.interrupt_after == epoch:
                        raise KeyboardInterrupt("UNIT_TEST_INTERRUPTION")
                return SimpleNamespace(metrics={"train_runtime": 0.0})

        self.trainer_double = TrainerDouble
        module = SimpleNamespace(Trainer=TrainerDouble, TrainerCallback=object,
                                 TrainingArguments=ArgumentDouble,
                                 DataCollatorWithPadding=lambda **kwargs: kwargs)
        stack.enter_context(patch.dict(sys.modules, {"transformers": module}))
        stack.enter_context(patch.dict(os.environ, {"WORLD_SIZE": "1"}))
        stack.enter_context(patch.object(t, "_require_training_backend"))
        stack.enter_context(patch.object(t, "detect_runtime", return_value=runtime()))
        stack.enter_context(patch.object(t, "set_seed"))
        stack.enter_context(patch.object(t, "version", return_value="UNIT_TEST_VERSION"))
        stack.enter_context(patch.object(t, "guard_development_dataset", side_effect=lambda d, *a: d))
        self.original_init = stack.enter_context(patch.object(t, "initialize_original_model"))
        self.git_metadata = stack.enter_context(patch.object(t, "git_metadata", return_value={
            "commit": "a" * 40, "dirty": False, "diff_sha256": "b" * 64,
            "transformer_script_sha256": "c" * 64}))
        stack.enter_context(patch.object(t, "load_tokenizer", return_value=SimpleNamespace(name_or_path=t.MODEL_ID)))
        stack.enter_context(patch.object(t, "prepare_training_datasets", return_value=self.datasets))

    def build(self, experiment="T1", run_id=None, datasets=None):
        return t.build_trainer(self.config, None, datasets or self.datasets, experiment,
                               t.EXPERIMENT_LEARNING_RATES[experiment], 16, "fp16", "cuda",
                               self.base, run_id=run_id)

    def test_independent_builds_original_model_directories_and_unique_reruns(self):
        trainers = [self.build(e) for e in ("T1", "T2", "T3", "T1")]
        self.assertEqual(len({x.args.output_dir for x in trainers}), 4)
        self.assertEqual(len({x.run_manifest["run_id"] for x in trainers}), 4)
        self.assertEqual(self.original_init.call_count, 4)
        for call in self.original_init.call_args_list:
            self.assertEqual(call.args[0], self.config)
            self.assertEqual(call.args[0]["model_id"], t.MODEL_ID)
            self.assertEqual(call.args[0]["model_revision"], self.config["model_revision"])
        for trainer in trainers:
            self.assertFalse(trainer.run_manifest["resume_from_checkpoint"])

    def test_existing_directory_and_result_directory_never_overwritten(self):
        trainer = self.build()
        snapshot = (Path(trainer.args.output_dir) / "run_manifest.json").read_bytes()
        with self.assertRaises(ValueError):
            self.build(run_id=trainer.run_manifest["run_id"])
        self.assertEqual((Path(trainer.args.output_dir) / "run_manifest.json").read_bytes(), snapshot)
        run_id = "RUN_" + "d" * 32
        (self.base / "results/transformer/T1" / run_id).mkdir()
        with self.assertRaises(ValueError):
            self.build(run_id=run_id)

    def test_interrupted_metadata_write_keeps_previous_json(self):
        path = self.base / "atomic_unit_test.json"
        t._write_run_json(path, {"status": "UNIT_TEST_PREVIOUS"})
        with patch.object(Path, "replace", side_effect=OSError("UNIT_TEST_FAILED_REPLACE")):
            with self.assertRaises(OSError):
                t._write_run_json(path, {"status": "UNIT_TEST_NEXT"})
        self.assertEqual(json.loads(path.read_text()), {"status": "UNIT_TEST_PREVIOUS"})

    def test_test_key_rejected_before_initializing_model(self):
        with self.assertRaises(ValueError):
            self.build(datasets={**self.datasets, "test": self.datasets["validation"]})
        self.original_init.assert_not_called()

    def test_public_resume_legacy_resume_and_second_train_rejected(self):
        trainer = self.build()
        for checkpoint in (True, "models/transformer/T2/checkpoint-1"):
            with self.assertRaises(ValueError):
                trainer.train(resume_from_checkpoint=checkpoint)
        with self.assertRaises(ValueError):
            trainer.train(model_path="models/transformer/T2/checkpoint-1")
        trainer._started = True
        with self.assertRaises(ValueError):
            trainer.train()

    def test_mutated_arguments_rejected_and_no_hidden_hf_selection(self):
        trainer = self.build()
        self.assertEqual(trainer.protocol_arguments["report_to"], [])
        self.assertFalse(trainer._determine_best_metric({"eval_f1_macro": 0.9}, None))
        self.assertIsNone(trainer.state.best_metric)
        trainer.args.learning_rate = 2e-5
        with self.assertRaises(ValueError):
            trainer.train()

    def run_fixture(self, experiment="T1"):
        return t.run_official_experiment(experiment, self.base)

    def test_dirty_official_run_aborts_before_any_preparation_or_model(self):
        self.git_metadata.return_value = {**self.git_metadata.return_value, "dirty": True}
        with patch.object(t, "load_config") as config_loader, \
             patch.object(t, "detect_runtime") as detect, \
             patch.object(t, "load_tokenizer") as tokenizer, \
             patch.object(t, "prepare_training_datasets") as datasets, \
             patch.object(t, "build_trainer") as builder:
            with self.assertRaisesRegex(ValueError, "git status --porcelain"):
                self.run_fixture()
            for operation in (config_loader, detect, tokenizer, datasets, builder, self.original_init):
                operation.assert_not_called()
        self.assertFalse((self.base / "results").exists())
        self.assertFalse((self.base / "models").exists())

    def test_clean_official_run_reaches_preparation_without_training(self):
        with patch.object(t, "load_tokenizer", side_effect=RuntimeError("UNIT_TEST_STOP")) as tokenizer, \
             patch.object(t, "build_trainer") as builder:
            with self.assertRaisesRegex(RuntimeError, "UNIT_TEST_STOP"):
                self.run_fixture()
            tokenizer.assert_called_once()
            builder.assert_not_called()
            self.original_init.assert_not_called()

    def test_official_builder_also_rejects_dirty_tree_before_initialization(self):
        self.git_metadata.return_value = {**self.git_metadata.return_value, "dirty": True}
        with self.assertRaisesRegex(ValueError, "git status --porcelain"):
            self.build()
        self.original_init.assert_not_called()
        self.assertFalse((self.base / "models").exists())

    def test_persisted_technical_metadata_epoch_metrics_and_selection(self):
        result = self.run_fixture()
        self.assertEqual(result["selected"]["epoch"], 2)
        self.assertEqual(len(result["epochs"]), 3)
        self.assertEqual(result["configuration_sha256"], t._file_hash(self.base / "configs/transformer.json"))
        for field in ("experiment_id", "run_id", "git", "runtime", "accelerate", "configuration",
                      "data", "runtime_seconds", "selection_rule", "selected", "training_arguments"):
            self.assertIn(field, result)
        self.assertEqual(result["git"]["commit"], "a" * 40)
        self.assertEqual(result["git_commit"], "a" * 40)
        self.assertIs(result["working_tree_clean"], True)
        for field in ("gpu", "cuda_runtime", "torch", "transformers"):
            self.assertIn(field, result["runtime"])
        self.assertEqual(set(result["data"]), {"train", "validation"})
        for row in result["epochs"]:
            self.assertEqual(set(row["metrics"]), set(metrics()))
            self.assertIsInstance(row["metrics"]["support_class_0"], int)
            self.assertTrue((self.base / row["checkpoint"] / "model.safetensors").exists())
        self.assertIs(t.validate_official_result(result, self.base), result)
        directory = self.base / "results/transformer/T1" / result["run_id"]
        self.assertEqual(json.loads((directory / "validation_result.json").read_text()), result)
        self.assertEqual(json.loads((directory / "validation_epochs.json").read_text())["status"], "COMPLETED")
        self.assertFalse(result["test_used"])
        self.assertTrue(result["official"])
        self.assertNotIn("predictions", result)

    def test_interruption_preserves_epoch_then_rerun_is_new_from_original(self):
        self.trainer_double.interrupt_after = 1
        with self.assertRaises(KeyboardInterrupt):
            self.run_fixture()
        directory = next((self.base / "results/transformer/T1").iterdir())
        failed = json.loads((directory / "run_manifest.json").read_text())
        self.assertEqual(failed["status"], "FAILED_OR_INTERRUPTED")
        self.assertFalse((directory / "validation_result.json").exists())
        self.assertEqual(len(json.loads((directory / "validation_epochs.json").read_text())["epochs"]), 1)
        self.trainer_double.interrupt_after = None
        complete = self.run_fixture()
        self.assertNotEqual(complete["run_id"], failed["run_id"])
        self.assertEqual(self.original_init.call_count, 4)  # constructor + inicio por cada run

    def test_selection_load_is_local_and_separate_from_initialization(self):
        result = self.run_fixture()
        model = SimpleNamespace(eval=lambda: "UNIT_TEST_MODEL_IN_EVAL")
        from unittest.mock import Mock
        loader = Mock(return_value=model)
        module = SimpleNamespace(AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=loader))
        with patch.dict(sys.modules, {"transformers": module}):
            self.assertEqual(t.load_selected_checkpoint(result, self.base), "UNIT_TEST_MODEL_IN_EVAL")
        loader.assert_called_once_with(str(self.base / result["selected"]["checkpoint"]),
                                       local_files_only=True, trust_remote_code=False)

    def test_incomplete_smoke_altered_selection_and_missing_checkpoint_rejected(self):
        result = self.run_fixture()
        for change in ({"official": False, "purpose": "SMOKE_TEST_NO_OFICIAL"},
                       {"status": "RUNNING"}, {"test_used": True},
                       {"working_tree_clean": False}, {"git_commit": "b" * 40},
                       {"selected": result["epochs"][0]}, {"learning_rate": 2e-5}):
            with self.assertRaises(ValueError):
                t.validate_official_result({**result, **change}, self.base)
        incomplete = copy.deepcopy(result)
        incomplete["epochs"].pop()
        with self.assertRaises(ValueError):
            t.validate_official_result(incomplete, self.base)
        # Renombrar una fixture de checkpoint demuestra que las tres se exigen.
        checkpoint = self.base / result["epochs"][0]["checkpoint"] / "model.safetensors"
        checkpoint.rename(checkpoint.with_suffix(".unit_test_missing"))
        with self.assertRaises(ValueError):
            t.validate_official_result(result, self.base)

    def result_paths(self):
        return [self.base / "results/transformer" / result["experiment_id"] / result["run_id"]
                / "validation_result.json" for result in
                [self.run_fixture(e) for e in ("T1", "T2", "T3")]]

    def test_summary_selects_global_once_and_does_not_overwrite(self):
        paths = self.result_paths()
        with patch.object(t, "validate_official_result", wraps=t.validate_official_result) as validate:
            summary = t.summarize_validation(paths, self.base)
            self.assertEqual(validate.call_count, 3)
        self.assertEqual(summary["selected_experiment_id"], "T1")
        self.assertEqual([r["selected"] for r in summary["experiments"]], [True, False, False])
        self.assertFalse(summary["test_used"])
        self.assertEqual({row["git_commit"] for row in summary["experiments"]}, {"a" * 40})
        self.assertEqual([row["result_path"] for row in summary["experiments"]],
                         [path.relative_to(self.base).as_posix() for path in paths])
        with self.assertRaises(FileExistsError):
            t.summarize_validation(paths, self.base)
        other = self.base / "results/transformer/another_summary.json"
        self.assertEqual(t.summarize_validation(paths, self.base, other), summary)

    def test_summary_rejects_duplicate_ids_inconsistent_code_and_data(self):
        paths = self.result_paths()
        with self.assertRaises(ValueError):
            t.summarize_validation([paths[0], paths[0], paths[2]], self.base)
        modified = json.loads(paths[1].read_text())
        modified["git"]["transformer_script_sha256"] = "d" * 64
        paths[1].write_bytes(t._json_bytes(modified))
        with self.assertRaises(ValueError):
            t.summarize_validation(paths, self.base)
        modified["data"]["validation"]["sha256"] = "e" * 64
        paths[1].write_bytes(t._json_bytes(modified))
        with self.assertRaises(ValueError):
            t.summarize_validation(paths, self.base)

    def assert_summary_rejected(self, paths):
        output = self.base / "results/transformer/validation_summary.json"
        self.assertFalse(output.exists())
        with self.assertRaises(ValueError):
            t.summarize_validation(paths, self.base)
        self.assertFalse(output.exists())

    def test_summary_rejects_duplicate_experiments_with_distinct_runs(self):
        paths = self.result_paths()
        rerun = self.run_fixture("T1")
        paths[1] = self.base / "results/transformer/T1" / rerun["run_id"] / "validation_result.json"
        self.assert_summary_rejected(paths)

    def test_summary_rejects_missing_experiment_before_loading_results(self):
        paths = self.result_paths()
        with patch.object(t, "validate_official_result") as validate:
            self.assert_summary_rejected(paths[:2])
            validate.assert_not_called()

    def test_summary_rejects_four_results(self):
        paths = self.result_paths()
        self.assert_summary_rejected([*paths, paths[0]])

    def test_summary_rejects_distinct_commits_even_with_identical_script(self):
        paths = self.result_paths()
        changed = json.loads(paths[1].read_text())
        changed["git_commit"] = changed["git"]["commit"] = "b" * 40
        # Cada resultado sigue siendo válido individualmente; lo incompatible
        # aquí son los commits de la terna, aunque el script sea idéntico.
        t.validate_official_result(changed, self.base)
        paths[1].write_bytes(t._json_bytes(changed))
        self.assert_summary_rejected(paths)

    def check_incompatible_reference(self, reference):
        paths = self.result_paths()
        changed = json.loads(paths[1].read_text())
        if reference == "config":
            changed["configuration_sha256"] = "d" * 64
        elif reference in ("train", "validation"):
            changed["data"][reference]["sha256"] = "d" * 64
        else:
            changed["initial_checkpoint"]["revision"] = "d" * 40
        paths[1].write_bytes(t._json_bytes(changed))
        self.assert_summary_rejected(paths)
        # Aislar además la coherencia entre resultados de la comprobación de
        # archivos locales: incluso con un validator doble se exige igualdad.
        with patch.object(t, "validate_official_result", side_effect=lambda result, *a: result), \
             patch.object(t, "select_validation_candidate") as select:
            self.assert_summary_rejected(paths)
            select.assert_not_called()

    def test_summary_rejects_incompatible_configuration_hash(self):
        self.check_incompatible_reference("config")

    def test_summary_rejects_incompatible_train_hash(self):
        self.check_incompatible_reference("train")

    def test_summary_rejects_incompatible_validation_hash(self):
        self.check_incompatible_reference("validation")

    def test_summary_rejects_incompatible_beto_revision(self):
        self.check_incompatible_reference("revision")

    def test_summary_rejects_incorrect_mandatory_learning_rate(self):
        paths = self.result_paths()
        changed = json.loads(paths[1].read_text())
        changed["learning_rate"] = 3e-5  # T2 debe ser 2e-5.
        paths[1].write_bytes(t._json_bytes(changed))
        self.assert_summary_rejected(paths)

    def test_summary_rejects_duplicate_run_ids(self):
        paths = self.result_paths()
        changed = json.loads(paths[1].read_text())
        changed["run_id"] = json.loads(paths[0].read_text())["run_id"]
        paths[1].write_bytes(t._json_bytes(changed))
        self.assert_summary_rejected(paths)
        with patch.object(t, "validate_official_result", side_effect=lambda result, *a: result), \
             patch.object(t, "select_validation_candidate") as select:
            self.assert_summary_rejected(paths)
            select.assert_not_called()

    def test_unpaired_save_and_duplicate_epoch_rejected(self):
        trainer = self.build()
        recorder = t.official_epoch_recorder(trainer, self.base)
        with self.assertRaises(ValueError):
            recorder.on_save(trainer.args, trainer.state, None)
        with self.assertRaises(ValueError):
            t.persist_validation_epoch(trainer.run_manifest, [], {"eval_" + k: v for k, v in metrics().items()},
                                       1, 10, trainer.args.output_dir, trainer.result_dir, self.base)


if __name__ == "__main__":
    unittest.main()
