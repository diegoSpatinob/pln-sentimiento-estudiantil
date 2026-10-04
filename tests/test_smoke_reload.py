"""Regresión AMP/save-reload: sin corpus, checkpoints BETO ni entrenamiento.

Los dobles cubren el contrato sin instalar torch. La prueba adicional con
PyTorch/Accelerate reales se ejecuta solo cuando ambos están disponibles.
"""

import importlib.util
import json
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scripts import transformer as t


class TensorDouble:
    def __init__(self, values, dtype="torch.float32"):
        self.values = np.asarray(values)
        self.dtype = dtype
        self.shape = self.values.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def is_floating_point(self):
        return self.dtype in {"torch.float32", "torch.float64"}

    def double(self):
        return TensorDouble(self.values.astype(np.float64), "torch.float64")

    def __sub__(self, other):
        return TensorDouble(self.values - other.values, self.dtype)

    def abs(self):
        return TensorDouble(np.abs(self.values), self.dtype)

    def max(self):
        return self.values.max()

    def mean(self):
        return self.values.mean()


class ModelDouble:
    def __init__(self, wrapped=False):
        self.wrapped = wrapped
        self.training = True
        self.calls = []
        self.state = {"weight": TensorDouble([0.123]),
                      "buffer": TensorDouble([1], "torch.int64")}
        self.delta = np.zeros((1, 2))
        self.loss = 0.5

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False
        return self

    def state_dict(self):
        return self.state

    def __call__(self, **batch):
        self.calls.append((self.training, self.device, batch))
        # Igual dtype visible, distinto cálculo interno: replica el wrapper AMP.
        amp_difference = np.array([[0.001, 0.002]]) if self.wrapped else 0
        return SimpleNamespace(logits=TensorDouble([[1., 2.]] + self.delta + amp_difference),
                               loss=TensorDouble(self.loss))


class SaveReloadRegressionTests(unittest.TestCase):
    def setUp(self):
        self.original, self.reloaded = ModelDouble(wrapped=True), ModelDouble()
        self.device = SimpleNamespace(type="cuda")

        def unwrap(model, **kwargs):
            model.wrapped = False
            return model

        self.unwrap = Mock(side_effect=unwrap)
        self.trainer = SimpleNamespace(model=self.original, args=SimpleNamespace(device=self.device),
                                       accelerator=SimpleNamespace(unwrap_model=self.unwrap))
        self.autocast = Mock(side_effect=lambda **kwargs: nullcontext())
        self.torch = SimpleNamespace(
            no_grad=nullcontext, autocast=self.autocast,
            equal=lambda a, b: np.array_equal(a.values, b.values),
            isfinite=lambda a: np.isfinite(a.values),
            allclose=lambda a, b, **kwargs: np.allclose(a.values, b.values, **kwargs),
        )
        self.batch = {"input_ids": TensorDouble([[1, 2]], "torch.int64"),
                      "labels": TensorDouble([0], "torch.int64")}

    def verify(self, path):
        with patch.dict(sys.modules, {"torch": self.torch}), patch("builtins.print"):
            return t.verify_smoke_save_reload(self.trainer, self.reloaded, self.batch, path)

    def test_hidden_amp_difference_is_measured_then_removed(self):
        with TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "diagnostics.json"
            result = self.verify(path)
            self.assertEqual(json.loads(path.read_text()), result)
        before = result["before_removing_amp_wrapper"]
        self.assertFalse(before["allclose"])
        self.assertAlmostEqual(before["max_abs_diff"], 0.002)
        self.assertAlmostEqual(before["mean_abs_diff"], 0.0015)
        self.assertEqual(before["original_dtype"], before["reloaded_dtype"])
        self.assertEqual(result["equivalent_fp32_forwards"]["max_abs_diff"], 0)
        self.assertTrue(result["state_dict_verified"])
        self.assertTrue(result["save_reload_verified"])
        self.assertEqual(result["state_dict_tensor_count"], 2)
        self.unwrap.assert_called_once_with(self.original, keep_fp32_wrapper=False,
                                            keep_torch_compile=False)
        for model in (self.original, self.reloaded):
            self.assertEqual(len(model.calls), 2)
            for training, device, batch in model.calls:
                self.assertFalse(training)
                self.assertIs(device, self.device)
                for key in self.batch:
                    self.assertIs(batch[key], self.batch[key])
                    self.assertIs(batch[key].device, self.device)
        for call in self.autocast.call_args_list:
            self.assertEqual(call.kwargs, {"device_type": "cuda", "enabled": False})

    def test_tiny_weight_difference_fails_even_when_logits_match(self):
        self.reloaded.state["weight"] = TensorDouble([0.1230001])
        with TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "diagnostics.json"
            with self.assertRaisesRegex(ValueError, "recarga técnica"):
                self.verify(path)
            result = json.loads(path.read_text())
        self.assertFalse(result["save_reload_verified"])
        self.assertFalse(result["state_dict_verified"])
        self.assertEqual(result["unequal_state_tensors"], ["weight"])
        self.assertTrue(result["equivalent_fp32_forwards"]["allclose"])

    def test_state_keys_shapes_and_dtypes_are_checked(self):
        for replacement in (None, TensorDouble([[0.123]]),
                            TensorDouble([0.123], "torch.float64")):
            with self.subTest(replacement=replacement), TemporaryDirectory(dir="/tmp") as tmp:
                if replacement is None:
                    self.reloaded.state.pop("weight")
                else:
                    self.reloaded.state["weight"] = replacement
                path = Path(tmp) / "diagnostics.json"
                with self.assertRaises(ValueError):
                    self.verify(path)
                result = json.loads(path.read_text())
                self.assertFalse(result["state_dict_verified"])
                self.assertFalse(result["save_reload_verified"])

    def test_real_logit_difference_still_fails_at_unchanged_tolerance(self):
        self.reloaded.delta = np.array([[0.01, 0.02]])
        with TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "diagnostics.json"
            with self.assertRaises(ValueError):
                self.verify(path)
            result = json.loads(path.read_text())
        self.assertTrue(result["state_dict_verified"])
        self.assertFalse(result["save_reload_verified"])
        self.assertEqual((result["rtol"], result["atol"]), (1e-4, 1e-4))
        self.assertAlmostEqual(result["equivalent_fp32_forwards"]["max_abs_diff"], 0.02)
        self.assertAlmostEqual(result["equivalent_fp32_forwards"]["mean_abs_diff"], 0.015)

    def test_nonfinite_logits_or_loss_cannot_verify_reload(self):
        for field in ("delta", "loss"):
            with self.subTest(field=field), TemporaryDirectory(dir="/tmp") as tmp:
                self.reloaded.delta = np.zeros((1, 2))
                self.reloaded.loss = 0.5
                setattr(self.reloaded, field, np.array([[np.nan, 0]]) if field == "delta" else np.inf)
                path = Path(tmp) / "diagnostics.json"
                with self.assertRaises(ValueError):
                    self.verify(path)
                result = json.loads(path.read_text())
                self.assertFalse(result["save_reload_verified"])
                if field == "delta":
                    self.assertIsNone(result["equivalent_fp32_forwards"]["max_abs_diff"])


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("accelerate"),
                     "PyTorch/Accelerate no instalados; prueba real de AMP omitida")
class RealAccelerateSaveReloadTests(unittest.TestCase):
    def test_actual_amp_wrapper_and_torch_save_reload(self):
        import torch
        from accelerate import Accelerator

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)
                self.register_buffer("trace_buffer", torch.tensor([1]))

            def forward(self, input_ids, labels):
                logits = self.linear(input_ids)
                return SimpleNamespace(logits=logits, loss=logits.square().mean())

        # Se usa CPU y bf16 únicamente en esta prueba sintética, sin backward.
        torch.manual_seed(42)
        accelerator = Accelerator(cpu=True, mixed_precision="bf16")
        original = accelerator.prepare_model(TinyModel())
        restored = TinyModel()
        with TemporaryDirectory(dir="/tmp") as tmp:
            weights = Path(tmp) / "weights.pt"
            torch.save(original.state_dict(), weights)
            restored.load_state_dict(torch.load(weights, weights_only=True))
            trainer = SimpleNamespace(model=original, accelerator=accelerator,
                                       args=SimpleNamespace(device=accelerator.device))
            batch = {"input_ids": torch.tensor([[0.12345, 0.76543]]), "labels": torch.tensor([0])}
            result = t.verify_smoke_save_reload(trainer, restored, batch, Path(tmp) / "diagnostics.json")
        self.assertTrue(result["state_dict_verified"])
        self.assertTrue(result["save_reload_verified"])
        self.assertGreater(result["before_removing_amp_wrapper"]["max_abs_diff"], 0)
        self.assertEqual(result["equivalent_fp32_forwards"]["max_abs_diff"], 0)


if __name__ == "__main__":
    unittest.main()
