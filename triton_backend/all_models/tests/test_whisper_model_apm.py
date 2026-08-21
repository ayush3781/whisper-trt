import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock


MODEL_ROOT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "whisper"
    / "whisper_bls"
    / "1"
)
PACKAGE_NAME = "whisper_backend_under_test"


def load_model_module():
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(MODEL_ROOT)]

    torch = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch.utils")
    torch_dlpack = types.ModuleType("torch.utils.dlpack")
    torch_dlpack.to_dlpack = mock.Mock()
    torch_utils.dlpack = torch_dlpack
    torch.utils = torch_utils

    fbank = types.ModuleType(f"{PACKAGE_NAME}.fbank")
    fbank.FeatureExtractor = mock.Mock()
    tokenizer = types.ModuleType(f"{PACKAGE_NAME}.tokenizer")
    tokenizer.LANGUAGES = {}
    tokenizer.get_tokenizer = mock.Mock()

    modules = {
        PACKAGE_NAME: package,
        "torch": torch,
        "torch.utils": torch_utils,
        "torch.utils.dlpack": torch_dlpack,
        "triton_python_backend_utils": types.ModuleType(
            "triton_python_backend_utils"
        ),
        f"{PACKAGE_NAME}.fbank": fbank,
        f"{PACKAGE_NAME}.tokenizer": tokenizer,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            f"{PACKAGE_NAME}.model", MODEL_ROOT / "model.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


MODEL_MODULE = load_model_module()


class Logger:
    def __init__(self):
        self.warnings = []

    def log_warn(self, message):
        self.warnings.append(message)


class ModelApmTransactionTests(unittest.TestCase):
    def make_model(self, client):
        model = object.__new__(MODEL_MODULE.TritonPythonModel)
        model._apm_client = client
        model._failed_requests = 0
        model.logger = Logger()
        return model

    def test_successful_batch_ends_successfully(self):
        client = mock.Mock()
        model = self.make_model(client)
        model._execute = mock.Mock(return_value=["response"])

        result = model.execute(["request"])

        self.assertEqual(result, ["response"])
        client.begin_transaction.assert_called_once_with("inference")
        client.end_transaction.assert_called_once_with(
            "whisper_bls.execute", "success"
        )

    def test_handled_request_failure_marks_batch_failed(self):
        client = mock.Mock()
        model = self.make_model(client)

        def execute_with_handled_failure(requests):
            model._failed_requests += 1
            return ["error response"]

        model._execute = execute_with_handled_failure
        self.assertEqual(model.execute(["request"]), ["error response"])
        client.end_transaction.assert_called_once_with(
            "whisper_bls.execute", "failure"
        )

    def test_unhandled_failure_is_captured_and_reraised(self):
        client = mock.Mock()
        model = self.make_model(client)
        model._execute = mock.Mock(side_effect=RuntimeError("inference failed"))

        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            model.execute(["request"])

        client.capture_exception.assert_called_once_with(handled=True)
        client.end_transaction.assert_called_once_with(
            "whisper_bls.execute", "failure"
        )

    def test_transaction_start_failure_does_not_block_inference(self):
        client = mock.Mock()
        client.begin_transaction.side_effect = RuntimeError("APM unavailable")
        model = self.make_model(client)
        model._execute = mock.Mock(return_value=["response"])

        self.assertEqual(model.execute(["request"]), ["response"])
        client.end_transaction.assert_not_called()
        self.assertEqual(len(model.logger.warnings), 1)

    def test_transaction_end_failure_does_not_replace_response(self):
        client = mock.Mock()
        client.end_transaction.side_effect = RuntimeError("APM unavailable")
        model = self.make_model(client)
        model._execute = mock.Mock(return_value=["response"])

        self.assertEqual(model.execute(["request"]), ["response"])
        self.assertEqual(len(model.logger.warnings), 1)

    def test_finalize_closes_client_safely(self):
        client = mock.Mock()
        model = self.make_model(client)
        model.finalize()
        client.close.assert_called_once_with()

        client.close.side_effect = RuntimeError("APM unavailable")
        model.finalize()
        self.assertEqual(len(model.logger.warnings), 1)


if __name__ == "__main__":
    unittest.main()
