import importlib.util
import os
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
APM_SPEC = importlib.util.spec_from_file_location(
    "whisper_apm_under_test", MODEL_ROOT / "apm.py"
)
apm = importlib.util.module_from_spec(APM_SPEC)
APM_SPEC.loader.exec_module(apm)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload


class ApmMetricTests(unittest.TestCase):
    def test_gpu_metrics_are_aggregated_by_gpu(self):
        payload = """
        nv_gpu_utilization{gpu_uuid="one"} 0.25
        nv_gpu_utilization{gpu_uuid="two"} 0.75
        nv_gpu_memory_total_bytes{gpu_uuid="one"} 100
        nv_gpu_memory_total_bytes{gpu_uuid="two"} 300
        nv_gpu_memory_used_bytes{gpu_uuid="one"} 40
        nv_gpu_memory_used_bytes{gpu_uuid="two"} 210
        """
        with mock.patch.object(
            apm.urllib.request, "urlopen", return_value=FakeResponse(payload)
        ) as urlopen:
            metrics = apm.read_triton_gpu_metrics()

        urlopen.assert_called_once_with("http://localhost:8002/metrics", timeout=2)
        self.assertEqual(metrics["triton.gpu.metrics.available"], 1.0)
        self.assertEqual(metrics["triton.gpu.count"], 2.0)
        self.assertEqual(metrics["triton.gpu.utilization.pct"], 0.5)
        self.assertEqual(metrics["triton.gpu.memory.total.bytes"], 400.0)
        self.assertEqual(metrics["triton.gpu.memory.used.bytes"], 250.0)
        self.assertEqual(metrics["triton.gpu.memory.free.bytes"], 150.0)
        self.assertEqual(metrics["triton.gpu.memory.used.pct"], 0.625)

    def test_duplicate_gpu_series_replaces_the_previous_value(self):
        payload = """
        nv_gpu_utilization{gpu="0"} 0.1
        nv_gpu_utilization{gpu="0"} 0.8
        """
        with mock.patch.object(
            apm.urllib.request, "urlopen", return_value=FakeResponse(payload)
        ):
            metrics = apm.read_triton_gpu_metrics()

        self.assertEqual(metrics["triton.gpu.count"], 1.0)
        self.assertEqual(metrics["triton.gpu.utilization.pct"], 0.8)

    def test_unavailable_metrics_endpoint_is_non_fatal(self):
        with mock.patch.object(
            apm.urllib.request, "urlopen", side_effect=OSError("unavailable")
        ):
            metrics = apm.read_triton_gpu_metrics()

        self.assertEqual(metrics, {"triton.gpu.metrics.available": 0.0})

    def test_metric_set_removes_stale_gpu_values(self):
        class Gauge:
            val = None

        class MetricSet:
            def __init__(self):
                self.gauges = {}

            def gauge(self, name):
                self.gauges.setdefault(name, Gauge())
                return self.gauges[name]

        registered = []
        client = types.SimpleNamespace(
            metrics=types.SimpleNamespace(register=registered.append)
        )
        elasticapm = types.ModuleType("elasticapm")
        metrics_module = types.ModuleType("elasticapm.metrics")
        base_metrics = types.ModuleType("elasticapm.metrics.base_metrics")
        base_metrics.MetricSet = MetricSet
        modules = {
            "elasticapm": elasticapm,
            "elasticapm.metrics": metrics_module,
            "elasticapm.metrics.base_metrics": base_metrics,
        }
        with mock.patch.dict(sys.modules, modules), mock.patch.object(
            apm,
            "read_triton_gpu_metrics",
            return_value={"triton.gpu.metrics.available": 0.0},
        ):
            self.assertTrue(apm.register_apm_gpu_metrics(client))
            metric_set = registered[0]()
            metric_set.before_collect()
            data = {
                "samples": {
                    "triton.gpu.metrics.available": 0.0,
                    "triton.gpu.memory.used.bytes": 100.0,
                    "python.gc.count": 1.0,
                }
            }
            result = metric_set.before_yield(data)

        self.assertNotIn("triton.gpu.memory.used.bytes", result["samples"])
        self.assertIn("triton.gpu.metrics.available", result["samples"])
        self.assertIn("python.gc.count", result["samples"])


class ApmInitializationTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_apm_is_disabled_without_server_url(self):
        self.assertIsNone(apm.init_apm())

    def test_client_uses_safe_defaults_and_api_key_precedence(self):
        configs = []

        class Client:
            def __init__(self, config):
                configs.append(config)

        elasticapm = types.ModuleType("elasticapm")
        elasticapm.Client = Client
        os.environ.update(
            {
                "ELASTIC_APM_SERVER_URL": "http://apm:8200",
                "ELASTIC_APM_API_KEY": "api-key",
                "ELASTIC_APM_SECRET_TOKEN": "secret",
                "ENV_VERSION": "stage",
            }
        )
        with mock.patch.dict(sys.modules, {"elasticapm": elasticapm}), mock.patch.object(
            apm, "register_apm_gpu_metrics", return_value=True
        ):
            client = apm.init_apm()

        self.assertIsInstance(client, Client)
        self.assertEqual(configs[0]["SERVICE_NAME"], "whisper")
        self.assertEqual(configs[0]["ENVIRONMENT"], "stage")
        self.assertEqual(configs[0]["API_KEY"], "api-key")
        self.assertNotIn("SECRET_TOKEN", configs[0])
        self.assertEqual(configs[0]["COLLECT_LOCAL_VARIABLES"], "off")
        self.assertEqual(configs[0]["CAPTURE_BODY"], "off")
        self.assertFalse(configs[0]["CAPTURE_HEADERS"])

    def test_client_initialization_failure_is_non_fatal(self):
        class Client:
            def __init__(self, config):
                raise RuntimeError("bad config")

        elasticapm = types.ModuleType("elasticapm")
        elasticapm.Client = Client
        os.environ["ELASTIC_APM_SERVER_URL"] = "http://apm:8200"
        with mock.patch.dict(sys.modules, {"elasticapm": elasticapm}):
            self.assertIsNone(apm.init_apm())

    def test_exception_capture_failure_is_non_fatal(self):
        client = mock.Mock()
        client.capture_exception.side_effect = RuntimeError("offline")
        apm.capture_apm_exception(client)
        client.capture_exception.assert_called_once_with(handled=True)


if __name__ == "__main__":
    unittest.main()
