"""Best-effort Elastic APM support for the Whisper Triton backend."""

import math
import os
import urllib.request
from typing import Dict


_DEFAULT_TRITON_METRICS_URL = "http://localhost:8002/metrics"
_TRITON_GPU_METRIC_NAMES = {
    "nv_gpu_utilization",
    "nv_gpu_memory_total_bytes",
    "nv_gpu_memory_used_bytes",
}


def _triton_metrics_url() -> str:
    return os.getenv("TRITON_METRICS_URL") or _DEFAULT_TRITON_METRICS_URL


def read_triton_gpu_metrics() -> Dict[str, float]:
    """Read and aggregate Triton's per-GPU Prometheus gauges."""
    result = {"triton.gpu.metrics.available": 0.0}
    try:
        with urllib.request.urlopen(_triton_metrics_url(), timeout=2) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except Exception:
        # Triton may still be starting or metrics may be disabled. Metric
        # collection must never interfere with inference.
        return result

    samples: Dict[str, Dict[str, float]] = {
        name: {} for name in _TRITON_GPU_METRIC_NAMES
    }
    for line in payload.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        series = fields[0]
        name = series.partition("{")[0]
        if name not in samples:
            continue
        try:
            value = float(fields[1])
        except ValueError:
            continue
        if not math.isfinite(value):
            continue

        gpu_id = None
        labels = series.partition("{")[2].rpartition("}")[0]
        for key in ("gpu_uuid", "gpu"):
            marker = f'{key}="'
            start = labels.find(marker)
            if start >= 0:
                start += len(marker)
                end = labels.find('"', start)
                if end >= 0:
                    gpu_id = f"{key}:{labels[start:end]}"
                    break
        if gpu_id is None:
            gpu_id = f"sample:{len(samples[name])}"
        # Duplicate exposition rows for a GPU replace the prior value.
        samples[name][gpu_id] = value

    utilization = samples["nv_gpu_utilization"]
    memory_total = samples["nv_gpu_memory_total_bytes"]
    memory_used = samples["nv_gpu_memory_used_bytes"]
    gpu_ids = set(utilization) | set(memory_total) | set(memory_used)
    if not gpu_ids:
        return result

    result["triton.gpu.metrics.available"] = 1.0
    result["triton.gpu.count"] = float(len(gpu_ids))
    if utilization:
        result["triton.gpu.utilization.pct"] = sum(utilization.values()) / len(
            utilization
        )
    if memory_total:
        result["triton.gpu.memory.total.bytes"] = sum(memory_total.values())
    if memory_used:
        result["triton.gpu.memory.used.bytes"] = sum(memory_used.values())
    if memory_total and memory_total.keys() == memory_used.keys():
        total = result["triton.gpu.memory.total.bytes"]
        used = result["triton.gpu.memory.used.bytes"]
        result["triton.gpu.memory.free.bytes"] = max(total - used, 0.0)
        if total > 0:
            result["triton.gpu.memory.used.pct"] = used / total
    return result


def register_apm_gpu_metrics(client) -> bool:
    """Register periodic Triton GPU gauges with an Elastic APM client."""
    try:
        from elasticapm.metrics.base_metrics import MetricSet

        class TritonGpuMetricSet(MetricSet):
            def before_collect(self):
                values = read_triton_gpu_metrics()
                self._current_gpu_metric_names = set(values)
                for name, value in values.items():
                    # elastic-apm 6.26.x exposes Gauge.val as its setter.
                    self.gauge(name).val = value

            def before_yield(self, data):
                # Gauges retain their previous value. Do not resend stale GPU
                # values when a later scrape is unavailable or partial.
                current = getattr(self, "_current_gpu_metric_names", set())
                for name in list(data.get("samples", {})):
                    if name.startswith("triton.gpu.") and name not in current:
                        data["samples"].pop(name, None)
                return data

        client.metrics.register(TritonGpuMetricSet)
    except Exception as exc:
        print(
            f"[whisper_bls] WARNING: Elastic APM GPU metric registration "
            f"failed: {exc}",
            flush=True,
        )
        return False
    return True


def init_apm():
    """Create an Elastic APM client when a server URL is configured."""
    server_url = os.getenv("ELASTIC_APM_SERVER_URL", "").strip()
    if not server_url:
        print(
            "[whisper_bls] Elastic APM disabled "
            "(ELASTIC_APM_SERVER_URL is not set).",
            flush=True,
        )
        return None

    try:
        from elasticapm import Client
    except ImportError:
        print(
            "[whisper_bls] WARNING: ELASTIC_APM_SERVER_URL is set, but "
            "elastic-apm is not installed; Elastic APM is disabled.",
            flush=True,
        )
        return None

    service_name = os.getenv("ELASTIC_APM_SERVICE_NAME") or "whisper"
    environment = (
        os.getenv("ELASTIC_APM_ENVIRONMENT")
        or os.getenv("ENV_VERSION")
        or "production"
    )
    config = {
        "SERVICE_NAME": service_name,
        "SERVER_URL": server_url,
        "ENVIRONMENT": environment,
        # Inference errors can contain audio arrays and transcripts in local
        # variables. Keep those values out of APM error events.
        "COLLECT_LOCAL_VARIABLES": "off",
        "CAPTURE_BODY": "off",
        "CAPTURE_HEADERS": False,
    }

    api_key = os.getenv("ELASTIC_APM_API_KEY", "").strip()
    secret_token = os.getenv("ELASTIC_APM_SECRET_TOKEN", "").strip()
    if api_key:
        config["API_KEY"] = api_key
    elif secret_token:
        config["SECRET_TOKEN"] = secret_token

    try:
        client = Client(config)
    except Exception as exc:
        print(
            f"[whisper_bls] WARNING: Elastic APM initialization failed: {exc}",
            flush=True,
        )
        return None

    print(
        f"[whisper_bls] Elastic APM enabled: service={service_name} "
        f"environment={environment} server={server_url}",
        flush=True,
    )
    if register_apm_gpu_metrics(client):
        print(
            f"[whisper_bls] Elastic APM GPU metrics enabled: "
            f"source={_triton_metrics_url()}",
            flush=True,
        )
    return client


def capture_apm_exception(client) -> None:
    """Capture the active exception without allowing APM to affect inference."""
    if client is None:
        return
    try:
        client.capture_exception(handled=True)
    except Exception as exc:
        print(
            f"[whisper_bls] WARNING: Elastic APM exception capture failed: {exc}",
            flush=True,
        )
