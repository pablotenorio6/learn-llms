"""GPU exporter mínimo basado en nvidia-smi.

Expone en :9835/metrics gauges Prometheus por GPU:
  - gpu_memory_used_mb
  - gpu_memory_total_mb
  - gpu_memory_free_mb
  - gpu_utilization_percent
  - gpu_temperature_celsius
  - gpu_power_draw_watts
  - gpu_fan_speed_percent

dcgm-exporter es el estándar pero en Windows + Docker Desktop + WSL2 da más
guerra de la que merece para una sola GPU local. Este script hace lo justo,
soporta multi-GPU y se acaba en ~70 líneas.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

from prometheus_client import Gauge, start_http_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s gpu-exporter: %(message)s",
)
log = logging.getLogger(__name__)

PORT = int(os.getenv("GPU_EXPORTER_PORT", "9835"))
INTERVAL = float(os.getenv("GPU_SCRAPE_INTERVAL", "5"))

# nvidia-smi --query-gpu= soporta exactamente estos campos.
QUERY_FIELDS = [
    ("index", "index"),
    ("name", "name"),
    ("uuid", "uuid"),
    ("memory.used", "memory_used_mb"),
    ("memory.total", "memory_total_mb"),
    ("memory.free", "memory_free_mb"),
    ("utilization.gpu", "utilization_percent"),
    ("temperature.gpu", "temperature_celsius"),
    ("power.draw", "power_draw_watts"),
    ("fan.speed", "fan_speed_percent"),
]

LABELS = ("gpu", "name", "uuid")

# Gauges por métrica numérica.
gauges: dict[str, Gauge] = {
    "memory_used_mb": Gauge("gpu_memory_used_mb", "VRAM usada (MB)", LABELS),
    "memory_total_mb": Gauge("gpu_memory_total_mb", "VRAM total (MB)", LABELS),
    "memory_free_mb": Gauge("gpu_memory_free_mb", "VRAM libre (MB)", LABELS),
    "utilization_percent": Gauge("gpu_utilization_percent", "Utilización del GPU (%)", LABELS),
    "temperature_celsius": Gauge("gpu_temperature_celsius", "Temperatura (°C)", LABELS),
    "power_draw_watts": Gauge("gpu_power_draw_watts", "Potencia draw (W)", LABELS),
    "fan_speed_percent": Gauge("gpu_fan_speed_percent", "Velocidad del ventilador (%)", LABELS),
}


def _query_once() -> list[dict[str, str]]:
    fields_csv = ",".join(q for q, _ in QUERY_FIELDS)
    cmd = [
        "nvidia-smi",
        f"--query-gpu={fields_csv}",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=10, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        log.error("nvidia-smi no encontrado en el contenedor")
        return []
    except subprocess.CalledProcessError as e:
        log.warning("nvidia-smi falló: %s", e.output.strip() if e.output else e)
        return []
    except subprocess.TimeoutExpired:
        log.warning("nvidia-smi tardó >10s, saltando este ciclo")
        return []

    rows: list[dict[str, str]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(QUERY_FIELDS):
            log.warning("nvidia-smi: línea con %d campos esperados %d: %r",
                        len(parts), len(QUERY_FIELDS), line)
            continue
        rows.append({key: parts[i] for i, (_, key) in enumerate(QUERY_FIELDS)})
    return rows


def _to_float(s: str) -> float | None:
    """Convierte cadenas tipo '52', '52.3', 'N/A', '[Not Supported]' → float|None."""
    if not s or s.lower() in ("n/a", "[not supported]", "[insufficient permissions]"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def scrape_loop() -> None:
    while True:
        rows = _query_once()
        for row in rows:
            labels = {
                "gpu": row.get("index", "?"),
                "name": row.get("name", "unknown"),
                "uuid": row.get("uuid", "unknown"),
            }
            for key, gauge in gauges.items():
                val = _to_float(row.get(key, ""))
                if val is None:
                    continue
                gauge.labels(**labels).set(val)
        time.sleep(INTERVAL)


def main() -> int:
    log.info("starting GPU exporter on :%d (interval=%.1fs)", PORT, INTERVAL)
    start_http_server(PORT)
    try:
        scrape_loop()
    except KeyboardInterrupt:
        log.info("shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
