"""Best-effort NVIDIA GPU telemetry collection."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSample:
    timestamp_s: float
    memory_used_mib: float
    utilization_pct: float
    power_w: float
    temperature_c: float


class GpuMonitor:
    def __init__(self, gpu_index: int = 0, interval_s: float = 0.5):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.samples: list[GpuSample] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> GpuMonitor:
        if shutil.which("nvidia-smi") is None:
            self.error = "nvidia-smi not found"
            return self
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.interval_s * 3))

    def _run(self) -> None:
        query = "memory.used,utilization.gpu,power.draw,temperature.gpu"
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.gpu_index}",
                        f"--query-gpu={query}",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                fields = [
                    float(item.strip()) for item in completed.stdout.splitlines()[0].split(",")
                ]
                self.samples.append(GpuSample(time.time(), *fields))
            except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                return
            self._stop.wait(self.interval_s)

    def summary(self) -> dict[str, object]:
        if not self.samples:
            return {"available": False, "error": self.error}
        return {
            "available": True,
            "sample_count": len(self.samples),
            "peak_memory_used_mib": max(v.memory_used_mib for v in self.samples),
            "mean_gpu_utilization_pct": sum(v.utilization_pct for v in self.samples)
            / len(self.samples),
            "peak_power_w": max(v.power_w for v in self.samples),
            "peak_temperature_c": max(v.temperature_c for v in self.samples),
            "error": self.error,
        }
