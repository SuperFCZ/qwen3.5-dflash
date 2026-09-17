"""Managed vLLM server process."""

from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import ExperimentConfig, render_vllm_command


class ServerError(RuntimeError):
    pass


def server_is_ready(base_url: str, timeout_s: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(
            f"{base_url.rstrip('/')}/health", timeout=timeout_s
        ) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError):
        return False


class ManagedServer:
    def __init__(self, config: ExperimentConfig, log_path: str | Path):
        self.config = config
        self.log_path = Path(log_path)
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle = None

    def __enter__(self) -> ManagedServer:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()

    def start(self) -> None:
        if server_is_ready(self.config.base_url):
            raise ServerError(
                f"{self.config.base_url} is already serving; use --no-launch to benchmark it"
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("wb")
        environment = os.environ.copy()
        environment.update(self.config.server.environment)
        try:
            self.process = subprocess.Popen(
                render_vllm_command(self.config),
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
        except OSError as exc:
            self._log_handle.close()
            self._log_handle = None
            raise ServerError(f"could not launch vLLM: {exc}") from exc

        deadline = time.monotonic() + self.config.benchmark.startup_timeout_s
        while time.monotonic() < deadline:
            if server_is_ready(self.config.base_url):
                return
            if self.process.poll() is not None:
                message = (
                    f"vLLM exited with code {self.process.returncode}. Log tail:\n{self.log_tail()}"
                )
                self.stop()
                raise ServerError(message)
            time.sleep(1.0)
        message = (
            f"vLLM did not become healthy within {self.config.benchmark.startup_timeout_s}s. "
            f"Log tail:\n{self.log_tail()}"
        )
        self.stop()
        raise ServerError(message)

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

    def log_tail(self, lines: int = 80) -> str:
        if self._log_handle:
            self._log_handle.flush()
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return "<log unavailable>"
