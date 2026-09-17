"""Experiment configuration loading and validation."""

from __future__ import annotations

import dataclasses
import json
import shlex
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment file is incomplete or inconsistent."""


@dataclass(frozen=True)
class SpeculativeConfig:
    model: str
    method: str = "dflash"
    num_speculative_tokens: int = 15
    quantization: str | None = None
    revision: str | None = None
    dtype: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_vllm_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "method": self.method,
            "model": self.model,
            "num_speculative_tokens": self.num_speculative_tokens,
        }
        for key in ("quantization", "revision", "dtype"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        data.update(self.extra)
        return data


@dataclass(frozen=True)
class ServerConfig:
    model: str
    served_model_name: str = "qwen3.5-4b"
    executable: str = "vllm"
    host: str = "127.0.0.1"
    port: int = 8000
    dtype: str = "bfloat16"
    quantization: str | None = None
    trust_remote_code: bool = True
    max_model_len: int = 8192
    max_num_seqs: int = 4
    max_num_batched_tokens: int = 16384
    gpu_memory_utilization: float = 0.90
    attention_backend: str | None = "FLASH_ATTN"
    enforce_eager: bool = False
    extra_args: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkConfig:
    prompt_file: str = "../prompts/benchmark.jsonl"
    repetitions: int = 1
    warmup_requests: int = 2
    max_tokens: int = 128
    concurrency: int = 1
    temperature: float = 0.0
    seed: int = 0
    request_timeout_s: float = 300.0
    startup_timeout_s: float = 1800.0
    gpu_index: int = 0
    gpu_poll_interval_s: float = 0.5


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    server: ServerConfig
    benchmark: BenchmarkConfig
    speculative: SpeculativeConfig | None = None
    source_path: Path | None = field(default=None, compare=False, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://{self.server.host}:{self.server.port}"

    def prompt_path(self) -> Path:
        path = Path(self.benchmark.prompt_file).expanduser()
        if path.is_absolute():
            return path
        base = self.source_path.parent if self.source_path else Path.cwd()
        return (base / path).resolve()


def _only(mapping: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"unknown key(s) in [{section}]: {', '.join(unknown)}")


def _positive(value: int | float, label: str) -> None:
    if value <= 0:
        raise ConfigError(f"{label} must be > 0, got {value!r}")


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).expanduser().resolve()
    try:
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read {source}: {exc}") from exc

    _only(raw, {"name", "description", "server", "speculative", "benchmark"}, "root")
    if "name" not in raw or "server" not in raw:
        raise ConfigError("the root 'name' and [server] section are required")

    server_raw = dict(raw["server"])
    environment = {str(k): str(v) for k, v in server_raw.pop("environment", {}).items()}
    if "extra_args" in server_raw:
        server_raw["extra_args"] = tuple(str(v) for v in server_raw["extra_args"])
    try:
        server = ServerConfig(environment=environment, **server_raw)
    except TypeError as exc:
        raise ConfigError(f"invalid [server] section: {exc}") from exc

    speculative = None
    if "speculative" in raw:
        spec_raw = dict(raw["speculative"])
        extra = dict(spec_raw.pop("extra", {}))
        try:
            speculative = SpeculativeConfig(extra=extra, **spec_raw)
        except TypeError as exc:
            raise ConfigError(f"invalid [speculative] section: {exc}") from exc

    try:
        benchmark = BenchmarkConfig(**raw.get("benchmark", {}))
    except TypeError as exc:
        raise ConfigError(f"invalid [benchmark] section: {exc}") from exc

    config = ExperimentConfig(
        name=str(raw["name"]),
        description=str(raw.get("description", "")),
        server=server,
        speculative=speculative,
        benchmark=benchmark,
        source_path=source,
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if not config.server.model.strip():
        raise ConfigError("server.model cannot be empty")
    if not 1 <= config.server.port <= 65535:
        raise ConfigError(f"server.port must be in 1..65535, got {config.server.port}")
    if not 0 < config.server.gpu_memory_utilization <= 1:
        raise ConfigError("server.gpu_memory_utilization must be in (0, 1]")
    for value, label in (
        (config.server.max_model_len, "server.max_model_len"),
        (config.server.max_num_seqs, "server.max_num_seqs"),
        (config.server.max_num_batched_tokens, "server.max_num_batched_tokens"),
        (config.benchmark.max_tokens, "benchmark.max_tokens"),
        (config.benchmark.concurrency, "benchmark.concurrency"),
        (config.benchmark.repetitions, "benchmark.repetitions"),
        (config.benchmark.request_timeout_s, "benchmark.request_timeout_s"),
        (config.benchmark.startup_timeout_s, "benchmark.startup_timeout_s"),
        (config.benchmark.gpu_poll_interval_s, "benchmark.gpu_poll_interval_s"),
    ):
        _positive(value, label)
    if config.benchmark.warmup_requests < 0:
        raise ConfigError("benchmark.warmup_requests must be >= 0")
    if config.benchmark.temperature < 0:
        raise ConfigError("benchmark.temperature must be >= 0")
    if config.benchmark.gpu_index < 0:
        raise ConfigError("benchmark.gpu_index must be >= 0")
    if config.speculative:
        _positive(
            config.speculative.num_speculative_tokens,
            "speculative.num_speculative_tokens",
        )
        if config.speculative.method != "dflash":
            raise ConfigError("this harness currently supports speculative.method='dflash'")


def render_vllm_command(config: ExperimentConfig) -> list[str]:
    server = config.server
    command = [
        server.executable,
        "serve",
        server.model,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--served-model-name",
        server.served_model_name,
        "--dtype",
        server.dtype,
        "--max-model-len",
        str(server.max_model_len),
        "--max-num-seqs",
        str(server.max_num_seqs),
        "--max-num-batched-tokens",
        str(server.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(server.gpu_memory_utilization),
    ]
    if server.quantization:
        command.extend(["--quantization", server.quantization])
    if server.trust_remote_code:
        command.append("--trust-remote-code")
    if server.attention_backend:
        command.extend(["--attention-backend", server.attention_backend])
    if server.enforce_eager:
        command.append("--enforce-eager")
    if config.speculative:
        payload = json.dumps(
            config.speculative.as_vllm_dict(), separators=(",", ":"), sort_keys=True
        )
        command.extend(["--speculative-config", payload])
    command.extend(server.extra_args)
    return command


def render_shell_command(config: ExperimentConfig) -> str:
    env = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in sorted(config.server.environment.items())
    )
    command = shlex.join(render_vllm_command(config))
    return f"{env} {command}" if env else command


def config_as_dict(config: ExperimentConfig) -> dict[str, Any]:
    result = dataclasses.asdict(config)
    result.pop("source_path", None)
    return result
