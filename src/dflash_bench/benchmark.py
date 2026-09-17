"""Benchmark orchestration and result aggregation."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .client import CompletionResult, chat_completion
from .config import ExperimentConfig, config_as_dict, render_shell_command
from .gpu import GpuMonitor
from .prometheus import parse_prometheus, speculative_stats


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    text: str


def load_prompts(path: str | Path) -> list[Prompt]:
    source = Path(path)
    prompts: list[Prompt] = []
    seen: set[str] = set()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read prompt file {source}: {exc}") from exc
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
        prompt_id = str(item.get("id", f"line-{line_number}"))
        text = item.get("prompt")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{source}:{line_number}: 'prompt' must be a non-empty string")
        if prompt_id in seen:
            raise ValueError(f"{source}:{line_number}: duplicate id {prompt_id!r}")
        seen.add(prompt_id)
        prompts.append(Prompt(prompt_id, text))
    if not prompts:
        raise ValueError(f"{source} contains no prompts")
    return prompts


def fetch_metrics(base_url: str, timeout_s: float = 10.0) -> str:
    try:
        with urllib.request.urlopen(
            f"{base_url.rstrip('/')}/metrics", timeout=timeout_s
        ) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"cannot read vLLM metrics: {exc}") from exc


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    location = (len(ordered) - 1) * percentile
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _timing(values: list[float]) -> dict[str, float | None]:
    return {
        "mean_ms": statistics.fmean(values) * 1000 if values else None,
        "p50_ms": _percentile(values, 0.50) * 1000 if values else None,
        "p95_ms": _percentile(values, 0.95) * 1000 if values else None,
    }


def _hardware() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        result["gpus"] = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        result["gpus"] = []
    return result


def _one_request(
    config: ExperimentConfig,
    base_url: str,
    prompt: Prompt,
    repetition: int,
    *,
    max_tokens: int | None = None,
) -> CompletionResult:
    bench = config.benchmark
    return chat_completion(
        base_url=base_url,
        model=config.server.served_model_name,
        prompt_id=prompt.prompt_id,
        prompt=prompt.text,
        repetition=repetition,
        max_tokens=max_tokens or bench.max_tokens,
        temperature=bench.temperature,
        seed=bench.seed,
        timeout_s=bench.request_timeout_s,
    )


def run_benchmark(config: ExperimentConfig, *, base_url: str | None = None) -> dict[str, Any]:
    base_url = (base_url or config.base_url).rstrip("/")
    prompt_path = config.prompt_path()
    prompts = load_prompts(prompt_path)
    bench = config.benchmark

    warmup_errors: list[str] = []
    for index in range(bench.warmup_requests):
        # Do not warm up with measured prompts: prefix-cache hits would make the
        # first measured requests artificially cheap.
        prompt = Prompt(
            f"__warmup_{index}",
            f"Warm-up request {index}. Briefly explain why deterministic benchmarks "
            "need a warm-up phase.",
        )
        try:
            _one_request(config, base_url, prompt, -1, max_tokens=min(32, bench.max_tokens))
        except Exception as exc:  # keep the measured run useful when a single warmup fails
            warmup_errors.append(f"{type(exc).__name__}: {exc}")

    before = parse_prometheus(fetch_metrics(base_url))
    jobs = [(prompt, repetition) for repetition in range(bench.repetitions) for prompt in prompts]
    results: list[CompletionResult] = []
    errors: list[dict[str, Any]] = []
    start = time.perf_counter()
    with GpuMonitor(bench.gpu_index, bench.gpu_poll_interval_s) as monitor:
        with ThreadPoolExecutor(max_workers=bench.concurrency) as executor:
            future_to_job = {
                executor.submit(_one_request, config, base_url, prompt, repetition): (
                    prompt,
                    repetition,
                )
                for prompt, repetition in jobs
            }
            for future in as_completed(future_to_job):
                prompt, repetition = future_to_job[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append(
                        {
                            "prompt_id": prompt.prompt_id,
                            "repetition": repetition,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    wall_s = time.perf_counter() - start
    after = parse_prometheus(fetch_metrics(base_url))
    spec = speculative_stats(before, after)

    results.sort(key=lambda item: (item.repetition, item.prompt_id))
    total_tokens = sum(item.output_tokens for item in results)
    tpots = [item.tpot_s for item in results if item.tpot_s is not None]
    aggregate = {
        "planned_requests": len(jobs),
        "completed_requests": len(results),
        "failed_requests": len(errors),
        "wall_time_s": wall_s,
        "request_throughput_per_s": len(results) / wall_s if wall_s else None,
        "output_throughput_tokens_per_s": total_tokens / wall_s if wall_s else None,
        "total_output_tokens": total_tokens,
        "latency": _timing([item.latency_s for item in results]),
        "ttft": _timing([item.ttft_s for item in results]),
        "tpot": _timing(tpots),
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": config_as_dict(config),
        "server_command": render_shell_command(config),
        "workload": {
            "prompt_file": str(prompt_path),
            "prompt_file_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
            "unique_prompts": len(prompts),
        },
        "base_url": base_url,
        "hardware": _hardware(),
        "aggregate": aggregate,
        "speculative": spec.as_dict() if spec else None,
        "gpu": monitor.summary(),
        "warmup_errors": warmup_errors,
        "errors": errors,
        "requests": [item.as_dict() for item in results],
    }


def write_result(result: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
