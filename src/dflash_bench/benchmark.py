"""Benchmark orchestration and result aggregation."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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
    input_field: str = "prompt"
    source_record: dict[str, Any] = field(default_factory=dict)


def _prompt_id_sort_key(value: object) -> tuple[tuple[tuple[int, int | str], ...], str, str]:
    """Sort embedded numbers numerically while keeping arbitrary IDs deterministic."""
    text = str(value)
    chunks: list[tuple[int, int | str]] = []
    for chunk in re.split(r"(\d+)", text):
        if not chunk:
            continue
        chunks.append((0, int(chunk)) if chunk.isdigit() else (1, chunk.casefold()))
    return tuple(chunks), text.casefold(), text


def discover_prompt_files(path: str | Path) -> list[Path]:
    """Return one JSONL file, or the sorted JSONL files directly inside a directory."""
    source = Path(path).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() != ".jsonl":
            raise ValueError(f"prompt file must have a .jsonl suffix: {source}")
        return [source]
    if source.is_dir():
        try:
            files = sorted(
                (
                    item.resolve()
                    for item in source.iterdir()
                    if item.is_file() and item.suffix.lower() == ".jsonl"
                ),
                key=lambda item: item.name,
            )
        except OSError as exc:
            raise ValueError(f"cannot list prompt directory {source}: {exc}") from exc
        if not files:
            raise ValueError(f"prompt directory {source} contains no .jsonl files")
        return files
    raise ValueError(f"prompt path does not exist: {source}")


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
        if not isinstance(item, dict):
            raise ValueError(f"{source}:{line_number}: each line must be a JSON object")
        raw_id = item.get("id")
        prompt_id = f"line-{line_number}" if raw_id is None else str(raw_id)
        if not prompt_id.strip():
            raise ValueError(f"{source}:{line_number}: 'id' cannot be empty")
        input_field = "prompt"
        text = item.get(input_field)
        if not isinstance(text, str) or not text.strip():
            input_field = "question"
            text = item.get(input_field)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"{source}:{line_number}: 'question' or 'prompt' must be a non-empty string"
            )
        if prompt_id in seen:
            raise ValueError(f"{source}:{line_number}: duplicate id {prompt_id!r}")
        seen.add(prompt_id)
        prompts.append(Prompt(prompt_id, text, input_field, item))
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


def run_benchmark(
    config: ExperimentConfig,
    *,
    base_url: str | None = None,
    prompt_path: str | Path | None = None,
) -> dict[str, Any]:
    base_url = (base_url or config.base_url).rstrip("/")
    source = (
        Path(prompt_path).expanduser().resolve()
        if prompt_path is not None
        else config.prompt_path()
    )
    prompts = load_prompts(source)
    prompts_by_id = {prompt.prompt_id: prompt for prompt in prompts}
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
                            "input_field": prompt.input_field,
                            "input": prompt.text,
                            "source_record": prompt.source_record,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    wall_s = time.perf_counter() - start
    after = parse_prometheus(fetch_metrics(base_url))
    spec = speculative_stats(before, after)

    # Keep every prompt's repeated measurements adjacent in serialized results.
    results.sort(key=lambda item: (_prompt_id_sort_key(item.prompt_id), item.repetition))
    errors.sort(key=lambda item: (_prompt_id_sort_key(item["prompt_id"]), int(item["repetition"])))
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
    request_records: list[dict[str, Any]] = []
    for item in results:
        prompt = prompts_by_id[item.prompt_id]
        record = item.as_dict()
        record.update(
            {
                "input_field": prompt.input_field,
                "input": prompt.text,
                "source_record": prompt.source_record,
            }
        )
        request_records.append(record)

    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": config_as_dict(config),
        "server_command": render_shell_command(config),
        "workload": {
            "prompt_file": str(source),
            "prompt_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "unique_prompts": len(prompts),
        },
        "base_url": base_url,
        "hardware": _hardware(),
        "aggregate": aggregate,
        "speculative": spec.as_dict() if spec else None,
        "gpu": monitor.summary(),
        "warmup_errors": warmup_errors,
        "errors": errors,
        "requests": request_records,
    }


def write_result(result: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def write_responses_jsonl(result: dict[str, Any], path: str | Path) -> Path:
    """Write a compact, line-oriented response file including failed requests."""
    records: list[dict[str, Any]] = []
    for item in result.get("requests", []):
        record = {
            "status": "ok",
            "prompt_id": item["prompt_id"],
            "repetition": item["repetition"],
            "input_field": item["input_field"],
            "input": item["input"],
            "source_record": item["source_record"],
            "response": item["text"],
            "output_tokens": item["output_tokens"],
            "latency_s": item["latency_s"],
            "ttft_s": item["ttft_s"],
            "tpot_s": item["tpot_s"],
            "finish_reason": item["finish_reason"],
        }
        record[str(item["input_field"])] = item["input"]
        records.append(record)
    for item in result.get("errors", []):
        record = {
            "status": "error",
            "prompt_id": item["prompt_id"],
            "repetition": item["repetition"],
            "input_field": item["input_field"],
            "input": item["input"],
            "source_record": item["source_record"],
            "error": item["error"],
        }
        record[str(item["input_field"])] = item["input"]
        records.append(record)
    records.sort(
        key=lambda item: (
            _prompt_id_sort_key(item["prompt_id"]),
            int(item["repetition"]),
            str(item["status"]),
        )
    )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records)
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)
    return destination
