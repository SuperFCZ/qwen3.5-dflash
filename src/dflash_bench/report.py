"""Markdown comparison reports for benchmark JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_result(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        result = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read result {source}: {exc}") from exc
    if result.get("schema_version") != 1:
        raise ValueError(f"{source}: unsupported or missing schema_version")
    return result


def _get(data: dict[str, Any], *path: str) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _number(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}{suffix}"


def _percent(value: Any) -> str:
    return _number(float(value) * 100, 2, "%") if value is not None else "—"


def exact_match(baseline: dict[str, Any], candidate: dict[str, Any]) -> tuple[int, int]:
    reference = {
        (item["prompt_id"], item["repetition"]): item.get("text", "")
        for item in baseline.get("requests", [])
    }
    matches = 0
    total = 0
    for item in candidate.get("requests", []):
        key = (item["prompt_id"], item["repetition"])
        if key not in reference:
            continue
        total += 1
        matches += item.get("text", "") == reference[key]
    return matches, total


def comparison_markdown(results: list[dict[str, Any]], labels: list[str] | None = None) -> str:
    if not results:
        raise ValueError("at least one result is required")
    if labels and len(labels) != len(results):
        raise ValueError("labels and results must have the same length")
    baseline = results[0]
    baseline_target = _get(baseline, "experiment", "server", "model")
    baseline_prompts = _get(baseline, "workload", "prompt_file_sha256")
    warnings: list[str] = []
    rows: list[str] = []
    for index, result in enumerate(results):
        experiment = result.get("experiment", {})
        label = labels[index] if labels else experiment.get("name", f"result-{index + 1}")
        speculative = result.get("speculative") or {}
        spec_config = experiment.get("speculative") or {}
        gpu = result.get("gpu") or {}
        matches, total = exact_match(baseline, result)
        match = f"{matches}/{total}" if total else "—"
        target = _get(result, "experiment", "server", "model")
        prompt_hash = _get(result, "workload", "prompt_file_sha256")
        if index and baseline_target and target and target != baseline_target:
            warnings.append(f"{label}: target differs from the baseline ({target}).")
        if index and baseline_prompts and prompt_hash and prompt_hash != baseline_prompts:
            warnings.append(f"{label}: prompt-file SHA-256 differs from the baseline.")
        rows.append(
            "| "
            + " | ".join(
                [
                    str(label),
                    str(spec_config.get("num_speculative_tokens", "—")),
                    _number(speculative.get("mean_accepted_draft_tokens"), 3),
                    _number(speculative.get("mean_accept_length"), 3),
                    _percent(speculative.get("acceptance_rate")),
                    _number(_get(result, "aggregate", "request_throughput_per_s"), 3),
                    _number(_get(result, "aggregate", "output_throughput_tokens_per_s"), 1),
                    _number(_get(result, "aggregate", "ttft", "p50_ms"), 1),
                    _number(_get(result, "aggregate", "tpot", "p50_ms"), 2),
                    _number(gpu.get("peak_memory_used_mib"), 0),
                    match,
                ]
            )
            + " |"
        )

    lines = [
        "# DFlash benchmark comparison",
        "",
        "The first result is the exact-output baseline. Metrics are computed from the measured",
        "request interval only; warm-up traffic is excluded from Prometheus counter deltas.",
        "",
        *(["Warnings:", *(f"- {warning}" for warning in warnings), ""] if warnings else []),
        "| Experiment | K | Accepted/step | Mean length (+bonus) | Acceptance | req/s | "
        "output tok/s | p50 TTFT ms | p50 TPOT ms | Peak VRAM MiB | Exact match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "Exact match is only meaningful for greedy runs using the same target, prompts, and",
        "generation settings. The Nota W4 target/drafter track is a system comparison and must",
        "not be interpreted as a pure W4-versus-W8 drafter quantization comparison.",
        "",
    ]
    return "\n".join(lines)
