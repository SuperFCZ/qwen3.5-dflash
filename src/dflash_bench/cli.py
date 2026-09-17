"""Command line interface."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from .benchmark import run_benchmark, write_result
from .config import (
    ConfigError,
    ExperimentConfig,
    load_config,
    render_shell_command,
    validate_config,
)
from .report import comparison_markdown, load_result
from .server import ManagedServer, ServerError, server_is_ready


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dflash-bench",
        description="Benchmark quantized Qwen3.5-4B DFlash drafters with vLLM.",
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    validate = subparsers.add_parser("validate", help="validate one or more TOML configs")
    validate.add_argument("configs", nargs="+")

    command = subparsers.add_parser("command", help="print the exact vLLM launch command")
    command.add_argument("config")
    command.add_argument("--json", action="store_true", help="emit machine-readable metadata")

    run = subparsers.add_parser("run", help="launch vLLM and run a benchmark")
    run.add_argument("config")
    run.add_argument("--no-launch", action="store_true", help="benchmark an already-running server")
    run.add_argument("--base-url", help="server URL used with --no-launch")
    run.add_argument("--output", help="result JSON path")
    run.add_argument("--prompts", help="override the configured JSONL prompt file")
    run.add_argument("--concurrency", type=int)
    run.add_argument("--repetitions", type=int)
    run.add_argument("--max-tokens", type=int)
    run.add_argument("--warmup-requests", type=int)
    run.add_argument("--k", type=int, help="override num_speculative_tokens")

    compare = subparsers.add_parser("compare", help="compare result JSON files")
    compare.add_argument("results", nargs="+")
    compare.add_argument("--label", action="append", dest="labels")
    compare.add_argument("--output", help="write Markdown to this path")
    return parser


def _override(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    bench_changes = {}
    for argument, field_name in (
        ("prompts", "prompt_file"),
        ("concurrency", "concurrency"),
        ("repetitions", "repetitions"),
        ("max_tokens", "max_tokens"),
        ("warmup_requests", "warmup_requests"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            if argument == "prompts":
                value = str(Path(value).expanduser().resolve())
            bench_changes[field_name] = value
    benchmark = dataclasses.replace(config.benchmark, **bench_changes)
    speculative = config.speculative
    if getattr(args, "k", None) is not None:
        if speculative is None:
            raise ConfigError("--k cannot be used with a target-only config")
        speculative = dataclasses.replace(speculative, num_speculative_tokens=args.k)
    result = dataclasses.replace(config, benchmark=benchmark, speculative=speculative)
    validate_config(result)
    return result


def _default_output(config: ExperimentConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("results") / f"{config.name}-{timestamp}.json"


def _validate_base_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError(f"invalid base URL: {value!r}")
    return value.rstrip("/")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command_name == "validate":
            for item in args.configs:
                config = load_config(item)
                print(f"OK  {item}  ({config.name})")
            return 0

        if args.command_name == "command":
            config = load_config(args.config)
            shell = render_shell_command(config)
            if args.json:
                print(json.dumps({"name": config.name, "command": shell}, ensure_ascii=False))
            else:
                print(shell)
            return 0

        if args.command_name == "compare":
            results = [load_result(item) for item in args.results]
            markdown = comparison_markdown(results, args.labels)
            if args.output:
                destination = Path(args.output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(markdown, encoding="utf-8")
                print(destination)
            else:
                print(markdown)
            return 0

        config = _override(load_config(args.config), args)
        output = Path(args.output) if args.output else _default_output(config)
        log_path = output.with_suffix(".server.log")
        base_url = _validate_base_url(args.base_url) if args.base_url else config.base_url
        if args.no_launch:
            if not server_is_ready(base_url):
                raise ServerError(f"no healthy vLLM server at {base_url}")
            manager = nullcontext()
        else:
            if args.base_url:
                raise ConfigError("--base-url is only valid with --no-launch")
            print(f"Launching: {render_shell_command(config)}", flush=True)
            print(f"Server log: {log_path}", flush=True)
            manager = ManagedServer(config, log_path)
        with manager:
            result = run_benchmark(config, base_url=base_url)
        destination = write_result(result, output)
        aggregate = result["aggregate"]
        print(f"Result: {destination}")
        completed = aggregate["completed_requests"]
        planned = aggregate["planned_requests"]
        print(
            f"Completed {completed}/{planned} requests; "
            f"{aggregate['output_throughput_tokens_per_s']:.2f} output tok/s"
        )
        return 0 if aggregate["failed_requests"] == 0 else 2
    except (ConfigError, ServerError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
