"""Command line interface."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from .benchmark import (
    discover_prompt_files,
    load_prompts,
    run_benchmark,
    write_responses_jsonl,
    write_result,
)
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
    run.add_argument(
        "--output",
        help="result JSON path, or output directory when --prompts is a directory",
    )
    run.add_argument(
        "--prompts",
        help="override the configured JSONL prompt file or directory of JSONL files",
    )
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


def _default_output(config: ExperimentConfig, *, directory: bool = False) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "" if directory else ".json"
    return Path("results") / f"{config.name}-{timestamp}{suffix}"


def _response_output(result_output: Path) -> Path:
    return result_output.with_name(f"{result_output.stem}.responses.jsonl")


def _print_aggregate(label: str, result: dict) -> None:
    aggregate = result["aggregate"]
    completed = aggregate["completed_requests"]
    planned = aggregate["planned_requests"]
    throughput = aggregate["output_throughput_tokens_per_s"]
    throughput_text = f"{throughput:.2f}" if throughput is not None else "n/a"
    print(f"{label}: completed {completed}/{planned}; {throughput_text} output tok/s")
    for record in result.get("cuda_event_profile") or []:
        print(f"CUDA Event worker pid={record['pid']} device={record['device']} (ms per batch):")
        for phase, stats in record["metrics"].items():
            print(
                f"  {phase}: count={stats['count']} mean={stats['mean_ms']} "
                f"p50={stats['p50_ms']} p95={stats['p95_ms']}"
            )


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
        prompt_source = config.prompt_path()
        prompt_files = discover_prompt_files(prompt_source)
        directory_mode = prompt_source.is_dir()
        if directory_mode and len({item.stem for item in prompt_files}) != len(prompt_files):
            raise ConfigError(
                "prompt filenames must have unique stems because output names are stem-based"
            )
        # Validate every shard before spending time loading the model onto the GPU.
        for prompt_file in prompt_files:
            load_prompts(prompt_file)
        output = (
            Path(args.output).expanduser()
            if args.output
            else _default_output(config, directory=directory_mode)
        )
        if directory_mode:
            if output.exists() and not output.is_dir():
                raise ConfigError(
                    f"directory prompt input requires an output directory, but {output} is a file"
                )
            output.mkdir(parents=True, exist_ok=True)
            log_path = output / "server.log"
        else:
            if output.exists() and output.is_dir():
                raise ConfigError(
                    f"single prompt file requires an output JSON path, but {output} is a directory"
                )
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
        failed_requests = 0
        manifest_files: list[dict] = []
        with manager:
            for index, prompt_file in enumerate(prompt_files, 1):
                if directory_mode:
                    print(
                        f"[{index}/{len(prompt_files)}] Benchmarking {prompt_file.name}",
                        flush=True,
                    )
                result = run_benchmark(config, base_url=base_url, prompt_path=prompt_file)
                failed_requests += result["aggregate"]["failed_requests"]

                if directory_mode:
                    result_output = output / f"{prompt_file.stem}.result.json"
                    responses_output = output / f"{prompt_file.stem}.responses.jsonl"
                else:
                    result_output = output
                    responses_output = _response_output(output)
                destination = write_result(result, result_output)
                response_destination = write_responses_jsonl(result, responses_output)
                print(f"Result: {destination}")
                print(f"Responses: {response_destination}")
                _print_aggregate(prompt_file.name, result)

                if directory_mode:
                    manifest_files.append(
                        {
                            "input_file": prompt_file.name,
                            "input_path": str(prompt_file),
                            "result_json": result_output.name,
                            "responses_jsonl": responses_output.name,
                            "workload": result["workload"],
                            "aggregate": result["aggregate"],
                            "speculative": result["speculative"],
                        }
                    )

        if directory_mode:
            manifest = {
                "schema_version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "experiment_name": config.name,
                "input_directory": str(prompt_source),
                "server_log": None if args.no_launch else log_path.name,
                "file_count": len(manifest_files),
                "failed_requests": failed_requests,
                "files": manifest_files,
            }
            manifest_output = write_result(manifest, output / "manifest.json")
            print(f"Manifest: {manifest_output}")
        return 0 if failed_requests == 0 else 2
    except (ConfigError, ServerError, ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
