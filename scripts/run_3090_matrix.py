#!/usr/bin/env python3
"""Run the recommended paired RTX 3090 experiment matrix."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACKS = {
    "w8": {
        "target": "w8_target_only.toml",
        "drafts": ["w8_bf16_draft.toml", "w8_draft.toml"],
    },
    "w4": {
        "target": "w4_target_only.toml",
        "drafts": ["w4_draft_full.toml", "w4_draft_swa1024.toml"],
    },
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--track", choices=["w8", "w4", "all"], default="all")
    result.add_argument("--ks", nargs="+", type=int, default=[3, 7, 15])
    result.add_argument("--concurrency", nargs="+", type=int, default=[1, 4])
    result.add_argument("--repetitions", type=int, default=3)
    result.add_argument("--output-dir")
    result.add_argument("--quick", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--keep-going", action="store_true")
    return result


def run(command: list[str], *, dry_run: bool) -> int:
    print(f"$ {shlex.join(command)}", flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main() -> int:
    args = parser().parse_args()
    if args.quick:
        args.ks = [15]
        args.concurrency = [1]
        args.repetitions = 1
    if any(value <= 0 for value in [*args.ks, *args.concurrency, args.repetitions]):
        parser().error("K, concurrency, and repetitions must all be positive")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ROOT / "results" / f"matrix-{timestamp}"
    )
    tracks = list(TRACKS) if args.track == "all" else [args.track]
    prompt_args = (
        ["--prompts", str(ROOT / "prompts/smoke.jsonl"), "--max-tokens", "64"] if args.quick else []
    )

    for track in tracks:
        definition = TRACKS[track]
        for concurrency in args.concurrency:
            produced: list[Path] = []
            target_output = output_dir / f"{track}-target-c{concurrency}.json"
            target_command = [
                sys.executable,
                "-m",
                "dflash_bench",
                "run",
                str(ROOT / "configs" / definition["target"]),
                "--concurrency",
                str(concurrency),
                "--repetitions",
                str(args.repetitions),
                "--output",
                str(target_output),
                *prompt_args,
            ]
            code = run(target_command, dry_run=args.dry_run)
            if code and not args.keep_going:
                return code
            target_succeeded = code == 0
            if not code:
                produced.append(target_output)

            for config_name in definition["drafts"]:
                stem = Path(config_name).stem
                for k in args.ks:
                    output = output_dir / f"{stem}-k{k}-c{concurrency}.json"
                    command = [
                        sys.executable,
                        "-m",
                        "dflash_bench",
                        "run",
                        str(ROOT / "configs" / config_name),
                        "--k",
                        str(k),
                        "--concurrency",
                        str(concurrency),
                        "--repetitions",
                        str(args.repetitions),
                        "--output",
                        str(output),
                        *prompt_args,
                    ]
                    code = run(command, dry_run=args.dry_run)
                    if code and not args.keep_going:
                        return code
                    if not code:
                        produced.append(output)

            if not args.dry_run and target_succeeded and len(produced) > 1:
                report = output_dir / f"{track}-c{concurrency}-report.md"
                compare = [
                    sys.executable,
                    "-m",
                    "dflash_bench",
                    "compare",
                    *(str(path) for path in produced),
                    "--output",
                    str(report),
                ]
                code = run(compare, dry_run=False)
                if code and not args.keep_going:
                    return code
    print(f"Matrix output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
