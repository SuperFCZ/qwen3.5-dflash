from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dflash_bench.cli import main

ROOT = Path(__file__).resolve().parents[1]


def fake_result(prompt_path: Path) -> dict:
    return {
        "workload": {
            "prompt_file": str(prompt_path),
            "prompt_file_sha256": "abc",
            "unique_prompts": 1,
        },
        "aggregate": {
            "planned_requests": 1,
            "completed_requests": 1,
            "failed_requests": 0,
            "output_throughput_tokens_per_s": 10.0,
        },
        "speculative": None,
        "errors": [],
        "requests": [
            {
                "prompt_id": "line-1",
                "repetition": 0,
                "input_field": "question",
                "input": "one",
                "source_record": {"question": "one"},
                "text": "answer",
                "output_tokens": 1,
                "latency_s": 0.2,
                "ttft_s": 0.1,
                "tpot_s": None,
                "finish_reason": "stop",
            }
        ],
    }


class CliTests(unittest.TestCase):
    @patch("dflash_bench.cli.server_is_ready", return_value=True)
    @patch("dflash_bench.cli.run_benchmark")
    def test_directory_input_writes_one_result_pair_per_file(
        self, run_benchmark_mock, _server_ready_mock
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            for name in ("part-b.jsonl", "part-a.jsonl"):
                (inputs / name).write_text('{"question":"one"}\n', encoding="utf-8")
            output = root / "output"
            run_benchmark_mock.side_effect = lambda _config, *, base_url, prompt_path: fake_result(
                Path(prompt_path)
            )

            code = main(
                [
                    "run",
                    str(ROOT / "configs/w8_draft.toml"),
                    "--no-launch",
                    "--prompts",
                    str(inputs),
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(code, 0)
            self.assertEqual(run_benchmark_mock.call_count, 2)
            self.assertTrue((output / "part-a.result.json").is_file())
            self.assertTrue((output / "part-a.responses.jsonl").is_file())
            self.assertTrue((output / "part-b.result.json").is_file())
            self.assertTrue((output / "part-b.responses.jsonl").is_file())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [item["input_file"] for item in manifest["files"]],
                ["part-a.jsonl", "part-b.jsonl"],
            )
            self.assertIsNone(manifest["server_log"])


if __name__ == "__main__":
    unittest.main()
