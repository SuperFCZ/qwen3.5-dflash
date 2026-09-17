from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dflash_bench.benchmark import (
    _percentile,
    discover_prompt_files,
    load_prompts,
    write_responses_jsonl,
)


class BenchmarkTests(unittest.TestCase):
    def test_load_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text(
                '# comment\n{"id":"a","prompt":"one"}\n{"id":"b","prompt":"two"}\n',
                encoding="utf-8",
            )
            prompts = load_prompts(path)
            self.assertEqual(
                [(item.prompt_id, item.text) for item in prompts], [("a", "one"), ("b", "two")]
            )

    def test_duplicate_prompt_ids_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text(
                '{"id":"a","prompt":"one"}\n{"id":"a","prompt":"two"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_prompts(path)

    def test_loads_question_records_with_generated_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            path.write_text(
                '{"question":"How many eggs?","answer":"9"}\n'
                '{"id":"custom","question":"How many bolts?"}\n',
                encoding="utf-8",
            )
            prompts = load_prompts(path)
            self.assertEqual(prompts[0].prompt_id, "line-1")
            self.assertEqual(prompts[0].text, "How many eggs?")
            self.assertEqual(prompts[0].input_field, "question")
            self.assertEqual(prompts[0].source_record["answer"], "9")
            self.assertEqual(prompts[1].prompt_id, "custom")

    def test_discovers_jsonl_files_in_name_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.jsonl").write_text('{"question":"b"}\n', encoding="utf-8")
            (root / "a.jsonl").write_text('{"question":"a"}\n', encoding="utf-8")
            (root / "notes.txt").write_text("ignored", encoding="utf-8")
            self.assertEqual(
                [item.name for item in discover_prompt_files(root)],
                ["a.jsonl", "b.jsonl"],
            )

    def test_empty_prompt_directory_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "contains no .jsonl"):
                discover_prompt_files(directory)

    def test_response_jsonl_keeps_source_and_errors(self) -> None:
        result = {
            "requests": [
                {
                    "prompt_id": "line-1",
                    "repetition": 0,
                    "input_field": "question",
                    "input": "How many?",
                    "source_record": {"question": "How many?", "answer": "2"},
                    "text": "2",
                    "output_tokens": 1,
                    "latency_s": 0.2,
                    "ttft_s": 0.1,
                    "tpot_s": None,
                    "finish_reason": "stop",
                }
            ],
            "errors": [
                {
                    "prompt_id": "line-2",
                    "repetition": 0,
                    "input_field": "question",
                    "input": "Broken?",
                    "source_record": {"question": "Broken?"},
                    "error": "timeout",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = write_responses_jsonl(result, Path(directory) / "responses.jsonl")
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([item["status"] for item in records], ["ok", "error"])
        self.assertEqual(records[0]["source_record"]["answer"], "2")
        self.assertEqual(records[0]["question"], "How many?")
        self.assertEqual(records[0]["response"], "2")

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(_percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(_percentile([0.0, 10.0], 0.95) or 0, 9.5)


if __name__ == "__main__":
    unittest.main()
