from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dflash_bench.benchmark import _percentile, load_prompts


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

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(_percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(_percentile([0.0, 10.0], 0.95) or 0, 9.5)


if __name__ == "__main__":
    unittest.main()
