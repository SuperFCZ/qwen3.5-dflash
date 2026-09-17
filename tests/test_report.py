from __future__ import annotations

import unittest

from dflash_bench.report import comparison_markdown, exact_match


def result(name: str, texts: list[str]) -> dict:
    return {
        "schema_version": 1,
        "experiment": {
            "name": name,
            "speculative": {"num_speculative_tokens": 15},
        },
        "aggregate": {
            "request_throughput_per_s": 1.5,
            "output_throughput_tokens_per_s": 100,
            "ttft": {"p50_ms": 12},
            "tpot": {"p50_ms": 5},
        },
        "speculative": {
            "mean_accepted_draft_tokens": 2.5,
            "mean_accept_length": 3.5,
            "acceptance_rate": 0.2,
        },
        "gpu": {"peak_memory_used_mib": 12345},
        "requests": [
            {"prompt_id": str(index), "repetition": 0, "text": text}
            for index, text in enumerate(texts)
        ],
    }


class ReportTests(unittest.TestCase):
    def test_exact_match(self) -> None:
        self.assertEqual(exact_match(result("a", ["x", "y"]), result("b", ["x", "z"])), (1, 2))

    def test_markdown(self) -> None:
        markdown = comparison_markdown([result("base", ["x"]), result("w8", ["x"])])
        self.assertIn("| w8 | 15 | 2.500 | 3.500 |", markdown)
        self.assertIn("1/1", markdown)
        self.assertIn("20.00%", markdown)


if __name__ == "__main__":
    unittest.main()
