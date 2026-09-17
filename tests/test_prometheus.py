from __future__ import annotations

import unittest

from dflash_bench.prometheus import parse_prometheus, speculative_stats

BEFORE = """
# TYPE vllm:spec_decode_num_drafts counter
vllm:spec_decode_num_drafts_total 100
vllm:spec_decode_num_draft_tokens_total 1500
vllm:spec_decode_num_accepted_tokens_total 250
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0",model_name="qwen"} 80
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="1",model_name="qwen"} 50
"""

AFTER = """
vllm:spec_decode_num_drafts_total 110
vllm:spec_decode_num_draft_tokens_total 1650
vllm:spec_decode_num_accepted_tokens_total 277
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="qwen",position="0"} 89
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="qwen",position="1"} 56
"""


class PrometheusTests(unittest.TestCase):
    def test_counter_deltas_and_formulas(self) -> None:
        stats = speculative_stats(parse_prometheus(BEFORE), parse_prometheus(AFTER))
        assert stats is not None
        self.assertEqual(stats.draft_steps, 10)
        self.assertEqual(stats.draft_tokens, 150)
        self.assertEqual(stats.accepted_tokens, 27)
        self.assertAlmostEqual(stats.mean_accepted_draft_tokens or 0, 2.7)
        self.assertAlmostEqual(stats.mean_accept_length or 0, 3.7)
        self.assertAlmostEqual(stats.acceptance_rate or 0, 0.18)
        self.assertEqual(stats.per_position_acceptance, {0: 0.9, 1: 0.6})

    def test_absent_metrics_return_none(self) -> None:
        self.assertIsNone(speculative_stats({}, {}))

    def test_parser_handles_timestamp_and_escaped_label(self) -> None:
        parsed = parse_prometheus('metric_total{label="a\\\\b"} 1.25 123\n')
        self.assertEqual(parsed[("metric_total", (("label", "a\\b"),))], 1.25)


if __name__ == "__main__":
    unittest.main()
