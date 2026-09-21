# ruff: noqa: E402, I001
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PLUGIN_ROOT = Path(__file__).resolve().parents[1] / "plugins/dflash_vllm_patch"
sys.path.insert(0, str(PLUGIN_ROOT))

from dflash_vllm_patch import (
    _patch_decoder_quant_config,
    _patch_sliding_window,
)
from dflash_vllm_patch.cuda_profile import (
    DFLASH_PROPOSAL,
    TARGET_ONLY_DECODE,
    TARGET_VERIFY,
    CudaEventProfiler,
    _target_phase,
    install_cuda_event_profiling,
)


class _FakeEvent:
    clock = 0.0

    def __init__(self, *, enable_timing):
        assert enable_timing
        self.timestamp = None

    def record(self):
        self.timestamp = self.clock
        type(self).clock += 1.0

    def query(self):
        return self.timestamp is not None

    def elapsed_time(self, other):
        return other.timestamp - self.timestamp


class _FakeCuda:
    Event = _FakeEvent

    def __init__(self):
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1

    @staticmethod
    def current_device():
        return 0


class _FakeTorch:
    def __init__(self):
        self.cuda = _FakeCuda()


class PluginTests(unittest.TestCase):
    def test_existing_quant_config_is_forwarded(self) -> None:
        class Layer:
            def __init__(
                self,
                vllm_config,
                *,
                config,
                cache_config=None,
                quant_config=None,
                prefix="",
            ):
                self.received = (vllm_config, config, cache_config, quant_config, prefix)

        module = SimpleNamespace(DFlashQwen3DecoderLayer=Layer)
        _patch_decoder_quant_config(module)
        instance = Layer("vllm", config="model", quant_config="quant", prefix="draft")
        self.assertEqual(instance.received, ("vllm", "model", None, "quant", "draft"))

    def test_conditional_swa_is_a_noop_below_window(self) -> None:
        class Implementation:
            sliding_window = None

            def forward(self, *args, **kwargs):
                return self.sliding_window

        class Attention:
            def __init__(self):
                self.attn = SimpleNamespace(impl=Implementation())

        module = SimpleNamespace(DFlashQwen3Attention=Attention)
        _patch_sliding_window(module, 1024, static=False)
        attention = Attention()
        short = attention.attn.impl.forward(attn_metadata=SimpleNamespace(max_seq_len=1024))
        long = attention.attn.impl.forward(attn_metadata=SimpleNamespace(max_seq_len=1025))
        self.assertEqual(short, (-1, -1))
        self.assertEqual(long, (1023, 1023))

    def test_static_swa_is_symmetric(self) -> None:
        class Implementation:
            sliding_window = None

            def forward(self, *args, **kwargs):
                return None

        class Attention:
            def __init__(self):
                self.attn = SimpleNamespace(impl=Implementation())

        module = SimpleNamespace(DFlashQwen3Attention=Attention)
        _patch_sliding_window(module, 8, static=True)
        self.assertEqual(Attention().attn.impl.sliding_window, (7, 7))

    def test_cuda_profiler_only_synchronizes_for_final_report(self) -> None:
        torch = _FakeTorch()
        emitted = []
        profiler = CudaEventProfiler(emitted.append, torch_module=torch, drain_interval=1)

        for phase in (DFLASH_PROPOSAL, TARGET_VERIFY, TARGET_ONLY_DECODE):
            pair = profiler.begin()
            profiler.finish(phase, pair)

        self.assertEqual(torch.cuda.synchronize_calls, 0)
        payload = profiler.report()
        self.assertEqual(torch.cuda.synchronize_calls, 1)
        self.assertIsNotNone(payload)
        for phase in (DFLASH_PROPOSAL, TARGET_VERIFY, TARGET_ONLY_DECODE):
            self.assertEqual(payload["metrics"][phase]["count"], 1)
            self.assertEqual(payload["metrics"][phase]["mean_ms"], 1.0)
        self.assertTrue(emitted[-1].startswith("CUDA_EVENT_PROFILE "))

    def test_cuda_profile_summary_reports_interpolated_percentiles(self) -> None:
        summary = CudaEventProfiler._summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean_ms"], 2.5)
        self.assertEqual(summary["p50_ms"], 2.5)
        self.assertEqual(summary["p95_ms"], 3.85)

    def test_target_phase_requires_a_pure_decode_or_verify_batch(self) -> None:
        runner = SimpleNamespace(speculative_config=None)
        decode = SimpleNamespace(
            num_scheduled_tokens={"request": 1},
            scheduled_spec_decode_tokens={},
            scheduled_new_reqs=[],
            scheduled_encoder_inputs={},
        )
        prefill = SimpleNamespace(
            num_scheduled_tokens={"request": 8},
            scheduled_spec_decode_tokens={},
            scheduled_new_reqs=[object()],
            scheduled_encoder_inputs={},
        )
        self.assertEqual(_target_phase(runner, decode), TARGET_ONLY_DECODE)
        self.assertIsNone(_target_phase(runner, prefill))

        runner.speculative_config = SimpleNamespace(use_dflash=lambda: True)
        verify = SimpleNamespace(
            num_scheduled_tokens={"request": 16},
            scheduled_spec_decode_tokens={"request": list(range(15))},
        )
        mixed = SimpleNamespace(
            num_scheduled_tokens={"request": 16, "prefill": 4},
            scheduled_spec_decode_tokens={"request": list(range(15))},
        )
        self.assertEqual(_target_phase(runner, verify), TARGET_VERIFY)
        self.assertIsNone(_target_phase(runner, mixed))

    def test_cuda_profile_patches_preserve_results_and_report_together(self) -> None:
        class Runner:
            def __init__(self):
                self.speculative_config = None

            def execute_model(self, scheduler_output):
                self._model_forward()
                return "execute-result"

            def _model_forward(self):
                return "forward-result"

            def _sample(self):
                return "sample-result"

            def shutdown(self):
                return "shutdown-result"

        class DFlashProposer:
            def propose(self):
                return "proposal-result"

        torch = _FakeTorch()
        emitted = []
        profiler = CudaEventProfiler(emitted.append, torch_module=torch)
        install_cuda_event_profiling(
            emitted.append,
            runner_module=SimpleNamespace(GPUModelRunner=Runner),
            dflash_module=SimpleNamespace(DFlashProposer=DFlashProposer),
            profiler=profiler,
        )

        runner = Runner()
        decode = SimpleNamespace(
            num_scheduled_tokens={"request": 1},
            scheduled_spec_decode_tokens={},
            scheduled_new_reqs=[],
            scheduled_encoder_inputs={},
        )
        self.assertEqual(runner.execute_model(decode), "execute-result")
        self.assertEqual(runner._sample(), "sample-result")

        runner.speculative_config = SimpleNamespace(use_dflash=lambda: True)
        verify = SimpleNamespace(
            num_scheduled_tokens={"request": 16},
            scheduled_spec_decode_tokens={"request": list(range(15))},
        )
        self.assertEqual(runner.execute_model(verify), "execute-result")
        self.assertEqual(runner._sample(), "sample-result")
        self.assertEqual(DFlashProposer().propose(), "proposal-result")
        self.assertEqual(torch.cuda.synchronize_calls, 0)

        self.assertEqual(runner.shutdown(), "shutdown-result")
        self.assertEqual(torch.cuda.synchronize_calls, 1)
        payload = profiler.report()
        self.assertIsNone(payload)
        profile_line = next(line for line in emitted if line.startswith("CUDA_EVENT_PROFILE "))
        self.assertIn('"dflash_proposal":{"count":1', profile_line)
        self.assertIn('"target_verify":{"count":1', profile_line)
        self.assertIn('"target_only_single_token_decode":{"count":1', profile_line)


if __name__ == "__main__":
    unittest.main()
