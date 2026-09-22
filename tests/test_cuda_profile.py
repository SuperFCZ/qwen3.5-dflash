# ruff: noqa: E402
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

PLUGIN_ROOT = Path(__file__).resolve().parents[1] / "plugins/dflash_vllm_patch"
sys.path.insert(0, str(PLUGIN_ROOT))

from dflash_vllm_patch.cuda_profile import (
    DFLASH_PROPOSAL,
    TARGET_ONLY_DECODE,
    TARGET_VERIFY,
    CudaEventProfiler,
    _target_phase,
    bind_runner,
    install_cuda_event_profiling,
)


class FakeEvent:
    clock = 0.0
    ready = False

    def __init__(self, *, enable_timing):
        assert enable_timing
        self.timestamp = None

    def record(self, stream):
        self.stream = stream
        self.timestamp = FakeEvent.clock
        FakeEvent.clock += 1.0

    def query(self):
        return self.ready

    def elapsed_time(self, other):
        assert other.query(), "elapsed_time must never read an unfinished event"
        assert self.stream is other.stream
        return other.timestamp - self.timestamp


class FakeCuda:
    def __init__(self):
        self.events = []
        self.synchronize_calls = 0
        self.stream = object()
        self.capturing = False

    def Event(self, *, enable_timing):
        event = FakeEvent(enable_timing=enable_timing)
        self.events.append(event)
        return event

    def synchronize(self, device):
        assert device == 2
        self.synchronize_calls += 1
        for event in self.events:
            event.ready = True

    def current_device(self):
        return 2

    def current_stream(self):
        return self.stream

    def is_current_stream_capturing(self):
        return self.capturing

    def device(self, device):
        assert device == 2
        return nullcontext()


class BaseProposer:
    num_speculative_tokens = 15
    parallel_drafting = True

    def propose(self):
        return NS(shape=(1, 15))


class Proposer(BaseProposer):
    pass  # Like v0.22.1 DFlashProposer, inherits propose from its base.


class Runner:
    def __init__(self, dflash=False):
        self.speculative_config = NS(use_dflash=lambda: True) if dflash else None
        self.vllm_config = NS(
            parallel_config=NS(pipeline_parallel_size=1, data_parallel_size=1, enable_dbo=False)
        )
        self.input_batch = NS(num_reqs=1, num_computed_tokens_cpu=[100], num_prompt_tokens=[100])
        self.drafter = Proposer()
        self.fail_forward = False
        self.fail_sample = False

    def execute_model(self, scheduler_output):
        self._model_forward()
        return None

    def _model_forward(self):
        if self.fail_forward:
            raise ValueError("forward failed")
        return "forward"

    def _sample(self):
        if self.fail_sample:
            raise ValueError("sample failed")
        return "sample"

    def sample_tokens(self):
        self._sample()
        if self.speculative_config:
            self.drafter.propose()
        return "output"


def schedule(k=0):
    return NS(
        num_scheduled_tokens={"r": 1 + k},
        scheduled_spec_decode_tokens={"r": list(range(k))} if k else {},
        scheduled_new_reqs=[],
        scheduled_encoder_inputs={},
    )


class ProfilerTests(unittest.TestCase):
    def setUp(self):
        self.cuda = FakeCuda()
        self.lines = []
        self.profiler = CudaEventProfiler(self.lines.append, torch_module=NS(cuda=self.cuda))

    def bind(self, dflash=False):
        runner = Runner(dflash)
        bind_runner(runner, self.profiler)
        return runner

    def test_nonblocking_snapshot_then_explicit_stop_covers_tail(self):
        p = self.profiler
        p.finish(DFLASH_PROPOSAL, p.begin(), 1)
        incomplete = p.snapshot()
        self.assertEqual(incomplete["metrics"][DFLASH_PROPOSAL]["count"], 0)
        self.assertEqual(incomplete["diagnostics"]["pending_counts"][DFLASH_PROPOSAL], 1)
        self.assertEqual(self.cuda.synchronize_calls, 0)
        final = p.stop()
        self.assertEqual(final["metrics"][DFLASH_PROPOSAL]["count"], 1)
        self.assertEqual(final["metrics"][DFLASH_PROPOSAL]["mean_ms"], 1.0)
        self.assertEqual(self.cuda.synchronize_calls, 1)
        self.assertIs(p.stop("worker_shutdown"), final)
        self.assertIsNone(p.begin())

    def test_reset_excludes_warmup_and_invalidates_unfinished_pairs(self):
        p = self.profiler
        p.finish(TARGET_VERIFY, p.begin(), 1)
        old_pair = p.begin()
        p.start()
        p.finish(TARGET_VERIFY, old_pair, 1)
        p.finish(TARGET_VERIFY, p.begin(), 1)
        final = p.stop()
        self.assertEqual(final["metrics"][TARGET_VERIFY]["count"], 1)
        self.assertEqual(final["generation"], 2)
        p.start()
        self.assertEqual(p.stop()["metrics"][TARGET_VERIFY]["count"], 0)

    def test_actual_execute_sample_split_and_proposal_do_not_overlap(self):
        runner = self.bind(True)
        self.assertIsNone(runner.execute_model(schedule(15)))
        self.assertEqual(len(self.cuda.events), 2)
        self.assertIsNone(self.cuda.events[1].timestamp)
        self.assertEqual(runner.sample_tokens(), "output")
        self.assertLess(self.cuda.events[1].timestamp, self.cuda.events[2].timestamp)
        self.assertEqual(self.cuda.synchronize_calls, 0)
        final = self.profiler.stop()
        self.assertEqual(final["metrics"][TARGET_VERIFY]["count"], 1)
        self.assertEqual(final["metrics"][DFLASH_PROPOSAL]["count"], 1)
        # The inherited base method and other proposer instances stay unchanged.
        self.assertIs(Proposer.propose, BaseProposer.propose)
        self.assertEqual(Proposer().propose().shape, (1, 15))

    def test_decode_prefill_short_tail_and_mixed_batches(self):
        runner = self.bind()
        # A single remaining prompt token in an existing/cached request.
        runner.input_batch.num_computed_tokens_cpu = [99]
        runner.execute_model(schedule())
        runner.sample_tokens()
        self.assertEqual(len(self.cuda.events), 0)
        runner.input_batch.num_computed_tokens_cpu = [100]
        runner.execute_model(schedule())
        runner.sample_tokens()
        final = self.profiler.stop()
        self.assertEqual(final["metrics"][TARGET_ONLY_DECODE]["count"], 1)
        draft = Runner(True)
        self.assertIsNone(_target_phase(draft, schedule(14)))
        self.assertIsNone(_target_phase(draft, schedule(0)))
        mixed = schedule(15)
        mixed.num_scheduled_tokens["prefill"] = 4
        self.assertIsNone(_target_phase(draft, mixed))
        self.assertEqual(_target_phase(draft, schedule(15)), TARGET_VERIFY)

    def test_first_post_prefill_proposal_is_excluded(self):
        runner = self.bind(True)
        runner.input_batch.num_computed_tokens_cpu = [0]
        runner.execute_model(schedule())
        # Simulate bookkeeping updating computed tokens before propose.
        runner.input_batch.num_computed_tokens_cpu = [100]
        runner.sample_tokens()
        self.assertEqual(self.profiler.stop()["metrics"][DFLASH_PROPOSAL]["count"], 0)

    def test_capture_is_skipped_but_every_replay_is_recorded(self):
        runner = self.bind()
        self.cuda.capturing = True
        runner.execute_model(schedule())
        runner.sample_tokens()
        self.cuda.capturing = False
        for _ in range(3):
            runner.execute_model(schedule())
            runner.sample_tokens()
        final = self.profiler.stop()
        self.assertEqual(final["metrics"][TARGET_ONLY_DECODE]["count"], 3)
        self.assertEqual(final["diagnostics"]["calls"]["capture_skips"], 1)

    def test_failed_forward_or_sample_does_not_create_latency(self):
        runner = self.bind()
        runner.fail_forward = True
        with self.assertRaisesRegex(ValueError, "forward"):
            runner.execute_model(schedule())
        runner.fail_forward = False
        runner.fail_sample = True
        runner.execute_model(schedule())
        with self.assertRaisesRegex(ValueError, "sample"):
            runner.sample_tokens()
        self.assertEqual(self.profiler.stop()["metrics"][TARGET_ONLY_DECODE]["count"], 0)

    def test_zero_sample_and_disabled_diagnostics_are_visible(self):
        self.assertEqual(self.profiler.snapshot("worker_ready")["pid"], os.getpid())
        with patch.object(self.cuda, "Event", side_effect=RuntimeError("broken")):
            self.profiler.begin()
        record = self.profiler.stop()
        self.assertTrue(record["diagnostics"]["disabled"])
        self.assertIn("broken", record["diagnostics"]["disable_reason"])
        self.assertTrue(self.lines[-1].startswith("CUDA_EVENT_PROFILE "))

    def test_unsupported_parallel_or_non15_configuration_fails_loudly(self):
        runner = Runner(True)
        runner.drafter.num_speculative_tokens = 8
        with self.assertRaisesRegex(RuntimeError, "k=15"):
            bind_runner(runner, self.profiler)
        runner = Runner()
        runner.vllm_config.parallel_config.pipeline_parallel_size = 2
        with self.assertRaisesRegex(RuntimeError, "PP=1"):
            bind_runner(runner, self.profiler)

    def test_idle_reporter_drains_without_another_step_or_shutdown(self):
        emitted = threading.Event()
        records = []

        def emit(line):
            record = json.loads(line.removeprefix("CUDA_EVENT_PROFILE "))
            records.append(record)
            if record["metrics"][TARGET_VERIFY]["count"]:
                emitted.set()

        p = CudaEventProfiler(emit, torch_module=NS(cuda=self.cuda), interval_s=0.01)
        p.start_reporting()
        p.finish(TARGET_VERIFY, p.begin(), 1)
        for event in self.cuda.events:
            event.ready = True
        self.assertTrue(emitted.wait(2))
        self.assertEqual(self.cuda.synchronize_calls, 0)
        p.close()
        self.assertEqual(records[0]["trigger"], "worker_ready")

    def test_worker_lifecycle_owns_collector_and_rpc(self):
        lifecycle = []

        class Worker:
            rank = 0
            local_rank = 2

            def init_device(self):
                self.model_runner = Runner()
                self.model_runner.vllm_config.model_config = NS(model="W4", quantization="q")
                return "init"

            def shutdown(self):
                lifecycle.append("release")
                return "shutdown"

        def factory(*a, **kw):
            return CudaEventProfiler(
                self.lines.append, torch_module=NS(cuda=self.cuda), metadata=kw["metadata"]
            )

        with patch("dflash_vllm_patch.cuda_profile.CudaEventProfiler", side_effect=factory):
            install_cuda_event_profiling(self.lines.append, worker_module=NS(Worker=Worker))
            worker = Worker()
            self.assertFalse(hasattr(worker, "_eqc_cuda_event_profiler"))
            with patch.object(CudaEventProfiler, "start_reporting"):
                self.assertEqual(worker.init_device(), "init")
        worker.model_runner.execute_model_state = object()
        with self.assertRaisesRegex(RuntimeError, "in-flight"):
            worker.eqc_cuda_event_profile("start")
        worker.model_runner.execute_model_state = None
        worker.eqc_cuda_event_profile("start")
        worker.model_runner.execute_model(schedule())
        worker.model_runner.sample_tokens()
        final = worker.eqc_cuda_event_profile("stop")
        self.assertEqual(final["metrics"][TARGET_ONLY_DECODE]["count"], 1)
        self.assertEqual(final["metadata"]["target_model"], "W4")
        self.assertEqual(worker.shutdown(), "shutdown")
        self.assertEqual(lifecycle, ["release"])
        self.assertIs(worker.eqc_cuda_event_profile("stop"), final)

    def test_percentiles(self):
        self.assertEqual(
            CudaEventProfiler._summary([1, 2, 3, 4]),
            {
                "count": 4,
                "mean_ms": 2.5,
                "p50_ms": 2.5,
                "p95_ms": 3.85,
            },
        )

    def test_abrupt_worker_exit_still_leaves_periodic_report(self):
        # A real child with os._exit: no shutdown or atexit hook can help it.
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "worker.log"
            script = """
import os, threading
from types import SimpleNamespace as NS
from test_cuda_profile import CudaEventProfiler, FakeCuda, TARGET_VERIFY
ready = threading.Event()
with open(os.environ['PROFILE_TEST_LOG'], 'w') as output:
    def emit(line):
        output.write(line + '\\n')
        output.flush()
        if '"count": 1' in line:
            ready.set()
    cuda = FakeCuda()
    p = CudaEventProfiler(emit, torch_module=NS(cuda=cuda), interval_s=0.01)
    p.start_reporting()
    p.finish(TARGET_VERIFY, p.begin(), 1)
    for event in cuda.events:
        event.ready = True
    if not ready.wait(3):
        os._exit(2)
    os._exit(0)
"""
            environment = dict(os.environ, PROFILE_TEST_LOG=str(log))
            environment["PYTHONPATH"] = str(Path(__file__).parent)
            subprocess.run([sys.executable, "-c", script], env=environment, check=True, timeout=5)
            records = [
                json.loads(line.removeprefix("CUDA_EVENT_PROFILE "))
                for line in log.read_text().splitlines()
            ]
        self.assertTrue(any(r["metrics"][TARGET_VERIFY]["count"] == 1 for r in records))
        self.assertTrue(all(r["pid"] != os.getpid() and not r["final"] for r in records))


if __name__ == "__main__":
    unittest.main()
