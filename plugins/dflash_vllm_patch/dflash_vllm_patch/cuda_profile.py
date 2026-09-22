"""Worker-owned, opt-in CUDA timeline measurements for vLLM 0.22.1."""

from __future__ import annotations

import functools
import json
import math
import os
import statistics
import threading
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from queue import Empty, SimpleQueue
from typing import Any

DFLASH_PROPOSAL = "dflash_proposal"
TARGET_VERIFY = "target_verify"
TARGET_ONLY_DECODE = "target_only_single_token_decode"
_PHASES = (DFLASH_PROPOSAL, TARGET_VERIFY, TARGET_ONLY_DECODE)


@dataclass(frozen=True)
class _EventPair:
    start: Any
    end: Any
    stream: Any
    generation: int


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class CudaEventProfiler:
    """No inference-step synchronization; only explicit start/stop may wait.

    A worker-local daemon queries completed events, including when the engine is
    idle. It never records events, waits for GPU work, or calls model code.
    """

    def __init__(
        self,
        emit: Callable[[str], None],
        *,
        torch_module: Any | None = None,
        metadata: dict[str, Any] | None = None,
        interval_s: float = 5.0,
    ) -> None:
        if torch_module is None:
            import torch

            torch_module = torch
        self._torch = torch_module
        self._emit = emit
        self.metadata = metadata or {}
        self.device = int(self._torch.cuda.current_device())
        self.pid = os.getpid()
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._interval_s = interval_s
        self._generation = 0
        self._sequence = 0
        self._reset()

    def _reset(self) -> None:
        self._generation += 1
        self._incoming: SimpleQueue = SimpleQueue()
        self._pending = {phase: deque() for phase in _PHASES}
        self._elapsed_ms: dict[str, list[float]] = {phase: [] for phase in _PHASES}
        self._batch_sizes = {phase: Counter() for phase in _PHASES}
        self._calls: Counter = Counter()
        self._disabled = False
        self._disable_reason: str | None = None
        self._active = True
        self._final: dict[str, Any] | None = None

    def _disable(self, operation: str, exc: Exception) -> None:
        self._disabled = True
        self._disable_reason = f"{operation}: {type(exc).__name__}: {exc}"

    def note(self, name: str) -> None:
        if self._active:
            self._calls[name] += 1

    def begin(self) -> _EventPair | None:
        if not self._active or self._disabled:
            return None
        self._calls["begin_calls"] += 1
        try:
            # These wrappers are outside the graph, so graph replay is timed
            # on every invocation. Never insert our events inside capture.
            if self._torch.cuda.is_current_stream_capturing():
                self._calls["capture_skips"] += 1
                return None
            stream = self._torch.cuda.current_stream()
            start = self._torch.cuda.Event(enable_timing=True)
            end = self._torch.cuda.Event(enable_timing=True)
            start.record(stream)
            return _EventPair(start, end, stream, self._generation)
        except Exception as exc:
            self._disable("begin", exc)
            return None

    def finish(self, phase: str, pair: _EventPair | None, batch_size: int) -> None:
        if pair is None or pair.generation != self._generation or not self._active:
            return
        self._calls["finish_calls"] += 1
        try:
            # Never acquire the reporter's lock around an event boundary.
            # SimpleQueue.put is nonblocking; aggregation stays off this path.
            pair.end.record(pair.stream)
            self._incoming.put((phase, pair, batch_size))
        except Exception as exc:
            self._disable("finish", exc)

    def _receive(self) -> None:
        while True:
            try:
                phase, pair, batch_size = self._incoming.get_nowait()
            except Empty:
                return
            self._pending[phase].append(pair)
            self._batch_sizes[phase][str(batch_size)] += 1

    def _drain_ready(self) -> None:
        self._receive()
        for phase, pending in self._pending.items():
            while pending and pending[0].end.query():
                pair = pending[0]
                elapsed = pair.start.elapsed_time(pair.end)
                pending.popleft()
                self._elapsed_ms[phase].append(elapsed)

    @staticmethod
    def _summary(values: list[float]) -> dict[str, float | int | None]:
        return {
            "count": len(values),
            "mean_ms": round(statistics.fmean(values), 6) if values else None,
            "p50_ms": round(_percentile(values, 0.50), 6) if values else None,
            "p95_ms": round(_percentile(values, 0.95), 6) if values else None,
        }

    def _report(self, trigger: str, final: bool) -> dict[str, Any]:
        try:
            self._drain_ready()
        except Exception as exc:
            self._disable("event query/elapsed", exc)
        self._sequence += 1
        payload = {
            "schema_version": 3,
            "clock": "cuda_event",
            "scope": "worker_batch",
            "pid": self.pid,
            "device": self.device,
            "generation": self._generation,
            "sequence": self._sequence,
            "trigger": trigger,
            "final": final,
            "metadata": self.metadata,
            "diagnostics": {
                "calls": dict(self._calls),
                "pending_counts": {p: len(q) for p, q in self._pending.items()},
                "batch_sizes": {p: dict(v) for p, v in self._batch_sizes.items()},
                "disabled": self._disabled,
                "disable_reason": self._disable_reason,
            },
            "metrics": {p: self._summary(v) for p, v in self._elapsed_ms.items()},
        }
        try:
            self._emit("CUDA_EVENT_PROFILE " + json.dumps(payload, sort_keys=True))
        except Exception as exc:
            # Control RPC must not falsely acknowledge a successful export.
            payload["diagnostics"]["emit_error"] = f"{type(exc).__name__}: {exc}"
        return payload

    def snapshot(self, trigger: str = "periodic") -> dict[str, Any]:
        with self._lock:
            return self._final or self._report(trigger, False)

    def start(self) -> dict[str, Any]:
        with self._lock:
            # Caller drains requests first. This also resolves the preceding
            # warm-up window, outside the measured wall-time interval.
            self._torch.cuda.synchronize(self.device)
            self._reset()
            return self._report("start", False)

    def stop(self, trigger: str = "stop_rpc") -> dict[str, Any]:
        with self._lock:
            if self._final is not None:
                return self._final
            self._active = False
            self._receive()
            try:
                if any(self._pending.values()):
                    self._torch.cuda.synchronize(self.device)
            except Exception as exc:
                self._disable("stop synchronization", exc)
            self._final = self._report(trigger, True)
            return self._final

    def start_reporting(self) -> None:
        self.snapshot("worker_ready")
        if self._thread is not None:
            return

        def report_loop() -> None:
            # CUDA's current device is thread-local; never assume device 0.
            with self._torch.cuda.device(self.device):
                while not self._closed.wait(self._interval_s):
                    self.snapshot()

        self._thread = threading.Thread(target=report_loop, name="eqc-cuda-events", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._closed.set()
        self.stop("worker_shutdown")


def _uses_dflash(runner: Any) -> bool:
    config = getattr(runner, "speculative_config", None)
    return config is not None and config.use_dflash()


def _target_phase(runner: Any, scheduler_output: Any) -> str | None:
    scheduled = scheduler_output.num_scheduled_tokens
    if not scheduled or getattr(scheduler_output, "scheduled_new_reqs", None):
        return None
    if getattr(scheduler_output, "scheduled_encoder_inputs", None):
        return None
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    if _uses_dflash(runner):
        # 15 candidates + the previous/bonus token, never a truncated tail.
        if set(spec_tokens) == set(scheduled) and all(
            len(spec_tokens[r]) == 15 and n == 16 for r, n in scheduled.items()
        ):
            return TARGET_VERIFY
        return None
    if runner.speculative_config is None and not spec_tokens:
        if all(n == 1 for n in scheduled.values()):
            return TARGET_ONLY_DECODE
    return None


def _is_decode_batch(runner: Any) -> bool:
    # Called at _model_forward, AFTER _update_states/_prepare_inputs. A one-token
    # cached/resumed prefill must not be mistaken for single-token decode.
    batch = runner.input_batch
    return all(
        batch.num_computed_tokens_cpu[i] >= batch.num_prompt_tokens[i]
        for i in range(batch.num_reqs)
    )


def bind_runner(runner: Any, profiler: CudaEventProfiler) -> None:
    """Bind to the actual runner instance, preserving all original bound calls."""
    for name in ("execute_model", "_model_forward", "_sample", "sample_tokens"):
        if not callable(getattr(runner, name, None)):
            raise RuntimeError(f"unsupported runner {type(runner)}: missing {name}")
    parallel = runner.vllm_config.parallel_config
    if (
        parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.enable_dbo
    ):
        raise RuntimeError("CUDA Event profiling requires PP=1, DP=1 and DBO disabled")
    if _uses_dflash(runner):
        drafter = runner.drafter
        if drafter.num_speculative_tokens != 15 or not drafter.parallel_drafting:
            raise RuntimeError("CUDA Event profiling requires DFlash parallel drafting with k=15")
    elif runner.speculative_config is not None:
        raise RuntimeError("CUDA Event profiling supports DFlash or target-only")

    phase: str | None = None
    decode_context = False
    pending: tuple[str, _EventPair | None, int] | None = None
    original_execute = runner.execute_model
    original_forward = runner._model_forward
    original_sample = runner._sample
    original_sample_tokens = runner.sample_tokens

    @functools.wraps(original_execute)
    def execute(scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal phase, pending, decode_context
        profiler.note("execute_calls")
        if pending is not None:
            profiler.note("abandoned_targets")
        pending = None
        decode_context = False
        phase = _target_phase(runner, scheduler_output)
        try:
            return original_execute(scheduler_output, *args, **kwargs)
        except BaseException:
            pending = None
            raise
        finally:
            phase = None

    @functools.wraps(original_forward)
    def forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal pending, decode_context
        profiler.note("forward_calls")
        decode_context = _is_decode_batch(runner)
        if phase is not None and decode_context:
            profiler.note("target_phase_matches")
            pending = (phase, profiler.begin(), runner.input_batch.num_reqs)
        try:
            return original_forward(*args, **kwargs)
        except BaseException:
            pending = None
            raise

    @functools.wraps(original_sample)
    def sample(*args: Any, **kwargs: Any) -> Any:
        nonlocal pending
        profiler.note("sample_calls")
        target = pending
        pending = None
        # Do not turn failed sampling into a valid latency sample.
        result = original_sample(*args, **kwargs)
        if target is not None:
            profiler.finish(*target)
        return result

    @functools.wraps(original_sample_tokens)
    def sample_tokens(*args: Any, **kwargs: Any) -> Any:
        nonlocal pending
        try:
            return original_sample_tokens(*args, **kwargs)
        finally:
            if pending is not None:
                profiler.note("abandoned_targets")
            pending = None

    if _uses_dflash(runner):
        original_propose = runner.drafter.propose

        @functools.wraps(original_propose)
        def propose(*args: Any, **kwargs: Any) -> Any:
            profiler.note("propose_calls")
            # Include only decode-context proposals; the first post-prefill
            # proposal has a different context-K/V precompute workload.
            if not decode_context:
                profiler.note("prefill_proposal_skips")
                return original_propose(*args, **kwargs)
            pair = profiler.begin()
            result = original_propose(*args, **kwargs)
            if tuple(result.shape) == (runner.input_batch.num_reqs, 15):
                profiler.finish(DFLASH_PROPOSAL, pair, runner.input_batch.num_reqs)
            else:
                profiler.note("proposal_shape_skips")
            return result

        runner.drafter.propose = propose

    runner.execute_model = execute
    runner._model_forward = forward
    runner._sample = sample
    runner.sample_tokens = sample_tokens


def install_cuda_event_profiling(
    emit: Callable[[str], None], *, worker_module: Any | None = None
) -> None:
    """Install lifecycle/RPC only; allocate the collector inside the GPU worker."""
    if worker_module is None:
        import vllm.v1.worker.gpu_worker as worker_module

    worker_class = worker_module.Worker
    if getattr(worker_class, "_eqc_profile_installed", False):
        return
    original_init = worker_class.init_device
    original_shutdown = worker_class.shutdown

    @functools.wraps(original_init)
    def init_device(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_init(self, *args, **kwargs)
        runner = self.model_runner
        model = runner.vllm_config.model_config
        draft = getattr(runner.speculative_config, "draft_model_config", None)
        try:
            vllm_version = version("vllm")
        except PackageNotFoundError:
            vllm_version = "unknown"
        profiler = CudaEventProfiler(
            emit,
            metadata={
                "runner": f"{type(runner).__module__}.{type(runner).__qualname__}",
                "vllm_version": vllm_version,
                "implementation": "worker-events-v3",
                "async_scheduling": getattr(runner, "use_async_scheduling", None),
                "rank": self.rank,
                "local_rank": self.local_rank,
                "target_model": model.model,
                "target_quantization": model.quantization,
                "draft_model": getattr(draft, "model", None),
                "draft_quantization": getattr(draft, "quantization", None),
                "proposal_tokens": 15 if _uses_dflash(runner) else None,
                "plugin_file": __file__,
            },
        )
        self._eqc_cuda_event_profiler = profiler
        try:
            bind_runner(runner, profiler)
        except Exception as exc:
            profiler._disable("bind_runner", exc)
            profiler.snapshot("unsupported_runner")
            raise
        profiler.start_reporting()
        return result

    def control(self: Any, action: str) -> dict[str, Any]:
        profiler = getattr(self, "_eqc_cuda_event_profiler", None)
        if profiler is None or profiler.pid != os.getpid():
            raise RuntimeError("CUDA Event collector is not bound in this worker")
        if (
            action in {"start", "stop"}
            and getattr(self.model_runner, "execute_model_state", None) is not None
        ):
            raise RuntimeError(
                "finish in-flight execute_model/sample_tokens before profile control"
            )
        if action == "start":
            return profiler.start()
        if action == "stop":
            return profiler.stop()
        if action == "snapshot":
            return profiler.snapshot("snapshot_rpc")
        raise ValueError(f"invalid CUDA Event profile action: {action}")

    @functools.wraps(original_shutdown)
    def shutdown(self: Any, *args: Any, **kwargs: Any) -> Any:
        profiler = getattr(self, "_eqc_cuda_event_profiler", None)
        if profiler is not None:
            profiler.close()
        return original_shutdown(self, *args, **kwargs)

    worker_class.init_device = init_device
    worker_class.shutdown = shutdown
    worker_class.eqc_cuda_event_profile = control
    worker_class._eqc_profile_installed = True
