"""Opt-in CUDA Event timings for the vLLM 0.22.1 DFlash path."""

from __future__ import annotations

import atexit
import functools
import json
import math
import os
import statistics
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DFLASH_PROPOSAL = "dflash_proposal"
TARGET_VERIFY = "target_verify"
TARGET_ONLY_DECODE = "target_only_single_token_decode"
_PHASES = (DFLASH_PROPOSAL, TARGET_VERIFY, TARGET_ONLY_DECODE)


@dataclass(frozen=True)
class _EventPair:
    start: Any
    end: Any


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class CudaEventProfiler:
    """Collect CUDA Event pairs without synchronizing the inference loop."""

    def __init__(
        self,
        emit: Callable[[str], None],
        *,
        torch_module: Any | None = None,
        drain_interval: int = 256,
    ) -> None:
        if torch_module is None:
            import torch

            torch_module = torch
        self._torch = torch_module
        self._emit = emit
        self._drain_interval = drain_interval
        self._pending = {phase: deque() for phase in _PHASES}
        self._elapsed_ms = {phase: [] for phase in _PHASES}
        self._finished_since_drain = 0
        self._reported = False
        self._disabled = False
        self._error_reported = False
        self._disable_reason: str | None = None
        self._propose_hook_calls = 0
        self._target_phase_matches = 0
        self._target_phase_matches_by_phase = {
            TARGET_VERIFY: 0,
            TARGET_ONLY_DECODE: 0,
        }
        self._begin_calls = 0
        self._finish_calls = 0

    def _disable(self, operation: str, exc: Exception) -> None:
        self._disabled = True
        self._disable_reason = f"{operation}: {type(exc).__name__}: {exc}"
        if not self._error_reported:
            try:
                self._emit(
                    "CUDA Event profiling disabled after "
                    f"{operation} failed: {type(exc).__name__}: {exc}"
                )
            except Exception:  # pragma: no cover - diagnostics must stay best-effort
                pass
            self._error_reported = True

    def note_propose_hook(self) -> None:
        self._propose_hook_calls += 1

    def note_target_phase(self, phase: str) -> None:
        self._target_phase_matches += 1
        self._target_phase_matches_by_phase[phase] += 1

    def begin(self) -> _EventPair | None:
        self._begin_calls += 1
        if self._disabled or self._reported:
            return None
        try:
            start = self._torch.cuda.Event(enable_timing=True)
            end = self._torch.cuda.Event(enable_timing=True)
            start.record()
            return _EventPair(start=start, end=end)
        except Exception as exc:  # pragma: no cover - depends on CUDA runtime
            self._disable("event creation", exc)
            return None

    def finish(self, phase: str, pair: _EventPair | None) -> None:
        self._finish_calls += 1
        if pair is None or self._disabled or self._reported:
            return
        try:
            pair.end.record()
            self._pending[phase].append(pair)
            self._finished_since_drain += 1
            if self._finished_since_drain >= self._drain_interval:
                self._drain_ready()
                self._finished_since_drain = 0
        except Exception as exc:  # pragma: no cover - depends on CUDA runtime
            self._disable("event recording", exc)

    def _drain_ready(self) -> None:
        for phase in _PHASES:
            pending = self._pending[phase]
            while pending and pending[0].end.query():
                pair = pending.popleft()
                self._elapsed_ms[phase].append(pair.start.elapsed_time(pair.end))

    @staticmethod
    def _summary(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {
                "count": 0,
                "mean_ms": None,
                "p50_ms": None,
                "p95_ms": None,
            }
        return {
            "count": len(values),
            "mean_ms": round(statistics.fmean(values), 6),
            "p50_ms": round(_percentile(values, 0.50), 6),
            "p95_ms": round(_percentile(values, 0.95), 6),
        }

    def _phase_counts(self, values: dict[str, Any]) -> dict[str, int]:
        counts = {phase: len(values[phase]) for phase in _PHASES}
        counts["total"] = sum(counts.values())
        return counts

    def report(
        self,
        *,
        force: bool = False,
        trigger: str = "manual",
    ) -> dict[str, Any] | None:
        if self._reported:
            return None
        has_samples = any(self._pending.values()) or any(self._elapsed_ms.values())
        if not force and not has_samples and not self._disabled:
            return None
        self._reported = True
        pending_counts = self._phase_counts(self._pending)
        elapsed_counts = self._phase_counts(self._elapsed_ms)
        if not self._disabled and has_samples:
            try:
                self._torch.cuda.synchronize()
                for phase in _PHASES:
                    pending = self._pending[phase]
                    while pending:
                        pair = pending.popleft()
                        self._elapsed_ms[phase].append(pair.start.elapsed_time(pair.end))
            except Exception as exc:  # pragma: no cover - depends on CUDA runtime
                self._disable("final report", exc)

        try:
            device = int(self._torch.cuda.current_device())
        except Exception:  # pragma: no cover - depends on CUDA shutdown order
            device = None
        payload = {
            "schema_version": 2,
            "clock": "cuda_event",
            "device": device,
            "pid": os.getpid(),
            "trigger": trigger,
            "diagnostics": {
                "propose_hook_calls": self._propose_hook_calls,
                "target_phase_matches": self._target_phase_matches,
                "target_phase_matches_by_phase": dict(
                    self._target_phase_matches_by_phase
                ),
                "begin_calls": self._begin_calls,
                "finish_calls": self._finish_calls,
                "pending_counts": pending_counts,
                "elapsed_counts": elapsed_counts,
                "final_elapsed_counts": self._phase_counts(self._elapsed_ms),
                "disabled": self._disabled,
                "disable_reason": self._disable_reason,
            },
            "metrics": {
                phase: self._summary(self._elapsed_ms[phase]) for phase in _PHASES
            },
        }
        try:
            self._emit(
                "CUDA_EVENT_PROFILE "
                + json.dumps(payload, sort_keys=True, separators=(",", ":"))
            )
        except Exception:  # pragma: no cover - logging must not block shutdown
            pass
        return payload


def _uses_dflash(runner: Any) -> bool:
    speculative_config = getattr(runner, "speculative_config", None)
    if speculative_config is None:
        return False
    use_dflash = getattr(speculative_config, "use_dflash", None)
    return bool(callable(use_dflash) and use_dflash())


def _target_phase(runner: Any, scheduler_output: Any) -> str | None:
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", {})
    if not scheduled:
        return None

    spec_tokens = getattr(scheduler_output, "scheduled_spec_decode_tokens", {})
    if _uses_dflash(runner):
        if spec_tokens and set(spec_tokens) == set(scheduled):
            return TARGET_VERIFY
        return None

    if getattr(runner, "speculative_config", None) is not None:
        return None
    if getattr(scheduler_output, "scheduled_new_reqs", None):
        return None
    if getattr(scheduler_output, "scheduled_encoder_inputs", None):
        return None
    if all(int(num_tokens) == 1 for num_tokens in scheduled.values()):
        return TARGET_ONLY_DECODE
    return None


def _timed_call(
    profiler: CudaEventProfiler,
    phase: str,
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    pair = profiler.begin()
    try:
        return function(*args, **kwargs)
    finally:
        profiler.finish(phase, pair)


def install_cuda_event_profiling(
    emit: Callable[[str], None],
    *,
    runner_module: Any | None = None,
    dflash_module: Any | None = None,
    engine_core_module: Any | None = None,
    profiler: CudaEventProfiler | None = None,
) -> CudaEventProfiler:
    """Patch vLLM phase boundaries while preserving their return values."""
    if runner_module is None:
        import vllm.v1.worker.gpu_model_runner as runner_module
    if dflash_module is None:
        import vllm.v1.spec_decode.dflash as dflash_module
    if engine_core_module is None:
        import vllm.v1.engine.core as engine_core_module

    runner_class = runner_module.GPUModelRunner
    proposer_class = dflash_module.DFlashProposer
    engine_core_class = engine_core_module.EngineCore
    existing = getattr(runner_class, "_eqc_cuda_event_profiler", None)
    if existing is not None:
        return existing

    profiler = profiler or CudaEventProfiler(emit)

    phase_attribute = "_eqc_cuda_profile_next_target_phase"
    pending_attribute = "_eqc_cuda_profile_pending_target"

    original_execute = runner_class.execute_model

    @functools.wraps(original_execute)
    def execute_model(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        phase = _target_phase(self, scheduler_output)
        if phase is not None:
            profiler.note_target_phase(phase)
        if hasattr(self, pending_attribute):
            delattr(self, pending_attribute)
        setattr(self, phase_attribute, phase)
        try:
            return original_execute(self, scheduler_output, *args, **kwargs)
        finally:
            setattr(self, phase_attribute, None)

    original_model_forward = runner_class._model_forward

    @functools.wraps(original_model_forward)
    def model_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        phase = getattr(self, phase_attribute, None)
        if phase is None:
            return original_model_forward(self, *args, **kwargs)
        pair = profiler.begin()
        setattr(self, pending_attribute, (phase, pair))
        try:
            return original_model_forward(self, *args, **kwargs)
        except BaseException:
            delattr(self, pending_attribute)
            raise

    original_sample = runner_class._sample

    @functools.wraps(original_sample)
    def sample(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return original_sample(self, *args, **kwargs)
        finally:
            pending = getattr(self, pending_attribute, None)
            if pending is not None:
                delattr(self, pending_attribute)
                phase, pair = pending
                profiler.finish(phase, pair)

    original_propose = proposer_class.propose

    @functools.wraps(original_propose)
    def propose(self: Any, *args: Any, **kwargs: Any) -> Any:
        profiler.note_propose_hook()
        return _timed_call(
            profiler,
            DFLASH_PROPOSAL,
            original_propose,
            self,
            *args,
            **kwargs,
        )

    original_shutdown = runner_class.shutdown

    @functools.wraps(original_shutdown)
    def shutdown(self: Any, *args: Any, **kwargs: Any) -> Any:
        profiler.report(force=True, trigger="gpu_model_runner_shutdown")
        return original_shutdown(self, *args, **kwargs)

    original_engine_core_shutdown = engine_core_class.shutdown

    @functools.wraps(original_engine_core_shutdown)
    def engine_core_shutdown(self: Any, *args: Any, **kwargs: Any) -> Any:
        # EngineCore.shutdown tears down model_executor first in vLLM 0.22.1.
        profiler.report(force=True, trigger="engine_core_shutdown")
        return original_engine_core_shutdown(self, *args, **kwargs)

    runner_class.execute_model = execute_model
    runner_class._model_forward = model_forward
    runner_class._sample = sample
    runner_class.shutdown = shutdown
    runner_class._eqc_cuda_event_profiler = profiler
    proposer_class.propose = propose
    engine_core_class.shutdown = engine_core_shutdown
    atexit.register(functools.partial(profiler.report, trigger="atexit"))
    return profiler


__all__ = [
    "CudaEventProfiler",
    "DFLASH_PROPOSAL",
    "TARGET_ONLY_DECODE",
    "TARGET_VERIFY",
    "install_cuda_event_profiling",
]
