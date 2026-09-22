# Troubleshooting

## Server exits while loading `weight_packed` or has no `.weight`

Confirm all three conditions:

```bash
python -m pip show vllm dflash-vllm-patch compressed-tensors
dflash-bench command configs/w8_draft.toml
```

The environment must contain `dflash-vllm-patch`, the vLLM version must be 0.22.1, and the
printed command must start with `EQC_DFLASH_QUANT_PATCH=1`. Do not install the plugin only in
a parent process and then launch workers from a different virtual environment.

## Fused-KV build is deferred

One log message during model loading is expected. Quantization backends create their runtime
buffers after weights load, so the plugin deliberately retries the fused-KV construction on
the first inference path. Repeated deferrals followed by a request failure are not expected;
save the complete `.server.log`, `pip freeze`, and `nvidia-smi` output.

## CUDA out of memory on a 3090

Try, in order:

1. ensure no other process is using the GPU;
2. reduce `gpu_memory_utilization` slightly only if startup reservation fails;
3. reduce `max_model_len` from 8192 to 4096;
4. reduce `max_num_seqs` from 4 to 1;
5. test K=7 or K=3;
6. start with the W4 target/draft pair, which has lower weight residency.

Keep the changed setting identical across every run being compared.

## `/metrics` has no speculative counters

Target-only runs correctly produce no speculative block in the result. For a draft run,
check that the rendered command contains `--speculative-config`, send at least one measured
request long enough to draft, and inspect the service log for a fallback that disabled
speculation.

## Exact-match rate is below 100%

- Compare only runs with the same target model and tokenizer/chat template.
- Keep temperature at zero and all generation options identical.
- Verify that W8 BF16 uses the pinned five-layer revision.
- Disable SWA and retry before investigating quantization.
- Inspect per-position acceptance. Near-zero or structurally impossible rates often point to
  a bad runtime integration rather than ordinary quantization error.
- Re-run on a quiet GPU to rule out an interrupted or partially failed request.

Do not publish speed numbers from a run that fails the correctness gate.

## The W8 and W4 numbers disagree strongly

That is not automatically a bug. W8 uses a BF16 Qwen target and a weight-only RTN derivative;
Nota W4 uses a QAD target, a separately aligned draft, GPTQ, and optional SWA. Treat them as
separate tracks. Compare W8 only with its pinned BF16 draft, and compare W4 against the Nota
target-only baseline.

## First run is much slower

Model downloads, kernel compilation, CUDA graph capture, and cache population happen on the
first launch. The harness excludes request warm-up from metrics, but cannot make the first
server start representative. Complete one smoke run before collecting repeated measurements.

## No `CUDA_EVENT_PROFILE` line appears

Use plugin 0.3.0 or later in the **same Python environment as vLLM**, then restart the
server. `registered` includes PID, detected vLLM version and plugin path; it is not evidence
of GPU execution. Look for the worker's `CUDA_EVENT_PROFILE` with `trigger=worker_ready`
and check its actual runner/rank/model metadata. `unsupported_runner` means the plugin
refused an unvalidated execution path (including V2 runner, PP/DP > 1, DBO or DFlash k != 15).

The worker emits a nonblocking snapshot every 5 seconds, including while idle. An explicit
`POST /eqc_cuda_profile/stop` returns final worker summaries while they are still alive;
`dflash-bench run` does this automatically and stores them in the result JSON. Neither
EngineCore shutdown nor Python atexit is required for this export. A 404 means the API
process did not load the updated plugin/flag; an RPC error means the worker is missing it
or collection failed. Do not continue benchmarking an old server after changing the plugin.

Inspect `diagnostics.calls`: `execute_calls`, `forward_calls`, `sample_calls`,
`propose_calls`, `target_phase_matches`, `begin_calls` and `finish_calls`. No matched phase
can be legitimate for prefill, mixed batches or short outputs. `prefill_proposal_skips`,
`proposal_shape_skips`, `capture_skips` and `abandoned_targets` explain skipped work.
`pending_counts` reports recorded but unresolved pairs, and must be zero in a complete
final report. `disabled` and `disable_reason` identify CUDA failures. Empty distributions
have count 0 and null latencies, never fabricated zero-millisecond timings.

See [CUDA Event profiling](CUDA_EVENT_PROFILING.md) for lifecycle analysis, manual RPC/HTTP
commands and a GPU validation checklist. Abrupt termination can lose events since the last
snapshot; only a successful explicit stop guarantees the tail was collected.
