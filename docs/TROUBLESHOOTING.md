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

Confirm that the rendered command contains `EQC_DFLASH_CUDA_PROFILE=1`, let the harness stop
the managed server normally, and inspect the worker's complete server log. The report is
emitted only during graceful worker shutdown and only after at least one qualifying phase was
recorded. Target-only prefill and multi-token batches do not count as single-token decode;
mixed DFlash verify/prefill batches are deliberately excluded as well.
