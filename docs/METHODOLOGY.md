# Benchmark methodology

## 1. Questions and controls

There are two deliberately separate experiments.

### W8 draft quantization

Hold the target (`Qwen/Qwen3.5-4B`), prompts, decoding parameters, vLLM stack,
and DFlash architecture constant. Compare the five-layer BF16 source checkpoint at
revision `96899cc270945f554998309580b08a04a05a3187` with the W8A16 derivative.
This isolates draft weight quantization as closely as the public artifacts allow.

### Nota W4 system

Hold the QAD W4A16 target constant. Compare target-only, target + W4A16 draft with
full attention, and target + W4A16 draft with conditional SWA-1024. This evaluates
the deployed system, not a pure bit-width ablation: the target is different from
the W8 track and the draft was trained against the QAD target.

## 2. Reproducibility controls

- Pin vLLM to 0.22.1 and record driver/GPU data in every result.
- Use `temperature=0`, a fixed seed, identical prompts, and `n=1`.
- Warm up before taking the first Prometheus snapshot.
- Compute speculative metrics from counter deltas, never lifetime totals.
- Change one variable at a time. Do not compare runs with different targets,
  prompt files, output caps, context limits, or concurrency as though they were paired.
- Run every point at least three times after the model cache is warm. Report median and
  dispersion across runs for publication-quality results.
- Keep clocks, power limit, ambient temperature, and background GPU processes stable.

When `--prompts` points to a directory, the server stays loaded while each JSONL file is
measured as a separate workload. Every file gets a fresh warm-up and fresh before/after
counter snapshots, and produces its own result JSON. Keep the same file boundaries and
names across configurations so results remain directly pairable.

## 3. Grid

The suggested starting grid is:

| Variable | Values |
|---|---|
| speculative tokens (`K`) | 3, 7, 15 |
| concurrency | 1, 4 |
| max output tokens | 128 (fixed within one comparison) |
| repetitions | at least 3 |
| W8 variants | target-only, BF16 draft, W8A16 draft |
| W4 variants | target-only, W4 full attention, W4 SWA-1024 |

The bundled 20-prompt file is a deterministic engineering smoke benchmark, not a
quality leaderboard. For conclusions about production traffic, replace it with a
version-controlled JSONL sample from that traffic. Use distinct, fixed prompt sets for
short, 1K, 4K, and 8K contexts when studying SWA or context scaling.

## 4. Metrics

For counter deltas over the measured request interval:

```text
mean_accepted_draft_tokens = accepted_draft_tokens / draft_steps
mean_acceptance_length = 1 + accepted_draft_tokens / draft_steps
draft_acceptance_rate  = accepted_draft_tokens / drafted_tokens
position_i_rate        = accepted_at_position_i / draft_steps
```

Mean acceptance length includes the verifier's bonus token, matching vLLM's convention.
Some model cards call `mean_accepted_draft_tokens` the acceptance length instead; the two
numbers differ by exactly one when the same counters are used. Report both definitions.
Position rates are unconditional: position 5 counts only steps that reached and accepted
position 5.

Performance metrics have different meanings:

- TTFT measures prompt processing and queue effects before the first streamed text.
- TPOT is `(end-to-end - TTFT) / (output_tokens - 1)` per request.
- Output tok/s is total completed output tokens divided by measured wall time.
- Request/s is useful only when output lengths are held comparable.
- Peak VRAM is sampled after server readiness; it is runtime residency, not load peak.

The opt-in CUDA Event profiler (`EQC_DFLASH_CUDA_PROFILE=1`) reports three GPU timeline
intervals at worker shutdown. `dflash_proposal` wraps the full DFlash proposer.
`target_verify` and `target_only_single_token_decode` start immediately before the target
model forward and end after rejection/ordinary sampling, respectively. Events are resolved
opportunistically only after completion and synchronized once for the final report; no
per-step synchronization is introduced. Only pure verify or pure single-token decode batches
are classified. The server-side profiler includes harness warm-up requests, unlike the
Prometheus counter deltas and end-to-end benchmark aggregates.

## 5. Correctness gate

For greedy decoding, speculative decoding is intended to preserve the target model's
output. Compare each candidate with a target-only run using the same target and inputs.
Require 100% exact output match before trusting performance results. A mismatch can come
from a runtime patch problem, target mismatch, tokenizer/template drift, non-deterministic
kernels, or a changed generation parameter. Acceptance rate alone cannot prove correctness.

The W8 BF16/W8 pair should also have similar per-position acceptance curves. A sudden
collapse at every position is a stronger signal of an integration bug than of normal INT8
quantization loss.

## 6. Interpretation

A useful draft must improve the whole chain. Higher acceptance can still lose if the draft
cost, synchronization, or batching overhead is too high. Conversely, a quantized draft can
be worthwhile with nearly unchanged throughput if the memory saving permits a longer context
or larger batch. Report speed, acceptance, correctness, and memory together.
