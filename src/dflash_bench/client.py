"""OpenAI-compatible streaming client implemented with the Python standard library."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any


class RequestError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompletionResult:
    prompt_id: str
    repetition: int
    text: str
    output_tokens: int
    latency_s: float
    ttft_s: float
    tpot_s: float | None
    finish_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "repetition": self.repetition,
            "text": self.text,
            "output_tokens": self.output_tokens,
            "latency_s": self.latency_s,
            "ttft_s": self.ttft_s,
            "tpot_s": self.tpot_s,
            "finish_reason": self.finish_reason,
        }


def _chunk_text(chunk: dict[str, Any]) -> str:
    pieces: list[str] = []
    for choice in chunk.get("choices", []):
        delta = choice.get("delta") or {}
        reasoning = delta.get("reasoning_content")
        content = delta.get("content")
        if isinstance(reasoning, str):
            pieces.append(reasoning)
        if isinstance(content, str):
            pieces.append(content)
    return "".join(pieces)


def iter_sse_lines(lines: Iterable[bytes | str]) -> Iterator[dict[str, Any]]:
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            yield json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RequestError(f"invalid SSE JSON: {payload[:200]!r}") from exc


def parse_sse_lines(lines: Iterable[bytes | str]) -> list[dict[str, Any]]:
    """Materialize an SSE stream; primarily useful for tests and diagnostics."""
    return list(iter_sse_lines(lines))


def chat_completion(
    *,
    base_url: str,
    model: str,
    prompt_id: str,
    prompt: str,
    repetition: int,
    max_tokens: int,
    temperature: float,
    seed: int,
    timeout_s: float,
) -> CompletionResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "seed": seed,
        "n": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_token_at: float | None = None
    text_parts: list[str] = []
    output_tokens = 0
    finish_reason: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for chunk in iter_sse_lines(response):
                piece = _chunk_text(chunk)
                if piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(piece)
                usage = chunk.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                    output_tokens = int(usage["completion_tokens"])
                for choice in chunk.get("choices", []):
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RequestError(f"HTTP {exc.code}: {detail[:1000]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RequestError(str(exc)) from exc

    end = time.perf_counter()
    latency = end - start
    ttft = (first_token_at - start) if first_token_at is not None else latency
    if output_tokens <= 0 and text_parts:
        # Usage should be present on vLLM; this fallback preserves timing when it is not.
        output_tokens = 1
    tpot = (latency - ttft) / (output_tokens - 1) if output_tokens > 1 else None
    return CompletionResult(
        prompt_id=prompt_id,
        repetition=repetition,
        text="".join(text_parts),
        output_tokens=output_tokens,
        latency_s=latency,
        ttft_s=ttft,
        tpot_s=tpot,
        finish_reason=finish_reason,
    )
