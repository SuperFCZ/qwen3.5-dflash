"""Tiny Prometheus text parser and vLLM speculative-decoding statistics."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

MetricKey = tuple[str, tuple[tuple[str, str], ...]]
Snapshot = dict[MetricKey, float]

_METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?Inf|NaN)"
    r"(?:\s+\d+)?$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"(?:,|$)')


def _unescape_label(value: str) -> str:
    return value.replace(r"\\", "\\").replace(r"\"", '"').replace(r"\n", "\n")


def parse_prometheus(text: str) -> Snapshot:
    snapshot: Snapshot = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_RE.match(line)
        if not match:
            continue
        labels: list[tuple[str, str]] = []
        label_text = match.group("labels")
        if label_text:
            labels = [
                (m.group(1), _unescape_label(m.group(2))) for m in _LABEL_RE.finditer(label_text)
            ]
        value = float(match.group("value"))
        snapshot[(match.group("name"), tuple(sorted(labels)))] = value
    return snapshot


def _value(snapshot: Mapping[MetricKey, float], name: str) -> float:
    return sum(value for (metric, _labels), value in snapshot.items() if metric == name)


def _delta(before: Mapping[MetricKey, float], after: Mapping[MetricKey, float], name: str) -> float:
    return _value(after, name) - _value(before, name)


@dataclass(frozen=True)
class SpeculativeStats:
    draft_steps: int
    draft_tokens: int
    accepted_tokens: int
    mean_accepted_draft_tokens: float | None
    mean_accept_length: float | None
    acceptance_rate: float | None
    per_position_acceptance: dict[int, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "draft_steps": self.draft_steps,
            "draft_tokens": self.draft_tokens,
            "accepted_tokens": self.accepted_tokens,
            "mean_accepted_draft_tokens": self.mean_accepted_draft_tokens,
            "mean_accept_length": self.mean_accept_length,
            "acceptance_rate": self.acceptance_rate,
            "per_position_acceptance": {
                str(key): value for key, value in self.per_position_acceptance.items()
            },
        }


def speculative_stats(before: Snapshot, after: Snapshot) -> SpeculativeStats | None:
    drafts_name = "vllm:spec_decode_num_drafts_total"
    tokens_name = "vllm:spec_decode_num_draft_tokens_total"
    accepted_name = "vllm:spec_decode_num_accepted_tokens_total"
    steps = max(0.0, _delta(before, after, drafts_name))
    draft_tokens = max(0.0, _delta(before, after, tokens_name))
    accepted = max(0.0, _delta(before, after, accepted_name))

    known_names = {key[0] for key in before} | {key[0] for key in after}
    if not ({drafts_name, tokens_name, accepted_name} & known_names):
        return None

    position_name = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
    per_position: dict[int, float] = {}
    all_keys = set(before) | set(after)
    for name, labels_tuple in all_keys:
        if name != position_name:
            continue
        labels = dict(labels_tuple)
        raw_position = labels.get("position")
        if raw_position is None:
            continue
        try:
            position = int(raw_position)
        except ValueError:
            continue
        delta = max(
            0.0, after.get((name, labels_tuple), 0.0) - before.get((name, labels_tuple), 0.0)
        )
        if steps:
            per_position[position] = delta / steps

    return SpeculativeStats(
        draft_steps=round(steps),
        draft_tokens=round(draft_tokens),
        accepted_tokens=round(accepted),
        mean_accepted_draft_tokens=(accepted / steps) if steps else None,
        mean_accept_length=(1.0 + accepted / steps) if steps else None,
        acceptance_rate=(accepted / draft_tokens) if draft_tokens else None,
        per_position_acceptance=dict(sorted(per_position.items())),
    )


def finite_snapshot(snapshot: Snapshot) -> Snapshot:
    """Drop NaN/Inf values before serializing diagnostic snapshots."""
    return {key: value for key, value in snapshot.items() if math.isfinite(value)}
