"""CUDA Event control is explicit and completes while workers are still alive."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .config import ExperimentConfig


def enabled(config: ExperimentConfig) -> bool:
    return (
        config.server.environment.get(
            "EQC_DFLASH_CUDA_PROFILE", os.environ.get("EQC_DFLASH_CUDA_PROFILE")
        )
        == "1"
    )


def control(base_url: str, action: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/eqc_cuda_profile/{action}", data=b"", method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=360) as response:
            workers = json.load(response)["workers"]
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
        raise RuntimeError(
            f"CUDA Event {action} failed; check that dflash-vllm-patch>=0.3.0 and "
            f"EQC_DFLASH_CUDA_PROFILE=1 are active in API and worker processes: {exc}"
        ) from exc
    if not workers or not isinstance(workers, list):
        raise RuntimeError("CUDA Event control returned no worker records")
    for record in workers:
        if record.get("schema_version") != 3:
            raise RuntimeError("unexpected CUDA Event worker schema")
        diagnostics = record["diagnostics"]
        if diagnostics["disabled"] or diagnostics.get("emit_error"):
            raise RuntimeError(f"CUDA Event worker failed: {diagnostics}")
        if action == "stop" and (
            not record["final"] or any(diagnostics["pending_counts"].values())
        ):
            raise RuntimeError("CUDA Event stop returned an incomplete worker report")
    return workers
