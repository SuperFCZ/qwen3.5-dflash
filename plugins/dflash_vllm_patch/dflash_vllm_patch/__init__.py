"""Compatibility patches for quantized DFlash drafters on vLLM 0.22.1.

The plugin is loaded in every vLLM process through the ``vllm.general_plugins``
entry-point. It remains a no-op unless ``EQC_DFLASH_QUANT_PATCH=1`` or
``EQC_DFLASH_SWA_WINDOW`` is set.
"""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

QUANT_ENV = "EQC_DFLASH_QUANT_PATCH"
SWA_ENV = "EQC_DFLASH_SWA_WINDOW"
SWA_STATIC_ENV = "EQC_DFLASH_SWA_STATIC"


def _log(message: str) -> None:
    print(f"[dflash_vllm_patch] {message}", file=sys.stderr, flush=True)


def _floating_dtype(module: Any, fallback: Any) -> Any:
    import torch

    for attribute in ("weight_scale", "input_scale", "scale", "weight"):
        tensor = getattr(module, attribute, None)
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            return tensor.dtype
    return fallback


def _plain_float_weight(module: Any) -> bool:
    import torch

    weight = getattr(module, "weight", None)
    return isinstance(weight, torch.Tensor) and weight.is_floating_point()


def _dequantized_weight(linear: Any) -> Any:
    """Recover effective ``[out, in]`` weights with a quant-aware identity probe."""
    import torch

    hidden = getattr(linear, "input_size", None)
    if hidden is None:
        hidden = linear.input_size_per_partition
    hidden = int(hidden)
    parameter = next(linear.parameters())
    dtype = _floating_dtype(linear, torch.bfloat16)
    identity = torch.eye(hidden, dtype=dtype, device=parameter.device)
    with torch.inference_mode():
        output = linear(identity)
    if isinstance(output, tuple):
        output = output[0]
    bias = getattr(linear, "bias", None)
    if isinstance(bias, torch.Tensor):
        output = output - bias
    return output.transpose(0, 1).contiguous()


def _patch_decoder_quant_config(module: Any) -> None:
    layer = module.DFlashQwen3DecoderLayer
    original = layer.__init__
    if getattr(original, "_dflash_quant_patch", False):
        return

    def patched_init(
        self: Any,
        vllm_config: Any,
        *,
        config: Any,
        cache_config: Any = None,
        quant_config: Any = None,
        prefix: str = "",
    ) -> None:
        if quant_config is None:
            try:
                from vllm.config import get_current_vllm_config
                from vllm.model_executor.models.utils import get_draft_quant_config

                quant_config = get_draft_quant_config(get_current_vllm_config())
            except Exception as exc:  # pragma: no cover - depends on vLLM internals
                _log(f"could not resolve draft quant_config: {type(exc).__name__}: {exc}")
        original(
            self,
            vllm_config,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

    patched_init._dflash_quant_patch = True  # type: ignore[attr-defined]
    layer.__init__ = patched_init


def _patch_fused_kv(module: Any) -> None:
    model = module.DFlashQwen3Model
    original = model._build_fused_kv_buffers
    if getattr(original, "_dflash_quant_patch", False):
        return

    def patched_build(self: Any) -> Any:
        for layer in self.layers:
            projection = layer.self_attn.qkv_proj
            if _plain_float_weight(projection):
                continue
            try:
                weight = _dequantized_weight(projection)
            except (AttributeError, RuntimeError, TypeError) as exc:
                # During load, Marlin/Machete has not completed post-processing. The
                # existing lazy call from precompute_and_store_context_kv retries later.
                _log(f"deferring fused-KV build until quant buffers are ready: {exc}")
                return None
            object.__setattr__(projection, "weight", weight)
            _log(f"recovered fused-KV source weight {tuple(weight.shape)}")
        return original(self)

    patched_build._dflash_quant_patch = True  # type: ignore[attr-defined]
    model._build_fused_kv_buffers = patched_build


def _patch_attention_forward(module: Any) -> None:
    attention = module.DFlashQwen3Attention
    if getattr(attention.forward, "_dflash_quant_patch", False):
        return

    def forward(self: Any, positions: Any, hidden_states: Any) -> Any:
        qkv = self.qkv_proj(hidden_states)
        if isinstance(qkv, tuple):
            qkv = qkv[0]
        query, key, value = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        query_shape, key_shape = query.shape, key.shape
        query = self.q_norm(
            query.view(*query_shape[:-1], query_shape[-1] // self.head_dim, self.head_dim)
        ).view(query_shape)
        key = self.k_norm(
            key.view(*key_shape[:-1], key_shape[-1] // self.head_dim, self.head_dim)
        ).view(key_shape)
        query, key = self.rotary_emb(positions, query, key)
        output = self.attn(query, key, value)
        output, _ = self.o_proj(output)
        return output

    forward._dflash_quant_patch = True  # type: ignore[attr-defined]
    attention.forward = forward


def _patch_combine_hidden_states(module: Any) -> None:
    causal_lm = module.DFlashQwen3ForCausalLM
    original = causal_lm.combine_hidden_states
    if getattr(original, "_dflash_quant_patch", False):
        return

    def combine_hidden_states(self: Any, hidden_states: Any) -> Any:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        dtype = _floating_dtype(self.model.fc, hidden_states.dtype)
        result = self.model.fc(hidden_states.to(dtype=dtype))
        if isinstance(result, tuple):
            result = result[0]
        return result.squeeze(0) if needs_squeeze else result

    combine_hidden_states._dflash_quant_patch = True  # type: ignore[attr-defined]
    causal_lm.combine_hidden_states = combine_hidden_states


def _patch_sliding_window(module: Any, window: int, *, static: bool) -> None:
    attention = module.DFlashQwen3Attention
    original = attention.__init__
    if getattr(original, "_dflash_swa_patch", False):
        return
    symmetric = (window - 1, window - 1)
    full = (-1, -1)

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        implementation = getattr(self.attn, "impl", None)
        if implementation is None:
            return
        if static:
            implementation.sliding_window = symmetric
            return
        if getattr(implementation, "_dflash_conditional_swa", False):
            return
        implementation.sliding_window = full
        original_forward = implementation.forward

        def conditional_forward(*forward_args: Any, **forward_kwargs: Any) -> Any:
            metadata = forward_kwargs.get("attn_metadata")
            if metadata is None and len(forward_args) >= 6:
                metadata = forward_args[5]
            sequence_length = getattr(metadata, "max_seq_len", None)
            implementation.sliding_window = (
                symmetric if sequence_length is not None and sequence_length > window else full
            )
            return original_forward(*forward_args, **forward_kwargs)

        implementation.forward = conditional_forward
        implementation._dflash_conditional_swa = True

    patched_init._dflash_swa_patch = True  # type: ignore[attr-defined]
    attention.__init__ = patched_init


def register() -> None:
    quantized = os.environ.get(QUANT_ENV) == "1"
    raw_window = os.environ.get(SWA_ENV)
    window: int | None = None
    if raw_window:
        try:
            window = int(raw_window)
        except ValueError:
            _log(f"ignoring invalid {SWA_ENV}={raw_window!r}")
    if not quantized and not (window and window > 0):
        return
    try:
        detected_version = version("vllm")
    except PackageNotFoundError:  # pragma: no cover - import below reports the useful error
        detected_version = "unknown"
    if detected_version != "0.22.1":
        _log(f"warning: patches are validated on vLLM 0.22.1, found {detected_version}")
    try:
        import vllm.model_executor.models.qwen3_dflash as module
    except Exception as exc:  # pragma: no cover - only exercised in the GPU env
        _log(f"could not import vLLM DFlash module: {type(exc).__name__}: {exc}")
        return
    if quantized:
        _patch_decoder_quant_config(module)
        _patch_fused_kv(module)
        _patch_attention_forward(module)
        _patch_combine_hidden_states(module)
        _log("quantized DFlash compatibility patches active")
    if window and window > 0:
        static = os.environ.get(SWA_STATIC_ENV) == "1"
        _patch_sliding_window(module, window, static=static)
        _log(f"DFlash SWA active: window={window}, static={static}")


__all__ = ["register"]
