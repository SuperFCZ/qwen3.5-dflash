# ruff: noqa: E402, I001
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PLUGIN_ROOT = Path(__file__).resolve().parents[1] / "plugins/dflash_vllm_patch"
sys.path.insert(0, str(PLUGIN_ROOT))

from dflash_vllm_patch import (
    _patch_decoder_quant_config,
    _patch_sliding_window,
)


class PluginTests(unittest.TestCase):
    def test_existing_quant_config_is_forwarded(self) -> None:
        class Layer:
            def __init__(
                self,
                vllm_config,
                *,
                config,
                cache_config=None,
                quant_config=None,
                prefix="",
            ):
                self.received = (vllm_config, config, cache_config, quant_config, prefix)

        module = SimpleNamespace(DFlashQwen3DecoderLayer=Layer)
        _patch_decoder_quant_config(module)
        instance = Layer("vllm", config="model", quant_config="quant", prefix="draft")
        self.assertEqual(instance.received, ("vllm", "model", None, "quant", "draft"))

    def test_conditional_swa_is_a_noop_below_window(self) -> None:
        class Implementation:
            sliding_window = None

            def forward(self, *args, **kwargs):
                return self.sliding_window

        class Attention:
            def __init__(self):
                self.attn = SimpleNamespace(impl=Implementation())

        module = SimpleNamespace(DFlashQwen3Attention=Attention)
        _patch_sliding_window(module, 1024, static=False)
        attention = Attention()
        short = attention.attn.impl.forward(attn_metadata=SimpleNamespace(max_seq_len=1024))
        long = attention.attn.impl.forward(attn_metadata=SimpleNamespace(max_seq_len=1025))
        self.assertEqual(short, (-1, -1))
        self.assertEqual(long, (1023, 1023))

    def test_static_swa_is_symmetric(self) -> None:
        class Implementation:
            sliding_window = None

            def forward(self, *args, **kwargs):
                return None

        class Attention:
            def __init__(self):
                self.attn = SimpleNamespace(impl=Implementation())

        module = SimpleNamespace(DFlashQwen3Attention=Attention)
        _patch_sliding_window(module, 8, static=True)
        self.assertEqual(Attention().attn.impl.sliding_window, (7, 7))


if __name__ == "__main__":
    unittest.main()
