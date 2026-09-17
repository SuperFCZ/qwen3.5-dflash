from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from dflash_bench.cli import _override
from dflash_bench.config import ConfigError, load_config, render_shell_command, render_vllm_command

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_all_repository_configs_validate(self) -> None:
        configs = sorted((ROOT / "configs").glob("*.toml"))
        self.assertEqual(len(configs), 6)
        for path in configs:
            with self.subTest(path=path.name):
                config = load_config(path)
                self.assertTrue(config.prompt_path().is_file())

    def test_w8_command_has_revision_and_quantization(self) -> None:
        config = load_config(ROOT / "configs/w8_draft.toml")
        command = render_vllm_command(config)
        spec = json.loads(command[command.index("--speculative-config") + 1])
        self.assertEqual(spec["method"], "dflash")
        self.assertEqual(spec["num_speculative_tokens"], 15)
        self.assertEqual(spec["quantization"], "compressed-tensors")
        self.assertIn("EQC_DFLASH_QUANT_PATCH=1", render_shell_command(config))

        baseline = load_config(ROOT / "configs/w8_bf16_draft.toml")
        baseline_command = render_vllm_command(baseline)
        baseline_spec = json.loads(
            baseline_command[baseline_command.index("--speculative-config") + 1]
        )
        self.assertEqual(baseline_spec["revision"], "96899cc270945f554998309580b08a04a05a3187")

    def test_target_only_has_no_speculative_flag(self) -> None:
        config = load_config(ROOT / "configs/w4_target_only.toml")
        self.assertNotIn("--speculative-config", render_vllm_command(config))

    def test_invalid_port_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            path.write_text('name="bad"\n[server]\nmodel="x"\nport=70000\n', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "1..65535"):
                load_config(path)

    def test_invalid_cli_override_is_rejected(self) -> None:
        config = load_config(ROOT / "configs/w8_draft.toml")
        args = Namespace(
            prompts=None,
            concurrency=0,
            repetitions=None,
            max_tokens=None,
            warmup_requests=None,
            k=None,
        )
        with self.assertRaisesRegex(ConfigError, "concurrency"):
            _override(config, args)


if __name__ == "__main__":
    unittest.main()
