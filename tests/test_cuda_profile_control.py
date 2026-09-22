from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from dflash_bench import cuda_profile
from dflash_bench.benchmark import run_benchmark
from dflash_bench.config import BenchmarkConfig, ExperimentConfig, ServerConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins/dflash_vllm_patch"))
from dflash_vllm_patch.cuda_profile_api import install_api_route, router  # noqa: E402


def worker_record():
    return {
        "schema_version": 3,
        "final": True,
        "diagnostics": {"disabled": False, "pending_counts": {"target_verify": 0}},
    }


class ControlTests(unittest.TestCase):
    def test_route_drains_then_calls_worker_rpc_and_returns_records(self):
        calls = []

        class Engine:
            async def wait_for_requests_to_drain(self):
                calls.append("drain")

            async def collective_rpc(self, method, *, timeout, args):
                calls.append((method, timeout, args))
                return [worker_record()]

        app = FastAPI()
        app.state.engine_client = Engine()
        app.include_router(router)
        with TestClient(app) as client:
            response = client.post("/eqc_cuda_profile/stop")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"workers": [worker_record()]})
            self.assertEqual(calls, ["drain", ("eqc_cuda_event_profile", 60.0, ("stop",))])
            calls.clear()
            client.post("/eqc_cuda_profile/snapshot")
            self.assertEqual(calls, [("eqc_cuda_event_profile", 60.0, ("snapshot",))])
            self.assertEqual(client.post("/eqc_cuda_profile/invalid").status_code, 400)

    def test_factory_retains_existing_routers_and_is_idempotent(self):
        calls = []
        serve = ModuleType("vllm.entrypoints.serve")
        serve.register_vllm_serve_api_routers = lambda app: calls.append(app)
        vllm = ModuleType("vllm")
        entrypoints = ModuleType("vllm.entrypoints")
        vllm.entrypoints = entrypoints
        entrypoints.serve = serve
        with patch.dict(
            sys.modules,
            {
                "vllm": vllm,
                "vllm.entrypoints": entrypoints,
                "vllm.entrypoints.serve": serve,
            },
        ):
            install_api_route()
            original = serve.register_vllm_serve_api_routers
            install_api_route()
            self.assertIs(serve.register_vllm_serve_api_routers, original)
            app = FastAPI()
            serve.register_vllm_serve_api_routers(app)
            self.assertEqual(calls, [app])
            self.assertIn("/eqc_cuda_profile/{action}", app.openapi()["paths"])

    def test_client_rejects_incomplete_and_disabled_reports(self):
        for mutation in ("pending", "disabled", "schema", "nonfinal"):
            record = worker_record()
            if mutation == "pending":
                record["diagnostics"]["pending_counts"]["target_verify"] = 1
            elif mutation == "disabled":
                record["diagnostics"]["disabled"] = True
            elif mutation == "schema":
                record["schema_version"] = 2
            else:
                record["final"] = False
            from io import BytesIO

            with (
                patch(
                    "urllib.request.urlopen",
                    return_value=nullcontext(BytesIO(json.dumps({"workers": [record]}).encode())),
                ),
                self.assertRaises(RuntimeError),
            ):
                cuda_profile.control("http://localhost:8000", "stop")

    def test_config_environment_overrides_inherited_flag(self):
        config = ExperimentConfig("test", "", ServerConfig("model"), BenchmarkConfig())
        with patch.dict("os.environ", EQC_DFLASH_CUDA_PROFILE="1"):
            self.assertTrue(cuda_profile.enabled(config))
            config = dataclasses.replace(
                config,
                server=dataclasses.replace(
                    config.server, environment={"EQC_DFLASH_CUDA_PROFILE": "0"}
                ),
            )
            self.assertFalse(cuda_profile.enabled(config))

    def test_benchmark_resets_after_warmup_and_flushes_before_return(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "prompts.jsonl"
            prompt.write_text('{"prompt":"test"}\n')
            config = ExperimentConfig(
                "test",
                "",
                ServerConfig("model", environment={"EQC_DFLASH_CUDA_PROFILE": "1"}),
                BenchmarkConfig(prompt_file=str(prompt), warmup_requests=1),
            )
            result = NS(
                prompt_id="line-1",
                repetition=0,
                output_tokens=2,
                latency_s=0.2,
                ttft_s=0.1,
                tpot_s=0.1,
                as_dict=lambda: {},
            )

            def request(_config, _base_url, _prompt, repetition, **kw):
                events.append("warmup" if repetition == -1 else "request")
                return result

            def control(_url, action):
                events.append(action)
                return [worker_record()]

            monitor = Mock()
            monitor.summary.return_value = {}
            with (
                patch("dflash_bench.benchmark._one_request", side_effect=request),
                patch("dflash_bench.benchmark.cuda_profile.control", side_effect=control),
                patch("dflash_bench.benchmark.fetch_metrics", return_value=""),
                patch("dflash_bench.benchmark._hardware", return_value={}),
                patch("dflash_bench.benchmark.GpuMonitor", return_value=nullcontext(monitor)),
            ):
                record = run_benchmark(config)
                self.assertEqual(events, ["warmup", "start", "request", "stop"])
                self.assertEqual(record["cuda_event_profile"], [worker_record()])
                events.clear()
                with patch(
                    "dflash_bench.benchmark.fetch_metrics", side_effect=RuntimeError("metrics")
                ):
                    with self.assertRaisesRegex(RuntimeError, "metrics"):
                        run_benchmark(config)
                self.assertEqual(events, ["warmup", "start", "stop"])
