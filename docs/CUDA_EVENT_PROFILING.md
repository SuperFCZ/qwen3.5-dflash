# CUDA Event profiling on vLLM 0.22.1

## 执行与退出路径

本次按 vLLM 官方 **v0.22.1 tag** 核对，而不是把本机另一个 vLLM checkout 当作目标版本。
对应源码：

- [GPU Worker](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/v1/worker/gpu_worker.py)：`init_device` 选择 V1/V2 runner；`execute_model` 与 `sample_tokens` 进入实际 runner。
- [GPUModelRunner](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/v1/worker/gpu_model_runner.py)：`execute_model` 进行 forward、compute_logits，保存 `execute_model_state`；后续 `sample_tokens` 调用 `_sample`，然后才调用 drafter。
- [SpecDecodeBaseProposer](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/v1/spec_decode/llm_base_proposer.py)：DFlash 继承 `propose`；`parallel_drafting` 分支一次 forward 后返回 `[batch, k]` 候选，没有执行后面的自回归循环。
- [EngineCore](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/v1/engine/core.py)：`EngineCoreProc` 使用继承的 shutdown；run loop 的 finally 调用 shutdown 前将 SIGTERM 恢复为默认处理。
- [UniProcExecutor](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/v1/executor/uniproc_executor.py) 的 worker 与 EngineCore 同进程；[MultiprocExecutor](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/v1/executor/multiproc_executor.py) 的 Event 位于独立 worker，父进程的 collector 无法看到子进程内存。退出先关闭 death pipe，超时可升级 SIGTERM/SIGKILL。

旧方案的 `active` 仅说明 general plugin 已在某个进程修改类。其 `report` 只由 shutdown/atexit
触发，因此运行期间无输出是设计结果；退出报告还依赖实际 worker、相同插件版本、对应方法被调用以及
CUDA/日志仍可用。仓库 `ManagedServer.stop` 对整个进程组发 SIGTERM，不能把它当作跨进程 flush 屏障。
父进程补一个 shutdown hook 或多注册一次 atexit 都不能提供这个屏障。

若旧实现确实被实际 worker 调用且正常走完 shutdown，即使零样本也应输出。仅凭“active 且无结果”
不能断言是过滤条件、某个具体 signal，或插件部署路径问题；需要同一次 GPU 运行的进程日志才能区分。
本机没有该失败运行的服务日志和 CUDA 环境，因此这里不给出未经实测的唯一退出故障归因。

## 现在的采集/导出方式

1. general plugin 只安装 Worker 生命周期和专用 HTTP 控制入口，不在 API/EngineCore 创建 collector。
2. `Worker.init_device` 返回后绑定**实际 runner/drafter 实例**，记录 PID、device、rank、模型、量化配置和插件路径。
3. target Event 跨越 forward → logits → `_sample`；proposal 单独在继承的 `propose` 周围记录。
   不改模型 tensor、返回值、采样或调度逻辑，不插入 kernel 内 hook，不要求关闭 CUDA Graph。
4. 推理路径只记录 Event 并放入非阻塞队列。后台线程在正确 CUDA device 上 `query` 完成的 Event，
   每 5 秒输出统计；记录边界不等待 reporter 锁，不逐 step synchronize。
5. `/eqc_cuda_profile/start`、`/stop` 经已有 `AsyncLLM.collective_rpc` 调用每个 worker 的
   `eqc_cuda_event_profile`。start 重置窗口，stop 同步剩余 Event 并返回完整 JSON；没有 native torch
   profiler/trace 导出的依赖，也不覆盖 vLLM 的 `/start_profile`、`/stop_profile` 行为。
6. worker shutdown 仅兜底。重复 stop/shutdown 返回相同 final 记录，不重复记样本；下一次 start 开新窗口。

自动基准在自身 warm-up 后 start，所有测量请求结束后 stop，在退出服务器之前把返回值写入
`cuda_event_profile`，不需要从交错的多进程日志里猜测最终报告。HTTP/RPC 失败会导致运行明确失败。
专用控制接口仅在开关启用时注册；用没有其他并发客户端的独立基准服务。

## 手动服务和离线 LLM

使用与 vLLM 相同的 Python 环境安装插件并以 `EQC_DFLASH_CUDA_PROFILE=1` 启动服务。完成预热后：

```bash
curl --fail -X POST http://127.0.0.1:8000/eqc_cuda_profile/start
# 在此发送测量请求，并等待它们结束。
curl --fail -X POST http://127.0.0.1:8000/eqc_cuda_profile/stop > cuda-events.json
# stop 成功后再终止服务。运行中只读快照：
curl --fail -X POST http://127.0.0.1:8000/eqc_cuda_profile/snapshot
```

如果配置了 API key，使用服务要求的 Authorization header。离线 `vllm.LLM` 同样能经 worker RPC
控制，无须 HTTP：

```python
# 设置环境开关后创建 llm，先 warm-up，再执行：
llm.collective_rpc("eqc_cuda_event_profile", args=("start",))
outputs = llm.generate(prompts, sampling_params)
records = llm.collective_rpc("eqc_cuda_event_profile", args=("stop",))
```

## 统计定义和限制

- 支持 v0.22.1 的 `vllm.v1.worker.gpu_model_runner.GPUModelRunner`、PP=1、DP=1、DBO 关闭。
  TP 多 rank 的报告保留各自身份，不拼成单 GPU latency。遇到缺失边界或不支持配置直接诊断失败。
- DFlash 要求 `num_speculative_tokens=15` 且 `parallel_drafting=True`；返回 shape 也必须为 `[batch, 15]`。
  `dflash_proposal` 是一轮 15 候选的总区间，不除以 15，不逐候选计数。排除包含 prefill 请求的 proposal。
- `target_verify` 每请求实际调度量必须为 16（15 候选加前一/bonus token），不按接受 token 数归一化。
  不足 15 候选的尾轮、prefill/verify 混合轮不计。接受 0 个或 15 个候选均是一轮 verify。
- target-only 每请求调度 1 token，且准备好的 CPU batch 状态表明已经完成 prompt。
  这排除了 chunked/cached/resumed prefill 恰好剩一个 token 的情况。
- 每个完整 Event pair 贡献一个 batch 样本。`diagnostics.batch_sizes` 标记已记录样本的 batch size；
  并发变化时不要称为单请求延迟。单请求实验固定 concurrency=1，并使用独立服务。
- `mean_ms/p50_ms/p95_ms` 是毫秒；分位数使用 `(N-1)*q` 线性插值。零样本返回 null，不是 0 ms。
  日志同一 `(pid, generation)` 的周期记录是累计快照；只用最终记录，不相加、不平均各 rank 的 p95。
- Event 衡量 CUDA stream 时间轴跨度，可能包含 CPU 调度、RPC 间隔和 GPU idle；不是所有 kernel 的
  duration 总和。target 区间包括 logits 与普通/拒绝采样，proposal 包括 K/V 预计算与草稿采样。
- start/stop 同步发生在 wall-time 测量窗之外。后台查询、Event 与统计仍有开销，正式吞吐对比应关闭开关。
- 异常中断只能保证已写出的周期快照；SIGKILL 之前未完成或尚未导出的样本不能凭空补齐。
- 仓库 `w8_target_only.toml` 的 target 实际为 BF16；W8 指草稿实验线。现有量化 target-only 是
  `w4_target_only.toml`。若要 W8 target latency，另配真实 W8 target checkpoint，并检查输出 metadata。

## 验证

本机 CPU 测试覆盖跨 execute/sample 调用、继承 proposer、graph capture 跳过/replay 每次记录、
短 verify/混合轮/单-token prefill 排除、窗口重置、异常采样、尾 Event 收齐、HTTP → worker RPC、
以及真实子进程 `os._exit()` 不走退出 hook 时仍保留周期记录。fake CUDA tests 只能验证控制语义，
不能证明真实 GPU 毫秒数或 vLLM 启动兼容性。

在目标 CUDA 主机验收：

1. 更新插件并重启服务；确认 registered 的插件路径、版本，以及 worker_ready 的 runner/模型身份。
2. 运行 README 的 W4/W8 draft 和 W4 target-only 命令。应看到非零的对应 phase count、有限正延迟，
   `final=true`、`disabled=false`、所有 pending_counts 为 0。
3. 同一服务重复 start/请求/stop，检查 generation 递增且 count 不累加前一窗口；空窗口应全为 0/null。
4. 分别验证默认 CUDA Graph 与 `--enforce-eager`；比较同配置开关 profiling 前后的 greedy 输出应完全一致。
5. 严格记录 GPU 型号、batch size、prompt/context 长度和量化 metadata，避免混淆不同 workload 的 latency。
