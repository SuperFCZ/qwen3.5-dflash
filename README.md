# Qwen3.5-4B DFlash 量化基准

这个仓库用于在单张 NVIDIA RTX 3090（24 GB）上，复现并比较 Qwen3.5-4B
DFlash 草稿模型的量化效果：

- [`naveenrajk/Qwen3.5-4B-DFlash-W8A16`](https://huggingface.co/naveenrajk/Qwen3.5-4B-DFlash-W8A16)
- [`nota-ai/Qwen3.5-4B-DFlash-GPTQ-W4A16`](https://huggingface.co/nota-ai/Qwen3.5-4B-DFlash-GPTQ-W4A16)

它不是新的推理引擎，而是围绕 **vLLM 0.22.1** 的可复现实验层：负责启动服务、应用量化
DFlash 兼容补丁、发送成对请求、读取 Prometheus 接受率计数器、采集 `nvidia-smi`
遥测，并生成可直接比较的 JSON/Markdown 结果。

## 为什么单独建仓库

这是比直接修改 `day8reak/qwen3.5-4B-dflash` 更合适的边界。后者主要是严格 BF16、六层草稿
模型的正确性参考实现；这里的两个量化 checkpoint 是五层 `compressed-tensors` 草稿，且 Nota
W4 还绑定了单独训练的 QAD W4 目标模型和可选 SWA。把它们塞进参考仓库会混合模型实现、运行时
补丁和实验方法，也容易做出不公平的 W4/W8 横向比较。

本仓库因此拆成两个实验轨道：

| 轨道 | 固定目标模型 | 对照 | 待测草稿 | 能回答的问题 |
|---|---|---|---|---|
| W8 | `Qwen/Qwen3.5-4B` BF16 | 同源五层 BF16 草稿的固定历史 revision | W8A16 草稿 | 仅把草稿权重变成 INT8 后，接受率、速度和显存如何变化？ |
| W4 | `nota-ai/Qwen3.5-4B-QAD-W4A16` | 同一目标、不使用草稿 | W4A16 草稿，full attention / SWA-1024 | Nota 的完整 W4 推测解码系统相对目标单跑是否获益？ |

> W4 和 W8 **不是一条纯量化阶梯**。它们的目标模型、草稿训练过程和运行策略不同，不能把两者
> 的差值直接归因于 4-bit 与 8-bit。

## 仓库内容

```text
configs/                    六个可复现实验配置
environment.yml             Conda 环境定义（Python 3.12）
plugins/dflash_vllm_patch/  vLLM 0.22.1 量化 DFlash + 可选 SWA 插件
prompts/                    smoke 与 20 条成对基准提示
src/dflash_bench/           服务管理、请求、指标、GPU 遥测与报告工具
scripts/run_3090_matrix.py  K×并发实验矩阵
docs/                       方法与排错说明
tests/                      不依赖 GPU 的单元测试
```

## 环境要求

- Linux x86_64；RTX 3090 24 GB；可被当前 vLLM/CUDA 镜像支持的 NVIDIA 驱动
- Python 3.12（Nota 的已验证环境为 `>=3.12,<3.13`）
- 充足磁盘空间和 Hugging Face 访问权限
- 建议不要在现有训练环境中原地安装；为本仓库创建独立 Conda 环境

先确认 GPU：

```bash
nvidia-smi
```

### 方式一：Conda（推荐）

在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate qwen35-dflash
```

以后依赖文件有更新时，可以原地同步：

```bash
conda env update -f environment.yml --prune
```

`environment.yml` 会创建 Python 3.12 环境，并安装固定的 vLLM 栈、本仓库的 DFlash 插件和
`dflash-bench` 命令。

### 方式二：Python venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-vllm.txt
python -m pip install -e plugins/dflash_vllm_patch
python -m pip install -e '.[dev]'
```

这里固定 `vllm==0.22.1`，因为两个模型的已知可运行方案和本仓库补丁都针对这个版本。不要在同一
轮比较中途升级 vLLM、Transformers 或 CUDA 栈。

### 方式三：Docker

```bash
docker compose build bench
docker compose run --rm bench
```

容器内已经安装基准工具和插件，当前仓库与 Hugging Face cache 会被挂载到 `/workspace`。如模型
需要认证，先在宿主机设置 `HF_TOKEN`。Docker 使用官方 `vllm/vllm-openai:v0.22.1` 作为默认
基础镜像，可用 `--build-arg VLLM_IMAGE=...` 替换为适配本机驱动的等价镜像。

## 先做 smoke test

验证配置并查看实际启动命令：

```bash
dflash-bench validate configs/*.toml
dflash-bench command configs/w8_draft.toml
```

先分别跑一条 W8 和 W4 小实验：

```bash
dflash-bench run configs/w8_draft.toml \
  --prompts prompts/smoke.jsonl --max-tokens 64 --output results/w8-smoke.json

dflash-bench run configs/w4_draft_full.toml \
  --prompts prompts/smoke.jsonl --max-tokens 64 --output results/w4-smoke.json
```

`run` 会启动 vLLM，等待 `/health`，预热，读取一次 `/metrics`，执行测量请求，再读取计数器差值并
停止服务。服务日志与结果同名，后缀为 `.server.log`；逐题输出另存为
`*.responses.jsonl`。模型首次下载不计入基准时间。

如果你已经手工启动了服务：

```bash
dflash-bench run configs/w8_draft.toml \
  --no-launch --base-url http://127.0.0.1:8000 \
  --output results/w8-existing-server.json
```

此时配置仍决定请求模型名和基准参数，但工具不会管理服务进程。

## 使用你的多文件 JSONL 数据集

`--prompts` 既可以指向单个 `.jsonl`，也可以指向包含多个 `.jsonl` 的目录。目录只扫描第一层，
并按文件名排序。每个非空行必须是一个独立 JSON 对象，支持 `question` 或 `prompt` 字段；`id`
可选，缺省时自动使用该文件内的行号。比如：

```jsonl
{"question":"Janet's ducks lay 16 eggs per day. How much does she make?"}
{"id":"bolts-1","question":"A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total?"}
```

原始记录里的其他字段（例如标准答案、类别）会原样保存在结果的 `source_record` 中。假设目录为
`/data/gsm8k_parts/`，可以分别运行两个量化草稿：

```bash
conda activate qwen35-dflash

dflash-bench run configs/w8_draft.toml \
  --prompts /data/gsm8k_parts \
  --repetitions 3 \
  --output results/w8-gsm8k

dflash-bench run configs/w4_draft_full.toml \
  --prompts /data/gsm8k_parts \
  --repetitions 3 \
  --output results/w4-gsm8k
```

每条命令只加载一次模型，然后依次测量各个文件。每个文件有独立的预热、Prometheus 前后快照、
GPU 遥测和聚合指标。输出结构如下：

```text
results/w8-gsm8k/
├── manifest.json
├── server.log
├── part-000.result.json
├── part-000.responses.jsonl
├── part-001.result.json
└── part-001.responses.jsonl
```

`*.result.json` 包含完整配置、接受率、吞吐、延迟和显存；`*.responses.jsonl` 方便逐题分析，包含
输入、模型输出、耗时、token 数以及失败记录。`manifest.json` 汇总所有分片并给出对应文件名。
目录模式下 `--output` 表示目录，而单文件模式下仍表示结果 JSON 路径。

逐题记录按 `prompt_id` 自然排序，再按 `repetition` 升序排列。例如顺序为
`prompt1/r0, prompt1/r1, prompt2/r0, prompt2/r1, prompt10/r0, ...`，不会出现字符串排序导致的
`1, 10, 100, 2`。各次 repetition 保留为独立记录，便于检查输出一致性和运行波动，不会被平均值
掩盖。

如果要严格判断“草稿量化”本身的影响，不要直接把 W4 与 W8 相减：应在各自轨道内使用相同数据
分别跑本 README 开头表格中的 target/baseline/draft 配置，再比较对应分片的结果。

## 推荐实验顺序

### W8：干净的草稿量化对照

```bash
dflash-bench run configs/w8_target_only.toml --output results/w8-target.json
dflash-bench run configs/w8_bf16_draft.toml --output results/w8-bf16-k15.json
dflash-bench run configs/w8_draft.toml --output results/w8-int8-k15.json

dflash-bench compare \
  results/w8-target.json results/w8-bf16-k15.json results/w8-int8-k15.json \
  --output results/w8-report.md
```

BF16 配置固定到上游 revision
`96899cc270945f554998309580b08a04a05a3187`。当前上游主分支已换成六层结构；如果不固定
revision，就不再是 W8 checkpoint 的同源五层基线。

### W4：完整系统对照

```bash
dflash-bench run configs/w4_target_only.toml --output results/w4-target.json
dflash-bench run configs/w4_draft_full.toml --output results/w4-full-k15.json
dflash-bench run configs/w4_draft_swa1024.toml --output results/w4-swa-k15.json

dflash-bench compare \
  results/w4-target.json results/w4-full-k15.json results/w4-swa-k15.json \
  --output results/w4-report.md
```

SWA-1024 在上下文不超过 1024 时会保持 full-attention 路径；长上下文时才切到对称滑动窗口。

### K 与并发扫描

任何草稿配置都可在命令行覆盖 `K`、并发和重复次数：

```bash
dflash-bench run configs/w8_draft.toml --k 3  --concurrency 1 --repetitions 3
dflash-bench run configs/w8_draft.toml --k 7  --concurrency 1 --repetitions 3
dflash-bench run configs/w8_draft.toml --k 15 --concurrency 4 --repetitions 3
```

也可以运行完整矩阵（会多次加载模型，可能耗时数小时）：

```bash
python scripts/run_3090_matrix.py --track all --repetitions 3
```

先用 `--dry-run` 查看全部命令；用 `--quick` 只跑 smoke prompts、`K=15`、并发 1。

## 结果指标

每个 `*.result.json` 保存完整配置、硬件信息、逐请求输出和以下聚合指标：

- `mean_accepted_draft_tokens = accepted_draft_tokens / draft_steps`；
- `mean_acceptance_length = 1 + mean_accepted_draft_tokens`，包含目标模型 bonus token；
- `acceptance_rate = accepted_draft_tokens / drafted_tokens`；
- 每个草稿位置的无条件接受率；
- 请求吞吐、输出 token 吞吐、端到端延迟、TTFT、TPOT；
- 测量区间的峰值显存、平均 GPU 利用率、峰值功耗和温度；
- 相对报告中第一份结果的 greedy 完整文本一致率。

结果还记录实际 vLLM 命令、prompt 文件绝对路径及 SHA-256；修改 prompt 文件后不会被误认为同一
工作负载。

Prometheus 计数器在预热后读取，并用前后差值计算，因此不会把预热请求混入接受率。显存指标是
服务已启动后的运行期峰值，不代表模型加载过程的瞬时峰值。

两种“平均接受长度”口径经常混用。vLLM 日志使用包含 bonus token 的后一种；W8 模型卡中约
`3.60` 的数字与 `K × acceptance_rate` 一致，实际上是不含 bonus 的
`mean_accepted_draft_tokens`。本仓库同时输出两列，避免相差 1 的伪回归。

## 如何判断量化是否“有效”

至少同时看四件事：

1. **正确性**：同目标、greedy、同 prompts 时，目标单跑与推测解码应保持完整输出一致。
2. **草稿质量**：对照平均接受长度、接受率和逐位置曲线，而不只看单个平均数。
3. **链路性能**：看 output tok/s 与 TPOT；草稿自身更快不等于端到端一定更快，验证目标常是瓶颈。
4. **资源收益**：比较运行期显存。W8 模型卡的公开 3090 数据显示约节省 480 MiB，但端到端吞吐
   变化很小；应以你的驱动、上下文和并发下的实测为准。

建议每个点至少重复三次，保持 prompt 顺序、目标、vLLM 版本、最大长度、并发、温度和 GPU 功耗
状态一致。详细协议见 [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)。

## 量化兼容补丁

vLLM 0.22.1 的 DFlash 实现存在直接读取线性层浮点 `.weight` 的路径，量化层可能因此加载失败，
或绕过量化感知 forward。`plugins/dflash_vllm_patch` 通过 vLLM general-plugin entry point 在 API
进程和 worker 中统一修补：

1. 确保草稿 decoder layer 收到 draft quantization config；
2. 量化权重完成 post-processing 后，通过量化层的 identity probe 构造 fused K/V 权重；
3. 常规 QKV 路径调用量化线性层自身的 forward；
4. `fc` 路径根据量化 scale 的 dtype 转换输入；
5. 可选启用 Nota 的条件式、对称 SWA。

补丁默认不生效。量化配置通过 `EQC_DFLASH_QUANT_PATCH=1` 开启；SWA 配置再设置
`EQC_DFLASH_SWA_WINDOW=1024`。实现改编自 Nota AI 的 Apache-2.0 插件，并保留了 NOTICE。

### CUDA Event 分阶段计时

需要区分草稿开销与目标验证开销时，在待测 TOML 中开启：

```toml
[server.environment]
EQC_DFLASH_CUDA_PROFILE = "1"
```

量化草稿配置已有 `[server.environment]`，直接在同一节追加这一行；target-only 配置则新建该节。
也可以只对单次命令临时开启，例如
`EQC_DFLASH_CUDA_PROFILE=1 dflash-bench run configs/w8_draft.toml --output results/profile.json`。
插件只记录 CUDA Event，不在每个 decode step 同步。vLLM V1 EngineCore shutdown 时、释放
model executor 之前统一同步一次，并在
`*.server.log`（目录输入模式为 `server.log`）输出一行机器可读 JSON：

```text
[dflash_vllm_patch] CUDA_EVENT_PROFILE {"clock":"cuda_event",...,"metrics":{"dflash_proposal":{"count":...,"mean_ms":...,"p50_ms":...,"p95_ms":...},"target_verify":{...},"target_only_single_token_decode":{...}}}
```

```bash
rg 'CUDA_EVENT_PROFILE' results/*.server.log results/*/server.log
```

三个区间的口径如下：

- `dflash_proposal`：完整 `DFlashProposer.propose`，包括 context-K/V 预计算、草稿 forward 和草稿采样；
- `target_verify`：纯 DFlash verify batch，从目标模型 forward 开始，到 rejection sampling 结束；
- `target_only_single_token_decode`：无 speculative config 的纯单-token decode batch，边界同上。

prefill、混合 prefill/decode batch，以及只有部分请求携带 draft token 的混合 batch 不计入这三项。
CUDA Event profiler 位于 vLLM worker，因而也会看到 harness 的 warm-up 请求；做严格的 measured-only
采样时，可将 `warmup_requests=0`，并在正式实验前另跑一次 smoke warm-up。profiling 本身会增加少量
Event 记录开销，因此端到端吞吐结论仍应以关闭该开关的正式运行结果为准。

## 测试

控制面测试不需要 GPU 或 vLLM：

```bash
python -m unittest discover -s tests -v
python -m compileall -q src plugins/dflash_vllm_patch
```

GPU smoke test 仍是发布实验结果前的必要步骤。常见问题见
[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)。

## 上游资料

- [DFlash 参考实现](https://github.com/day8reak/qwen3.5-4B-dflash)
- [Nota AdaptFM 量化 DFlash](https://github.com/nota-github/adaptfm-quant-dflash)
- [vLLM DFlash 量化草稿问题 #51581](https://github.com/vllm-project/vllm/issues/51581)
- [DFlash 论文](https://arxiv.org/abs/2602.06036)

## License

Apache License 2.0。模型权重不包含在本仓库中，分别遵循其模型卡标注的许可证和使用条款。
