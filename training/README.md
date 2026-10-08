# 三语言 XLM-R 训练与迁移

一个 XLM-R Base 模型混合学习英文、简体中文和泰文，输出固定 25 类意图的集合。
这是监督微调；不训练 ASR，不做额外 MLM 预训练，不调用生成模型 API。

训练工程有独立 `pyproject.toml`、`uv.lock` 和 `.venv`，不修改根目录生成工程的锁文件。
所有命令在 **WSL / Linux 的 `Bert_lab/training` 目录**执行。

## 1. 本地安装和验证

需要 Python 3.12、uv。CPU 与 CUDA 两组依赖互斥；不要把它们同时安装。

```bash
cd /home/panhao/Bert_lab/training
uv sync --frozen --extra cpu --extra dev
uv run --frozen --extra cpu pytest -q
uv run --frozen --extra cpu ruff check src tests
uv run --frozen --extra cpu watch-intents-train check
uv run --frozen --extra cpu watch-intents-train smoke --forward-only
```

本地真实权重检查只运行前向，适合当前约 7.5GB RAM 的 WSL。
测试使用临时生成的小型 XLM-R 权重，覆盖反向更新、保存重载、断点恢复、五次实验和测试门禁；
小模型结果只证明工程功能，不是业务指标。真实权重的完整反向与保存重载留给 GPU。

默认配置为 `configs/baseline.toml`。路径相对于配置文件解析，迁移整个目录后不需要改路径。
模型和 tokenizer 只从本地读取，正式运行不依赖 Hugging Face 网络。

## 2. 打包和上传

```bash
uv run --frozen --extra cpu watch-intents-train pack \
  --output ../outputs/transfer/bert-lab-training.tar.gz
sha256sum ../outputs/transfer/bert-lab-training.tar.gz
```

归档使用白名单，仅包含训练源代码、配置、锁文件、测试和说明、本地基础模型、冻结数据导出。
不会包含 API 凭证、虚拟环境、Git 目录、生成数据库或构建日志。
归档内的 `TRANSFER.json` 记录逐文件 SHA256，包括尚未提交但属于白名单的源文件。
输出文件已存在时拒绝覆盖，重新打包请使用新名字。

通过平台上传或 SCP 把归档上传到 GPU 数据盘。在云端执行：

```bash
cd /root/autodl-tmp
tar -xzf bert-lab-training.tar.gz
cd Bert_lab/training
```

如果没有 uv，可按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)安装，
或在所选 Python 3.12 镜像中执行 `python -m pip install uv==0.11.18`。

## 3. GPU 环境

推荐 1 张 RTX 4090 24GB，8 核以上 CPU、32GB 以上 RAM、100GB 可用数据盘。
镜像选择 PyTorch 2.5.1 / Python 3.12 / Ubuntu 22.04 / CUDA 12.4。
本项目仍在独立环境中按锁文件安装；CUDA 构建与镜像一致，防止镜像隐含依赖改变实验。

```bash
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
uv sync --frozen --extra cuda --extra dev
nvidia-smi
uv run --frozen --extra cuda watch-intents-train verify-transfer --root ..
uv run --frozen --extra cuda watch-intents-train check
uv run --frozen --extra cuda watch-intents-train smoke
uv run --frozen --extra cuda watch-intents-train overfit --steps 200 --samples 96
uv run --frozen --extra cuda watch-intents-train benchmark --warmup 20 --steps 100
```

检查命令只做文件、标签、token 长度和分组验证，读取测试集是验证完整性，不生成测试预测、不用于调参。
`overfit` 仅从训练集轮流抽取三语言×零/单/多标签样本，以诊断学习率 1e-4 检查可学习性，
达到集合正确率 0.98 结束；它不产出正式候选。未达到时返回退出码 2，应先检查诊断结果。
硬件计时包含反向和优化器，不能用推理吞吐代替。预计纯训练秒数≈9000×epochs/实测 samples/s，
另加验证、保存及加载开销。CPU 检查不替代 GPU 冒烟和显存验证。

显存不足时，在**开始正式实验前**把 `micro_batch` 从 16 改为 8、
`gradient_accumulation` 从 2 改为 4，有效 batch 保持 32，然后重跑诊断。
不能用更改后的配置续跑旧 checkpoint；新配置使用新实验名。

## 4. 一次命令完成基线实验

先确认前述命令成功。建议在 `tmux` 会话中执行，断开 SSH 后进程仍可继续。

```bash
mkdir -p ../outputs/training
set -o pipefail
uv run --frozen --extra cuda watch-intents-train experiment --name baseline \
  2>&1 | tee ../outputs/training/baseline-console.log
```

顺序固定：种子 42 的学习率 2e-5、1e-5、3e-5；仅用验证集选择最佳学习率；
补跑该学习率的种子 43、44。最多 5 次正式训练，单卡顺序执行。
每次最多 5 epochs，验证集合正确率连续 2 轮无改善则提前停止。
每次训练按阈值 0.5 的验证指标选择最佳 epoch，同分保留较早 epoch。
随后仅在验证集扫描统一阈值 0.10–0.90，步长 0.01；同分优先接近 0.5，再取较小值。
最佳学习率按校准后集合正确率、Macro F1、预定学习率顺序选取。
三种子中发布候选也只用验证指标选取，同分按 42、43、44。

所有候选和阈值冻结后，才为最佳学习率的三个种子生成测试报告。
没有明确冻结文件，`evaluate --split test` 会拒绝执行。
程序门禁防止误用命令，不能替代研究纪律：不要查看测试误例后回头调整同一轮配置、
阈值或训练数据，再把同一测试集称为独立盲测。

中断后保持配置、代码、模型和依赖版本相同，用：

```bash
uv run --frozen --extra cuda watch-intents-train experiment --name baseline --resume
```

只保留最近一个**完整 epoch** 的恢复点；中断的 epoch 从头重跑。
如果首个 epoch 前就中断、没有 `latest.pt`，使用新实验名重试。
恢复校验涵盖数据、模型资产、配置、代码及依赖版本；不承诺跨硬件位级一致。
单个 run 也可使用 `train --run NAME` 和 `train --run NAME --resume`。
结束后先下载模型与报告，再关机释放 GPU 计费；数据盘费用遵循平台规则。

## 5. 输出与使用

输出在仓库 `outputs/training/`，默认已被 Git 忽略。

| 文件 | 内容 |
|---|---|
| `baseline-report.md` / `baseline-summary.json` | 最佳配置、验证集选择的候选、三种子均值与样本标准差 |
| `baseline-experiment.json` | 实验状态、失败类型、冻结记录 |
| `freeze-*.json` | 封存测试前的候选、阈值和模型哈希 |
| `RUN/run.json` | 可复现实验身份、每轮 loss 和验证指标 |
| `RUN/latest.pt` | 最近完整 epoch 的训练、优化器、调度器和随机状态 |
| `RUN/best-epoch-*/` | 未校准的最佳模型 |
| `RUN/model/` | 最佳模型加验证集阈值；可独立迁移 |
| `RUN/calibration.json` / `threshold-curve.csv` | 验证集阈值选择和风险—覆盖率曲线数据 |
| `RUN/test-report.md` / `test-metrics.json` | 总体、分语言、场景及路由指标 |
| `RUN/test-per-label.csv` | 各语言逐标签 precision、recall、F1 和支持数 |
| `RUN/test-predictions.jsonl` | 完整预测、概率和错误样本 |
| `RUN/performance-epoch-*.json` | 训练秒数、训练吞吐与峰值显存 |
| `benchmark-*.json` | 预热后性能及实际硬件信息 |

指标包括集合完全正确率、Micro/Macro F1、简单链路错误率/正确率/覆盖率/正确覆盖率、
可简单请求命中率、零/多标签误入简单链路及 0/1/≥2 混淆矩阵。
分语言、逐标签、场景及语言×场景单列。无分母时使用 null，不填 0 或 100%；
Macro F1 只平均有定义标签，同时报告参与数量。比例指标给 Wilson 95% 区间，
该区间的独立样本假设可能不适用于合成模板数据，不能当作生产错误率保证。

将 `RUN` 替换为 summary.json 中的 `selected_run`，检查离线预测和本机延迟：

```bash
uv run --frozen --extra cuda watch-intents-train predict \
  --bundle ../outputs/training/RUN/model --text '打开蓝牙，再打开运动页面'
uv run --frozen --extra cuda watch-intents-train benchmark \
  --bundle ../outputs/training/RUN/model --warmup 20 --steps 1000
```

预测标准输出为标签数组，固定 ID 顺序，无执行顺序语义。空白或超长输入返回错误，
不会伪装为 `[]`，也不静默截断。预测 1 个标签对应简单链路，0 或 ≥2 个对应 Agent。
延迟包含本地 tokenizer、传输、forward 和标签序列化，不包含服务排队、网络或生产并发。
模型包含同一套 tokenizer、标签、阈值、长度策略和文件哈希；不要只复制某个权重文件。

50GB 数据盘需特别控制磁盘余量：每个正式 run 留一个最佳原始包、一个校准包和一个完整
恢复点，五次实验再加环境/缓存可能接近可用空间。推荐在开始前扩容到 100GB；
保存失败时先处理空间，再从上一个完整 epoch 恢复。
