# Bert_lab — 三语言手表指令数据集

为本地 `xlm-roberta-base` 微调建设合成数据。生成模型和独立盲审模型统一由 `dashscope.env` 配置。
数据完成后，使用独立的 [训练与 GPU 迁移工程](training/README.md) 完成监督微调、阈值校准及指标报告。
根目录命令仍只负责数据建设；训练环境及锁文件位于 `training/`，不会改变已冻结数据的生成指纹。

标签和标注边界见 [标签规范](docs/label-guide.md)，固定字典见 [labels.v0.1.json](configs/labels.v0.1.json)。业务目标为当前完整文本的去重标签集合，不输出参数、执行顺序或回复。零标签使用 `[]`，不设 `OTHER`。

## 本地运行（WSL / Linux）

在 WSL 终端执行，使用 Python 3.12。当前机器已安装 uv 和对应 Python。

```bash
cd /home/panhao/Bert_lab
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen --extra dev
uv run watch-intents check
```

`check` 不调用模型、不生成数据；检查配置、25 标签映射、配额、凭证是否存在、tokenizer 和真实 PyTorch safetensors 文件。密钥有效性只有实际 API 调用才能验证。已有 `dashscope.env` 可直接使用，不要用示例文件覆盖真实配置。`dashscope.env` 中的配置优先于终端环境变量；仅文件未定义的项才读取环境变量。显式留空会报错，不会偷偷使用旧值。

更换模型、Key 或接口地址只需修改 `dashscope.env`，下次运行命令即生效，无需修改代码或数据配置。生成端使用 `SUUNTO_CHAT_MODEL_CN_NAME`、`DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`；审核端使用 `SUUNTO_EVAL_JUDGE_NAME`、`SUUNTO_EVAL_JUDGE_API_KEY`、`SUUNTO_EVAL_JUDGE_BASE_URL`。模型名必填，不设白名单；接口需兼容 chat completions 协议，审核 provider 保持 `dashscope`。未定义接口地址时使用 `https://dashscope.aliyuncs.com/compatible-mode/v1`。真实配置和示例配置均可显式设置上述两个 BaseURL；地址不包含 `/chat/completions`，由程序追加。Key 必须对所填地址有访问权限；显式填写同一个默认地址不会修复无效 Key。`check` 输出实际生效的模型和地址，不输出 Key。Key、模型名和接口地址属于运行连接配置，修改后不阻断断点续跑。原样本保持不变；每次构建的实际模型和地址记录在 `provider_runs`，新样本和调用记录通过 `provider_profile_id` 关联来源。更换模型后数据会包含多个模型来源。

**下面的批量命令由你手动执行。构建会调用付费模型。**

```bash
# 每语言100条，共300条试跑；成功数据保存在训练集的固定配额中
uv run watch-intents build --stage pilot --resume
uv run watch-intents verify --stage pilot

# 只有试跑完整且 verify 成功，才允许补齐总计13,500条
uv run watch-intents build --stage full --resume
uv run watch-intents verify --stage full
uv run watch-intents report --stage full
```

`build` 默认持续补齐所有合格配额：不再受每槽位3次上限限制，不一致、不合格、重复或格式错误的候选仍严格淘汰。临时网络故障、429、5xx 在有限单次重试后继续等待15/30/60秒重试；审核临时中断时保留当前内存候选重试审核，避免重新生成。连续多批没有增长时等待60秒并继续，质量标准不变。持续模式没有总调用次数或费用上限；永久鉴权、模型、参数、版本或文件错误仍会退出，需要解决原因后恢复。

按 Ctrl+C 可停止，退出后重复同一条 `build ... --resume` 即可续跑。已完成且审核通过的样本不会重复调用模型；进程退出时尚未落盘的候选可能需要重新生成。执行中只打印进度和脱敏错误码，不打印原始文本或密钥。

在 WSL Bash 终端粘贴下面整段，追加日志并自动依次完成试跑、验证、正式生成、验证和报告。`&&` 确保当前阶段成功后才进入下一阶段；终端需保持运行，系统休眠/关机会暂停或中断任务。

```bash
(
  export PYTHONUNBUFFERED=1
  printf '\n===== %s =====\n' "$(date -Is)"
  uv run watch-intents build --stage pilot --resume &&
  uv run watch-intents verify --stage pilot &&
  uv run watch-intents build --stage full --resume &&
  uv run watch-intents verify --stage full &&
  uv run watch-intents report --stage full
) >> build.log.txt 2>&1
```

默认使用 **2 个生成工作线程＋2 个审核工作线程**，每批仍为10条。配额分配、候选校验、独立审核结果处理、最终跨批次去重和SQLite写入只在主线程执行。所有在途/待审批次合计最多4个，避免审核变慢时无限生成；只有候选通过原有全部检查才能入库。模型、提示词、配额和质量阈值均保持原样。

并发与速率统一在 `concurrent.env` 管理，模型与Key仍放在 `dashscope.env`。运行配置优先级为命令行并发参数 > `concurrent.env` > 终端环境变量 > 内置默认值；不再从 `dashscope.env` 读取运行参数。示例见 [concurrent.env.example](concurrent.env.example)：

```dotenv
WATCH_INTENTS_GENERATOR_WORKERS=2
WATCH_INTENTS_JUDGE_WORKERS=2
WATCH_INTENTS_GENERATOR_RPM=6
WATCH_INTENTS_JUDGE_RPM=6
WATCH_INTENTS_GENERATOR_TPM=120000
WATCH_INTENTS_JUDGE_TPM=120000
```

这些是本地请求上限，不代表账号实际获批额度。按接口地址和模型共享滑动60秒窗口；生成和审核若使用同一模型，即使Key不同也共用限速器，采用两侧更低的RPM/TPM上限。发请求前按提示词UTF-8字节数、响应上限及消息开销保守预留token预算，收到实际用量后回补差额；未知用量不回补。服务端计量仍以账单和429反馈为准。

收到429时将该模型允许的在途请求数减半（最低1），等待后重试；不影响另一个模型的并发。冷却期结束且累计20次成功后逐步恢复到配置上限。不会自动切换模型或降低审核标准。运行参数记录在报告的 `execution_runs`，不进入数据版本指纹；只改并发/速率不需要迁移。

稳定后，可停机并把`concurrent.env` 中两个 `WORKERS` 值改成3，再执行原来的命令。也可以仅对单次命令覆盖：

```bash
uv run watch-intents build --stage full --resume --generator-workers 3 --judge-workers 3
```

按一次 Ctrl+C 后会停止新请求，取消尚未发出的请求，等待在途调用完成或超时，并保存已经完成审核且合格的结果；请等待退出再续跑。未完成审核的正文不落盘，恢复后可能重新生成并产生费用。

在另一终端用 `tail -n 50 -f build.log.txt` 实时查看。若要使用原来的有界行为，给单条构建命令加 `--bounded`，它仍遵循 `semantic_retries` 并保留累计次数。

已诊断的旧版检查点可离线执行一次 `uv run watch-intents upgrade`。它支持已知的数字标签修复前版本、串行v0.2版本、运行配置拆分前的并发版本及连接配置与版本分离前的版本，要求标签、数据配置及资产完全一致，检查原样本和导出文件后备份SQLite、记录迁移历史并刷新导出指纹。不会改动原样本文本、标签、配额、审核或调用计数，也不会发起模型调用。任意其他版本变化仍拒绝混用；完整冻结数据不允许原地迁移。迁移后先执行 `uv run watch-intents verify --stage pilot` 恢复试跑验证门槛，再继续正式生成。

需要另做小规模接口验证时，可执行以下**独立付费命令**；同一输出目录只允许运行一次，生成最多9个候选，没有网络重试和语义补生成，与训练数据完全隔离。交付时的实际验证记录见 [验证记录](docs/validation.md)，无需重复执行已完成的冒烟。

```bash
uv run watch-intents smoke
uv run watch-intents verify --stage smoke
```

退出码：`0` 完成/通过；`2` 配额未补齐或验证失败；`1` 配置、接口或状态错误；`130` 用户中断。只有构建和验证均成功，才能把对应版本视为可交付数据。

## 配额与生成规则

每语言的数据数量如下；语言为 `en`、`zh-Hans`、`th`：

| 划分 | 总量 | 单标签60% | 多标签20% | 零标签20% | 每类单标签 |
|---|---:|---:|---:|---:|---:|
| train | 3,000 | 1,800 | 600 | 600 | 72 |
| validation | 500 | 300 | 100 | 100 | 12 |
| test | 1,000 | 600 | 200 | 200 | 24 |

多标签包含两个或三个有效类型，一条文本只计一条数据。零标签默认一半为纯闲聊（占整个数据10%），另一半为否定、撤回、引用、过去叙述、咨询、对象不明及参数型不支持请求。细分权重可以在 [dataset.toml](configs/dataset.toml) 修改，和为100；这不是线上流量估计。

生成前固定 split、语义根、表达家族和目标标签。三种语言各自原生生成，不从其他集合翻译或扩写。试跑是训练计划的固定前缀，计入13,500条总量。验证及测试使用独立表达家族；测试多标签中的20%（每语言40条）属于 `unseen_combination` 挑战切片，其标签组合未出现在训练计划中；报告中单列场景。

审核模型只看到完整文本、语言和标签规范，不看到生成标签、目标标签数、场景或 split。结构正确、标签与计划及独立判别一致、语义明确且语言质量合格才能入库。缺乏可确定对象的“打开它”按规则为 `[]`；存在多个同样合理的目标解释才作为真正歧义淘汰。

**不一致、不确定、不合格或重复的候选直接淘汰，正文不写入数据库、缓存、审核日志或最终文件。** 仅保留本地分配的编号、场景、拒绝原因代码等元数据。生成响应和待判别文本只短暂保留在内存中；如果判别前中断，会重新生成未完成候选，可能增加费用。不会自动改标签、清空标签或保存拒绝语料。

## 恢复、版本和费用边界

- 默认持续补生成，已有历史尝试数继续累计；已达到旧版3次上限的缺口在迁移后也能继续补。不会自动改标签、降级模型、清空标签或放宽审核。
- `--bounded` 模式仍按配置每槽位最多3次（首次＋最多2次补生成），达到上限返回2；两种模式都不重置历史计数。HTTP超时可能已在服务端执行，用量报告只汇总服务端返回的 token，不能当作完整账单。
- 标签、行为配置、提示词、实现代码、依赖锁和本地模型/tokenizer资产纳入一致性检查，变化后拒绝直接续跑。Key 不进入指纹或审计输出；远程模型名和 BaseURL 只作来源记录，不阻断续跑。并发与限速参数同样不阻断续跑。原始版本清单保留首次使用的连接信息，后续变动见 `provider_runs`。
- `verify --stage full` 成功后冻结完整数据签名；后续发现缺失或修改会报错，不会重新生成封存测试数据。最终测试只用于候选/阈值冻结后的评估，不能据其结果反复补训练数据。
- 同一输出目录一次只允许一个构建、验证或报告命令，避免并发覆盖；并发发生在同一构建进程内部的接口调用阶段。

使用其他配置时，把全局参数放在子命令之前：

```bash
uv run watch-intents --config configs/dataset.toml build --stage pilot --resume
```

## 输出与后续训练接口

默认输出根目录为 `data/generated`，已被 Git 忽略：

```text
data/generated/
  dataset/state.sqlite3              # 已通过样本、调用用量、无正文拒绝记录、恢复状态
  dataset/pilot-slots.json           # 调用前固定的来源与配额计划
  dataset/full-slots.json
  dataset/exports/pilot/
  dataset/exports/full/
    train.jsonl                     # 三语言合并
    validation.jsonl
    test.jsonl
    en/{train,validation,test}.jsonl # zh-Hans、th 同结构
    labels.json
    label2id.json
    id2label.json
    rejections.jsonl                # 只有元数据，不含被拒绝文本
    manifest.json                   # 数据版本、来源、指纹、数量及文件SHA256
    verification.json               # verify命令产生
    report.json                     # 配额、标签、场景、组合、长度及用量
  smoke/                            # 独立9条接口测试，不并入任何正式划分
```

记录核心字段为 `sample_id/text/labels/locale/group_id/split/source/label_version/review_status`，并附 `family_id/kind/scenario/n_tokens`、生成/判别模型及尝试次数。`labels` 按固定编号排列，空集合是 JSON 数组 `[]`。`source=llm_synthetic`，`review_status=machine_accepted`。只将完整 `text` 和标签的25维浮点 multi-hot 用于后续训练，其他字段用于治理和分层评估。

tokenizer 使用本地 `tokenizer.json`，包含特殊 token，无截断计数；超过512 token淘汰补齐。报告给出各语言 P50/P95/P99 和超过128/256 token的数量，后续据此选择训练长度。

去重只在比较时做 NFC 和空白规范化，保存的原文不变。完全重复全局检查；近重复使用同语言、同最终标签集合且双方至少20字符的词面相似度，默认阈值0.90。不同标签的否定对照不会被近重复规则误删。这是保守的自动排查，不保证检测所有语义改写、短句变体或跨语言翻译；不要把它表述为真实独立盲测。

当前数据和评估均为全合成，模型之间的一致不等于人工正确率；不据此宣布生产验收通过。泰语自然度仍需后续有该语言能力的人复核。GPU训练与阈值选择见独立训练工程；线上部署另行建设。

## 开发验证

```bash
uv run pytest -q
uv run ruff check src tests
```

离线测试不调用付费API；使用真实本地 tokenizer 加上模拟生成/判别响应，验证流程与边界。它们不证明生成模型具有业务语义准确率。
