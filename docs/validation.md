# 实施验证记录

验证日期：2026-09-29。执行环境：WSL Ubuntu，uv，Python 3.12.13；依赖固定在仓库 `uv.lock`。

## 已完成

| 检查 | 结果 |
|---|---|
| `uv sync --frozen --extra dev` | 成功，锁定环境可同步 |
| `uv run watch-intents check` | 成功；25标签，三语言，试跑计划300条，正式计划13,500条；未调用API |
| `uv run pytest -q` | **120 passed** |
| `uv run ruff check src tests` | **All checks passed** |
| `uv run watch-intents smoke` | **9/9合格**，无补生成 |
| `uv run watch-intents verify --stage smoke` | **通过，0项错误** |
| `uv run watch-intents report --stage smoke` | 成功，输出分布、长度及用量报告 |

真实API调用严格限定在9条冒烟：英语、简中、泰语各一条单标签、一条多标签和一条零标签。共1次 `glm-5.2` 生成请求和1次 `qwen3.8-max` 盲审请求，没有网络重试。服务端报告输入11,445 token、输出2,873 token，共14,318 token；这里不估算费用。

9条完整文本使用本地 XLM-R tokenizer 编码，长度14–24 token（包含特殊token），没有截断。所有候选经独立标签一致性、质量、结构、去重及分组检查后入库；这只验证接口与数据流程，不能推导真实业务准确率。

冒烟数据、机器审核标记、版本指纹与实际验证结果位于：

```text
data/generated/smoke/exports/smoke/smoke.jsonl
data/generated/smoke/exports/smoke/manifest.json
data/generated/smoke/exports/smoke/verification.json
data/generated/smoke/exports/smoke/report.json
```

## 覆盖的故障与边界

- 固定编号、全量配额、单标签均衡、试跑与训练计划一致、三语言原生种子及组合挑战切片。
- 用固定判别夹具验证否定、撤回、改口、部分撤回、参数型不支持请求、完整标签集合和原文保留；不把离线夹具测试当作LLM语义能力评测。
- Qwen输入不含生成目标；非法JSON、漏返回、重复/未知ID、非法布尔及理由字段均不能入库。
- 认证/参数失败立即停止；限流、超时和5xx有限重试；不输出错误响应正文或密钥。
- 拒绝文本在数据库、导出、缓存和日志中均不存在；只保存受控原因代码和来源元数据。
- 中断后恢复已通过数据；判别服务失败也不会重置生成尝试计数；达到上限保留缺口并报告。
- 本地真实tokenizer无截断计数、Unicode原文保护、精确/保守近重复和跨集合分组检查。
- 导出内容与检查点及哈希比对；试跑验证门槛；正式数据冻结后缺行不能通过恢复重新生成。

独立代码检查发现的报告并发写入、判别失败恢复绕过次数限制、已知重复样本仍送付费判别问题均已修复。

## 留给用户执行

**没有执行300条试跑或13,500条正式构建，没有训练或修改模型权重。**

按README依次运行 `build --stage pilot --resume`、`verify --stage pilot`，再运行正式构建、验证和报告命令。凭证、虚拟环境及生成输出均已被Git忽略；工作区改动未提交、未推送。
