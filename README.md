# Pokemon Agent — 开放式 LLM 对战 Agent 框架

一个 **Claude Code 式**的宝可梦对战 agent 框架：每回合经过「推理 → 并行工具调用 → 再推理 → 终端工具提交决策」的完整 agentic 循环，配决策前知识注入 hook、function call 工具、后台子 agent、上下文压缩与逐字 CoT 落盘。

对战跑在**本地** [pokemon-showdown](https://github.com/smogon/pokemon-showdown) 服务器上（不连外部网站、无真人对手），通过 [poke-env](https://github.com/harrit-labs/poke-env) 与服务器通信。灵感来自 [HF Agents Course Bonus Unit 3](https://huggingface.co/learn/agents-course/bonus-unit3/building_your_pokemon_agent)，但走的是完整 agentic 架构而非教程的"单次 LLM 调用出一招"模式。

## 架构总览

```
                    ┌────────────────────────── 关键路径（零子 agent） ──────────────────────────┐
Showdown 服务器 ───▶ PokeAgent (pokeagent/player.py, poke-env Player 子类)
  :8964              │  choose_move(battle)
                    │    └─ AgentLoop.run_turn (loop.py)
                    │         ① 新对手宝可梦检测 ──fire-and-forget──▶ TaskRunner.submit(NewMonThreatAgent)
                    │         ② hooks.run_pre ── 并行注入知识片段（5s 时间盒）
                    │         ③ ContextManager.assemble_prompt ── 组装 [INTEL/KNOWLEDGE/HISTORY/RECENT/NOW]
                    │         ④ 工具循环（≤6 轮，单轮多调用并行执行，45s 硬预算）
                    │         ⑤ submit_decision（终端工具）→ 校验合法 → 出招
                    │         失败/超时 → 随机出招兜底，永不判负
                    └────────────────────────────────────────────────────────────────────────┘
                                        │ 后台路径（便宜模型，全部离关键路径）
                    TaskRunner (Semaphore=3, 120s 超时, 按对战取消)
                      ├─ NewMonThreatAgent   对方新登场宝可梦的威胁评估
                      ├─ TeamPreviewAnalysisAgent   队伍预览分析（随机对战无预览时自然不触发）
                      ├─ PostBattleReviewAgent   战后复盘
                      └─ LLMCompressor   历史摘要压缩（ContextManager 内触发）
                              └─ 摘要写入 IntelStore（限长）→ 注入后续每回合 prompt
```

- **Provider / Tool / Hook / SubAgent / Compressor 全部是抽象接口 + 默认实现**，可独立替换扩展
- 每场对战落盘 `runs/<时间戳>_<battle_tag>/`：`meta.json` + 逐回合 `events.jsonl`（含**逐字 CoT**）
- 所有 LLM 消息走归一化格式（`UserMessage / AssistantMessage / ToolResultMessage`），provider 负责翻译成各家 API 格式

## 快速开始

### 1. 环境

```bash
cd pokemon-agent
# venv 已就绪；重装的话：
python -m venv .venv
env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL .venv/bin/pip install -i https://pypi.org/simple \
    -e ../poke-env anthropic pytest
```

（本机 local pip 镜像缺包时用 `-i https://pypi.org/simple`。）

### 2. 启动本地 Showdown 服务器（8964 端口）

```bash
cd pokemon-showdown   # 独立目录，别在里面写脚本
node pokemon-showdown start --no-security --skip-build 8964
```

> 端口说明：poke-env 默认配置硬编码 8000，那是这台共享服务器上**其他用户的服务**。本框架默认用 8964（见 `config.toml` 的 `server_port`），框架代码会给**所有**玩家（含对手 bot）统一传正确的服务器配置。

### 3. 配置 LLM 凭据

```bash
export ANTHROPIC_API_KEY=<你的key>
export ANTHROPIC_BASE_URL=https://<网关地址>   # 可选，接内部网关时用
```

### 4. 跑对战 + 看报告

```bash
.venv/bin/python scripts/run_battle.py --opponent heuristic --battles 1
#   --opponent random|heuristic （RandomPlayer / SimpleHeuristicsPlayer）

.venv/bin/python scripts/postmortem.py            # 汇总 runs/ 下的所有对战
.venv/bin/python scripts/postmortem.py --last 3   # 只看最近 3 场
```

postmortem 每场输出一行，例如：

```
20261029T020119Z_battle-gen9randombattle-4: WIN | turns 26 | fallbacks 0 (0%)
  | CoT logged 26/26 | latency mean 1.2s p95 3.4s | tokens in 51k out 8k
  | sub-agents 7 | compressions 1
```

### 5. 测试（39 个，无需 API key）

```bash
.venv/bin/python -m pytest tests/ -q              # 全部（e2e 需本地服务器在跑）
.venv/bin/python -m pytest tests/ -q --ignore=tests/test_e2e_loop.py   # 纯离线
```

- `test_tools / test_hooks / test_tasks / test_context`：38 个单测，stub provider 驱动，~5s
- `test_e2e_loop.py`：**零 token 真实对战**——`DecisionStubProvider` 解析 `[NOW]` 段合法动作直接提交，跑完整协议、落盘、postmortem 全链路（需 8964 服务器）

## 配置（`config.toml`，可覆盖 `AgentConfig` 全部默认值）

| 配置项 | 默认 | 说明 |
|---|---|---|
| `provider` | `anthropic` | provider 注册表键名 |
| `model` | `claude-sonnet-4-5` | 主决策模型 |
| `compressor_model` | `claude-haiku-4-5` | 压缩/复盘等后台便宜模型（`subagent_model` 不设时也用它） |
| `battle_format` | `gen9randombattle` | 对战格式 |
| `server_host` / `server_port` | `localhost` / `8964` | 本地 Showdown 服务器地址 |
| `turn_budget` | 45.0s | 整回合硬预算（`asyncio.wait_for`），超时随机兜底 |
| `hook_budget` | 5.0s | 决策前 hook 共享时间盒（逐 hook 分片，慢 hook 不拖累其他） |
| `tool_call_timeout` | 8.0s | 单个工具调用超时 |
| `max_tool_rounds` | 6 | 每回合工具循环轮数上限 |
| `recent_turns_verbatim` | 6 | prompt 中逐字保留的最近回合数（更早的压缩为一行） |
| `context_token_soft_cap` | 12k | 组装后 prompt 软上限（chars/4 估算） |
| `compression_trigger_tokens` | 8k | 未压缩历史超过该值则后台触发摘要 |
| `max_concurrent_subagents` | 3 | 后台子 agent 全局并发上限 |
| `subagent_timeout` | 120s | 单个子 agent 超时 |
| `max_threat_analyses_per_battle` | 6 | 每场威胁分析次数上限 |
| `enable_smogon_hook` | false | Smogon 使用率注入（随机对战无意义，默认休眠；标准排位格式可开） |
| `runs_dir` | `runs` | 对战记录输出目录 |

## 扩展指南

所有扩展点都是"实现接口 + 注册"两步：

- **新 LLM 后端**：实现 `LLMProvider.complete()`（`providers/base.py`），在 `get_provider` 工厂注册。`openai_provider.py` 留了占位。
- **新工具**：继承 `Tool`（`tools/base.py`），给 `name/description/input_schema` + `execute`；`PokeAgent._default_tools()` 或运行时 `registry.register(...)` 挂载。
- **新 hook**：继承 `PreDecisionHook` / `PostDecisionHook`（`hooks/base.py`），返回一段 markdown 知识片段即可，超时/异常自动吞掉。
- **新子 agent**：继承 `SubAgent`（`tasks/base.py`），隔离上下文里跑自己的短对话，返回摘要字符串；`TaskRunner.register()` 后在任意位置 `submit()` fire-and-forget。
- **新压缩策略**：实现 `ContextCompressor` 协议（`context/compressor.py`），赋给 `cm.compressor`；内置 `TruncatingCompressor`（无 LLM 兜底）和 `LLMCompressor`。

## 目录结构

```
pokemon-agent/
├── config.toml               # 运行配置（覆盖 AgentConfig 默认值）
├── pokeagent/
│   ├── config.py             # AgentConfig dataclass + TOML 加载
│   ├── player.py             # PokeAgent：唯一与 poke-env 耦合的入口
│   ├── loop.py               # AgentLoop：回合工具循环 + submit_decision 终端工具
│   ├── providers/            # LLM 抽象 + Anthropic 实现
│   ├── tools/                # 工具抽象/注册表 + 伤害计算/静态KB/历史查询
│   ├── hooks/                # hook 系统 + 克制矩阵注入 + Smogon（休眠）
│   ├── tasks/                # 子 agent 抽象/TaskRunner + 威胁分析/队预/复盘
│   ├── context/              # 上下文管理：prompt 组装、JSONL 落盘、后台压缩
│   └── knowledge/static.py   # StaticKB：GenData 查询、对手能力值估算
├── scripts/
│   ├── run_battle.py         # CLI 跑对战
│   ├── postmortem.py         # runs/ 汇总指标
│   ├── smoke_env.py          # poke-env 连通性冒烟（RandomPlayer vs RandomPlayer）
│   └── diag_connect.py       # 诊断各玩家 websocket 登录状态
├── tests/                    # 39 个测试：38 单测 + 1 零 token e2e
└── runs/                     # 运行时生成：每场 <时间戳>_<battle_tag>/
```

详细代码说明见 [docs/architecture.md](docs/architecture.md)。

## 设计要点

- **终端工具制**：出招必须经 `submit_decision`，对照当前合法动作校验；非法提交返回错误 tool_result 让模型在剩余轮数内重试；轮数/时间耗尽抛 `TurnAbort` → 随机出招兜底，**永不因超时判负**。
- **速度设计**：关键路径零子 agent（靠单轮并行工具调用 + 预计算知识注入提速）；威胁分析、队预、复盘、压缩全部后台跑便宜模型，摘要经 `IntelStore`（1200 字符截断）注入 prompt，原始研究内容不污染主上下文。
- **磁盘即真相**：每回合 `TurnRecord`（模型看到的请求、决策、理由、逐字 CoT、工具调用、token、耗时）追加写 `events.jsonl`；压缩只影响 prompt 组装，落盘记录永不丢失。
- **共享服务器友好**：子 agent 全局并发上限、每回合 token 上限、所有后台调用有超时；Showdown 服务端跑在独立目录。
