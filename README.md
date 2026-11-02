# PokeAgent — 开放式 LLM 宝可梦对战 Agent 框架

让 LLM 像人一样打宝可梦对战：每回合经过「推理 → 并行工具调用 → 再推理 → 终端工具提交决策」的完整 agentic 循环，可以查伤害计算、翻知识库、看对战历史，然后才决定出招。对战跑在**本地** [pokemon-showdown](https://github.com/smogon/pokemon-showdown) 服务器上（不连外部网站、无真人对手），通过 [poke-env](https://github.com/harrit-labs/poke-env) 与服务器通信。

灵感来自 [HF Agents Course Bonus Unit 3](https://huggingface.co/learn/agents-course/bonus-unit3/building_your_pokemon_agent)，但走的是完整 agentic 架构而非教程的"单次 LLM 调用出一招"模式。

## 特性

- **完整 agentic 决策循环**：每回合模型先思考，再并行调用领域工具（伤害计算、招式/种族值/克制知识查询、对战历史查询），最后必须通过 `submit_decision` 终端工具提交决策——框架对照当前合法动作校验，非法提交会带着错误信息重试。
- **两阶段决策**：队伍预览阶段让模型选首发（`submit_lead`），对战内逐回合出招（`submit_decision`），同一套循环骨架。
- **永不超时判负**：每个决策有硬时间预算，超时或出错自动回退随机出招——质量降级，但绝不因超时输掉对战。
- **全程落盘，可复现**：每场对战自动留下完整记录——逐回合的决策与理由、模型的逐字思考过程（CoT）、以及模型看到的**完整对话轨迹**（`llm.jsonl`，prompt/响应/工具结果逐字复现）。
- **可插拔后端**：任何 OpenAI 兼容端点（vLLM、Ollama、各类网关、官方 API）或 Anthropic；Provider / Tool / Hook / SubAgent / Compressor 全部是抽象接口，实现 + 注册即可扩展。
- **速度与成本设计**：决策关键路径零子 agent；威胁分析、战后复盘、历史压缩全部跑在后台的便宜模型上，摘要注入后续决策，不阻塞对战。

## 环境要求

- Python ≥ 3.10
- Node.js（运行本地 Showdown 服务器）
- 一个 LLM API：OpenAI 兼容端点或 Anthropic

## 安装

```bash
git clone https://github.com/Reshiramss/Pokemon_battle_agent.git
cd Pokemon_battle_agent
pip install -e ".[dev]"
```

### 导出 Showdown 静态数据（move_info / type_matchup / species_info 的主数据源）

```bash
node scripts/export_showdown_data.js /path/to/pokemon-showdown/dist/data pokeagent/data/showdown
```

从 pokemon-showdown 的编译产物导出 7 张表（招式/图鉴/属性表/别名/分级/文本）为 JSON。仓库里已带导出好的副本；升级 pokemon-showdown 后重跑一次即可。缺了它这三个工具自动回退到 poke-env 的 GenData，功能不中断。

## 快速开始

### 1. 启动本地 Showdown 服务器

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
node pokemon-showdown start --no-security --skip-build 8964
```

端口任意（默认配置用 8964），与下一步 `config.toml` 里的 `server_port` 保持一致即可。框架会给所有玩家（包括对手 bot）统一传入正确的服务器配置。

### 2. 配置 LLM

复制示例配置为 `config.toml` 并编辑（`config.toml` 在 .gitignore 中，私有端点和密钥不会被提交）：

```toml
[agent]
provider = "openai"                           # openai 或 anthropic
model = "your-model-name"
base_url = "https://api.example.com/v1"        # OpenAI 兼容端点；不设则读 OPENAI_BASE_URL
# api_key = "sk-..."                           # 不设则读 OPENAI_API_KEY 环境变量
server_port = 8964                             # 与你的 Showdown 服务器一致
```

用 Anthropic 的话把 `provider` 改为 `"anthropic"`，并设置环境变量：

```bash
export ANTHROPIC_API_KEY=<你的key>
export ANTHROPIC_BASE_URL=https://<网关地址>   # 可选，接内部网关时用
```

### 3. 跑对战 + 看报告

```bash
python scripts/run_battle.py --opponent heuristic --battles 1
#   --opponent random|heuristic    对手：随机 bot / 启发式 bot
#   --battles N                    连打 N 场

python scripts/postmortem.py            # 汇总 runs/ 下的所有对战
python scripts/postmortem.py --last 3   # 只看最近 3 场
```

### 4. 人机对战（人类 vs AI）——已写好、尚未实测

```bash
python scripts/human_battle.py --team "Shadow Bolt"
#   --team NAME        从 teams/ 库选队（或直接传 packed 队伍串）
#   --count N          连打 N 场后退出
#   --wait-for NAME    只接受来自该用户的挑战（默认接受任何人）
```

脚本启动后 agent 以 `PokeAgent` 用户名挂机等待挑战。用浏览器打开本地服务器
（`http://localhost:8964`），任选用户名登录，然后向 `PokeAgent` 发起
**[Gen 9] VGC 2025 Reg I (Hidden)** 格式的挑战（挑战下拉框里能选到），即可开打。
agent 侧的选 4/首发/出招流程与 AI vs AI 完全一致；出问题时看 `runs/logs/`
下当场的 `events.jsonl` / `llm.jsonl`。

> ⚠️ 此路径尚未经过实测，属实验性功能；首次试玩如遇挂起或非法指令，请带着
> 记录目录反馈。

运行结束后会打印胜负与记录目录：

```
Finished battles: 1/1
Agent wins: 1
Runs logged under: /…/Pokemon_battle_agent/runs
```

postmortem 每场输出一行，例如：

```
20261029T020119Z_battle-gen9randombattle-4: WIN | turns 26 | fallbacks 0 (0%)
  | CoT logged 26/26 | latency mean 1.2s p95 3.4s | tokens in 51k out 8k
  | sub-agents 7 | compressions 1
```

## 每场对战会得到什么

`runs/<时间戳>_<battle_tag>/` 下：

| 文件 | 内容 |
|---|---|
| `meta.json` | 该场运行的配置快照（模型/端点/参数，api_key 脱敏） |
| `events.jsonl` | 逐回合结构化记录：模型看到的局面、决策、理由、逐字 CoT、工具调用、token 与耗时，外加生命周期事件（预览、回退、对战结束） |
| `llm.jsonl` | 完整 LLM 对话轨迹：system prompt、每回合组装的 prompt、每轮响应、每次工具结果——逐字可复现（`log_llm_conversation = false` 可关） |

进程级日志在 `runs/logs/agent-<时间戳>.log`，框架全部 logging（provider 报错、hook 失败、子 agent 异常）都会落盘。

## 配置

`config.toml`（从 `config.example.toml` 复制）中的 `[agent]` 节可覆盖全部默认值，常用项：

| 配置项 | 默认 | 说明 |
|---|---|---|
| `provider` | `openai` | LLM 后端（`openai` / `anthropic`） |
| `model` | 见 `config.py` | 主决策模型 |
| `base_url` / `api_key` | 空 | OpenAI 兼容端点与密钥；不设时依次读 `OPENAI_BASE_URL` / `OPENAI_API_KEY` |
| `battle_format` | `gen9randombattle` | 对战格式（改为 `gen9ou` 等预览格式会启用阶段1 选首发） |
| `server_host` / `server_port` | `localhost` / `8964` | 本地 Showdown 服务器地址 |
| `turn_budget` / `preview_budget` | 45s / 20s | 两阶段决策的硬预算，超时回退 |
| `runs_dir` | `runs` | 对战记录输出目录 |

完整配置项与默认值见 `pokeagent/config.py`。

## 测试

```bash
python -m pytest tests/ -q                                          # 全部（e2e 需本地服务器在跑）
python -m pytest tests/ -q --ignore=tests/test_e2e_loop.py          # 纯离线
```

其中 e2e 测试是**零 token 真实对战**：用 stub provider 跑完整协议、落盘、复盘全链路。

## 架构一览

```
Showdown 服务器 ──▶ PokeAgent（poke-env Player 子类）
                     ├─ 阶段1 队伍预览: AgentLoop.run_preview
                     │     hook 注入知识 → submit_lead 终端工具选首发
                     │     失败/超时 → 回退基类随机首发
                     └─ 阶段2 对战决策: AgentLoop.run_turn（每回合）
                          ① 决策前 hook 并行注入知识（克制矩阵、使用率等）
                          ② 组装上下文 prompt（情报/知识/历史/当前局面/合法动作）
                          ③ 工具循环：并行调用伤害计算、知识查询、历史查询
                          ④ submit_decision 终端工具出招（校验合法性）
                          失败/超时 → 随机出招兜底，永不判负
                     后台（不阻塞决策，跑便宜模型）：
                       新宝可梦威胁分析 · 战后复盘 · 历史压缩
                       → 摘要注入后续回合的 prompt
```

扩展方式都是"实现接口 + 注册"两步：

- **新 LLM 后端**：实现 `LLMProvider.complete()`（`providers/base.py`），在 `get_provider` 工厂注册。
- **新工具**：继承 `Tool`（`tools/base.py`），`registry.register(...)` 挂载。
- **新 hook**：继承 `PreDecisionHook` / `PostDecisionHook`（`hooks/base.py`），返回一段知识片段即可。
- **新子 agent**：继承 `SubAgent`（`tasks/base.py`），`TaskRunner.register()` 后任意位置 fire-and-forget。
- **新压缩策略**：实现 `ContextCompressor` 协议（`context/compressor.py`）。
