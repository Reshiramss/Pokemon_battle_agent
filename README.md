# PokeAgent — 开放式 LLM 宝可梦对战 Agent

让 LLM 像人一样打宝可梦对战：它会查伤害、翻招式与特性资料、看对战历史和以往比赛记忆，再决定这一回合怎么打。对战跑在**本地** [pokemon-showdown](https://github.com/smogon/pokemon-showdown) 服务器上（不连外部网站、无真人对手），通过 [poke-env](https://github.com/harrit-labs/poke-env) 与服务器通信。

灵感来自 [HF Agents Course Bonus Unit 3](https://huggingface.co/learn/agents-course/bonus-unit3/building_your_pokemon_agent)，但走的不是“单次 LLM 调用出一招”，而是每回合完整的推理与决策流程。

## 它能做什么

- **像人一样查资料再出招**：每回合可以调用伤害计算、招式/种族值/特性/属性克制查询、对战历史检索、跨场次记忆、对局机制百科、对方配队流派识别等工具；最终决策会对照当前合法动作校验，非法选择会被打回并要求重新提交。
- **完整的 VGC 双打 6 选 4**：选队阶段从 6 只中选 4 只并指定首发；对战内每回合为两个槽位分别出招，支持 `opp1/opp2/ally/self` 目标、Tera/Mega 等机制。对手在选队时只暴露 6 只种类，招式/道具/特性随对战逐步揭示。
- **决策前自动注入知识**：每回合自动带上属性克制、已确认特性的效果说明、当前可用招式与对方已揭示招式的效果摘要、双打核心机制、天气/场地提示、对方使用率数据等，不需要模型每次自己查。
- **战役级策略主线**：每回合结束后，后台用一个子任务（主模型）结合本回合的思考、当时的局面与赛后结果，重写一份“这局怎么赢”的策略文本；下一次决策前回放给模型，并随 `state.json` 落盘，可复盘策略如何演变。
- **跨场次记忆**：每场结束把对手的招式/道具/特性/首发固化成不可变原子，合并成宝可梦、阵容与流派三类档案；预览时注入体系与个体先验，对局中按新揭示增量召回，模型也可以 `memory_grep` / `memory_read` 下钻。规则与赛季分区（Champions / 标准分开），冷启动由 curated 招式池、体系种子与队伍库兜底。
- **永不因超时判负**：每次决策有固定时间预算，超时或调用失败会自动回退为随机出招——质量会下降，但不会把胜利直接送给对手。
- **全程可复现**：每场对战都会保存逐回合局面、决策、理由、模型的完整思考过程，以及模型看到的完整对话轨迹；事后可以离线复盘“当时为什么这么打”。
- **两种规则模式**：常规 Gen 9 与 Pokémon Showdown 的 `champions` mod 共用同一套 agent，自动切换数值规则、可用机制（Tera/Mega）、PP 上限与数据表。
- **任意 LLM 后端**：OpenAI 兼容端点（vLLM、Ollama、各类网关、官方 API）或 Anthropic 均可。

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

### 可选：更新 Showdown 静态数据

仓库已经带了从 pokemon-showdown 导出的招式/图鉴/属性/特性/道具等数据，日常使用无需操作。升级 pokemon-showdown 后如需同步数据：

```bash
node scripts/export_showdown_data.js /path/to/pokemon-showdown/dist/data pokeagent/data/showdown
```

### 可选：Smogon 使用率快照

在有外网的机器上一次性下载真实天梯（如 `gen9vgc2025regi`）的使用率数据，之后即可完全离线使用：

```bash
python scripts/fetch_usage_snapshot.py            # 下载到 pokeagent/data/usage/
# 按脚本输出把 smogon_stats_snapshot 写进 config.toml
```

设置后，agent 会为对方已揭示的宝可梦注入使用率、常见招式、道具和特性。没有快照时会静默跳过，不影响对战。

## 快速开始

### 1. 启动本地 Showdown 服务器

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
# 安装自定义对战格式（一次性；config/custom-formats.ts 是官方扩展点，不改上游代码）
cp /path/to/Pokemon_battle_agent/server/custom-formats.ts config/custom-formats.ts
node build                                    # 一次性编译（秒级）
node pokemon-showdown start --no-security --skip-build 8964
```

端口可任意，默认配置使用 8964，与下一步 `config.toml` 中的 `server_port` 保持一致即可。

默认自定义格式 `gen9championsvgc2026regmchidden`（[Gen 9 Champions] VGC 2026 Reg M-C (Hidden)）是 Champions 规则：双打、Stat Points、无 Tera、有 Mega、6 选 4 + 选首发，双方配置在选队时保持隐藏。标准 Gen 9 仍可用 `--format gen9vgc2025regihidden`（官方 Reg I 规则，双打 6 选 4、限两只受限神）。

### 2. 配置 LLM

复制示例配置为 `config.toml` 并编辑（`config.toml` 已在 .gitignore 中，私有端点与密钥不会被提交）：

```toml
[agent]
provider = "openai"                           # openai 或 anthropic
model = "your-model-name"
base_url = "https://api.example.com/v1"        # OpenAI 兼容端点；不设则读 OPENAI_BASE_URL
# api_key = "sk-..."                           # 不设则读 OPENAI_API_KEY 环境变量
server_port = 8964                             # 与你的 Showdown 服务器一致
```

使用 Anthropic 时把 `provider` 改为 `"anthropic"`，并设置环境变量：

```bash
export ANTHROPIC_API_KEY=<你的key>
export ANTHROPIC_BASE_URL=https://<网关地址>   # 可选，接内部网关时用
```

### 3. 跑对战并查看报告

```bash
python scripts/run_battle.py --opponent heuristic --battles 1 --team "Champions A"
#   --opponent random|heuristic|agent  对手：随机 bot / 启发式 bot / 第二个 PokeAgent（AI vs AI）
#   --battles N                        连打 N 场
#   --team NAME                        我方队伍（teams/ 库名或 packed 队伍串）；预览格式必填
#   --opponent-team NAME               对手队伍；缺省时自动选库里另一套

python scripts/postmortem.py            # 汇总 runs/ 下的所有对战
python scripts/postmortem.py --last 3   # 只看最近 3 场
```

`--opponent agent` 是 AI vs AI：对手是第二个 PokeAgent，使用同样的模型配置、独立账号和另一套队伍，后台分析关闭以减少资源占用。双方各自的对战记录会落在 `runs/` 下两个目录中。

### 4. 人机对战（人类 vs AI）

```bash
python scripts/human_battle.py --team "Champions A"
#   --team NAME        从 teams/ 库选队（或直接传 packed 队伍串）
#   --count N          连打 N 场后退出
#   --wait-for NAME    只接受来自该用户的挑战（默认接受任何人）
```

脚本启动后 agent 以 `PokeAgent` 用户名等待挑战。用浏览器打开本地服务器（`http://localhost:8964`），任选用户名登录，然后向 `PokeAgent` 发起 **[Gen 9 Champions] VGC 2026 Reg M-C (Hidden)** 格式的挑战即可开打（标准 Gen 9 请显式传 `--format gen9vgc2025regihidden --team "Shadow Bolt"`）。

> 此路径尚未经过真实浏览器的端到端实测，属实验性功能；如遇挂起或非法指令，请附上 `runs/` 下对应的记录目录反馈。

运行结束后会打印胜负与记录目录：

```
Finished battles: 1/1
Agent wins: 1
Runs logged under: /…/Pokemon_battle_agent/runs
```

`postmortem.py` 每场输出一行，例如：

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
| `events.jsonl` | 逐回合结构化记录：模型看到的局面、决策、理由、逐字思考过程、工具调用、token 与耗时，以及预览/回退/对战结束等事件 |
| `llm.jsonl` | 完整 LLM 对话轨迹：system prompt、每回合 prompt、每轮响应、每次工具结果，以及推理模型的隐藏思考（`thinking`）与思考 token 数——逐字可复现（`log_llm_conversation = false` / `log_thinking = false` 可关闭） |

进程级日志位于 `runs/logs/agent-<时间戳>.log`，包含模型调用错误、工具与知识注入失败、子任务异常等排查信息。

跨场记忆位于 `runs/_memory/<game_type>/<ruleset>/<regulation>/`（例如 Champions Reg M-C 是 `doubles/champions/regmc/`）：

| 路径 | 内容 |
|---|---|
| `atoms/<date>_<player>_<tag>.ndjson` | 每场不可变原子（首行 battle_meta，其余为 fact 行） |
| `scenes/{species,teams,archetypes,our}/` | 合并后的档案 |
| `facts/` | scene 的确定性 grep 投影（`memory_grep` 检索的就是它） |
| `scene_index.json` | 记忆索引（summary/heat），`memory_read`/预览使用 |

维护脚本：`python scripts/seed_memory.py --coverage`（冷启动种子与覆盖率）、`python scripts/rebuild_memory.py`（从历史 `runs/` 回填原子）、`python scripts/memory_report.py`（scope 统计）。

## Champion 模式

同一套 agent 也支持 Pokémon Showdown 的 `champions` mod。Champion 规则下：

- 数值使用 **Stat Points**（每项 0–32、总和 ≤66），不使用 EV/IV 与等级公式；
- **无 Tera、有 Mega Evolution**（每回合可对一只宝可梦使用）；
- PP 上限为 20，数据表使用 Champion 覆盖表；
- 队伍必须 Champion 合法（仓库自带 `teams/champions-a.txt` / `champions-b.txt`）。

```bash
# 自定义隐藏 Champion VGC 格式（6 选 4、隐藏配置、AI 友好计时器）
python scripts/run_battle.py \
    --format gen9championsvgc2026regmchidden \
    --team "Champions A" --opponent agent --battles 1

# 官方格式（VGC Timer；agent 多轮调用可能超时）
python scripts/run_battle.py --format gen9championsvgc2026regmc --team "Champions A"
```

自定义 Champion 格式定义在 `server/custom-formats.ts`，安装后需 `node build` 并重启 8964 服务器。

## 配置

`config.toml`（从 `config.example.toml` 复制）中的 `[agent]` 节可覆盖全部默认值，常用项：

| 配置项 | 默认 | 说明 |
|---|---|---|
| `provider` | `openai` | LLM 后端（`openai` / `anthropic`） |
| `model` | `gpt-4o-mini` | 主决策模型 |
| `base_url` / `api_key` | 空 | OpenAI 兼容端点与密钥；不设时依次读 `OPENAI_BASE_URL` / `OPENAI_API_KEY` |
| `battle_format` | `gen9championsvgc2026regmchidden` | 对战格式（默认 Champions 6 选 4 双打；标准 Gen 9 用 `gen9vgc2025regihidden`） |
| `server_host` / `server_port` | `localhost` / `8964` | 本地 Showdown 服务器地址 |
| `turn_budget` / `preview_budget` | 45s / 30s | 每回合 / 选队决策的硬预算，超时回退 |
| `max_tokens_per_call` | 4000 | 单次模型调用的 token 上限；推理模型需要留出思考 token |
| `smogon_stats_snapshot` | 空 | 使用率快照文件（`scripts/fetch_usage_snapshot.py` 一次性下载）；设置后离线加载 |
| `smogon_stats_format` / `smogon_stats_fetch` | 自动 / `false` | 使用率对应的真实天梯格式 id（默认去掉隐藏格式的 `hidden` 后缀）；`fetch=true` 允许无快照时联网回退 |
| `enable_memory` | `true` | 启用三层记忆（atoms / scenes / facts）、记忆索引与 `memory_grep` / `memory_read` 工具 |
| `memory_tool_calls_per_turn` | `3` | 每回合 `memory_grep` + `memory_read` 的合计上限 |
| `seed_champions_sets` | `true` | 用 Champions curated 招式池做冷启动先验（无文件时静默跳过） |
| `allow_previous_seasons` | `true` | 旧赛季记忆可读但降权；规则本身始终隔离 |
| `enable_subagents` | `true` | 是否启用后台的威胁分析、选队分析与战后复盘 |
| `runs_dir` | `runs` | 对战记录输出目录 |

完整配置项及注释见 `config.example.toml`。

## 测试

```bash
python -m pytest tests/ -q                                          # 全部（e2e 需本地服务器在跑）
python -m pytest tests/ -q --ignore=tests/test_e2e_loop.py \
    --ignore=tests/test_e2e_doubles.py --ignore=tests/test_e2e_champions.py   # 纯离线
```

三个 e2e 测试是零 token 的真实对战（单打 / 双打 / Champion 双打 AI vs AI）：用 stub 模型跑完整协议、记录与复盘流程，不需要 API key。

## 路线图

- **人机对战实测**：`scripts/human_battle.py` 已写好，等待真实浏览器的端到端验证。
- **知识库扩展**：继续补充道具效果查询与决策前的机制提示，并增加“特性/道具克制关系”的软校验提醒。
- **对手知识蒸馏**：从高分对战记录与解说中提炼决策知识，扩充机制百科与配队流派库。