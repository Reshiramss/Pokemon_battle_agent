# 代码说明（Architecture & Internals）

本文逐模块说明 `pokeagent` 框架的实现。阅读顺序建议：先看「一次回合的完整时序」建立全局观，再按需查阅模块细节。

- 入门介绍见 [README.md](../README.md)
- 所有路径相对仓库根目录，行号以当前代码为准

---

## 1. 决策时序：两个阶段

决策循环分两种（同一套骨架：hook 注入 → prompt 组装 → 工具循环 → 终端工具校验）：

**阶段 1 `run_preview`（选宝可梦，仅预览格式触发）**

```
Showdown 服务器下发 teampreview 请求（gen9ou 等格式；gen9randombattle 无此阶段）
   ▼
PokeAgent.teampreview(battle)                     player.py
   │ cm = self._context_for(battle)
   │ runner.submit("team_preview", ...)           # 后台队预分析（fire-and-forget，便宜模型）
   ▼ asyncio.wait_for(..., preview_budget=20s)
AgentLoop.run_preview(battle, cm)                  loop.py
   │ ① hooks.run_pre → ② 组装 [TEAM PREVIEW] prompt（双方全队）
   │ ③ 工具循环（领域工具 + submit_lead 终端工具）
   │      submit_lead(slot 1..6, reason) → 校验槽位 → 首发排在 /team 串首位
   │ 轮数耗尽/provider 报错 → PreviewAbort → 回退基类随机首发
   ▼
返回 "/team 2..." 给服务器
```

**阶段 2 `run_turn`（对战内决策，每回合）**

```
Showdown 服务器 (ws://localhost:8964)
   │ request（"request" 协议消息）
   ▼
PokeAgent.choose_move(battle)                      player.py:161
   │ cm = self._context_for(battle)                 # 每场一个 ContextManager，惰性创建
   │
   ▼ asyncio.wait_for(..., turn_budget=45s)
AgentLoop.run_turn(battle, cm)                     loop.py:108
   │
   │ ① _maybe_fire_threat_analyses()               # 对比 cm.seen_species，发现新对手宝可梦
   │    └─ player._fire_threat_analysis()           #   → runner.submit(threat_analysis) 后台执行
   │ ② hooks.run_pre(ctx, 5s)                      # 并行跑 TypeMatchupHook / SmogonUsageHook
   │    └─ 返回知识片段列表（每段限 1500 字符）
   │ ③ cm.assemble_prompt(battle, fragments)       # 组装 [BATTLE INFO][INTEL][KNOWLEDGE][HISTORY][NOW]
   │
   │ ④ 工具循环，最多 max_tool_rounds=6 轮：
   │      provider.complete(system, messages, tools)   # tools = 领域工具 + submit_decision
   │      ├─ resp 带 tool_calls 且无 submit_decision
   │      │     → registry.execute_parallel(ctx, calls) # 单轮多调用 asyncio.gather 并行，每个 8s 超时
   │      │     → tool results 追加进 messages，进入下一轮
   │      ├─ resp 带 submit_decision
   │      │     → _validate_submit()               # 对照 available_moves / available_switches 校验
   │      │        ├─ 合法 → return TurnOutcome    # 回合结束
   │      │        └─ 非法 → 错误 tool_result 追加，模型重试
   │      └─ resp 纯文本 → 追加 NUDGE 消息，继续下一轮
   │
   │ ⑤ 轮数耗尽 → raise TurnAbort
   ▼
choose_move 捕获三类异常（TimeoutError / TurnAbort / 其他）
   ├─ 正常：TurnOutcome.order → 上交给 poke-env → 服务器
   └─ 异常：_record_fallback() 落盘 + choose_random_move() 兜底（绝不判负）

无论成功与否都构造 TurnRecord 调 cm.record_turn()：
   ├─ 逐字写 runs/<ts>_<tag>/events.jsonl（请求/决策/理由/CoT/工具调用/token/耗时）
   └─ _maybe_schedule_compression()                  # 历史超阈值则后台摘要（见 §6）
```

**关键路径上没有任何子 agent、没有任何压缩等待**——这两个都是 fire-and-forget 的后台任务。

---

## 2. `pokeagent/config.py` — AgentConfig

`@dataclass AgentConfig` 承载全部可调参数（字段清单见 README 配置表）。要点：

- `load_config(path, overrides)`：默认值 → TOML 文件（接受扁平结构或 `[agent]` 节）→ 显式覆盖，三层合并
- `effective_subagent_model` property：`subagent_model or compressor_model`——后台任务模型只需配一处
- 所有组件只从 config 读参数，不做自己的隐藏配置

## 3. `pokeagent/player.py` — PokeAgent（唯一与 poke-env 耦合的入口）

`PokeAgent(poke_env.player.Player)` 把 poke-env 的生命周期回调接到框架上。**框架其余部分不 import poke-env 的 Player**（tools/hooks/context 只用 `AbstractBattle` 的读接口）。

| 成员                                       | 行号   | 职责                                                                                                                                                 |
| ------------------------------------------ | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__init__`                               | :64    | super().__init__（传 `server_configuration`），然后装配 kb/provider/registry/hooks/runner/loop。所有部件都可注入，默认走 `_default_*` 工厂 |
| `build_server_configuration(config)`     | 模块级 | 8964 端口的 `ServerConfiguration`。**对手 bot 也必须用它**——poke-env 默认连 8000（本机是别人的服务）                                       |
| `_default_registry` / `_default_tools` | :126   | DamageCalcTool + move/type/species_info + BattleHistoryTool                                                                                          |
| `_default_hooks`                         | :138   | TypeMatchupHook（常开）+ SmogonUsageHook（`enable_smogon_hook` 控制）                                                                              |
| `_default_runner`                        | :102   | 后台 TaskRunner，三个子 agent 全注册，模型换成 `effective_subagent_model`                                                                          |
| `_context_for`                           | :149   | 每场一个 ContextManager；创建时把 `LLMCompressor(runner.provider)` 挂上去（便宜模型做压缩）                                                        |
| `choose_move`                            | :161   | 关键路径：`wait_for(loop.run_turn, turn_budget)`；成功构造 TurnRecord 落盘，异常走 `_record_fallback` + 随机兜底                                 |
| `_fire_threat_analysis`                  | :231   | 新对手宝可梦 → `runner.submit("threat_analysis")`，每场上限 `max_threat_analyses_per_battle`                                                    |
| `teampreview`                            | :253   | 阶段1 决策：`wait_for(loop.run_preview, preview_budget)` 让 LLM 经 `submit_lead` 终端工具定首发；失败/超时回退基类随机首发并落 `preview_fallback`。先 fire-and-forget 提交 team_preview 后台分析（其产出供后续回合用，不阻塞首发决策）。随机对战无预览 → 自然不触发 |
| `_battle_finished_callback`              | :268   | 先 `cancel_battle()`（丢弃战局分析），**再**提交 post_battle 复盘（否则复盘也会被取消）                                                      |

## 4. `pokeagent/loop.py` — AgentLoop 与两种决策循环

框架的心脏。**决策循环分两种**（同一骨架，仅 prompt 内容与终端工具不同）：

- **`run_preview`（阶段1：选宝可梦）**：仅预览格式触发（gen9ou 等；gen9randombattle 自动休眠）。hook 注入知识 → 组装 [TEAM PREVIEW] prompt → 工具循环 → `submit_lead` 终端工具校验槽位 1-6 → `PreviewOutcome`。失败抛 `PreviewAbort` → 回退基类随机首发。
- **`run_turn`（阶段2：对战内决策）**：流程见 §1。

- **`SUBMIT_DECISION`**（:38）：`submit_decision` 的 ToolSpec——`{action: move|switch, target, terastallize?, reason}`，`reason` 必填（验收要求每回合有理由）。它是**终端工具**，不在 ToolRegistry 里，由 loop 自己校验。
- **`SUBMIT_LEAD`**（:66）+ **`PREVIEW_SYSTEM_PROMPT`**（:88）：阶段1 的终端工具（`{slot: int, reason}`）与 prompt 模板——双方完整队伍（`describe_pokemon` 一行一只）+ 克制/速度/联防考量指引。
- **`PreviewOutcome`**（:138）：lead（槽位）+ reason + rounds + cot + tool_calls + token 计数，结构上与 `TurnOutcome`（:126）对齐——teampreview 落盘 `team_preview` 事件所需的全部。
- **`run_preview`**（:171）：与 run_turn 完全同构——hooks 注入 → 组 prompt → 工具循环（`registry.specs() + [SUBMIT_LEAD]`，`max_tool_rounds` 轮）。**逐轮落盘与 run_turn 一致**，仅多一个 `phase="preview"` 字段，所以 llm.jsonl 里能区分预览轨迹与对战轨迹。
- **`_validate_lead`**（:311）：槽位必须是 `1..len(team)` 的整数；非法 → 错误 tool_result（列出合法范围）让模型重试。`_validate_submit`（:477）同理：target 与 `battle.available_moves` / `available_switches` 做 `normalize_name` 匹配，错误信息里列出全部合法选项。
- **`PreviewAbort`**（:108）：预览版 TurnAbort——provider 错误或轮数耗尽时抛出，player 捕获后回退基类随机首发并落 `preview_fallback` 事件。
- **`_maybe_fire_threat_analyses`**（:464）：diff `battle.opponent_team` 与 `cm.seen_species`，新登场的 species 触发 `self.on_new_opponent_mon` 回调（player 注入 `_fire_threat_analysis`）。回调异常只 warning，不影响回合。
- 模型纯文本不出工具时追加 `NUDGE`（manager.py 定义）催促提交——两个循环共用。
- **对话轨迹落盘**：`log_llm_conversation=true`（默认）时，每回合先写一条 `turn_prompt`（system + 完整 prompt + 工具清单），之后每轮写 `llm_response`（文本/工具调用/stop_reason/tokens）和 `tool_results`——进 `cm.log_llm`，落 `llm.jsonl`。任何一回合都可离线复现。

## 5. `pokeagent/providers/` — LLM 抽象

### `base.py` — 归一化消息协议

框架内部**只**使用三种消息：`UserMessage` / `AssistantMessage` / `ToolResultMessage`（都带 `content: str`），加上 `ToolSpec`、`ToolCallRequest`（arguments 已解析为 dict）和 `LLMResponse`（`text`=CoT、`tool_calls`、归一化 `stop_reason ∈ {end_turn, tool_use, max_tokens, error}`、token usage）。

`LLMProvider.complete(*, system, messages, tools, max_tokens)` 是唯一抽象方法。provider 的职责是双向翻译归一化格式 ↔ 原生 API，框架其余代码完全不感知后端差异。

### `anthropic_provider.py`

- 凭据从环境读取：`ANTHROPIC_API_KEY`（必需，缺失时构造即报清晰错误）、`ANTHROPIC_BASE_URL`（可选，接内部网关）
- `_translate_messages`：归一化消息 → Anthropic wire 格式，**合并相邻同角色消息**（API 要求角色交替；多个 tool_result 连续出现时很常见）
- API 异常**永不冒泡**：捕获后返回 `stop_reason="error"` 的 LLMResponse，loop 收到后抛 TurnAbort → 随机兜底

### `openai_provider.py` — 默认实现

- 走 `chat.completions` wire 格式，兼容任何 OpenAI 风格端点（vLLM / 网关 / 官方）
- 端点解析顺序：`config.base_url`（`config.toml`）→ `OPENAI_BASE_URL` 环境变量 → api.openai.com；key 同理，最终兜底 `"EMPTY"`（本地 vLLM 不校验）
- `_translate_messages`：工具结果 → `role:"tool"` 消息（须紧跟持有对应 `tool_calls` 的 assistant 消息，loop 已保证顺序）；`finish_reason` 映射 `tool_calls→tool_use`、`length→max_tokens`，其余视为错误
- 工具参数 JSON 解析失败时降级为空参数 → 工具返回校验错误 → 模型重发
- API 异常同样**永不冒泡**，行为与 anthropic_provider 一致

## 6. `pokeagent/context/` — 上下文管理（核心资产）

### `records.py` — 快照与逐回合记录

- `snapshot(battle)` / `available_actions(battle)`：把 battle 对象渲染成紧凑文本（每只宝可梦一行：等级/血量/异常状态/属性/强化/道具；招式带威力/命中/PP/优先度）。`battle.turn` 可能在同一回合内重复出现（击倒后的强制换人请求），因此记录用**单调 seq** 而非 turn 编号。
- `TurnRecord`（:115）：一次决策点的全部——`seq`、`turn`、模型看到的 `request`、期间协议 `events`、`decision`/`reason`、**逐字 `cot`**、`tool_calls`、token 数、`fell_back`/`fallback_reason`、`elapsed_s`。

### `manager.py` — ContextManager（每场一个）

- **落盘即真相**：`record_turn()` 把 TurnRecord 原样追加进 `runs/<UTC时间戳>_<battle_tag>/events.jsonl`；`log_event()` 写其他事件（battle_start/fallback/subagent_done/compressed/…）；`log_llm()` 把逐轮原始对话追加进同一目录的 `llm.jsonl`（详见 §12）。压缩只影响 prompt，落盘永不丢失。
- **`_write_meta`**（构造时）：写 `meta.json` 配置快照（模型/provider/全部参数，`api_key` 脱敏为 `<set>`）——事后想知道"这局是什么配置跑的"不用翻日志。
- **`assemble_prompt(battle, fragments)`**（:200）按固定顺序拼段：
  `[BATTLE INFO]` → `[INTEL 子agent摘要]` → `[KNOWLEDGE hook片段]` → `[HISTORY]` → `[NOW 当前状态+合法动作]`。
  HISTORY 内部（`_history_text`）分两层：旧回合 → LLM 摘要（覆盖到 `summary_covered_seq`）+ 未覆盖旧回合的单行压缩（`_compact_record`：`T3 did move:x (理由) [fallback: ...] saw: ...`）；最近 `recent_turns_verbatim=6` 回合逐字保留。整体超过 `context_token_soft_cap`（chars/4 估算）则硬截断并注明。
- **压缩触发**（`_maybe_schedule_compression`，:120）：`record_turn` 末尾检查「未被摘要覆盖的历史字符 // 4 ≥ compression_trigger_tokens」且无压缩在飞 → `asyncio.create_task(_compress(end_seq))`。关键路径只做一次数值比较。
- **竞态安全**（`_compress`，:136）：`end_seq` 在调度瞬间快照；摘要只覆盖 `seq <= end_seq` 的记录。期间新记录照常追加；落成时原子换入 `summary` / `summary_covered_seq`。记录只增不改，所以换入永不覆盖新状态。`_compressing` 标志防止重复提交；压缩期间 prompt 对旧回合退回 `_compact_record` 单行（可读、有界、无需等待）。

### `compressor.py` — 两种压缩策略（可替换的 Protocol）

- `ContextCompressor`：`async compress(records, budget_tokens) -> str`
- `TruncatingCompressor`：零 LLM 兜底，每记录一行，超预算从最老开始丢弃并注明
- `LLMCompressor`：把 `_compact_record` 行喂给便宜模型做战斗摘要（保留局势走向/双方血量/已揭示招式/陷阱与制胜条件），输出限 `summary_token_budget`；失败 raise（由 `_compress` 记日志后吞掉，等价于本次没压缩）

## 7. `pokeagent/tools/` — Function-call 工具

### `base.py`

- `Tool`：`name/description/input_schema` + `async execute(ctx, **kwargs) -> str`
- `ToolContext`：battle + kb + context + config——工具能拿到的一切
- `ToolError`：参数非法专用；`ToolRegistry.execute_parallel` 把它转成错误 tool_result 让模型重试；**未知工具、超时、TypeError（参数不匹配）、意外异常都转成错误结果，永不把回合打崩**
- 并行执行：一轮里模型请求的多个调用 `asyncio.gather` 并发，每个 `wait_for(tool_call_timeout)`

### 具体工具

| 工具               | 文件            | 作用                                                                                                                                                                                                     |
| ------------------ | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `damage_calc`    | damage_calc.py  | 调 `poke_env.calc.damage_calc_gen9.calculate_damage` 算伤害区间。对手种族值未知时用 kb 估算（见 §9），结果标注 estimated。attacker/defender 用 species 名在 team/opponent_team 里匹配（含形态名归一） |
| `move_info`      | static_kb.py    | 招式数据：类型/分类/威力/命中/PP/优先度/目标/追加效果                                                                                                                                                    |
| `type_matchup`   | static_kb.py    | 攻击方 → 全属性倍率表，标注 x2/x4/x0.5 等                                                                                                                                                               |
| `species_info`   | static_kb.py    | 种族值/属性/特性/进化链                                                                                                                                                                                  |
| `battle_history` | battle_query.py | 无参数=一行决策摘要；`turn=N`=该回合完整记录（请求/决策/工具调用/协议事件）；`include_raw_protocol` 才给原始协议                                                                                     |

所有工具输出 `_clip` 限长，防止单次工具调用撑爆上下文。

## 8. `pokeagent/hooks/` — 决策前知识注入

### `base.py`

- `PreDecisionHook.run(ctx) -> str | None`：返回注入 `[KNOWLEDGE]` 段的 markdown 片段
- `HookSystem.run_pre(ctx, budget)`：全部并行，**每个 hook 独享共享 deadline 的一个 wait_for 分片**——单个慢/挂死 hook 只烧掉自己的时间，不拖累其他 hook。异常/超时一律 log 后跳过
- `PostDecisionHook`：观察已提交的决策（指标/日志用），当前无默认实现

### 实现

- `TypeMatchupHook`（static_kb_hook.py）：进攻视角（我方可用招式 → 对方属性倍率，标注 SUPER EFFECTIVE）+ 防守视角（对方已揭示招式 → 我方当前属性倍率）+ 已揭示招式清单。随机对战每回合都值得看，所以常开。
- `SmogonUsageHook`（smogon_usage_hook.py）：`SmogonStats.fetch` 按 format 抓使用率（`to_thread` + asyncio.Lock 缓存一次），给出每只已揭示宝可梦的使用率/热门配置。**随机对战默认休眠**（队伍随机生成，使用率无意义），`enable_smogon_hook=true` 给排位格式用。抓取失败粘性（`_fetch_failed`）后整场不再尝试。

## 9. `pokeagent/knowledge/static.py` — StaticKB

`GenData` 的薄封装（每代单例，进程内共享）：

- `move_data` / `species_data`（含形态名回退：`charizardgmax` → `charizard`）/ `learnset`（注意 GenData 的嵌套结构 `{learnset: [...], encounters: [...]}`，已解包）/ `type_chart`
- `type_effectiveness(attacker, defender)`：**chart 索引方向是 `[防守方][攻击方]`**（由各属性的 `damageTaken` 构建，键为大写枚举名）——曾因方向反了静默返回 1.0，已修正并有测试盯住
- `estimate_stats(species, level)`：`compute_raw_stats(species, evs=[85]*6, ivs=[31]*6, nature="serious", level=level)`——对手个体值未知时的均摊估算，伤害计算结果里明确标注

## 10. `pokeagent/tasks/` — 后台子 agent

### `base.py`

- `IntelStore`：key → (时间戳, 摘要)。`put` 时截断到 1200 字符——**后台研究的原始长文永远进不了主上下文**。`assemble_prompt` 每回合注入 `[INTEL]` 段。
- `SubAgentContext`：一次运行的隔离输入（battle 快照引用 + kb + intel + 自由 inputs）——子 agent 看不到主循环的对话历史。
- `SubAgent`：`name` + `async run(sctx) -> str`（返回值即摘要）。

### `runner.py` — TaskRunner

- `submit(battle_tag, agent_name, sctx, cm=None, intel_key=None)`：fire-and-forget `create_task`。任务内部：`Semaphore(max_concurrent_subagents)` 限全局并发 → `wait_for(agent.run, subagent_timeout)` → 摘要写入 intel → `cm.log_event("subagent_done", ...)`。超时/异常只记事件（`subagent_timeout` / `subagent_error`），**绝不冒泡**。
- `cancel_battle(battle_tag)`：对战结束时取消该场全部在飞任务。注意 `player._battle_finished_callback` 里必须**先** cancel 再提交 post_battle 复盘，否则复盘刚提交就被取消。

### 三个默认子 agent（全部用便宜模型、单次 LLM 调用）

| name                | 文件               | 触发点                                 | 产出                                                               |
| ------------------- | ------------------ | -------------------------------------- | ------------------------------------------------------------------ |
| `threat_analysis` | threat_analysis.py | loop 检测到新对手宝可梦（每场 ≤6 次） | `threat:<species>`：种族/属性/特性/已知招式 + 对我方全队的威胁线 |
| `team_preview`    | team_preview.py    | `teampreview` 回调（随机对战不触发） | `team_preview`：选队顺序建议                                     |
| `post_battle`     | post_battle.py     | `_battle_finished_callback`          | `post_battle_review`：整场决策复盘（从 cm.records 读决策日志）   |

## 11. `scripts/` — CLI 工具

- **run_battle.py**：`--opponent random|heuristic --battles N --config path`。启动先 `setup_logging`（`logging_setup.py`）：每次运行生成 `runs/logs/agent-<UTC时间戳>.log`，框架/poke-env 的 logging 与未捕获异常堆栈全部落盘，启动时打印路径。对手 bot 也传 `build_server_configuration(config)`（否则连 8000 端口）。
- **postmortem.py**：扫 `runs/` 下 events.jsonl，每场一行：胜负/回合数/回退率/CoT 完整率/延迟 mean-p95/token 总量/子 agent 数/压缩次数 + TOTAL 汇总。`--last N` 看最近几场。
- **smoke_env.py**：RandomPlayer vs RandomPlayer 连通性冒烟（`--port`）。
- **diag_connect.py**：逐玩家检查 websocket 登录状态（排查连错端口/登录失败用）。

## 12. 落盘格式：`runs/`

一次运行产生四层日志，各司其职：

```
runs/
├── logs/agent-<UTC时间戳>.log      进程级日志（每次 run_battle 一个文件）
└── <UTC时间戳>_<battle_tag>/       每场对战一个目录
    ├── meta.json                   该场配置快照（api_key 脱敏为 "<set>"）
    ├── events.jsonl                结构化事件流（postmortem 的数据源）
    └── llm.jsonl                   LLM 原始对话轨迹（调试复现用，MB 级）
```

### `events.jsonl` — 结构化事件流

每行一个 JSON 对象，`type` 字段区分：

| type                                                          | 内容                                                                                                              |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `battle_start`                                              | battle_tag / format / model                                                                                       |
| `turn`                                                      | 完整 TurnRecord（seq/turn/request/events/decision/reason/**cot 逐字**/tool_calls/tokens/elapsed/fell_back） |
| `fallback`                                                  | 走了随机兜底的额外标记（turn/reason/elapsed）                                                                     |
| `subagent_done` / `subagent_timeout` / `subagent_error` | 后台任务生命周期（agent/key/summary 前 1500 字符）                                                                |
| `compressed` / `compress_error`                           | 压缩换入（covered_seq/chars/**summary 全文**）或失败                                                               |
| `battle_finished`                                           | turn / won / lost                                                                                                 |

postmortem.py 只依赖这份文件——日志既是调试真相源，也是离线分析的数据集。

### `llm.jsonl` — LLM 对话轨迹（`log_llm_conversation`，默认开）

loop.py 逐轮写入，任何回合都能离线复现「模型到底看到了什么、说了什么」：

| type           | 内容                                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| `turn_prompt`  | 本回合完整输入：system prompt + 组装后的 user prompt（INTEL/KNOWLEDGE/HISTORY/NOW 全部）+ 工具名清单   |
| `llm_response` | 一轮模型输出：text（CoT）/ tool_calls（name+arguments）/ stop_reason / tokens                          |
| `tool_results` | 该轮工具执行结果：call_id / tool / content / is_error（非法 submit_decision 的报错也在这里）          |
| `llm_error`    | provider 网关级错误（该回合随后 TurnAbort）                                                             |

### 进程日志 `logs/agent-*.log`

`logging_setup.setup_logging(config)`（run_battle 启动时调用一次）给 root logger 挂 `FileHandler`（级别 `log_level`，默认 INFO）+ 控制台 `StreamHandler`（WARNING）。provider 报错、hook 超时、子 agent 失败、poke-env 玩家日志、未捕获异常堆栈全部落盘。重复调用幂等（换绑不叠加）。

## 13. 关键设计决策

1. **终端工具制而非解析自然语言**：出招必须是 `submit_decision` 的结构化调用。收益：校验合法性的位置单一、失败可给模型带明确错误的重试、reason 是 schema 必填字段（验收每回合有理由）。
2. **磁盘即真相，prompt 是派生物**：prompt 丢了随时可以重建，但逐字 CoT 一旦没落盘就永远没了。所以先落盘、后组装；压缩只改写派生物。为此落盘分三层：`events.jsonl`（结构化事件，postmortem 数据源）、`llm.jsonl`（模型看到的完整对话逐字复现，仅调试用）、`meta.json`（该场配置快照，api_key 脱敏）——事后排查「为什么这回合这么决策」不需要重跑，直接读文件。
3. **关键路径零子 agent**：每回合延迟预算 45s，任何后台 LLM 调用都可能吃掉它。子 agent 只做「下一回合才有价值」的分析，产出经 IntelStore 限长注入——用陈旧性换延迟。
4. **一切失败都兜底**：provider 异常、工具异常、hook 超时、子 agent 崩溃、压缩失败——全部被各自边界捕获并记录，最坏结果只是那一回合随机出招，对战永远不因框架崩溃而判负。
5. **seq 单调编号贯穿压缩**：turn 号会重复（强制换人请求）、记录只追加——用单调 seq 做「摘要覆盖到哪」的水位线，使后台摘要换入无需任何锁。
