"""Agent configuration: dataclass defaults, TOML file overrides."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass
class AgentConfig:
    """All tunables of the agent framework. Every component reads only this."""

    # --- LLM provider ---
    provider: str = "openai"  # key into the providers registry
    model: str = "gpt-4o-mini"  # main decision model
    compressor_model: str = "gpt-4o-mini"  # cheap model for background compression
    # OpenAI-compatible endpoint (vLLM / gateway / api.openai.com). None ->
    # OPENAI_BASE_URL env var, else api.openai.com. Key: None -> OPENAI_API_KEY
    # env var, else "EMPTY" (local vLLM endpoints don't check it).
    base_url: str | None = None
    api_key: str | None = None
    subagent_model: str | None = None  # defaults to compressor_model
    # Reasoning models (deepseek-flash etc.) emit hidden thinking tokens that
    # count against max_tokens: at 2000/call a long deliberation truncates to
    # an empty response (finish_reason "length"). Keep headroom, or lower the
    # effort so the model thinks less.
    max_tokens_per_call: int = 4000
    # Passed through to OpenAI-compatible endpoints as `reasoning_effort`
    # (deepseek-flash: low | high | max; default high). None -> not sent.
    reasoning_effort: str | None = None

    # --- battle ---
    # Doubles 6-pick-4 VGC without Open Team Sheets: team preview only reveals
    # the opponent's 6 species. Requires the custom format from
    # server/custom-formats.ts to be installed on the Showdown server.
    battle_format: str = "gen9championsvgc2026regmchidden"
    server: str = "localhost"  # "localhost" | "showdown"
    server_host: str = "localhost"
    server_port: int = 8964  # custom port; avoids colliding with poke-env's default 8000
    # Team for team-preview formats: a team-library name or a raw packed team
    # string (anything containing "|"). None -> server-assigned team
    # (random battles only).
    team: str | None = None
    teams_dir: Path | None = None  # optional user import dir merged into the library

    # --- per-turn budgets (seconds) ---
    turn_budget: float = 45.0  # hard asyncio.wait_for around the whole turn loop
    preview_budget: float = 30.0  # team-preview pick (a bigger decision than a turn)
    hook_budget: float = 5.0
    tool_call_timeout: float = 8.0
    max_tool_rounds: int = 6  # legacy single-loop cap; run_preview still uses it

    # --- two-phase turn loop (info gathering -> decision) ---
    # False reproduces the legacy single loop (regression guard)
    two_phase: bool = True
    info_rounds: int = 4  # max LLM rounds in the information phase
    decision_rounds: int = 3  # max submit/retry rounds in the decision phase

    # --- context management ---
    recent_turns_verbatim: int = 6  # how many recent decisions stay verbatim in prompt
    context_token_soft_cap: int = 12_000  # assembled-prompt soft cap (tokens, chars/4)
    compression_trigger_tokens: int = 8_000  # history size that triggers compression
    summary_token_budget: int = 800

    # --- reasoning notes (harness memory layer) ---
    # Raw CoT stays in llm.jsonl; each turn additionally yields a compact
    # structured note used by [REFLECTION]. With sub-agents disabled the
    # deterministic extractor runs instead (no extra model calls).
    enable_reasoning_notes: bool = True
    reasoning_note_input_chars: int = 6000  # CoT chars passed to the note model
    # reasoning models spend max_tokens on hidden thinking first; too small a
    # budget returns an empty answer (that is why notes were never produced)
    reasoning_note_max_tokens: int = 1500

    # --- rolling strategy brief (L2) ---
    # Every ``reasoning_brief_every`` new notes, a background model call
    # rewrites the brief in a fixed JSON schema ([STRATEGY] in the prompt).
    # Without a provider the deterministic brief lifts the plan/directives.
    enable_strategy_brief: bool = True
    reasoning_brief_every: int = 2
    reasoning_brief_input_chars: int = 4000
    reasoning_brief_max_tokens: int = 1200

    # --- battle-level COT (how this game is won) ---
    # After every turn a background sub-agent — on the MAIN model — rewrites
    # the COT from the previous text, this turn's hidden thinking, the board
    # the model saw, and what happened. It is rendered as [STRATEGY], so each
    # decision starts from the agent's own standing plan. Failures keep the
    # previous text; the turn never waits on it.
    enable_strategy_cot: bool = True
    strategy_cot_max_tokens: int = 3000  # thinking is spent from this budget
    strategy_cot_input_chars: int = 8000
    thinking_record_chars: int = 5000  # thinking text kept on TurnRecord

    # --- sub-agents ---
    # False -> the agent runs without the background TaskRunner entirely
    # (no cheap-model provider, no threat analyses/reviews) — used for the
    # AI-vs-AI opponent to halve resource use on a shared machine.
    enable_subagents: bool = True
    max_concurrent_subagents: int = 3
    subagent_timeout: float = 120.0
    max_threat_analyses_per_battle: int = 6

    # --- hooks ---
    # Usage stats hook: loads the local snapshot offline-first, network fetch
    # only as fallback (see hooks/smogon_usage_hook.py). The stats format is
    # the REAL ladder id — never the hidden battle format (Smogon's metagame
    # validation would reject it); None -> battle format with "hidden" stripped.
    enable_smogon_hook: bool = True
    smogon_stats_format: str | None = None
    smogon_stats_snapshot: Path | None = None  # chaos JSON(/.gz) from scripts/fetch_usage_snapshot.py
    # network fetch fallback when the snapshot is missing/broken — off by
    # default so battles never stall on a machine without internet
    smogon_stats_fetch: bool = False

    # --- cross-battle memory (S5) ---
    # Facts are immutable atoms; scenes are consolidated views; retrieval is
    # grep over a deterministic line projection. No embeddings, no server.
    enable_memory: bool = True
    memory_dir: Path | None = None  # defaults to <runs_dir>/_memory
    # Rule/season partitioning (see memory/scope.py).
    default_regulation: str = "auto"  # auto = parse regmc/regmb/regi from the id
    # Deterministic quality gate recorded on every battle meta atom.
    quality_min_lesson: float = 0.6
    quality_min_behavior: float = 0.5
    fallback_hard_gate: float = 0.25
    allow_previous_seasons: bool = True
    previous_season_decay: float = 0.6  # ranking multiplier for old seasons
    # Cold start: seed archetypes + our teams on first use; species skeletons
    # are generated lazily for exactly the keys a battle asks about.
    seed_on_first_run: bool = True
    seed_champions_sets: bool = True
    prior_rank_weight: float = 0.6  # priors rank below observations
    # Prompt budgets.
    memory_index_chars_preview: int = 1200
    memory_recall_chars_turn: int = 400
    memory_read_chars: int = 2000
    memory_tool_calls_per_turn: int = 3
    memory_recall_timeout_ms: int = 2000  # reserved; local reads are sync
    # Cross-rule transfer is deliberately off by default (docs §6.5).
    cross_ruleset_read: bool = False
    # Origin of the opponent, used by the quality gate.
    opponent_class: str = "unknown"  # human | agent | heuristic | random | unknown

    # --- misc ---
    poke_env_log_level: str = "WARNING"
    # run logs: per-invocation file log under runs/logs/, plus the per-battle
    # llm.jsonl raw-conversation dump. Both exist so a finished battle can be
    # debugged offline without re-running anything.
    log_level: str = "INFO"  # file log level (console stays WARNING)
    log_llm_conversation: bool = True  # full prompt/response/tool-result dump per turn
    # reasoning-model "thinking" text (Anthropic thinking blocks / OpenAI
    # reasoning_content). It is not assistant text, so it never reaches the
    # prompt, but it is the only reasoning a reasoning model shows us —
    # without it llm.jsonl records the visible answer and nothing else.
    log_thinking: bool = True

    # --- paths ---
    runs_dir: Path = field(default_factory=lambda: Path("runs"))

    @property
    def effective_subagent_model(self) -> str:
        return self.subagent_model or self.compressor_model

    @property
    def memory_path(self) -> Path:
        return Path(self.memory_dir) if self.memory_dir else Path(self.runs_dir) / "_memory"


def _discover_config() -> Path | None:
    """Find the project's config.toml when --config was not passed.

    Repo root first (deterministic regardless of cwd), then the working
    directory. Without this, a bare ``run_battle.py`` silently ran on the
    built-in defaults — wrong model and endpoint — and every LLM call failed.
    """
    candidates = (
        Path(__file__).resolve().parent.parent / "config.toml",
        Path.cwd() / "config.toml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_config(
    path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> AgentConfig:
    """Build an AgentConfig from defaults, then a TOML file, then explicit overrides."""
    values: dict[str, Any] = {}
    if path is None:
        path = _discover_config()
    if path is not None:
        with Path(path).open("rb") as f:
            data = tomllib.load(f)
        # Accept either a flat file or an [agent] section.
        values.update(data.get("agent", data))
    config = AgentConfig(**values)
    if overrides:
        config = replace(config, **overrides)
    config.runs_dir = Path(config.runs_dir)
    if config.memory_dir is not None:
        config.memory_dir = Path(config.memory_dir)
    return config
