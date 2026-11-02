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
    max_tokens_per_call: int = 2000

    # --- battle ---
    # Doubles 6-pick-4 VGC without Open Team Sheets: team preview only reveals
    # the opponent's 6 species. Requires the custom format from
    # server/custom-formats.ts to be installed on the Showdown server.
    battle_format: str = "gen9vgc2025regihidden"
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

    # --- misc ---
    poke_env_log_level: str = "WARNING"
    # run logs: per-invocation file log under runs/logs/, plus the per-battle
    # llm.jsonl raw-conversation dump. Both exist so a finished battle can be
    # debugged offline without re-running anything.
    log_level: str = "INFO"  # file log level (console stays WARNING)
    log_llm_conversation: bool = True  # full prompt/response/tool-result dump per turn

    # --- paths ---
    runs_dir: Path = field(default_factory=lambda: Path("runs"))

    @property
    def effective_subagent_model(self) -> str:
        return self.subagent_model or self.compressor_model


def load_config(
    path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> AgentConfig:
    """Build an AgentConfig from defaults, then a TOML file, then explicit overrides."""
    values: dict[str, Any] = {}
    if path is not None:
        with Path(path).open("rb") as f:
            data = tomllib.load(f)
        # Accept either a flat file or an [agent] section.
        values.update(data.get("agent", data))
    config = AgentConfig(**values)
    if overrides:
        config = replace(config, **overrides)
    config.runs_dir = Path(config.runs_dir)
    return config
