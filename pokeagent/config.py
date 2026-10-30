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
    provider: str = "anthropic"  # key into the providers registry
    model: str = "claude-sonnet-4-5"  # main decision model
    compressor_model: str = "claude-haiku-4-5"  # cheap model for background compression
    subagent_model: str | None = None  # defaults to compressor_model
    max_tokens_per_call: int = 2000

    # --- battle ---
    battle_format: str = "gen9randombattle"
    server: str = "localhost"  # "localhost" | "showdown"
    server_host: str = "localhost"
    server_port: int = 8964  # port 8000 is occupied by another service on this machine

    # --- per-turn budgets (seconds) ---
    turn_budget: float = 45.0  # hard asyncio.wait_for around the whole turn loop
    hook_budget: float = 5.0
    tool_call_timeout: float = 8.0
    max_tool_rounds: int = 6

    # --- context management ---
    recent_turns_verbatim: int = 6  # how many recent decisions stay verbatim in prompt
    context_token_soft_cap: int = 12_000  # assembled-prompt soft cap (tokens, chars/4)
    compression_trigger_tokens: int = 8_000  # history size that triggers compression
    summary_token_budget: int = 800

    # --- sub-agents ---
    max_concurrent_subagents: int = 3
    subagent_timeout: float = 120.0
    max_threat_analyses_per_battle: int = 6

    # --- hooks ---
    enable_smogon_hook: bool = False  # dormant for random battles; enable for ladder formats

    # --- misc ---
    poke_env_log_level: str = "WARNING"

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
