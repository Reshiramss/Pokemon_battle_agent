"""Sub-agent abstractions and the per-battle intel store."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poke_env.battle import AbstractBattle

    from ..knowledge.static import StaticKB


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class IntelStore:
    """Latest background-analysis summaries, keyed by topic.

    Sub-agent results land here; ``ContextManager.assemble_prompt`` injects
    the current state into every prompt. Summaries are capped so raw
    research data never pollutes the main context.
    """

    def __init__(self, summary_char_cap: int = 1200):
        self._cap = summary_char_cap
        self._items: dict[str, tuple[str, str]] = {}  # key -> (timestamp, summary)

    def put(self, key: str, summary: str) -> None:
        summary = (summary or "").strip()
        if len(summary) > self._cap:
            summary = summary[: self._cap - 1].rstrip() + "…"
        self._items[key] = (_utcnow(), summary)

    def latest(self) -> dict[str, str]:
        return {key: text for key, (_ts, text) in self._items.items()}

    def get(self, key: str) -> str | None:
        item = self._items.get(key)
        return item[1] if item else None

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class SubAgentContext:
    """Isolated inputs for one sub-agent run (a snapshot, never a live loop)."""

    battle: "AbstractBattle"
    kb: "StaticKB | None" = None
    intel: IntelStore | None = None
    inputs: dict[str, Any] = field(default_factory=dict)  # free-form per-agent inputs


class SubAgent(ABC):
    """A background LLM-driven analysis with an isolated context.

    Implementations run their own short LLM conversation and return a single
    summary string; the runner caps and stores it in the IntelStore.
    """

    name: str = "subagent"

    @abstractmethod
    async def run(self, sctx: SubAgentContext) -> str:
        """Execute the analysis and return a summary (kept short by the store)."""
