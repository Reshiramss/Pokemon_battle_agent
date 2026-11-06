"""Pre-decision hook: recent cross-battle memory for this matchup.

Keys are derived exactly like ``MemoryWriter`` derives them when a battle
ends (opponent species, opponent roster hash, our team key, detected
tactics), so the fragment is literally "what do we remember about these
opponents" — newest row per key, hard-capped to a few short lines. An
empty memory or a disabled config yields no fragment at all.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from ..knowledge.static import normalize_name
from ..memory.keys import opponent_key, our_team_key
from ..memory.store import MemoryStore
from ..memory.tactics import tactic_ids
from .base import PreDecisionHook

if TYPE_CHECKING:
    from ..config import AgentConfig
    from ..tools.base import ToolContext

logger = logging.getLogger(__name__)


class MemoryHook(PreDecisionHook):
    """Injects the newest remembered rows for the current matchup."""

    name = "memory"

    def __init__(self, config: "AgentConfig", store: MemoryStore | None = None):
        self.config = config
        self.store = store or MemoryStore(
            getattr(config, "memory_path", None) or "runs/_memory",
            lines_cap=getattr(config, "memory_lines_cap", 500),
        )

    async def run(self, ctx: "ToolContext") -> str | None:
        if not getattr(self.config, "enable_memory", False):
            return None
        try:
            return await asyncio.to_thread(self._build, ctx)
        except Exception as exc:
            logger.warning("memory hook failed: %s", exc)
            return None

    def _build(self, ctx: "ToolContext") -> str | None:
        battle = ctx.battle
        foes = list(getattr(battle, "opponent_team", {}).values())
        if not foes:
            return None
        foe_species = [normalize_name(str(m.species)) for m in foes]
        ours = [normalize_name(str(m.species)) for m in getattr(battle, "team", {}).values()]

        keys: list[str] = [s for s in foe_species if s]
        keys.append(opponent_key(foe_species))
        keys.append(our_team_key(self.config, ours))
        keys.extend(tactic_ids(foes))

        rows = self.store.recent_for_keys(keys)
        if not rows:
            return None

        tag = getattr(battle, "battle_tag", "") or ""
        max_lines = max(int(getattr(self.config, "memory_hook_max_lines", 3)), 1)
        max_chars = max(int(getattr(self.config, "memory_hook_chars", 250)), 60)

        lines: list[str] = []
        used = 0
        for line, total in rows:
            if line.battle and line.battle == tag:
                continue  # written by this very battle (should not happen)
            older = f" (+{total - 1} more)" if total > 1 else ""
            text = f"{line.key}: {line.text}{older}"
            room = max_chars - used
            if room <= 4:
                break
            if len(text) > room:
                text = text[: room - 1].rstrip() + "…"
            lines.append(text)
            used += len(text) + 1
            if len(lines) >= max_lines:
                break
        return "\n".join(lines) if lines else None