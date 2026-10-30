"""Pre-decision hook: Smogon usage stats for revealed opponent species.

Dormant by default (``enable_smogon_hook=false``): random-battle ladders
have no meaningful Smogon usage data, and the snapshot download is a slow
network call. For standard formats the user can enable it; the fetch runs
once (to_thread, cached by SmogonStats' cache_dir) and every following
turn only hits the in-memory snapshot.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .base import PreDecisionHook

if TYPE_CHECKING:
    from poke_env.data.smogon import SmogonStats

    from ..config import AgentConfig
    from ..tools.base import ToolContext

logger = logging.getLogger(__name__)


class SmogonUsageHook(PreDecisionHook):
    """Injects usage %, common moves/items/abilities for revealed species."""

    name = "smogon_usage"

    def __init__(self, config: "AgentConfig"):
        self.config = config
        self._stats: "SmogonStats | None" = None
        self._fetch_failed = False
        self._fetch_lock = asyncio.Lock()

    async def run(self, ctx: ToolContext) -> str | None:
        if not self.config.enable_smogon_hook:
            return None  # dormant for random battles
        stats = await self._get_stats()
        if stats is None:
            return None

        lines = []
        for mon in ctx.battle.opponent_team.values():
            data = stats.pokemon.get(mon.species)
            if data is None:
                continue
            parts = [f"{data.name}: {data.usage * 100:.1f}% usage"]
            if data.moves:
                top = sorted(data.moves.items(), key=lambda kv: -kv[1])[:4]
                parts.append("moves " + ", ".join(f"{m} {v:.0%}" for m, v in top))
            if data.items:
                top = sorted(data.items.items(), key=lambda kv: -kv[1])[:2]
                parts.append("items " + ", ".join(f"{i} {v:.0%}" for i, v in top))
            if data.abilities:
                top = sorted(data.abilities.items(), key=lambda kv: -kv[1])[:2]
                parts.append(
                    "abilities " + ", ".join(f"{a} {v:.0%}" for a, v in top)
                )
            lines.append(" | ".join(parts))
        return "\n".join(lines) if lines else None

    async def _get_stats(self) -> "SmogonStats | None":
        """Fetch once (cached); a failed fetch is remembered, not retried."""
        if self._stats is not None or self._fetch_failed:
            return self._stats
        async with self._fetch_lock:
            if self._stats is not None or self._fetch_failed:
                return self._stats
            try:
                from poke_env.data.smogon import SmogonStats

                cache = Path(self.config.runs_dir) / ".smogon_stats_cache"
                self._stats = await asyncio.to_thread(
                    SmogonStats.fetch,
                    self.config.battle_format,
                    cache_dir=cache,
                )
            except Exception as e:
                self._fetch_failed = True
                logger.warning(
                    "Smogon usage fetch failed (hook stays silent this battle): %s", e
                )
        return self._stats
