"""Pre-decision hook: Smogon usage stats for revealed opponent species.

Offline-first: with ``smogon_stats_snapshot`` set, stats load from that
local chaos-JSON file (a machine without internet still gets usage
data). The network ``SmogonStats.fetch`` runs only when explicitly
enabled (``smogon_stats_fetch``) and a snapshot is missing or fails —
never by default, so battles never stall on a dead network. The stats
format is NEVER the custom hidden battle format — its id fails Smogon's
metagame validation — so fetching uses ``smogon_stats_format`` when
set, else the battle format with the trailing "hidden" stripped.
Without any usable stats source the hook stays silent for the battle.
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
            return None
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
        """Load once — snapshot first, opt-in fetch fallback; sticky failure."""
        if self._stats is not None or self._fetch_failed:
            return self._stats
        async with self._fetch_lock:
            if self._stats is not None or self._fetch_failed:
                return self._stats
            try:
                from poke_env.data.smogon import SmogonStats

                snapshot = self.config.smogon_stats_snapshot
                if snapshot is not None:
                    path = Path(snapshot)
                    # fetch's cache layout is <dir>/<month>/<file>, so the
                    # month comes from the parent directory's name
                    try:
                        self._stats = await asyncio.to_thread(
                            SmogonStats.from_file,
                            path,
                            month=path.parent.name,
                        )
                    except Exception as e:
                        if not self.config.smogon_stats_fetch:
                            raise
                        logger.warning(
                            "Smogon snapshot at %s failed to load, falling "
                            "back to fetch: %s",
                            path,
                            e,
                        )
                if self._stats is None:
                    if not self.config.smogon_stats_fetch:
                        # no snapshot and no network opt-in: stay silent
                        # without burning a battle-long retry
                        self._fetch_failed = True
                        return None
                    self._stats = await asyncio.to_thread(
                        SmogonStats.fetch,
                        self.stats_format,
                        cache_dir=Path(self.config.runs_dir) / ".smogon_stats_cache",
                    )
            except Exception as e:
                self._fetch_failed = True
                logger.warning(
                    "Smogon usage stats unavailable (hook stays silent this "
                    "battle): %s",
                    e,
                )
        return self._stats

    @property
    def stats_format(self) -> str:
        """The real ladder format for usage stats — never the hidden id."""
        fmt = self.config.smogon_stats_format
        if fmt:
            return fmt
        battle = self.config.battle_format
        if battle.endswith("hidden"):
            return battle[: -len("hidden")]  # gen9vgc2025regihidden -> gen9vgc2025regi
        return battle
