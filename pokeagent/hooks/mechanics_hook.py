"""Pre-decision hook: core doubles rules plus a situational reminder.

Every doubles turn injects the three lines a VGC model most often needs
(turn order, spread moves, protect). When weather or a field effect is
active, the matching one-line reminder from ``data/mechanics.json`` is
appended — "Trick Room is up, the slower mon moves first" is exactly the
kind of fact a model forgets mid-battle. Singles battles get nothing:
their prompts already stay small and the doubles lines don't apply.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..knowledge.mechanics import get_mechanics_kb
from .base import PreDecisionHook

if TYPE_CHECKING:
    from ..tools.base import ToolContext


class MechanicsHook(PreDecisionHook):
    """Three core doubles lines + weather/field situational lines."""

    name = "mechanics"

    async def run(self, ctx: ToolContext) -> str | None:
        if not isinstance(ctx.battle.opponent_active_pokemon, list):
            return None  # doubles only
        kb = get_mechanics_kb()
        if not kb.topics:
            return None

        lines = list(kb.core_doubles_lines())
        profile = getattr(ctx.kb, "profile", None)
        if profile is not None and profile.is_champions:
            if profile.uses_stat_points:
                lines.append(
                    "CHAMPIONS: stats use stat points (0-32 each, 66 total), "
                    "not EVs/IVs/level; no Tera; Mega Evolution is available "
                    "when noted; PP is capped at 20."
                )
            else:
                lines.append(
                    "CHAMPIONS: mainline level stats (Level Clause), but "
                    "Champions move/item data and PP rules apply; no Tera."
                )
        for weather in ctx.battle.weather:
            line = kb.context_line("weather", weather.name)
            if line:
                lines.append(line)
        for field in ctx.battle.fields:
            line = kb.context_line("fields", field.name)
            if line:
                lines.append(line)
        return "\n".join(lines) if lines else None
