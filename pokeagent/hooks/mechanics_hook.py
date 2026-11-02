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
        for weather in ctx.battle.weather:
            line = kb.context_line("weather", weather.name)
            if line:
                lines.append(line)
        for field in ctx.battle.fields:
            line = kb.context_line("fields", field.name)
            if line:
                lines.append(line)
        return "\n".join(lines) if lines else None
