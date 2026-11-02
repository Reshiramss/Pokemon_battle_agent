"""Pre-decision hook: inject the opponent actives' knowledge-tree nodes.

The BattleKB tree (see ``knowledge/tree.py``) carries what static data plus
reveals the agent knows per species — possible abilities, revealed moves,
confirmed ability, item, tera state. This hook surfaces the nodes of the
mons currently on the other side of the field, so the model gets the
opponent-reading without spending a tool call on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import PreDecisionHook

if TYPE_CHECKING:
    from ..tools.base import ToolContext


class BattleKBHook(PreDecisionHook):
    """One compact line per opponent active, from the live knowledge tree."""

    name = "battle_kb"

    async def run(self, ctx: ToolContext) -> str | None:
        cm = ctx.context
        tree = getattr(cm, "battle_kb", None) if cm is not None else None
        if tree is None:
            return None

        foes = ctx.battle.opponent_active_pokemon
        if not isinstance(foes, list):  # singles: a single mon or None
            foes = [foes] if foes is not None else []

        lines = []
        seen: set[str] = set()
        for mon in foes:
            if mon is None or mon.species in seen:
                continue
            seen.add(mon.species)
            node = tree.them.get(mon.species)
            if node is not None:
                lines.append(node.render())
        return "\n".join(lines) if lines else None
