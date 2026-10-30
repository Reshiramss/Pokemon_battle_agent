"""Pre-decision hook: inject the turn's type matchups + revealed moves.

Pure-CPU over the static knowledge base; runs well inside the hook budget.
This precomputes what the model would otherwise spend tool calls on, keeping
the critical path fast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..knowledge.static import normalize_name
from .base import PreDecisionHook

if TYPE_CHECKING:
    from ..tools.base import ToolContext

_LABELS = {
    0.0: "no effect",
    0.25: "x0.25 double resisted",
    0.5: "x0.5 resisted",
    1.0: "x1 neutral",
    2.0: "x2 SUPER EFFECTIVE",
    4.0: "x4 double super effective",
}


def _tname(t) -> str:
    """PokemonType enum -> clean name ('GRASS'), not the repr-ish str()."""
    return getattr(t, "name", str(t))


def _label(mult: float) -> str:
    return _LABELS.get(round(mult, 2), f"x{mult:g}")


class TypeMatchupHook(PreDecisionHook):
    """Offense/defense type matrix for the current 1v1 plus revealed moves."""

    name = "type_matchups"

    def __init__(self, max_moves: int = 8):
        self.max_moves = max_moves

    async def run(self, ctx: ToolContext) -> str | None:
        battle, kb = ctx.battle, ctx.kb
        if kb is None:
            return None
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if me is None or opp is None:
            return None

        opp_types = [_tname(t) for t in opp.types]
        my_types = [_tname(t) for t in me.types]
        lines: list[str] = []

        # --- offense: my usable moves vs the opponent active ---
        moves = list(battle.available_moves or [])
        if moves:
            lines.append(
                f"Offense — your moves vs {opp.species} ({'/'.join(opp_types)}):"
            )
            seen: set[str] = set()
            for move in moves[: self.max_moves]:
                key = normalize_name(getattr(move, "id", str(move)))
                if key in seen:
                    continue
                seen.add(key)
                mult = kb.type_effectiveness(_tname(move.type), opp_types)
                lines.append(f"- {move.id} ({_tname(move.type)}): {_label(mult)}")

        # --- defense: opponent's revealed moves vs me ---
        opp_moves = list(opp.moves.values())
        if opp_moves:
            lines.append(
                f"Defense — {opp.species}'s revealed moves vs your {me.species} "
                f"({'/'.join(my_types)}):"
            )
            for move in opp_moves[: self.max_moves]:
                mult = kb.type_effectiveness(_tname(move.type), my_types)
                lines.append(f"- {move.id} ({_tname(move.type)}): {_label(mult)}")

        # --- revealed moves of the whole opponent team ---
        revealed = []
        for mon in battle.opponent_team.values():
            if mon.moves:
                names = ", ".join(m.id for m in mon.moves.values())
                revealed.append(f"{mon.species}: {names}")
        if revealed:
            lines.append("Opponent revealed moves so far: " + "; ".join(revealed))

        return "\n".join(lines) if lines else None
