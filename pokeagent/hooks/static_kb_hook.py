"""Pre-decision hook: inject the turn's type matchups + revealed moves.

Pure-CPU over the static knowledge base; runs well inside the hook budget.
This precomputes what the model would otherwise spend tool calls on, keeping
the critical path fast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..context.records import is_doubles
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
    """Offense/defense type matrix for the active mons plus revealed moves."""

    name = "type_matchups"

    def __init__(self, max_moves: int = 8):
        self.max_moves = max_moves

    async def run(self, ctx: ToolContext) -> str | None:
        battle, kb = ctx.battle, ctx.kb
        if kb is None:
            return None
        if is_doubles(battle):
            lines = self._doubles(battle, kb)
        else:
            lines = self._singles(battle, kb)

        if not lines:
            return None  # no actives to analyze — dormant, as before

        # --- revealed moves of the whole opponent team (shared) ---
        revealed = []
        for mon in battle.opponent_team.values():
            if mon.moves:
                names = ", ".join(m.id for m in mon.moves.values())
                revealed.append(f"{mon.species}: {names}")
        if revealed:
            lines.append("Opponent revealed moves so far: " + "; ".join(revealed))

        return "\n".join(lines)

    # --- singles (unchanged behavior) ---------------------------------------

    def _singles(self, battle, kb):
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if me is None or opp is None:
            return []

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

        return lines

    # --- doubles -------------------------------------------------------------

    def _doubles(self, battle, kb):
        """Cross matrix: each of our moves vs both foes, defense reversed.

        Deliberately terse — the whole hook fragment is capped at 1500
        chars by HookSystem, so per-move lines report both foes at once.
        """
        actives = [p for p in battle.active_pokemon if p is not None and not p.fainted]
        foes = [
            p for p in battle.opponent_active_pokemon
            if p is not None and not p.fainted
        ]
        if not actives or not foes:
            return []
        foe_types = {p.species: [_tname(t) for t in p.types] for p in foes}
        my_types = {p.species: [_tname(t) for t in p.types] for p in actives}
        lines: list[str] = []

        # --- offense: each active slot's moves vs each foe ---
        for slot, mon in enumerate(battle.active_pokemon):
            if mon is None or mon.fainted:
                continue
            moves = list(battle.available_moves[slot] or [])
            if not moves:
                continue
            lines.append(f"Offense — slot {slot + 1} {mon.species}:")
            seen: set[str] = set()
            for move in moves[: self.max_moves]:
                key = normalize_name(getattr(move, "id", str(move)))
                if key in seen:
                    continue
                seen.add(key)
                vs = " / ".join(
                    f"{f.species} {_label(kb.type_effectiveness(_tname(move.type), foe_types[f.species]))}"
                    for f in foes
                )
                lines.append(f"- {move.id} ({_tname(move.type)}): {vs}")

        # --- defense: each foe's revealed moves vs each of our actives ---
        for foe in foes:
            opp_moves = list(foe.moves.values())
            if not opp_moves:
                continue
            lines.append(f"Defense — {foe.species}'s revealed moves:")
            for move in opp_moves[: self.max_moves]:
                vs = " / ".join(
                    f"{m.species} {_label(kb.type_effectiveness(_tname(move.type), my_types[m.species]))}"
                    for m in actives
                )
                lines.append(f"- {move.id} ({_tname(move.type)}): {vs}")

        return lines
