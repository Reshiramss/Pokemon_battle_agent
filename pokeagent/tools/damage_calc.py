"""Damage calculation tool over poke-env's gen9 calculator.

Wraps ``poke_env.calc.damage_calc_gen9.calculate_damage``. Opponent stats
are often unknown (random battles); missing stats are estimated with the
standard even spread (85 EVs / 31 IVs / neutral nature) and the assumption
is labeled in the result. The calculator handles type effectiveness, STAB,
boosts, items and abilities to the extent battle state reveals them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from poke_env.battle import Move
from poke_env.calc.damage_calc_gen9 import calculate_damage

from ..knowledge.static import normalize_name
from .base import Tool, ToolContext, ToolError

if TYPE_CHECKING:
    from poke_env.battle import AbstractBattle, Pokemon

STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]


class DamageCalcTool(Tool):
    name = "damage_calc"
    description = (
        "Calculate the damage roll range (min-max, in HP points) of one move "
        "between two Pokemon currently in this battle, either direction. "
        "Identify Pokemon by species or nickname (e.g. 'Furret', 'Charizard'). "
        "Handles type effectiveness and STAB; unknown opponent stats are "
        "estimated and flagged in the result."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "attacker": {
                "type": "string",
                "description": "Name (species or nickname) of the attacking Pokemon.",
            },
            "defender": {
                "type": "string",
                "description": "Name (species or nickname) of the defending Pokemon.",
            },
            "move": {
                "type": "string",
                "description": "Name of the move being calculated.",
            },
            "critical": {
                "type": "boolean",
                "default": False,
                "description": "Assume the move lands as a critical hit.",
            },
        },
        "required": ["attacker", "defender", "move"],
    }

    async def execute(
        self,
        ctx: ToolContext,
        attacker: str = "",
        defender: str = "",
        move: str = "",
        critical: bool = False,
    ) -> str:
        battle: AbstractBattle = ctx.battle
        if not (attacker and defender and move):
            raise ToolError("attacker, defender and move are all required")

        atk = self._find(ctx, attacker)
        if atk is None:
            raise ToolError(f"attacker '{attacker}' not found in this battle")
        dfn = self._find(ctx, defender)
        if dfn is None:
            raise ToolError(f"defender '{defender}' not found in this battle")

        move_obj = self._resolve_move(ctx, atk[1], move)
        atk_estimated = self._ensure_stats(atk[1], ctx)
        dfn_estimated = self._ensure_stats(dfn[1], ctx)

        missing = [
            f"{p.species}:{k}"
            for p in (atk[1], dfn[1])
            for k, v in (p.stats or {}).items()
            if v is None
        ]
        if missing:
            raise ToolError(
                f"stats still unknown after estimation: {', '.join(missing)}"
            )

        try:
            min_dmg, max_dmg = calculate_damage(
                atk[0],
                dfn[0],
                move_obj,
                battle,
                is_critical=critical,
            )
        except Exception as e:
            raise ToolError(
                f"damage calc failed for {move_obj.id} "
                f"({atk[1].species} -> {dfn[1].species}): {e}"
            ) from e

        try:
            max_hp = dfn[1].max_hp
            percent = (
                f" ({min_dmg / max_hp * 100:.0f}-{max_dmg / max_hp * 100:.0f}%"
                f" of its {max_hp} HP)"
            )
        except Exception:
            percent = ""

        flags = []
        if atk_estimated:
            flags.append("attacker stats estimated")
        if dfn_estimated:
            flags.append("defender stats estimated")
        if critical:
            flags.append("assumed critical hit")
        note = (" Assumptions: " + "; ".join(flags) + ".") if flags else ""
        return (
            f"{move_obj.id}: {atk[1].species} -> {dfn[1].species}: "
            f"{min_dmg}-{max_dmg} damage{percent}.{note}"
        )

    # --- internals -----------------------------------------------------------

    def _find(self, ctx: ToolContext, name: str):
        """Resolve a name to (identifier, Pokemon) among both teams.

        The identifier is the battle-team dict key (e.g. 'p1: Nickname'),
        which is exactly what ``battle.get_pokemon`` expects.
        """
        battle = ctx.battle
        query = name.strip()
        if ":" in query:
            query = query.split(":", 1)[1].strip()
        norm = normalize_name(query)
        for team in (battle.team, battle.opponent_team):
            for key, p in team.items():
                if (
                    normalize_name(p.species) == norm
                    or normalize_name(p.name) == norm
                ):
                    return key, p
        return None

    def _resolve_move(self, ctx: ToolContext, pokemon: "Pokemon", move: str) -> Move:
        """Known move of the pokemon, else any legal move by name."""
        key = normalize_name(move)
        known = (pokemon.moves or {}).get(key)
        if known is not None:
            return known
        from poke_env.data import GenData

        gen_data = GenData.from_format(ctx.battle.format)
        if key in gen_data.moves:
            return Move(key, gen=gen_data.gen)
        known_moves = sorted(pokemon.moves) if pokemon.moves else []
        raise ToolError(
            f"move '{move}' not recognized for {pokemon.species}"
            + (f" (known moves: {known_moves})" if known_moves else "")
        )

    def _ensure_stats(self, pokemon: "Pokemon", ctx: ToolContext) -> bool:
        """Fill missing stats with even-spread estimates.

        Returns True if any stat was filled (so the result can flag the
        assumption). Pokemon uses __slots__, so the flag cannot live on the
        Pokemon object itself.
        """
        stats = pokemon.stats or {}
        if not any(v is None for v in stats.values()):
            return False
        if ctx.kb is None:
            return False
        estimates = ctx.kb.estimate_stats(pokemon.species, pokemon.level)
        if not estimates:
            return False
        filled = False
        for key, value in stats.items():
            if value is None and key in estimates:
                pokemon.stats[key] = estimates[key]
                filled = True
        return filled
