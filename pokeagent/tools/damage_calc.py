"""Damage calculation tool over poke-env's gen9 calculator.

Wraps ``poke_env.calc.damage_calc_gen9.calculate_damage``. Opponent stats
are often unknown (random battles, and every format before the opponent's
set is scouted); missing stats are estimated with the even spread (85 EVs /
31 IVs / neutral nature), used only for the duration of the calculation and
labelled in the result — nothing is written back onto the live Pokemon.

Damage is reported in HP points; percentages are taken over the defender's
**real** HP when it is ours, or over an estimated HP (prefixed ``~``) for an
opponent, because Showdown only sends opponents' HP as a percentage.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from poke_env.battle import DoubleBattle, Move
from poke_env.battle.target import Target
from poke_env.calc.damage_calc_gen9 import calculate_damage

from ..knowledge.static import normalize_name
from .base import Tool, ToolContext, ToolError

if TYPE_CHECKING:
    from poke_env.battle import AbstractBattle, Pokemon

STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]

# moves that hit every adjacent mon — they take the 0.75x spread reduction
# in doubles when two or more targets are on the field
SPREAD_TARGETS = {"ALL_ADJACENT", "ALL_ADJACENT_FOES"}


@dataclass(frozen=True)
class _ResolvedMon:
    """A Pokemon plus the identifier the calculator can resolve it by."""

    identifier: str
    pokemon: "Pokemon"
    side: str  # "us" | "them"


class DamageCalcTool(Tool):
    name = "damage_calc"
    description = (
        "Calculate the damage roll range (min-max, in HP points) of one move "
        "between two Pokemon currently in this battle, either direction — "
        "works during team preview too. Identify Pokemon by species or "
        "nickname (e.g. 'Furret', 'Charizard'). Handles type effectiveness "
        "and STAB. Opponent EVs are hidden: their stats and HP are estimated, "
        "flagged in the result, and percentages use '~N HP' for an estimated "
        "max HP."
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

        move_obj = self._resolve_move(ctx, atk.pokemon, move)

        # estimates are scoped to this call: filling them in permanently would
        # make later calls report guessed stats as if they were known
        atk_filled = self._fill_stats(atk.pokemon, ctx)
        dfn_filled = self._fill_stats(dfn.pokemon, ctx)
        # read the denominator while the estimates are still in place
        hp_stat = dfn.pokemon.stats.get("hp")
        hp_estimated = "hp" in dfn_filled
        try:
            missing = [
                f"{p.species}:{k}"
                for p in (atk.pokemon, dfn.pokemon)
                for k, v in (p.stats or {}).items()
                if v is None
            ]
            if missing:
                raise ToolError(
                    f"stats still unknown after estimation: {', '.join(missing)}"
                )
            calc_battle = self._battle_for_calc(battle, (atk, dfn))
            try:
                min_dmg, max_dmg = calculate_damage(
                    atk.identifier,
                    dfn.identifier,
                    move_obj,
                    calc_battle,
                    is_critical=critical,
                )
            except Exception as e:
                raise ToolError(
                    f"damage calc failed for {move_obj.id} "
                    f"({atk.pokemon.species} -> {dfn.pokemon.species}): {e}"
                ) from e
        finally:
            self._restore_stats(atk.pokemon, atk_filled)
            self._restore_stats(dfn.pokemon, dfn_filled)

        # Percent denominator: opponents' max_hp is Showdown's 100-point
        # percentage scale, not real HP, so use the estimated HP stat and mark
        # it with '~'. For our own Pokemon the stat is exact.
        percent = ""
        if hp_stat:
            hp_label = f"~{hp_stat}" if hp_estimated else f"{hp_stat}"
            percent = (
                f" ({min_dmg / hp_stat * 100:.0f}-{max_dmg / hp_stat * 100:.0f}%"
                f" of {hp_label} HP)"
            )

        flags = []
        if atk_filled:
            flags.append("attacker stats estimated")
        if dfn_filled:
            flags.append("defender stats estimated")
        if hp_estimated:
            flags.append(f"opponent HP ~{hp_stat} HP (EVs unknown)")
        if critical:
            flags.append("assumed critical hit")

        # doubles spread moves (e.g. Earthquake) hit 2+ targets at 0.75x —
        # poke-env's calculator does not apply the reduction
        spread = self._is_spread_hit(battle, atk.pokemon, move_obj)
        if spread:
            min_dmg = max(1, round(min_dmg * 0.75))
            max_dmg = max(1, round(max_dmg * 0.75))
            flags.append("spread-adjusted estimate: x0.75, move hits 2+ targets")
        note = (" Assumptions: " + "; ".join(flags) + ".") if flags else ""
        return (
            f"{move_obj.id}: {atk.pokemon.species} -> {dfn.pokemon.species}: "
            f"{min_dmg}-{max_dmg} damage{percent}.{note}"
        )

    # --- internals -----------------------------------------------------------

    def _is_spread_hit(
        self, battle: "AbstractBattle", attacker: "Pokemon", move_obj: Move
    ) -> bool:
        """True if this move lands as a reduced-power spread hit right now."""
        if not isinstance(battle, DoubleBattle):
            return False
        target = getattr(move_obj, "target", None)
        if target is None or target.name not in SPREAD_TARGETS:
            return False
        hits = [
            p
            for p in battle.opponent_active_pokemon
            if p is not None and not p.fainted
        ]
        if target is Target.ALL_ADJACENT:
            hits += [
                p
                for p in battle.active_pokemon
                if p is not None and not p.fainted and p is not attacker
            ]
        return len(hits) >= 2

    def _find(self, ctx: ToolContext, name: str) -> _ResolvedMon | None:
        """Resolve a name to a Pokemon and a calculator-safe identifier.

        During team preview ``battle.opponent_team`` is keyed by species
        (``calyrexice``), not by identifier; passing such a key to
        ``battle.get_pokemon`` mangles it and raises. Build the canonical
        ``p2: Species`` identifier instead (resolved via an alias in
        ``_battle_for_calc``).
        """
        battle = ctx.battle
        query = name.strip()
        if ":" in query:
            query = query.split(":", 1)[1].strip()
        norm = normalize_name(query)
        for side, team in (("us", battle.team), ("them", battle.opponent_team)):
            for key, p in team.items():
                if (
                    normalize_name(p.species) == norm
                    or normalize_name(p.name) == norm
                ):
                    return _ResolvedMon(self._identifier(battle, key, p, side), p, side)
        return None

    @staticmethod
    def _identifier(battle: "AbstractBattle", key: str, pokemon: "Pokemon", side: str) -> str:
        """Canonical ``pN: Name`` identifier for a battle-team entry."""
        role = battle.player_role if side == "us" else (battle.opponent_role or "p2")
        if role and key.startswith(role + ":"):
            return key
        return f"{role or ('p2' if side == 'them' else 'p1')}: {pokemon.species}"

    def _battle_for_calc(
        self, battle: "AbstractBattle", resolved: tuple[_ResolvedMon, ...]
    ) -> "AbstractBattle":
        """Battle object where every identifier resolves to the intended mon.

        Preview-time opponent keys are species, so the canonical identifier
        is not present in the team dicts. Shallow-copy the battle, copy both
        team dicts and insert aliases — the live battle is never mutated.
        """
        aliases = []
        for mon in resolved:
            team = battle._team if mon.side == "us" else battle._opponent_team
            if mon.identifier not in team:
                aliases.append(mon)
        if not aliases:
            return battle
        clone = copy.copy(battle)
        clone._team = dict(battle._team)
        clone._opponent_team = dict(battle._opponent_team)
        for mon in aliases:
            team = clone._team if mon.side == "us" else clone._opponent_team
            team[mon.identifier] = mon.pokemon
        return clone

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

    def _fill_stats(self, pokemon: "Pokemon", ctx: ToolContext) -> dict[str, int | None]:
        """Temporarily fill missing stats with even-spread estimates.

        Returns the original values (``None``) per filled key so the caller
        can restore them in ``finally``. Keeping the mutation call-scoped is
        what makes the "estimated" flag truthful on every later call.
        """
        stats = pokemon.stats or {}
        if not any(v is None for v in stats.values()):
            return {}
        if ctx.kb is None:
            return {}
        estimates = ctx.kb.estimate_stats(pokemon.species, pokemon.level)
        if not estimates:
            return {}
        originals: dict[str, int | None] = {}
        for key, value in stats.items():
            if value is None and key in estimates:
                originals[key] = value
                stats[key] = estimates[key]
        return originals

    @staticmethod
    def _restore_stats(pokemon: "Pokemon", originals: dict[str, int | None]) -> None:
        for key, value in originals.items():
            pokemon.stats[key] = value
