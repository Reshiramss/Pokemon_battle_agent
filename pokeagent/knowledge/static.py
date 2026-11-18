"""Static knowledge base over poke-env's bundled data files.

Thin async-friendly accessors over ``GenData`` (moves / pokedex / type chart /
learnsets) plus the opponent-stat estimator used by the damage-calc tool.
All lookups are CPU-cheap, in-memory, and safe inside ``choose_move``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from poke_env.data import GenData

from ..formats import FormatProfile, profile_for
from ..stats import estimate_opponent_stats

if TYPE_CHECKING:
    from .showdown_data import ShowdownData


def normalize_name(name: str) -> str:
    """Lowercase, strip to alphanumerics — Showdown-id-like normalization."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


class StaticKB:
    """Per-format static knowledge. Built once per agent (single-format player).

    ``data`` is the format's Showdown JSON bundle (standard or a mod overlay
    merged over it); GenData remains the fallback when the bundle is absent.
    """

    def __init__(
        self,
        battle_format: str,
        *,
        profile: FormatProfile | None = None,
        data: "ShowdownData | None" = None,
    ):
        self.profile = profile or profile_for(battle_format)
        self.gen_data: GenData = GenData.from_format(battle_format)
        if data is not None:
            self.data = data
        else:
            # local import avoids the showdown_data -> static.normalize cycle
            from .showdown_data import get_showdown_data

            self.data = get_showdown_data(self.profile.data_key)

    # --- raw data dicts -------------------------------------------------

    def move_data(self, move_name: str) -> dict[str, Any] | None:
        if self.data is not None:
            entry = self.data.move(move_name)
            if entry is not None:
                return entry
        return self.gen_data.moves.get(normalize_name(move_name))

    def species_data(self, species: str) -> dict[str, Any] | None:
        if self.data is not None:
            entry = self.data.species(species)
            if entry is not None:
                return entry
        key = normalize_name(species)
        # try exact id, then suffix-free match for forms ("charizardgmax" vs "charizard")
        dex = self.gen_data.pokedex
        if key in dex:
            return dex[key]
        for name, data in dex.items():
            if normalize_name(name) == key:
                return data
        base = re.sub(r"(gmax|mega|x|y)$", "", key)
        if base in dex:
            return dex[base]
        return None

    def species_key(self, species: str) -> str:
        """Team-identity key: Mega/Primal forms collapse to the base species."""
        entry = self.data.species(species) if self.data is not None else None
        if isinstance(entry, dict):
            base = entry.get("baseSpecies") or entry.get("changesFrom")
            if base:
                return normalize_name(base)
        return normalize_name(species)

    def learnset(self, species: str) -> list[str]:
        """Sorted learnable move ids, or [] if unknown.

        GenData nests each entry: learnset[key] = {"learnset": [...], "encounters": [...]}.
        """
        learnsets = self.gen_data.learnset
        key = normalize_name(species)
        entry = learnsets.get(key)
        if entry is None:
            for name, data in learnsets.items():
                if normalize_name(name) == key:
                    entry = data
                    break
        moves = entry.get("learnset") if isinstance(entry, dict) else None
        return sorted(moves) if moves else []

    # --- type chart ------------------------------------------------------

    def type_effectiveness(self, attacking: str, defending: list[str]) -> float:
        """Damage multiplier of ``attacking`` type vs the defending type combo.

        GenData's chart is indexed [defender][attacker] with UPPERCASE keys
        (built from each type's ``damageTaken``), i.e. type_chart[DEF][ATK].
        """
        chart = self.gen_data.type_chart
        atk_norm = normalize_name(str(attacking))
        multiplier = 1.0
        for d in defending:
            d = getattr(d, "name", d)
            def_norm = normalize_name(str(d))
            row = next(
                (v for k, v in chart.items() if normalize_name(k) == def_norm), None
            )
            if row is None:
                continue  # unknown defending type: treat as neutral
            value = next(
                (v for k, v in row.items() if normalize_name(k) == atk_norm), None
            )
            multiplier *= float(value) if value is not None else 1.0
        return multiplier

    # --- opponent stat estimation ----------------------------------------

    def estimate_stats(
        self,
        species: str,
        level: int,
        *,
        base_stats: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        """Estimate stats for an opponent with unknown spread.

        Standard: random-battle-style even spread (85 EVs / 31 IVs / neutral).
        Champions: 11 stat points per stat (66 total) and no level/IV term.
        ``base_stats`` should be the live Pokémon's base stats when available
        so a Mega form's new spread is used instead of the base species.
        Returns {"hp":..,"atk":..,"def":..,"spa":..,"spd":..,"spe":..} or {}.
        """
        if base_stats is None:
            entry = self.species_data(species) or {}
            base_stats = entry.get("baseStats") if isinstance(entry, dict) else None
        if not base_stats:
            return {}
        return estimate_opponent_stats(
            self.profile,
            base_stats,
            level,
            species=normalize_name(species),
        )
