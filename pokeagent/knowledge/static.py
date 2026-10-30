"""Static knowledge base over poke-env's bundled data files.

Thin async-friendly accessors over ``GenData`` (moves / pokedex / type chart /
learnsets) plus the opponent-stat estimator used by the damage-calc tool.
All lookups are CPU-cheap, in-memory, and safe inside ``choose_move``.
"""

from __future__ import annotations

import re
from typing import Any

from poke_env.data import GenData
from poke_env.stats import compute_raw_stats


def normalize_name(name: str) -> str:
    """Lowercase, strip to alphanumerics — Showdown-id-like normalization."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


class StaticKB:
    """Per-format static knowledge. Built once per agent (GenData is a singleton per gen)."""

    def __init__(self, battle_format: str):
        self.gen_data: GenData = GenData.from_format(battle_format)

    # --- raw data dicts -------------------------------------------------

    def move_data(self, move_name: str) -> dict[str, Any] | None:
        return self.gen_data.moves.get(normalize_name(move_name))

    def species_data(self, species: str) -> dict[str, Any] | None:
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

    def estimate_stats(self, species: str, level: int) -> dict[str, int]:
        """Estimate stats for an opponent with unknown EVs/IVs/nature.

        Uses the random-battle-style even spread (85 EVs / 31 IVs / neutral
        nature), which matches Showdown random-battle generation closely
        enough for damage *ranges*. Returns {"hp":..,"atk":..,"def":..,
        "spa":..,"spd":..,"spe":..} or {} if the species is unknown.
        """
        try:
            raw = compute_raw_stats(
                species=species,
                evs=[85] * 6,
                ivs=[31] * 6,
                level=level,
                nature="serious",
                data=self.gen_data,
            )
        except Exception:
            return {}
        if not raw or len(raw) != 6:
            return {}
        keys = ["hp", "atk", "def", "spa", "spd", "spe"]
        return dict(zip(keys, raw))
