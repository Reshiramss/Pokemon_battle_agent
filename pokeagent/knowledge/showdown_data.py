"""Static knowledge loaded from the local Pokemon Showdown checkout.

The data lives as JSON files under ``pokeagent/data/showdown/``, exported
once from the compiled Showdown tables (see
``scripts/export_showdown_data.js`` — rerun it after updating the
pokemon-showdown checkout). This is fresher and richer than what poke-env's
GenData exposes: move descriptions, tiers, aliases, flavor text.

Everything is loaded lazily, once per process, into plain dicts — lookups
are CPU-cheap and safe inside ``choose_move``. If the JSON bundle is
missing, :func:`get_showdown_data` returns None and callers fall back to
their GenData-based path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .static import normalize_name

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "showdown"

# Showdown's damageTaken encodes the multiplier as an index:
# 0 = 1x, 1 = 2x (weak), 2 = 0.5x (resist), 3 = 0x (immune).
_DAMAGE_TAKEN = {0: 1.0, 1: 2.0, 2: 0.5, 3: 0.0}

_FORM_SUFFIXES = ("gmax", "mega", "megax", "megay", "primal", "totem", "eternal")


def _load(name: str) -> dict[str, Any]:
    with (_DATA_DIR / f"{name}.json").open(encoding="utf-8") as f:
        return json.load(f)


def _load_optional(name: str) -> dict[str, Any]:
    path = _DATA_DIR / f"{name}.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _has_cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


def _fuzzy_cjk_match(index: dict[str, str], query: str) -> str | None:
    """Best official CJK name sharing characters with the query.

    Catches homophone typos ("大纽拉" -> "大狃拉"): score by shared
    characters, require the winner to cover all but one query char and to
    be unambiguous. Returns the table key, or None.
    """
    q_chars = set(query)
    if len(q_chars) < 2:
        return None
    best_score = 0
    winners: list[str] = []
    for zh_name, key in index.items():
        shared = len(q_chars & set(zh_name))
        if shared > best_score:
            best_score, winners = shared, [key]
        elif shared == best_score:
            winners.append(key)
    if best_score >= max(2, len(q_chars) - 1) and len(winners) == 1:
        return winners[0]
    return None


class ShowdownData:
    """All static tables from the Showdown data dump, with lookup helpers."""

    def __init__(self) -> None:
        self.moves: dict[str, Any] = _load("moves")
        self.pokedex: dict[str, Any] = _load("pokedex")
        self.typechart: dict[str, Any] = _load("typechart")
        self.aliases: dict[str, str] = _load("aliases")
        self.formats: dict[str, Any] = _load("formats")
        self.text_moves: dict[str, Any] = _load("text_moves")
        self.text_pokedex: dict[str, Any] = _load("text_pokedex")
        # Simplified-Chinese official names (may be absent in older bundles)
        self._zh: dict[str, dict[str, str]] = {}
        for table_name, json_name in (("moves", "zh_moves"), ("pokedex", "zh_pokedex")):
            index: dict[str, str] = {}
            for key, entry in _load_optional(json_name).items():
                if isinstance(entry, dict) and entry.get("name"):
                    index[entry["name"]] = key
            self._zh[table_name] = index
        # display name (normalized) -> table key, for alias resolution and
        # lookups by display name ("Nidoran-F" -> "nidoranf")
        self._by_name: dict[str, dict[str, str]] = {}
        for table_name, table in (("moves", self.moves), ("pokedex", self.pokedex)):
            index: dict[str, str] = {}
            for key, entry in table.items():
                if isinstance(entry, dict) and entry.get("name"):
                    index[normalize_name(entry["name"])] = key
            self._by_name[table_name] = index

    # --- lookups -----------------------------------------------------------

    def _resolve(self, table_name: str, name: str) -> str | None:
        """Name (id / display name / alias / Chinese) -> table key, or None."""
        table = getattr(self, table_name)
        if _has_cjk(name):
            zh = self._zh.get(table_name) or {}
            key = zh.get(name.strip())
            if key is not None:
                return key
            key = _fuzzy_cjk_match(zh, name.strip())
            if key is not None:
                return key
            return None  # CJK input never reaches the id/alias paths
        norm = normalize_name(name)
        if not norm:
            return None
        if norm in table:
            return norm
        alias = self.aliases.get(norm)
        if alias is not None:
            by_name = self._by_name[table_name]
            key = by_name.get(normalize_name(alias))
            if key is not None:
                return key
        key = self._by_name[table_name].get(norm)
        if key is not None:
            return key
        for suffix in _FORM_SUFFIXES:  # "charizardgmax" -> "charizard"
            if norm.endswith(suffix):
                base = norm[: -len(suffix)]
                if base in table:
                    return base
        return None

    def move(self, name: str) -> dict[str, Any] | None:
        key = self._resolve("moves", name)
        return self.moves.get(key) if key else None

    def move_text(self, name: str) -> dict[str, Any] | None:
        key = self._resolve("moves", name)
        return self.text_moves.get(key) if key else None

    def species(self, name: str) -> dict[str, Any] | None:
        key = self._resolve("pokedex", name)
        return self.pokedex.get(key) if key else None

    def species_text(self, name: str) -> dict[str, Any] | None:
        key = self._resolve("pokedex", name)
        return self.text_pokedex.get(key) if key else None

    def tier(self, name: str) -> str | None:
        """Competitive tier (OU/UU/.../ZU/LC/NFE) or None if unknown."""
        key = self._resolve("pokedex", name)
        entry = self.formats.get(key or normalize_name(name))
        return entry.get("tier") if isinstance(entry, dict) else None

    # --- type chart --------------------------------------------------------

    @property
    def types(self) -> list[str]:
        """The real type keys of the chart, capitalized for display."""
        return sorted(t.capitalize() for t in self.typechart)

    def type_multiplier(self, attacking: str, defending: list[str]) -> float:
        """Multiplier of one attacking type vs one or two defending types.

        Raises ValueError on an unknown attacking or defending type, so the
        tool layer can surface a clear error instead of a silent 1.0.
        """
        atk_norm = normalize_name(str(attacking))
        atk_key = atk_norm.capitalize()  # damageTaken keys are title case
        multiplier = 1.0
        for d in defending:
            def_norm = normalize_name(str(getattr(d, "name", d)))
            row = self.typechart.get(def_norm)
            if row is None:
                raise ValueError(f"unknown defending type '{d}'")
            if atk_key not in row["damageTaken"]:
                raise ValueError(f"unknown attacking type '{attacking}'")
            multiplier *= _DAMAGE_TAKEN[int(row["damageTaken"][atk_key])]
        return multiplier


_INSTANCE: ShowdownData | None = None


def get_showdown_data() -> ShowdownData | None:
    """Process-wide singleton; None if the exported JSON bundle is absent."""
    global _INSTANCE
    if _INSTANCE is None:
        if not (_DATA_DIR / "pokedex.json").exists():
            return None
        _INSTANCE = ShowdownData()
    return _INSTANCE
