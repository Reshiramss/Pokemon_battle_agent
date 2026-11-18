"""Per-format rules profile: standard Gen 9 vs Pokémon Champions.

One immutable value object captures every rule difference the harness has to
know about: static-data bundle, stat model (mainline level formula vs Champions
stat points), PP cap, and available gimmicks (Tera / Mega). ``PokeAgent`` is a
single-format player, so the profile is resolved once at construction and then
passed to the KB, prompt manager, tools and hooks — no process-global format
state.

Format metadata is exported from Showdown's ``Dex.formats`` into
``data/format_profiles.json`` (see ``scripts/export_showdown_data.js``). If the
file is absent, a conservative id-based heuristic keeps both modes working.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DATA_PATH = Path(__file__).resolve().parent / "data" / "format_profiles.json"

_META_CACHE: dict[str, dict[str, Any]] | None = None


def _normalize_format_id(format_id: str) -> str:
    """``"[Gen 9] ..."`` / ``id@@@rules`` -> Showdown id form."""
    raw = (format_id or "").split("@@@", 1)[0].strip().lower()
    return re.sub(r"[^a-z0-9]+", "", raw)


def _load_meta() -> dict[str, dict[str, Any]]:
    global _META_CACHE
    if _META_CACHE is None:
        try:
            with _DATA_PATH.open(encoding="utf-8") as f:
                data = json.load(f)
            _META_CACHE = data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            _META_CACHE = {}
    return _META_CACHE


@dataclass(frozen=True)
class FormatProfile:
    """Everything that differs between a standard and a Champions battle."""

    format_id: str
    mod: str = "gen9"
    data_key: str = "showdown"
    stat_model: str = "standard"  # standard | champions
    ev_limit: int | None = 510
    stat_point_cap: int | None = None
    pp_cap: int | None = None
    allows_tera: bool = True
    allows_mega: bool = False
    game_type: str | None = None
    memory_key: str | None = None
    smogon_format: str | None = None

    @property
    def is_champions(self) -> bool:
        return self.mod.startswith("champions")

    @property
    def uses_stat_points(self) -> bool:
        """True when stats are ``base + stat points + offset`` (no level/IV)."""
        return self.stat_model == "champions"

    @property
    def display_name(self) -> str:
        meta = _load_meta().get(self.format_id) or {}
        return str(meta.get("name") or self.format_id)


def _memory_key(mod: str) -> str | None:
    """Champion memories are partitioned; standard keeps the legacy root."""
    return "champions" if mod.startswith("champions") else None


def _smogon_format(format_id: str, is_champions: bool) -> str | None:
    if is_champions:
        return None  # no Champion usage snapshot yet
    return re.sub(r"hidden$", "", format_id) or None


def profile_for(format_id: str) -> FormatProfile:
    """Resolve a format id (or display name) into a :class:`FormatProfile`."""
    norm = _normalize_format_id(format_id) or str(format_id or "")
    meta = _load_meta().get(norm) or {}

    mod = str(meta.get("mod") or ("champions" if "champions" in norm else "gen9"))
    is_champions = mod.startswith("champions")

    # Champion random battles carry Level Clause Mod and therefore use the
    # mainline level-based stat formula despite living in the champions mod.
    level_clause = meta.get("levelClause")
    if level_clause is None:
        level_clause = is_champions and "random" in norm
    stat_model = "standard" if (not is_champions or level_clause) else "champions"

    game_type = meta.get("gameType")
    if not game_type:
        game_type = (
            "doubles"
            if ("doubles" in norm or "vgc" in norm)
            else "singles"
        )

    ev_limit = meta.get("evLimit")
    if ev_limit is None:
        ev_limit = 66 if is_champions else 510

    return FormatProfile(
        format_id=norm,
        mod=mod,
        data_key="showdown" if not is_champions else f"showdown_{mod}",
        stat_model=stat_model,
        ev_limit=int(ev_limit) if ev_limit is not None else None,
        stat_point_cap=32 if is_champions else None,
        pp_cap=20 if is_champions else None,
        allows_tera=not is_champions,
        allows_mega=is_champions,
        game_type=str(game_type),
        memory_key=_memory_key(mod),
        smogon_format=_smogon_format(norm, is_champions),
    )