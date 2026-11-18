"""Stat calculation for both battle modes.

Standard Gen 9 uses the mainline level formula (IVs + EVs, level, nature).
Pokémon Champions replaces it with stat points and fixed offsets:

    HP      = base + statPoints + 75
    ATK..SPE = (base + statPoints + 20) * natureMultiplier

No level and no IV term enter the Champions formula. ``evs`` in a packed
Champions team are Stat Points (0–32 per stat, total ≤ 66); the packed format
still calls the field ``EVs``.

The functions here are pure (aside from lazily reading GenData's nature table)
so they can be unit-tested without a battle and used from both the KB and the
damage calculator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

from poke_env.data import GenData

if TYPE_CHECKING:
    from .formats import FormatProfile

STAT_KEYS: tuple[str, ...] = ("hp", "atk", "def", "spa", "spd", "spe")

# Even spread assumption for a Champions opponent whose stat points are hidden
# (66 total points / 6 stats).
CHAMPION_EVEN_POINTS = 11
# Random-battle-style even spread used for standard opponents.
STANDARD_EVEN_EVS = 85
STANDARD_EVEN_IVS = 31


def _norm_nature(nature: str | None) -> str:
    raw = (nature or "").strip().lower()
    return "".join(ch for ch in raw if ch.isalnum()) or "serious"


def nature_multipliers(nature: str | None) -> dict[str, float]:
    """Nature multiplier per stat (1.0 for neutral stats)."""
    neutral = {key: 1.0 for key in STAT_KEYS}
    try:
        data = GenData.from_gen(9).natures[_norm_nature(nature)]
    except Exception:
        return neutral
    out = dict(neutral)
    for key, value in data.items():
        if key in out and isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _as_points(value: Any, *, default: int = 0) -> list[int]:
    """Normalize stat points/EVs to ``[hp, atk, def, spa, spd, spe]``."""
    if value is None:
        return [default] * 6
    if isinstance(value, dict):
        return [int(value.get(key, default) or default) for key in STAT_KEYS]
    if isinstance(value, Iterable):
        items = list(value)
        if len(items) >= 6:
            return [int(v or default) for v in items[:6]]
    return [default] * 6


def _base_table(base_stats: dict[str, Any] | None) -> dict[str, int]:
    table = base_stats or {}
    return {
        key: int(table.get(key, 0) or 0)
        for key in STAT_KEYS
    }


def standard_raw_stats(
    base_stats: dict[str, Any],
    evs: Any,
    *,
    ivs: Any = None,
    level: int = 50,
    nature: str | None = "serious",
) -> dict[str, int]:
    """Mainline Gen 9 stats from explicit base stats (level-clause formats)."""
    base = _base_table(base_stats)
    ev_list = _as_points(evs, default=0)
    iv_list = _as_points(ivs, default=STANDARD_EVEN_IVS)
    mult = nature_multipliers(nature)
    level = int(level or 50)

    out: dict[str, int] = {}
    hp_core = (ev_list[0] // 4) + iv_list[0] + 2 * base["hp"]
    out["hp"] = (hp_core * level) // 100 + level + 10
    for idx, key in enumerate(STAT_KEYS[1:], start=1):
        core = (ev_list[idx] // 4) + iv_list[idx] + 2 * base[key]
        raw = 5 + (core * level) // 100
        out[key] = int(raw * mult[key])
    return out


def champion_raw_stats(
    base_stats: dict[str, Any],
    stat_points: Any = None,
    *,
    nature: str | None = "serious",
) -> dict[str, int]:
    """Champions stats: base + stat points + fixed offset, then nature."""
    base = _base_table(base_stats)
    points = _as_points(stat_points, default=0)
    mult = nature_multipliers(nature)

    out: dict[str, int] = {"hp": base["hp"] + points[0] + 75}
    for idx, key in enumerate(STAT_KEYS[1:], start=1):
        value = base[key] + points[idx] + 20
        if mult[key] > 1.0:
            value = value * 110 // 100
        elif mult[key] < 1.0:
            value = value * 90 // 100
        out[key] = int(value)
    return out


def estimate_opponent_stats(
    profile: "FormatProfile | None",
    base_stats: dict[str, Any] | None,
    level: int = 50,
    *,
    species: str | None = None,
) -> dict[str, int]:
    """Estimate an opponent's stats when its spread is unknown.

    Champions: 11 stat points per stat, neutral nature. Standard: 85 EVs,
    31 IVs, neutral nature. ``base_stats`` should be the live Pokémon's base
    stats when available (it reflects a Mega form); otherwise ``species`` is
    looked up in GenData.
    """
    base = base_stats
    if not base and species:
        try:
            entry = GenData.from_gen(9).pokedex[species]
            base = entry.get("baseStats")
        except Exception:
            base = None
    if not base:
        return {}

    profile = profile
    if profile is not None and profile.uses_stat_points:
        return champion_raw_stats(
            base,
            [CHAMPION_EVEN_POINTS] * 6,
            nature="serious",
        )
    return standard_raw_stats(
        base,
        [STANDARD_EVEN_EVS] * 6,
        ivs=[STANDARD_EVEN_IVS] * 6,
        level=level,
        nature="serious",
    )


def own_stats(
    profile: "FormatProfile",
    pokemon: Any,
    *,
    stat_points: Any = None,
    nature: str | None = None,
) -> dict[str, int] | None:
    """Exact stats for one of our Pokémon under the profile's stat model.

    poke-env computes mainline stats from the teambuilder set before the
    server's request arrives. Champions needs the stat-point formula instead;
    reusing the live ``base_stats`` also fixes stats right after a Mega
    Evolution (poke-env updates base stats/types but not the ``stats`` dict).
    ``stat_points``/``nature`` override the live Pokémon's fields because the
    server request does not carry EVs (the agent keeps the packed team set).
    """
    if profile.uses_stat_points:
        base = getattr(pokemon, "base_stats", None)
        points = stat_points if stat_points is not None else getattr(pokemon, "evs", None)
        nat = nature if nature is not None else getattr(pokemon, "nature", None)
        if not base:
            return None
        return champion_raw_stats(base, points, nature=nat or "serious")

    stats = getattr(pokemon, "stats", None) or {}
    if all(isinstance(v, int) for v in stats.values()) and len(stats) == 6:
        return {key: int(stats[key]) for key in STAT_KEYS if key in stats}
    return None