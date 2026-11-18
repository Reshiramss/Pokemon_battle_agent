"""Champions stat formula, standard parity and opponent estimates."""

from __future__ import annotations

from poke_env.data import GenData
from poke_env.stats import compute_raw_stats

from pokeagent.formats import profile_for
from pokeagent.stats import (
    champion_raw_stats,
    estimate_opponent_stats,
    standard_raw_stats,
)

FARFETCHD_BASE = {"hp": 52, "atk": 90, "def": 55, "spa": 58, "spd": 62, "spe": 60}


def test_champion_formula_matches_showdown_fixture():
    # Farfetch'd, 11 stat points everywhere, Brave (+Atk / -Spe), level 50.
    # Expected values were computed from mods/champions statModify.
    stats = champion_raw_stats(
        FARFETCHD_BASE, [11] * 6, nature="brave"
    )
    assert stats == {
        "hp": 138,
        "atk": 133,
        "def": 86,
        "spa": 89,
        "spd": 93,
        "spe": 81,
    }


def test_champion_hp_ignores_nature():
    neutral = champion_raw_stats(FARFETCHD_BASE, [11] * 6, nature="serious")
    adamant = champion_raw_stats(FARFETCHD_BASE, [11] * 6, nature="adamant")
    assert neutral["hp"] == adamant["hp"] == 138
    assert adamant["atk"] > neutral["atk"]
    assert adamant["spa"] < neutral["spa"]


def test_standard_formula_matches_poke_env():
    evs = [85] * 6
    ivs = [31] * 6
    ours = standard_raw_stats(FARFETCHD_BASE, evs, ivs=ivs, level=50)
    reference = compute_raw_stats(
        "farfetchd", evs, ivs, 50, "serious", GenData.from_gen(9)
    )
    assert list(ours.values()) == reference


def test_champion_opponent_estimate_is_even_stat_points():
    profile = profile_for("gen9championsvgc2026regmc")
    est = estimate_opponent_stats(profile, FARFETCHD_BASE)
    assert est["hp"] == 52 + 86
    assert est["atk"] == 90 + 31
    assert est["spe"] == 60 + 31


def test_align_own_stats_uses_packed_stat_points():
    """PokeAgent fixes the level-formula stats before the first request."""
    from helpers import make_double_battle
    from poke_env.battle import Pokemon
    from poke_env.teambuilder import Teambuilder

    from pokeagent.knowledge.static import StaticKB
    from pokeagent.player import PokeAgent
    from pokeagent.teams import BUILTIN_TEAMS_DIR, TeamLibrary

    packed = TeamLibrary(BUILTIN_TEAMS_DIR).get("Champions A").packed
    sets = {s.species or s.nickname: s for s in Teambuilder.parse_packed_team(packed)}
    battle = make_double_battle()
    venusaur = Pokemon(9, teambuilder=sets["Venusaur"])
    battle._team = {"p1: Venusaur": venusaur}
    battle._active_pokemon = {"p1a": venusaur}

    class Dummy:
        pass

    agent = Dummy()
    agent.profile = profile_for("gen9championsvgc2026regmchidden")
    agent.kb = StaticKB("gen9championsvgc2026regmchidden")
    agent._team_sets = PokeAgent._parse_team_sets(packed)
    PokeAgent._align_own_stats(agent, battle)

    assert venusaur.stats == {
        "hp": 80 + 11 + 75,
        "atk": 82 + 11 + 20,
        "def": 83 + 11 + 20,
        "spa": 100 + 11 + 20,
        "spd": 100 + 11 + 20,
        "spe": 80 + 11 + 20,
    }


def test_standard_opponent_estimate_uses_even_evs():
    profile = profile_for("gen9vgc2025regihidden")
    est = estimate_opponent_stats(profile, FARFETCHD_BASE, level=50)
    ref = compute_raw_stats(
        "farfetchd", [85] * 6, [31] * 6, 50, "serious", GenData.from_gen(9)
    )
    assert list(est.values()) == ref