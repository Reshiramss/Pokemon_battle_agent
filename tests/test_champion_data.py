"""Champion static-data overlay: merged moves/items/tiers, isolated caches."""

from __future__ import annotations

import asyncio

import pytest
from helpers import make_ctx, make_double_battle

from pokeagent.knowledge.showdown_data import get_showdown_data
from pokeagent.knowledge.static import StaticKB
from pokeagent.tools.damage_calc import DamageCalcTool

sd = get_showdown_data("showdown_champions")
base = get_showdown_data("showdown")

needs_bundle = pytest.mark.skipif(
    sd is None or base is None, reason="showdown data bundles not exported"
)


@needs_bundle
def test_overlay_merges_move_overrides():
    assert sd.move("Anchor Shot")["basePower"] == 90
    assert sd.move("Snap Trap")["type"] == "Steel"
    assert sd.move("Growth")["type"] == "Grass"
    # inherited fields survive the overlay merge
    assert sd.move("Anchor Shot")["name"] == "Anchor Shot"


@needs_bundle
def test_champion_pp_cap():
    for move in ("Taunt", "Protect", "Flamethrower", "Anchor Shot", "Tackle"):
        assert sd.move(move)["pp"] <= 20


@needs_bundle
def test_items_bundle_and_mega_stone():
    item = sd.item("Venusaurite")
    assert item is not None
    assert item["megaStone"]["Venusaur"] == "Venusaur-Mega"
    assert sd.item("Black Glasses")["name"] == "Black Glasses"


@needs_bundle
def test_champion_tiers_differ_from_standard():
    # Tornadus is Past/Illegal in Champions, legal in standard Gen 9.
    assert sd.tier("Incineroar") == "RU"
    assert sd.tier("Tornadus") == "Illegal"
    assert base.tier("Tornadus") != "Illegal"


def test_team_sets_parse_for_stat_point_alignment():
    from pokeagent.player import PokeAgent
    from pokeagent.teams import BUILTIN_TEAMS_DIR, TeamLibrary

    packed = TeamLibrary(BUILTIN_TEAMS_DIR).get("Champions A").packed
    sets = PokeAgent._parse_team_sets(packed)
    assert sets["venusaur"].evs == [11] * 6
    # packed names are nicknames when the species field is empty
    assert "farfetchd" in sets and "raichualola" in sets


@needs_bundle
def test_champion_damage_calc_estimates_with_stat_points():
    kb = StaticKB("gen9championsvgc2026regmc")
    battle = make_double_battle()
    result = asyncio.run(
        DamageCalcTool().execute(
            make_ctx(battle, kb),
            attacker="Furret",
            defender="Torterra",
            move="flamethrower",
        )
    )
    assert "damage" in result
    assert "defender stats estimated" in result
    # Torterra base HP 95 + 11 stat points + 75 offset
    assert "~181 HP" in result


@needs_bundle
def test_mega_form_maps_to_base_team_node():
    from pokeagent.knowledge.tree import BattleKB

    kb = BattleKB()
    kb.data = sd
    # poke-env rewrites the species on detailschange; the team identity must
    # stay the base species so the KB does not grow a second node.
    assert kb.key_for("venusaurmega") == "venusaur"
    assert kb.key_for("venusaur") == "venusaur"


@needs_bundle
def test_static_kb_species_key_collapses_mega_forms():
    kb = StaticKB("gen9championsvgc2026regmc")
    assert kb.species_key("venusaurmega") == "venusaur"
    assert kb.species_key("venusaur") == "venusaur"


@needs_bundle
def test_bundles_are_isolated():
    assert base.move("Anchor Shot")["basePower"] != 90
    assert base.move("Tackle")["pp"] > 20
    assert sd.bundle == "showdown_champions"
    assert base.bundle == "showdown"