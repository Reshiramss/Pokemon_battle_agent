"""Tests for the Showdown data bundle and the upgraded static-KB tools.

Requires the exported JSON bundle under pokeagent/data/showdown/ (run
``node scripts/export_showdown_data.js <dist-data-dir> <out-dir>`` once).
The GenData fallback path of the same tools is covered by test_tools.py
via a mocked-out bundle check.
"""

from __future__ import annotations

import asyncio

import pytest
from helpers import make_battle, make_ctx

from pokeagent.knowledge.showdown_data import get_showdown_data
from pokeagent.tools.ability_info import AbilityInfoTool
from pokeagent.tools.base import ToolError
from pokeagent.tools.static_kb import (
    MoveInfoTool,
    SpeciesInfoTool,
    TypeInfoTool,
    static_kb_tools,
)

sd = get_showdown_data()

needs_bundle = pytest.mark.skipif(
    sd is None, reason="showdown data bundle not exported yet"
)


@pytest.fixture(scope="module")
def kb():
    from pokeagent.knowledge.static import StaticKB

    return StaticKB("gen9randombattle")


# --- data bundle -------------------------------------------------------------


@needs_bundle
def test_type_multiplier_directions():
    # damageTaken semantics: 0=1x, 1=2x, 2=0.5x, 3=0x
    assert sd.type_multiplier("Water", ["Fire"]) == 2.0
    assert sd.type_multiplier("Fire", ["Water"]) == 0.5
    assert sd.type_multiplier("Normal", ["Ghost"]) == 0.0
    assert sd.type_multiplier("Water", ["Ground", "Rock"]) == 4.0
    # the pair that historically catches manual charts: Fire IS 2x vs Bug
    assert sd.type_multiplier("Fire", ["Bug"]) == 2.0
    assert sd.type_multiplier("Fighting", ["Bug"]) == 0.5


@needs_bundle
def test_type_multiplier_rejects_unknown_types():
    with pytest.raises(ValueError):
        sd.type_multiplier("NotAType", ["Fire"])
    with pytest.raises(ValueError):
        sd.type_multiplier("Water", ["NotAType"])


@needs_bundle
def test_alias_and_display_name_resolution():
    assert sd.species("Nidoran-F")["name"] == "Nidoran-F"
    assert sd.species("Farfetch'd")["num"] == 83  # name stored with U+2019
    assert sd.species("Charizard-Gmax")["name"] == "Charizard-Gmax"
    assert sd.tier("Furret") == "ZU"
    assert sd.tier("Garchomp") in ("OU", "UUBL", "Uber")


@needs_bundle
def test_chinese_name_resolution():
    # exact official Chinese names
    assert sd.species("大狃拉")["name"] == "Sneasler"
    assert sd.species("狃拉")["name"] == "Sneasel"
    assert sd.species("玛狃拉")["name"] == "Weavile"
    assert sd.move("喷射火焰")["name"] == "Flamethrower"
    assert sd.move("近身战")["name"] == "Close Combat"
    # homophone typo: 大纽拉 (wrong 纽) must still resolve via char-overlap
    assert sd.species("大纽拉")["name"] == "Sneasler"
    assert sd.tier("大纽拉") == "Uber"
    # garbage CJK must not silently match something unrelated
    assert sd.species("甲乙丙丁") is None


# --- tools over the bundle ----------------------------------------------------


@needs_bundle
def test_move_info_includes_description():
    result = asyncio.run(MoveInfoTool().execute(None, move="Flamethrower"))
    assert "power 90" in result
    assert "secondary: 10% chance to burn" in result
    assert "Has a 10% chance to burn the target" in result


@needs_bundle
def test_move_info_unknown():
    with pytest.raises(ToolError):
        asyncio.run(MoveInfoTool().execute(None, move="Splash Dance"))


@needs_bundle
def test_ability_lookup_and_description():
    result = asyncio.run(AbilityInfoTool().execute(None, ability="Intimidate"))
    assert result.startswith("Intimidate")
    assert "lowers the Attack of opponents by 1 stage" in result
    # Chinese official name resolves through the zh index
    assert sd.ability("威吓")["name"] == "Intimidate"
    # unknown abilities raise, not silently match
    with pytest.raises(ToolError):
        asyncio.run(AbilityInfoTool().execute(None, ability="NotAnAbility"))


@needs_bundle
def test_ability_tool_registered_in_default_set():
    names = [t.name for t in static_kb_tools(None)]
    assert "ability_info" in names
    assert "move_info" in names and "species_info" in names


@needs_bundle
def test_type_matchup_tool_validates_types():
    result = asyncio.run(
        TypeInfoTool().execute(
            None, attacking_type="Water", defending_types=["Ground", "Rock"]
        )
    )
    assert "x4" in result and "double super effective" in result
    with pytest.raises(ToolError) as e:
        asyncio.run(
            TypeInfoTool().execute(
                None, attacking_type="Waterr", defending_types=["Rock"]
            )
        )
    assert "Valid types" in str(e.value)


@needs_bundle
def test_species_info_fuller_output():
    result = asyncio.run(SpeciesInfoTool().execute(None, species="Furret"))
    # base stats + BST + abilities + tier + evo line
    assert "HP 85" in result and "SPE 90" in result
    assert "BST 415" in result
    assert "Frisk (hidden)" in result
    assert "Tier: ZU" in result
    assert "Evolves from Sentret" in result
    assert "Fully evolved" in result


@needs_bundle
def test_species_info_evolves_into():
    result = asyncio.run(SpeciesInfoTool().execute(None, species="Sentret"))
    assert "Evolves into Furret" in result
    assert "Fully evolved" not in result


@needs_bundle
def test_species_info_chinese_query_with_typo():
    result = asyncio.run(SpeciesInfoTool().execute(None, species="大纽拉"))
    assert "Sneasler" in result
    assert "Fighting/Poison" in result
    assert "at level" not in result  # item-based evo has no level


@needs_bundle
def test_species_info_unknown():
    with pytest.raises(ToolError):
        asyncio.run(SpeciesInfoTool().execute(None, species="notapokemonnope"))


# --- GenData fallback (bundle monkeypatched away) ----------------------------


def test_fallback_to_gendata_when_bundle_missing(kb, monkeypatch):
    """With no showdown bundle the tools must degrade, not break."""
    import pokeagent.tools.static_kb as mod

    monkeypatch.setattr(mod, "get_showdown_data", lambda: None)
    ctx = make_ctx(make_battle(), kb)

    move = asyncio.run(MoveInfoTool(kb).execute(ctx, move="Flamethrower"))
    assert "power 90" in move and "Fire" in move

    matchup = asyncio.run(
        TypeInfoTool(kb).execute(ctx, attacking_type="Water", defending_types=["Fire"])
    )
    assert "x2" in matchup

    species = asyncio.run(SpeciesInfoTool(kb).execute(ctx, species="Furret"))
    assert "HP 85" in species and "Frisk" in species
