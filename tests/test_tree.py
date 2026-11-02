"""Tests for BattleKB, the per-battle per-Pokemon knowledge tree."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from helpers import make_ctx, make_double_battle, make_kb
from poke_env.battle import PokemonType, Status
from poke_env.battle.move import MoveSet

from pokeagent.hooks.battle_kb_hook import BattleKBHook
from pokeagent.knowledge.static import StaticKB
from pokeagent.knowledge.tree import BattleKB
from test_preview import make_preview_double_battle


def _kb():
    return StaticKB("gen9vgc2025regihidden")


def test_build_binds_static_data_both_sides():
    battle = make_preview_double_battle()
    # the preview fixture carries no stat spread; real preview requests do
    battle.team["p1: Furret"].stats = {"hp": 180, "atk": 76, "def": 64, "spa": 45, "spe": 90}
    # the fixture pre-reveals tackle on every mon; a real preview shows no
    # opponent moves yet, so strip the foes' to test the stub shape
    for mon in battle.opponent_team.values():
        mon._moves = MoveSet({})
    tree = BattleKB()
    tree.build(battle, _kb())

    # own side: full detail (stats, moves, known ability)
    mine = tree.us["furret"]
    assert mine.side == "us"
    assert mine.stats is not None and mine.stats["spe"] > 0
    assert mine.moves == ["tackle"]
    assert mine.possible_abilities  # bound from the static dex

    # opponent side: species stubs — identity data, nothing revealed
    foe = tree.them["incineroar"]
    assert foe.side == "them"
    assert foe.types
    assert foe.base_stats and foe.base_stats["spe"] > 0
    assert foe.possible_abilities  # every possible ability, unknown which
    assert foe.revealed_moves == []
    assert foe.known_ability is None
    assert foe.item is None

    # the fixture mirrors both teams, so query prefers the own-side node;
    # a species only on the opponent resolves to theirs
    assert tree.query("miraidon") is tree.us["miraidon"]
    del tree.us["miraidon"]
    assert tree.query("miraidon") is tree.them["miraidon"]


def test_refresh_binds_reveals():
    battle = make_double_battle()  # foes have tackle revealed
    tree = BattleKB()
    tree.build(battle, _kb())
    tree.refresh(battle, _kb())

    foe = tree.them["torterra"]
    assert foe.revealed_moves == ["tackle"]
    assert foe.fainted is False

    # a new reveal on a live object binds back into the same node
    live = battle._opponent_active_pokemon["p2a"]
    live._ability = "Intimidate"
    live._item = "leftovers"
    live._terastallized_type = PokemonType.from_name("Fire")
    live._terastallized = True
    tree.refresh(battle, _kb())
    foe = tree.them["torterra"]
    assert foe.known_ability == "Intimidate"
    assert foe.item == "leftovers"
    assert foe.tera_type == "FIRE"
    assert foe.terastallized is True

    # fainted state refreshes too (poke-env marks fainting via the FNT status)
    live._status = Status.FNT
    tree.refresh(battle, _kb())
    assert tree.them["torterra"].fainted is True


def test_refresh_introduces_unseen_species():
    battle = make_double_battle()
    tree = BattleKB()
    tree.refresh(battle, _kb())  # no build first — refresh must still work
    assert tree.query("gyarados") is not None


def test_json_round_trip():
    battle = make_double_battle()
    tree = BattleKB()
    tree.build(battle, _kb())
    tree.refresh(battle, _kb())

    clone = BattleKB.from_json(tree.to_json())
    assert clone.query("furret") is not None
    assert clone.them["incineroar"].revealed_moves == ["tackle"]
    assert clone.query("furret").stats == tree.query("furret").stats


def test_render_lines():
    battle = make_double_battle()
    tree = BattleKB()
    tree.build(battle, _kb())
    tree.refresh(battle, _kb())
    line = tree.them["incineroar"].render()
    assert "incineroar" in line
    assert "L50" in line
    assert "abilities:" in line  # possible abilities listed while unknown
    assert "moves: tackle" in line


def test_hook_injects_opponent_actives_only():
    battle = make_double_battle()
    tree = BattleKB()
    tree.build(battle, _kb())
    tree.refresh(battle, _kb())
    ctx = make_ctx(battle, make_kb())
    ctx.context = SimpleNamespace(battle_kb=tree)

    fragment = asyncio.run(BattleKBHook().run(ctx))
    assert fragment is not None
    assert "torterra" in fragment and "incineroar" in fragment
    assert "furret" not in fragment  # our side is not injected


def test_hook_dormant_without_tree():
    battle = make_double_battle()
    ctx = make_ctx(battle, make_kb())
    assert asyncio.run(BattleKBHook().run(ctx)) is None
