"""P0/P1 knowledge injection: ability ids, ability effects, move effects.

Regression driver: Tornadus' Prankster makes Taunt a priority move and
Farigiraf's Armor Tail blocks opposing priority moves. Both names used to
appear in the prompt with no effect text, so the model could not connect
them. These tests pin the two hooks plus the ``[NOW]`` ability field that
makes the names visible in the first place.
"""

from __future__ import annotations

import asyncio
import difflib

from helpers import make_battle, make_ctx, make_double_battle, make_kb
from poke_env.battle import Status

from pokeagent.context.records import snapshot
from pokeagent.hooks.ability_effects_hook import AbilityEffectsHook
from pokeagent.hooks.move_effects_hook import MoveEffectsHook
from pokeagent.state import BattleState, render_observed


def _assert_parity(battle) -> BattleState:
    state = BattleState.create(battle)
    want, got = snapshot(battle), render_observed(state)
    diff = "\n".join(
        difflib.unified_diff(
            want.splitlines(), got.splitlines(), "snapshot", "render", lineterm=""
        )
    )
    assert got == want, diff
    return state


# --- P0: ability is part of the observed board ---------------------------------


def test_snapshot_renders_known_ability_and_render_parity():
    battle = make_battle()
    battle.team["p1: Furret"].ability = "runaway"

    snap = snapshot(battle)
    assert "ability:runaway" in snap
    state = _assert_parity(battle)

    # an opponent ability becomes known when the protocol reveals it
    battle.opponent_team["p2: Torterra"].ability = "overgrow"
    state.refresh(battle)
    rendered = render_observed(state)
    assert "ability:overgrow" in rendered
    assert render_observed(state) == snapshot(battle)


def test_snapshot_does_not_invent_unknown_abilities():
    battle = make_double_battle()
    assert "ability:" not in snapshot(battle)


# --- P0: ability effects hook --------------------------------------------------


def test_ability_effects_hook_prankster_vs_armor_tail():
    battle = make_double_battle()
    ours = battle.team["p1: Furret"]
    ours._species = "tornadus"
    ours.ability = "prankster"
    foe = battle.opponent_team["p2: Torterra"]
    foe._species = "farigiraf"
    foe.ability = "armortail"

    fragment = asyncio.run(AbilityEffectsHook().run(make_ctx(battle, make_kb())))
    assert fragment is not None
    assert "our tornadus has Prankster" in fragment
    assert "Status moves have priority raised by 1" in fragment
    assert "foe farigiraf has Armor Tail" in fragment
    assert "protected from opposing priority moves" in fragment


def test_ability_effects_hook_is_dormant_without_confirmed_abilities():
    battle = make_battle()  # neither side has a revealed ability
    assert asyncio.run(AbilityEffectsHook().run(make_ctx(battle, make_kb()))) is None


def test_ability_effects_hook_skips_unpicked_and_fainted():
    battle = make_double_battle()
    battle.team["p1: Furret"].ability = "prankster"
    bench = battle.team["p1: Pikachu 1"]
    bench.ability = "static"
    ctx = make_ctx(battle, make_kb())

    fragment = asyncio.run(AbilityEffectsHook().run(ctx))
    assert "prankster" in fragment.lower() and "static" in fragment.lower()

    # once the preview marked picks, an unpicked bench mon cannot act
    battle.team["p1: Furret"]._selected_in_teampreview = True
    fragment = asyncio.run(AbilityEffectsHook().run(ctx))
    assert "prankster" in fragment.lower()
    assert "static" not in fragment.lower()

    # a fainted active cannot use its ability either (poke-env's ``fainted``
    # is the FNT status, not the HP field)
    battle.team["p1: Furret"]._status = Status.FNT
    assert asyncio.run(AbilityEffectsHook().run(ctx)) is None


# --- P1: move effects hook -----------------------------------------------------


def test_move_effects_hook_available_and_revealed():
    battle = make_battle()
    ctx = make_ctx(battle, make_kb())

    fragment = asyncio.run(MoveEffectsHook().run(ctx))
    assert fragment is not None
    # first prompt of the battle: no NEW markers for already-known reveals
    assert "NEW" not in fragment
    assert "yours flamethrower: 10% chance to burn the target." in fragment
    assert "yours quickattack: Usually goes first." in fragment
    assert "foe stoneedge: High critical hit ratio." in fragment


def test_move_effects_hook_marks_new_reveals_once():
    battle = make_battle()
    ctx = make_ctx(battle, make_kb())
    hook = MoveEffectsHook()

    asyncio.run(hook.run(ctx))  # baseline

    battle.opponent_team["p2: Torterra"]._add_move("earthquake")
    second = asyncio.run(hook.run(ctx))
    assert "NEW foe move earthquake: Hits adjacent Pokemon. Double damage on Dig." in second

    third = asyncio.run(hook.run(ctx))
    assert "NEW foe move earthquake" not in third
    assert "foe earthquake: Hits adjacent Pokemon. Double damage on Dig." in third


def test_move_effects_hook_marks_new_ability_and_item_once():
    battle = make_battle()
    ctx = make_ctx(battle, make_kb())
    hook = MoveEffectsHook()
    asyncio.run(hook.run(ctx))

    foe = battle.opponent_team["p2: Torterra"]
    foe.ability = "overgrow"
    foe._item = "leftovers"
    second = asyncio.run(hook.run(ctx))
    assert "NEW foe ability Overgrow" in second
    assert "NEW foe item leftovers" in second

    third = asyncio.run(hook.run(ctx))
    assert "NEW foe ability" not in third
    assert "NEW foe item" not in third