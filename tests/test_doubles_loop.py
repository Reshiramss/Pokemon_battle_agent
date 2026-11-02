"""Doubles decision-layer tests: submit_decision v2 validation matrix.

Covers the per-slot plan logic, the symbolic target vocabulary, forced
switches (including both-forced with a single bench mon), terastallization
rules, same-name bench claims, the early exits, and the doubles rendering
of the snapshot / available-actions builders. A few singles cases pin the
v2 schema on the unchanged singles path.
"""

from __future__ import annotations

import asyncio

import pytest
from helpers import ScriptedProvider, make_battle, make_double_battle
from poke_env.player import DoubleBattleOrder

from pokeagent.config import AgentConfig
from pokeagent.context.manager import ContextManager
from pokeagent.context.records import available_actions, snapshot
from pokeagent.hooks.base import HookSystem
from pokeagent.loop import AgentLoop, TurnAbort, _slot_plans
from pokeagent.providers.base import LLMResponse, ToolCallRequest
from pokeagent.tools.base import ToolRegistry


def make_turn_loop(tmp_path, responses):
    config = AgentConfig(runs_dir=tmp_path / "runs")
    provider = ScriptedProvider(responses)
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry(),
        hooks=HookSystem(),
        config=config,
        runner=None,
    )
    cm = ContextManager(
        battle_tag="battle-gen9vgc2025regihidden-1", config=config
    )
    return loop, cm, provider


def _submit(actions, reason="stub"):
    return LLMResponse(
        tool_calls=[
            ToolCallRequest(
                "c1", "submit_decision", {"actions": actions, "reason": reason}
            )
        ],
        stop_reason="tool_use",
    )


# --- happy paths ------------------------------------------------------------


def test_doubles_legal_both_slots(tmp_path):
    async def scenario():
        loop, cm, _provider = make_turn_loop(
            tmp_path,
            [
                _submit(
                    [
                        {
                            "slot": 1,
                            "kind": "move",
                            "move": "flamethrower",
                            "target": "opp1",
                        },
                        {
                            "slot": 2,
                            "kind": "move",
                            "move": "icefang",
                            "target": "opp2",
                        },
                    ]
                )
            ],
        )
        outcome = await loop.run_turn(make_double_battle(), cm)
        assert isinstance(outcome.order, DoubleBattleOrder)
        assert (
            outcome.decision
            == "slot1:move:flamethrower->opp1 | slot2:move:icefang->opp2"
        )

    asyncio.run(scenario())


def test_doubles_single_target_move_needs_no_target(tmp_path):
    async def scenario():
        loop, cm, _provider = make_turn_loop(
            tmp_path,
            [
                _submit(
                    [
                        {
                            "slot": 1,
                            "kind": "move",
                            "move": "flamethrower",
                            "target": "opp2",
                        },
                        {"slot": 2, "kind": "move", "move": "protect"},
                    ]
                )
            ],
        )
        battle = make_double_battle(moves=(("flamethrower",), ("protect",)))
        outcome = await loop.run_turn(battle, cm)
        # protect targets only the user — one legal target, omission accepted
        assert outcome.decision == "slot1:move:flamethrower->opp2 | slot2:move:protect"

    asyncio.run(scenario())


def test_doubles_terastallize_on_one_slot(tmp_path):
    async def scenario():
        loop, cm, _provider = make_turn_loop(
            tmp_path,
            [
                _submit(
                    [
                        {
                            "slot": 1,
                            "kind": "move",
                            "move": "flamethrower",
                            "target": "opp1",
                            "terastallize": True,
                        },
                        {
                            "slot": 2,
                            "kind": "move",
                            "move": "icefang",
                            "target": "opp2",
                        },
                    ]
                )
            ],
        )
        outcome = await loop.run_turn(
            make_double_battle(can_tera=(True, True)), cm
        )
        assert "slot1:move:flamethrower->opp1+tera" in outcome.decision

    asyncio.run(scenario())


def test_doubles_forced_switch_one_deciding_slot(tmp_path):
    async def scenario():
        loop, cm, _provider = make_turn_loop(
            tmp_path,
            [
                _submit(
                    [{"slot": 1, "kind": "switch", "switch_to": "pikachu"}],
                    reason="replacing the fainted lead",
                )
            ],
        )
        battle = make_double_battle(force_switch=(True, False), fainted=(True, False))
        outcome = await loop.run_turn(battle, cm)
        # slot 2 sits out the mid-turn forced-switch decision and auto-passes
        assert outcome.decision == "slot1:switch:pikachu | slot2:pass"
        assert isinstance(outcome.order, DoubleBattleOrder)

    asyncio.run(scenario())


def test_doubles_both_forced_single_bench_autopasses_other(tmp_path):
    async def scenario():
        loop, cm, _provider = make_turn_loop(
            tmp_path,
            [
                _submit(
                    [{"slot": 1, "kind": "switch", "switch_to": "pikachu"}],
                    reason="the only bench Pokemon goes to slot 1",
                )
            ],
        )
        battle = make_double_battle(
            force_switch=(True, True), fainted=(True, True), bench=("pikachu",)
        )
        outcome = await loop.run_turn(battle, cm)
        # slot 2 must pass — the lone bench mon is claimed by slot 1
        assert outcome.decision == "slot1:switch:pikachu | slot2:pass"

    asyncio.run(scenario())


def test_doubles_both_forced_passing_everything_rejected(tmp_path):
    async def scenario():
        loop, cm, provider = make_turn_loop(
            tmp_path,
            [
                _submit(
                    [{"slot": 1, "kind": "pass"}, {"slot": 2, "kind": "pass"}],
                    reason="passing both",
                ),
                _submit(
                    [{"slot": 2, "kind": "switch", "switch_to": "pikachu"}],
                    reason="slot 2 takes the bench mon instead",
                ),
            ],
        )
        battle = make_double_battle(
            force_switch=(True, True), fainted=(True, True), bench=("pikachu",)
        )
        outcome = await loop.run_turn(battle, cm)
        # omitting slot 1 auto-passes (pass is legal there); slot 2 switches
        assert outcome.decision == "slot1:pass | slot2:switch:pikachu"
        assert len(provider.calls) == 2  # one rejected round, one accepted

    asyncio.run(scenario())


def test_doubles_same_name_bench_two_switches(tmp_path):
    async def scenario():
        loop, cm, _provider = make_turn_loop(
            tmp_path,
            [
                _submit(
                    [
                        {"slot": 1, "kind": "switch", "switch_to": "pikachu"},
                        {"slot": 2, "kind": "switch", "switch_to": "pikachu"},
                    ]
                )
            ],
        )
        battle = make_double_battle(bench=("pikachu", "pikachu"))
        outcome = await loop.run_turn(battle, cm)
        # each slot claims a distinct Pikachu from the bench
        assert outcome.decision == "slot1:switch:pikachu | slot2:switch:pikachu"
        assert outcome.order.first_order.order is not outcome.order.second_order.order

    asyncio.run(scenario())


# --- validation errors, each with a retry ------------------------------------------------


def _last_error(provider_call):
    return [
        m for m in provider_call["messages"] if getattr(m, "is_error", False)
    ][-1].content


def test_doubles_missing_target_then_retry(tmp_path):
    async def scenario():
        loop, cm, provider = make_turn_loop(
            tmp_path,
            [
                _submit(
                    [
                        {"slot": 1, "kind": "move", "move": "flamethrower"},
                        {"slot": 2, "kind": "move", "move": "icefang", "target": "opp2"},
                    ]
                ),
                _submit(
                    [
                        {
                            "slot": 1,
                            "kind": "move",
                            "move": "flamethrower",
                            "target": "opp2",
                        },
                        {"slot": 2, "kind": "move", "move": "icefang", "target": "opp1"},
                    ]
                ),
            ],
        )
        outcome = await loop.run_turn(make_double_battle(), cm)
        assert outcome.rounds == 2
        assert "needs an explicit target" in _last_error(provider.calls[1])
        assert "Legal targets: ally, opp1, opp2" in _last_error(provider.calls[1])
        assert (
            outcome.decision
            == "slot1:move:flamethrower->opp2 | slot2:move:icefang->opp1"
        )

    asyncio.run(scenario())


def test_doubles_illegal_target_rejected(tmp_path):
    async def scenario():
        loop, cm, provider = make_turn_loop(
            tmp_path,
            [
                _submit(
                    [
                        {
                            "slot": 1,
                            "kind": "move",
                            "move": "flamethrower",
                            "target": "opp3",
                        },
                        {"slot": 2, "kind": "move", "move": "icefang", "target": "opp1"},
                    ]
                ),
                _submit(
                    [
                        {
                            "slot": 1,
                            "kind": "move",
                            "move": "flamethrower",
                            "target": "ally",
                        },
                        {"slot": 2, "kind": "move", "move": "icefang", "target": "opp1"},
                    ]
                ),
            ],
        )
        outcome = await loop.run_turn(make_double_battle(), cm)
        assert "invalid target 'opp3'" in _last_error(provider.calls[1])
        # targeting the ally is legal (a real VGC technique)
        assert "slot1:move:flamethrower->ally" in outcome.decision

    asyncio.run(scenario())


def test_doubles_missing_slot_rejected(tmp_path):
    async def scenario():
        loop, cm, provider = make_turn_loop(
            tmp_path,
            [
                _submit(
                    [
                        {
                            "slot": 1,
                            "kind": "move",
                            "move": "flamethrower",
                            "target": "opp1",
                        }
                    ]
                ),
                _submit(
                    [
                        {
                            "slot": 1,
                            "kind": "move",
                            "move": "flamethrower",
                            "target": "opp1",
                        },
                        {"slot": 2, "kind": "switch", "switch_to": "pikachu"},
                    ]
                ),
            ],
        )
        outcome = await loop.run_turn(make_double_battle(), cm)
        assert "missing action for slot 2" in _last_error(provider.calls[1])
        assert "slot2:switch:pikachu" in outcome.decision

    asyncio.run(scenario())


def test_doubles_duplicate_slot_rejected(tmp_path):
    async def scenario():
        loop, cm, provider = make_turn_loop(
            tmp_path,
            [
                _submit(
                    [
                        {"slot": 1, "kind": "move", "move": "flamethrower", "target": "opp1"},
                        {"slot": 1, "kind": "move", "move": "quickattack", "target": "opp1"},
                    ]
                ),
                _submit(
                    [
                        {"slot": 1, "kind": "move", "move": "flamethrower", "target": "opp1"},
                        {"slot": 2, "kind": "move", "move": "icefang", "target": "opp2"},
                    ]
                ),
            ],
        )
        outcome = await loop.run_turn(make_double_battle(), cm)
        assert "duplicate action for slot 1" in _last_error(provider.calls[1])
        assert outcome.rounds == 2

    asyncio.run(scenario())


def test_doubles_action_on_pass_slot_rejected(tmp_path):
    async def scenario():
        loop, cm, provider = make_turn_loop(
            tmp_path,
            [
                _submit(
                    [
                        {"slot": 1, "kind": "switch", "switch_to": "pikachu"},
                        {"slot": 2, "kind": "move", "move": "icefang", "target": "opp1"},
                    ]
                ),
                _submit([{"slot": 1, "kind": "switch", "switch_to": "pikachu"}]),
            ],
        )
        battle = make_double_battle(force_switch=(True, False), fainted=(True, False))
        outcome = await loop.run_turn(battle, cm)
        assert "no decision this round" in _last_error(provider.calls[1])
        assert outcome.decision == "slot1:switch:pikachu | slot2:pass"

    asyncio.run(scenario())


def test_doubles_double_tera_rejected(tmp_path):
    async def scenario():
        loop, cm, provider = make_turn_loop(
            tmp_path,
            [
                _submit(
                    [
                        {
                            "slot": 1,
                            "kind": "move",
                            "move": "flamethrower",
                            "target": "opp1",
                            "terastallize": True,
                        },
                        {
                            "slot": 2,
                            "kind": "move",
                            "move": "icefang",
                            "target": "opp2",
                            "terastallize": True,
                        },
                    ]
                ),
                _submit(
                    [
                        {
                            "slot": 1,
                            "kind": "move",
                            "move": "flamethrower",
                            "target": "opp1",
                            "terastallize": True,
                        },
                        {"slot": 2, "kind": "move", "move": "icefang", "target": "opp2"},
                    ]
                ),
            ],
        )
        outcome = await loop.run_turn(make_double_battle(can_tera=(True, True)), cm)
        assert "only one Pokemon can terastallize" in _last_error(provider.calls[1])
        assert outcome.decision.endswith("slot2:move:icefang->opp2")

    asyncio.run(scenario())


def test_doubles_tera_blocked_for_slot_without_it(tmp_path):
    async def scenario():
        loop, cm, provider = make_turn_loop(
            tmp_path,
            [
                _submit(
                    [
                        {
                            "slot": 1,
                            "kind": "move",
                            "move": "flamethrower",
                            "target": "opp1",
                            "terastallize": True,
                        },
                        {"slot": 2, "kind": "move", "move": "icefang", "target": "opp2"},
                    ]
                ),
                _submit(
                    [
                        {"slot": 1, "kind": "move", "move": "flamethrower", "target": "opp1"},
                        {
                            "slot": 2,
                            "kind": "move",
                            "move": "icefang",
                            "target": "opp2",
                            "terastallize": True,
                        },
                    ]
                ),
            ],
        )
        outcome = await loop.run_turn(make_double_battle(can_tera=(False, True)), cm)
        assert "not available" in _last_error(provider.calls[1])
        assert "slot2:move:icefang->opp2+tera" in outcome.decision

    asyncio.run(scenario())


def test_doubles_switch_into_claimed_bench_rejected(tmp_path):
    async def scenario():
        loop, cm, provider = make_turn_loop(
            tmp_path,
            [
                _submit(
                    [
                        {"slot": 1, "kind": "switch", "switch_to": "pikachu"},
                        {"slot": 2, "kind": "switch", "switch_to": "pikachu"},
                    ]
                ),
                _submit(
                    [
                        {"slot": 1, "kind": "switch", "switch_to": "pikachu"},
                        {"slot": 2, "kind": "switch", "switch_to": "charizard"},
                    ]
                ),
            ],
        )
        # one Pikachu on the bench: the second claim fails, retry uses Charizard
        battle = make_double_battle(bench=("pikachu", "charizard"))
        outcome = await loop.run_turn(battle, cm)
        assert "already claimed" in _last_error(provider.calls[1])
        assert outcome.decision == "slot1:switch:pikachu | slot2:switch:charizard"

    asyncio.run(scenario())


# --- early exits and aborts --------------------------------------------------


def test_doubles_wait_early_exit(tmp_path):
    async def scenario():
        loop, cm, provider = make_turn_loop(tmp_path, [])
        outcome = await loop.run_turn(make_double_battle(wait=True), cm)
        assert outcome.decision == "default:wait"
        assert provider.calls == []  # never consulted the LLM

    asyncio.run(scenario())


def test_doubles_rounds_exhausted_aborts(tmp_path):
    async def scenario():
        loop, cm, provider = make_turn_loop(tmp_path, [])
        with pytest.raises(TurnAbort):
            await loop.run_turn(make_double_battle(), cm)
        # two-phase flow: the empty info response ends the information phase
        # after one round (natural transition), then decision_rounds nudges
        assert len(provider.calls) == 1 + loop.config.decision_rounds

    asyncio.run(scenario())


# --- slot plans ------------------------------------------------------------------


def test_slot_plans_matrix():
    # normal turn: both slots decide
    assert _slot_plans(make_double_battle()) == ["act", "act"]
    # mid-turn forced switch: only the forced slot decides
    battle = make_double_battle(force_switch=(True, False), fainted=(True, False))
    assert _slot_plans(battle) == ["act", "pass"]
    # forced with no replacement available
    battle = make_double_battle(force_switch=(True, True), fainted=(True, True), bench=())
    assert _slot_plans(battle) == ["pass", "pass"]
    # both forced with a bench: both decide (passing one is up to the model)
    battle = make_double_battle(force_switch=(True, True), fainted=(True, True))
    assert _slot_plans(battle) == ["act", "act"]


# --- doubles rendering ------------------------------------------------------------


def test_doubles_snapshot_and_actions_rendering():
    battle = make_double_battle()
    snap = snapshot(battle)
    assert "Your slot 1: furret" in snap
    assert "Your slot 2: gyarados" in snap
    assert "Opponent slot 2 (opp2): incineroar" in snap
    assert "Speed order (fastest first):" in snap
    # own speeds exact, foe speeds flagged as estimates
    assert "furret 90" in snap and "(foe est.)" in snap

    actions = available_actions(battle)
    assert "Spread moves hit both foes at 0.75x power." in actions
    assert "Slot 1 (furret):" in actions
    # adjacent-foe moves list every non-auto target
    assert "targets: ally,opp1,opp2" in actions
    assert "Slot 2 (gyarados):" in actions


def test_doubles_forced_switch_rendering():
    battle = make_double_battle(force_switch=(True, False), fainted=(True, False))
    actions = available_actions(battle)
    assert "FORCED SWITCH for slot 1" in actions
    assert "pass (this slot sits out the forced-switch decision)" in actions
    assert "pikachu" in actions  # bench list for the forced slot


# --- singles path on the v2 schema -------------------------------------------------


def _singles_loop(tmp_path, responses):
    config = AgentConfig(runs_dir=tmp_path / "runs")
    provider = ScriptedProvider(responses)
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry(),
        hooks=HookSystem(),
        config=config,
        runner=None,
    )
    cm = ContextManager(battle_tag="battle-gen9randombattle-1", config=config)
    return loop, cm, provider


def test_singles_v2_move_decision(tmp_path):
    async def scenario():
        loop, cm, _provider = _singles_loop(
            tmp_path,
            [
                _submit(
                    [{"slot": 1, "kind": "move", "move": "flamethrower"}],
                    reason="super effective",
                )
            ],
        )
        outcome = await loop.run_turn(make_battle(), cm)
        assert outcome.decision == "move:flamethrower"

    asyncio.run(scenario())


def test_singles_tera_unavailable_is_rejected(tmp_path):
    async def scenario():
        loop, cm, provider = _singles_loop(
            tmp_path,
            [
                _submit(
                    [
                        {
                            "slot": 1,
                            "kind": "move",
                            "move": "flamethrower",
                            "terastallize": True,
                        }
                    ]
                ),
                _submit([{"slot": 1, "kind": "move", "move": "flamethrower"}]),
            ],
        )
        outcome = await loop.run_turn(make_battle(), cm)
        # can_tera is False on the fixture — the latent old bug passed it
        # through unchecked; now it is a validation error
        assert "terastallization is not available" in _last_error(provider.calls[1])
        assert outcome.decision == "move:flamethrower"
        assert outcome.rounds == 2

    asyncio.run(scenario())
