"""Mega Evolution actions in submit_decision (Champions' gimmick)."""

from __future__ import annotations

from helpers import ScriptedProvider, make_battle, make_double_battle

from pokeagent.config import AgentConfig
from pokeagent.context.records import available_actions
from pokeagent.hooks.base import HookSystem
from pokeagent.loop import AgentLoop
from pokeagent.providers.base import ToolCallRequest
from pokeagent.tools.base import ToolRegistry


def _loop():
    return AgentLoop(
        provider=ScriptedProvider([]),
        registry=ToolRegistry(),
        hooks=HookSystem(),
        config=AgentConfig(battle_format="gen9championsvgc2026regmc"),
    )


def _call(actions):
    return ToolCallRequest(
        "c1", "submit_decision", {"actions": actions, "reason": "test"}
    )


def _move(slot, move, target="opp1", **extra):
    action = {"slot": slot, "kind": "move", "move": move, "target": target}
    action.update(extra)
    return action


def test_doubles_mega_accepted():
    battle = make_double_battle(can_mega=(True, True), can_tera=(False, False))
    order, decision, reason, error = _loop()._validate_doubles(
        battle,
        _call(
            [
                _move(1, "flamethrower", mega=True),
                _move(2, "icefang", "opp2"),
            ]
        ),
    )
    assert error is None
    assert order is not None
    assert "+mega" in decision


def test_doubles_mega_rejected_when_unavailable():
    battle = make_double_battle(can_mega=(False, False))
    order, decision, reason, error = _loop()._validate_doubles(
        battle, _call([_move(1, "flamethrower", mega=True)])
    )
    assert order is None
    assert "mega" in error.lower()


def test_doubles_only_one_mega_per_turn():
    battle = make_double_battle(can_mega=(True, True))
    order, decision, reason, error = _loop()._validate_doubles(
        battle,
        _call(
            [
                _move(1, "flamethrower", "opp1", mega=True),
                _move(2, "icefang", "opp2", mega=True),
            ]
        ),
    )
    assert order is None
    assert "only one Pokemon can Mega Evolve" in error


def test_doubles_tera_and_mega_are_mutually_exclusive():
    battle = make_double_battle(can_tera=(True, True), can_mega=(True, True))
    order, decision, reason, error = _loop()._validate_doubles(
        battle,
        _call([_move(1, "flamethrower", terastallize=True, mega=True)]),
    )
    assert order is None
    assert "gimmick" in error


def test_singles_mega_accepted():
    battle = make_battle(can_mega=True)
    order, decision, reason, error = _loop()._validate_singles(
        battle, _call([{"slot": 1, "kind": "move", "move": "flamethrower", "mega": True}])
    )
    assert error is None
    assert order is not None
    assert decision == "move:flamethrower+mega"


def test_singles_mega_rejected_when_unavailable():
    battle = make_battle(can_mega=False)
    order, decision, reason, error = _loop()._validate_singles(
        battle, _call([{"slot": 1, "kind": "move", "move": "flamethrower", "mega": True}])
    )
    assert order is None
    assert "mega" in error.lower()


def test_available_actions_notes_mega():
    battle = make_battle(can_mega=True)
    text = available_actions(battle)
    assert "mega evolution available" in text


def test_available_actions_notes_mega_doubles():
    battle = make_double_battle(can_mega=(True, False))
    text = available_actions(battle)
    # slot 1 may Mega; slot 2 may not
    assert text.count("mega evolution available") == 1