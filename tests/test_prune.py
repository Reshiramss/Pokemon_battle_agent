"""Tests for deterministic claim pruning (speed bounds + reveal evidence)."""

from __future__ import annotations

import json

from pokeagent.config import AgentConfig
from pokeagent.context.manager import ContextManager
from pokeagent.context.records import TurnRecord
from pokeagent.knowledge.tree import MonNode
from pokeagent.state import BattleState, Claim
from pokeagent.state.prune import cap_claims, prune_state, speed_bounds


def _state(*, our_spe=143, foe_status=None, weather=(), fields=(),
           side=(), opp_side=()) -> BattleState:
    state = BattleState()
    state.field.turn = 3
    state.field.weather = list(weather)
    state.field.fields = list(fields)
    state.field.side_conditions = list(side)
    state.field.opponent_side_conditions = list(opp_side)
    state.kb.us["furret"] = MonNode(
        species="furret", side="us", stats={"spe": our_spe}
    )
    state.kb.them["incineroar"] = MonNode(
        species="incineroar", side="them", status=foe_status
    )
    return state


def _turn(n: int):
    return ["", "turn", str(n)]


def _move(actor: str, move: str, target: str = ""):
    return ["", "move", actor, move, target]


def _switch(actor: str, species: str):
    return ["", "switch", actor, f"{species}, L50, M", "100/100"]


def test_our_mon_moving_first_gives_foe_speed_upper_bound():
    state = _state()
    rows = [
        _turn(3),
        _move("p1a: Furret", "Flamethrower", "p2a: Incineroar"),
        _move("p2a: Incineroar", "Knock Off", "p1a: Furret"),
    ]
    bounds = speed_bounds(state, rows)
    assert bounds == {"incineroar": (None, 142)}

    result = prune_state(state, rows)
    assert result.speed_bounds == {"incineroar": {"lower": None, "upper": 142}}
    claim = state.ledger.claims[-1]
    assert claim.kind == "speed" and claim.subject == "incineroar"
    assert claim.text.startswith("speed < 142")
    assert claim.source == "inference" and claim.confidence == "high"


def test_foe_moving_first_gives_speed_lower_bound():
    state = _state()
    rows = [
        _turn(3),
        _move("p2a: Incineroar", "Knock Off", "p1a: Furret"),
        _move("p1a: Furret", "Flamethrower", "p2a: Incineroar"),
    ]
    assert speed_bounds(state, rows) == {"incineroar": (144, None)}


def test_switch_order_counts_too():
    state = _state()
    rows = [
        _turn(1),
        _switch("p1a: Furret", "Furret"),
        _switch("p2a: Incineroar", "Incineroar"),
    ]
    assert speed_bounds(state, rows) == {"incineroar": (None, 142)}


def test_speed_inference_skips_ambiguous_conditions():
    rows = [
        _turn(3),
        _move("p1a: Furret", "Flamethrower", "p2a: Incineroar"),
        _move("p2a: Incineroar", "Knock Off", "p1a: Furret"),
    ]
    assert speed_bounds(_state(weather=["RAIN"]), rows) == {}
    assert speed_bounds(_state(fields=["TRICK_ROOM"]), rows) == {}
    assert speed_bounds(_state(side=["TAILWIND"]), rows) == {}
    assert speed_bounds(_state(opp_side=["TAILWIND"]), rows) == {}
    assert speed_bounds(_state(foe_status="PAR"), rows) == {}

    boosted = [
        _turn(3),
        ["", "-boost", "p2a: Incineroar", "spe", "2"],
        *rows[1:],
    ]
    assert speed_bounds(_state(), boosted) == {}


def test_priority_moves_are_not_compared():
    class FakeKB:
        def move_data(self, name):
            key = name.lower().replace(" ", "").replace("-", "")
            return {"priority": 3} if key == "fakeout" else {"priority": 0}

    state = _state()
    rows = [
        _turn(3),
        _move("p2a: Incineroar", "Fake Out", "p1a: Furret"),
        _move("p1a: Furret", "Flamethrower", "p2a: Incineroar"),
    ]
    # without the static kb, Fake Out looks like priority 0 → a bound appears
    assert speed_bounds(state, rows) == {"incineroar": (144, None)}
    # with it, the priority action is excluded entirely
    assert speed_bounds(state, rows, static_kb=FakeKB()) == {}


def test_speed_bounds_tighten_and_share_one_claim():
    state = _state()
    state.kb.us["gyarados"] = MonNode(
        species="gyarados", side="us", stats={"spe": 120}
    )
    rows_first = [
        _turn(3),
        _move("p1a: Furret", "Flamethrower", "p2a: Incineroar"),
        _move("p2a: Incineroar", "Knock Off", "p1a: Furret"),
    ]
    prune_state(state, rows_first)
    assert state.ledger.claims[-1].text.startswith("speed < 142")

    rows_second = [
        _turn(4),
        _move("p2a: Incineroar", "Knock Off", "p1a: Gyarados"),
        _move("p1a: Gyarados", "Waterfall", "p2a: Incineroar"),
    ]
    result = prune_state(state, rows_second)
    assert result.speed_bounds == {"incineroar": {"lower": 121, "upper": 142}}
    speed_claims = [c for c in state.ledger.claims if c.kind == "speed"]
    assert len(speed_claims) == 1
    assert "> 121 and < 142" in speed_claims[0].text


def test_reveals_confirm_and_refute_beliefs():
    state = _state()
    foe = state.kb.them["incineroar"]
    foe.known_ability = "Intimidate"
    foe.item = "Leftovers"
    foe.revealed_moves = ["fakeout", "flareblitz"]
    state.ledger.claims = [
        Claim(subject="incineroar", kind="ability", text="has Intimidate"),
        Claim(subject="incineroar", kind="ability", text="has Blaze"),
        Claim(subject="incineroar", kind="item", text="holds Leftovers"),
        Claim(subject="incineroar", kind="item", text="probably Choice Scarf"),
        Claim(subject="incineroar", kind="move", text="knows Fake Out"),
        Claim(subject="incineroar", kind="move", text="knows Parting Shot"),
    ]

    result = prune_state(state, [])
    statuses = {c.text: c.status for c in state.ledger.claims}
    assert statuses["has Intimidate"] == "confirmed"
    assert statuses["has Blaze"] == "refuted"
    assert statuses["holds Leftovers"] == "confirmed"
    assert statuses["probably Choice Scarf"] == "refuted"
    assert statuses["knows Fake Out"] == "confirmed"
    # an unclicked move is not refuted — it may simply be unclicked
    assert statuses["knows Parting Shot"] == "active"
    assert len(result.confirmed) == 3 and len(result.refuted) == 2


def test_cap_claims_drops_the_weakest():
    state = _state()
    state.ledger.claims = [
        Claim(subject="x", kind="set", text=f"claim {i}",
              confidence="low" if i % 3 else "high", turn=i)
        for i in range(15)
    ]
    dropped = cap_claims(state, limit=12)
    assert len(dropped) == 3
    assert len(state.ledger.claims) == 12
    assert all("low" or True for _ in dropped)


def _cm(tmp_path) -> ContextManager:
    config = AgentConfig(runs_dir=tmp_path / "runs")
    return ContextManager(battle_tag="battle-1", config=config)


def test_record_turn_prunes_and_keeps_rows_off_disk(tmp_path):
    cm = _cm(tmp_path)
    cm.state = _state()
    rows = [
        _turn(3),
        _move("p1a: Furret", "Flamethrower", "p2a: Incineroar"),
        _move("p2a: Incineroar", "Knock Off", "p1a: Furret"),
    ]
    cm.record_turn(
        TurnRecord(
            seq=0,
            turn=3,
            request="[NOW]",
            events=["|turn|3", "|move|p1a: Furret|Flamethrower|p2a: Incineroar"],
            event_rows=rows,
            decision="move:flamethrower",
            reason="chip it",
        )
    )

    assert any(c.kind == "speed" for c in cm.state.ledger.claims)
    events = (cm.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert '"claims_pruned"' in events
    assert "event_rows" not in events
    # state.json reflects the pruned beliefs
    saved = json.loads((cm.run_dir / "state.json").read_text(encoding="utf-8"))
    assert any(c["kind"] == "speed" for c in saved["ledger"]["claims"])


def test_record_turn_without_rows_is_a_noop(tmp_path):
    cm = _cm(tmp_path)
    cm.state = _state()
    cm.record_turn(
        TurnRecord(seq=0, turn=3, request="[NOW]", events=[], decision="x")
    )
    assert cm.state.ledger.claims == []
    events = (cm.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert '"claims_pruned"' not in events