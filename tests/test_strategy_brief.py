"""Tests for the rolling strategy brief (L2 reasoning memory, S3.5)."""

from __future__ import annotations

import asyncio
import json

from helpers import make_battle, ScriptedProvider

from pokeagent.config import AgentConfig
from pokeagent.context.brief import (
    LLMBriefCompressor,
    deterministic_brief,
    sanitize_brief,
)
from pokeagent.context.manager import ContextManager
from pokeagent.context.reasoning import DeterministicNoteExtractor
from pokeagent.context.records import TurnRecord
from pokeagent.providers.base import LLMResponse
from pokeagent.state import (
    BattleState,
    Directive,
    PlanBoard,
    ReasoningLedger,
    ReasoningNote,
    StrategyBrief,
)


def _record(**kwargs) -> TurnRecord:
    base = dict(
        seq=0,
        turn=3,
        request="[NOW] board",
        events=["|move|p1a: Furret|Flamethrower|p2a: Torterra"],
        decision="move:flamethrower",
        reason="super effective and faster",
        cot="fire hits grass hard",
    )
    base.update(kwargs)
    return TurnRecord(**base)


def _cm(tmp_path, **overrides) -> ContextManager:
    config = AgentConfig(runs_dir=tmp_path / "runs", **overrides)
    return ContextManager(battle_tag="battle-1", config=config)


# --- schema sanitization -----------------------------------------------------


def test_sanitize_brief_caps_fields_and_ignores_unknown_keys():
    brief = sanitize_brief(
        {
            "stance": "s" * 500,
            "win_conditions": ["w1", "w2", "w3"],
            "threats": ["t1", "t2", "t3", "t4"],
            "commitments": ["c" * 400],
            "beliefs": ["b1"],
            "open_questions": [],
            "nonsense": "ignored",
        },
        covered_seq=4,
        turn=2,
    )
    assert brief is not None
    assert len(brief.stance) == 160
    assert brief.win_conditions == ["w1", "w2"]
    assert brief.threats == ["t1", "t2", "t3"]
    assert len(brief.commitments[0]) == 120
    assert brief.beliefs == ["b1"]
    assert brief.open_questions == []
    assert brief.covered_seq == 4 and brief.turn == 2
    assert brief.source == "llm"


def test_sanitize_brief_rejects_unusable_payloads():
    assert sanitize_brief(None, covered_seq=0, turn=0) is None
    assert sanitize_brief({"foo": "bar"}, covered_seq=0, turn=0) is None
    empty = sanitize_brief(
        {"stance": "", "win_conditions": [], "threats": []},
        covered_seq=0,
        turn=0,
    )
    assert empty is None


def test_deterministic_brief_lifts_plan_and_directives():
    ledger = ReasoningLedger(
        directives=[Directive(text="keep gyarados healthy", turn=2)],
        claims=[],
    )
    plan = PlanBoard(
        win_conditions=["outlast with tera"],
        commitments=["preserve tera for miraidon"],
        primary_threats=["incineroar"],
    )
    brief = deterministic_brief(
        ledger,
        plan=plan,
        records=[_record(reason="chip the fire slot")],
        covered_seq=3,
        turn=3,
    )
    assert brief.source == "deterministic"
    assert brief.stance == "chip the fire slot"
    assert brief.win_conditions == ["outlast with tera"]
    assert brief.commitments == ["preserve tera for miraidon"]
    assert brief.threats == ["incineroar"]
    assert brief.covered_seq == 3


# --- LLM compressor ----------------------------------------------------------


def test_llm_compressor_parses_fenced_json_and_bounds_input():
    payload = """```json
{"stance": "trade the fire slot, then clean with gyarados",
 "win_conditions": ["keep gyarados above 50%"],
 "threats": ["incineroar before it pivots"],
 "commitments": ["do not tera furret"],
 "beliefs": ["incineroar is faster than furret (T3)"],
 "open_questions": ["does it carry fake out?"]}
```"""
    provider = ScriptedProvider([LLMResponse(text=payload)])
    compressor = LLMBriefCompressor(provider, max_tokens=222, input_chars=500)

    brief = asyncio.run(
        compressor.compress(
            brief=None,
            notes=[ReasoningNote(seq=5, turn=3, intent="i" * 900)],
            records=[_record()],
            plan=None,
            turn=3,
        )
    )

    assert brief is not None
    assert brief.stance.startswith("trade the fire slot")
    assert brief.beliefs == ["incineroar is faster than furret (T3)"]
    assert brief.covered_seq == 5
    call = provider.calls[0]
    assert call["tools"] == []
    assert call["max_tokens"] == 222
    assert len(call["messages"][0].content) <= 500


def test_llm_compressor_returns_none_on_error_or_garbage():
    error = ScriptedProvider([LLMResponse(stop_reason="error", error="boom")])
    assert asyncio.run(
        LLMBriefCompressor(error).compress(
            brief=None, notes=[], records=[_record()], turn=3
        )
    ) is None
    garbage = ScriptedProvider([LLMResponse(text="just some prose")])
    assert asyncio.run(
        LLMBriefCompressor(garbage).compress(
            brief=None, notes=[], records=[_record()], turn=3
        )
    ) is None


# --- manager wiring ----------------------------------------------------------


def test_deterministic_brief_is_stored_and_rendered(tmp_path):
    cm = _cm(tmp_path)
    cm.note_extractor = DeterministicNoteExtractor()
    cm.state = BattleState()
    cm.state.plan.win_conditions = ["outlast with tera"]
    cm.state.plan.commitments = ["keep gyarados healthy"]

    cm.record_turn(_record())

    assert cm.ledger.brief is not None
    assert cm.ledger.brief.source == "deterministic"
    assert cm.ledger.brief_covered_seq == 1
    text = cm._strategy_text()
    assert "rolling brief" in text
    assert "Win: outlast with tera" in text
    assert "Commit: keep gyarados healthy" in text
    events = (cm.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert '"strategy_brief"' in events

    prompt = cm.assemble_prompt(make_battle(), [])
    user = prompt[1][0].content
    assert "[STRATEGY" in user
    assert user.index("[HISTORY]") < user.index("[STRATEGY") < user.index("[REFLECTION")


def test_brief_is_capped_in_the_prompt(tmp_path):
    cm = _cm(tmp_path)
    cm.ledger.brief = StrategyBrief(
        stance="s" * 1000,
        win_conditions=["w" * 400, "w" * 400],
        threats=["t" * 400, "t" * 400, "t" * 400],
        commitments=["c" * 400, "c" * 400, "c" * 400],
        beliefs=["b" * 400, "b" * 400, "b" * 400, "b" * 400],
        open_questions=["q" * 400, "q" * 400],
    )
    text = cm._strategy_text()
    assert len(text) <= 600
    assert text.endswith("…")


def test_background_brief_rewrite_is_stored(tmp_path):
    class FakeCompressor:
        async def compress(self, **kwargs):
            return StrategyBrief(stance="new plan", source="llm", covered_seq=0)

    cm = _cm(tmp_path, reasoning_brief_every=1)
    cm.note_extractor = DeterministicNoteExtractor()
    cm.brief_compressor = FakeCompressor()

    async def run():
        cm.record_turn(_record())
        for _ in range(20):
            await asyncio.sleep(0)

    asyncio.run(run())
    assert cm.ledger.brief.stance == "new plan"
    assert cm.ledger.brief_covered_seq == 1
    events = (cm.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert '"source": "llm"' in events


def test_failed_rewrite_keeps_the_previous_brief(tmp_path):
    class BoomCompressor:
        async def compress(self, **kwargs):
            raise RuntimeError("gateway down")

    cm = _cm(tmp_path, reasoning_brief_every=1)
    cm.note_extractor = DeterministicNoteExtractor()
    cm.brief_compressor = BoomCompressor()
    old = StrategyBrief(stance="old plan", source="llm")
    cm.ledger.brief = old

    async def run():
        cm.record_turn(_record())
        for _ in range(20):
            await asyncio.sleep(0)

    asyncio.run(run())
    assert cm.ledger.brief.stance == "old plan"
    events = (cm.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert '"strategy_brief_error"' in events


# --- reasoning_update / persistence -----------------------------------------


def test_apply_reasoning_update_replaces_plan_fields(tmp_path):
    cm = _cm(tmp_path)
    cm.state = BattleState()
    cm.apply_reasoning_update(
        {
            "win_conditions": ["a", "b", "c"],
            "commitments": ["keep tera"],
            "correction": "T3: the fire slot is physical, not special",
        },
        turn=3,
    )
    plan = cm.state.plan
    assert plan.win_conditions == ["a", "b"]
    assert plan.commitments == ["keep tera"]
    assert plan.revision == 1
    assert [c.text for c in cm.ledger.claims] == [
        "T3: the fire slot is physical, not special"
    ]
    assert cm.ledger.claims[0].kind == "plan"
    events = (cm.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert '"plan_update"' in events

    # replacement, not append
    cm.apply_reasoning_update({"win_conditions": []}, turn=4)
    assert plan.win_conditions == []
    assert plan.revision == 2


def test_apply_reasoning_update_is_failure_safe(tmp_path):
    cm = _cm(tmp_path)
    cm.apply_reasoning_update({"win_conditions": ["x"]}, turn=1)  # no state
    assert cm.ledger.claims == []
    cm.state = BattleState()
    cm.apply_reasoning_update(None, 2)
    cm.apply_reasoning_update("nonsense", 2)
    assert cm.state.plan.revision == 0


def test_brief_json_round_trip_and_legacy_migration():
    state = BattleState()
    state.ledger.brief = StrategyBrief(
        covered_seq=4, turn=4, stance="hold mid", win_conditions=["trade tera"]
    )
    state.ledger.brief_covered_seq = 4
    restored = BattleState.from_json(json.loads(json.dumps(state.to_json())))
    assert restored.ledger.brief.stance == "hold mid"
    assert restored.ledger.brief.win_conditions == ["trade tera"]
    assert restored.ledger.brief.covered_seq == 4

    legacy = json.loads(json.dumps(state.to_json()))
    legacy["ledger"]["brief"] = "plain string brief"
    migrated = BattleState.from_json(legacy)
    assert migrated.ledger.brief.stance == "plain string brief"
    assert migrated.ledger.brief.source == "legacy"