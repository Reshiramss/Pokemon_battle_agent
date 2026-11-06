"""Tests for structured reasoning notes (compressed per-turn CoT memory)."""

from __future__ import annotations

import asyncio
import json

from helpers import ScriptedProvider, make_battle

from pokeagent.config import AgentConfig
from pokeagent.context.manager import ContextManager
from pokeagent.context.records import TurnRecord
from pokeagent.context.reasoning import (
    DeterministicNoteExtractor,
    LLMNoteExtractor,
    deterministic_note,
)
from pokeagent.providers.base import LLMResponse
from pokeagent.state import ReasoningNote


def _record(**kwargs) -> TurnRecord:
    base = dict(
        seq=0,
        turn=3,
        request="[NOW] board",
        events=["|move|p1a: Furret|Flamethrower|p2a: Torterra", "|damage|p2a: Torterra|82/100"],
        decision="move:flamethrower",
        reason="super effective and faster",
        cot="I checked the type chart; fire hits grass hard.",
    )
    base.update(kwargs)
    return TurnRecord(**base)


def _cm(tmp_path, **overrides) -> ContextManager:
    config = AgentConfig(runs_dir=tmp_path / "runs", **overrides)
    return ContextManager(battle_tag="battle-1", config=config)


# --- deterministic extraction ------------------------------------------------


def test_deterministic_note_keeps_intent_only():
    note = deterministic_note(_record())
    assert note.intent == "super effective and faster"
    assert note.claims == [] and note.directives == []
    assert note.source == "deterministic"
    assert not note.fallback


def test_record_turn_stores_deterministic_note(tmp_path):
    cm = _cm(tmp_path)
    cm.note_extractor = DeterministicNoteExtractor()
    cm.record_turn(_record(cot="z" * 900))

    assert len(cm.ledger.notes) == 1
    assert cm.ledger.notes[0].seq == 1
    assert cm.ledger.notes[0].source == "deterministic"
    # persisted for replay
    lines = (cm.run_dir / "reasoning.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["intent"] == "super effective and faster"
    events = (cm.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert '"reasoning_note"' in events

    # a note that only repeats the reason does not displace the CoT excerpt
    reflection = cm._reflection_text()
    assert "z" * 600 in reflection


def test_no_extractor_means_no_notes_and_legacy_reflection(tmp_path):
    cm = _cm(tmp_path)
    cm.record_turn(_record(cot="w" * 900))
    assert cm.ledger.notes == []
    assert not (cm.run_dir / "reasoning.jsonl").exists()
    assert "w" * 600 in cm._reflection_text()


def test_structured_note_replaces_cot_excerpt(tmp_path):
    cm = _cm(tmp_path)
    cm.record_turn(_record(cot="r" * 900))
    cm.ledger.add_note(
        ReasoningNote(
            seq=1,
            turn=3,
            intent="chip the fire slot before it pivots",
            claims=["incineroar is faster than furret"],
            correction="waterfall was resisted last turn",
            directives=["keep gyarados healthy for the endgame"],
            open_questions=["does incineroar carry fake out?"],
            source="llm",
        )
    )

    reflection = cm._reflection_text()
    assert "[REFLECTION — your last decision and its outcome]" in reflection
    assert "Intent then: chip the fire slot" in reflection
    assert "Beliefs then: incineroar is faster than furret" in reflection
    assert "Correction: waterfall was resisted" in reflection
    assert "Carry forward: keep gyarados healthy" in reflection
    assert "Still unknown: does incineroar carry fake out?" in reflection
    # the structured note replaces the raw CoT excerpt (that is the point)
    assert "r" * 600 not in reflection


def test_note_directives_lift_into_ledger(tmp_path):
    cm = _cm(tmp_path)
    cm.ledger.add_note(
        ReasoningNote(seq=2, turn=4, directives=["preserve tera for miraidon"])
    )
    active = cm.ledger.active_directives()
    assert [d.text for d in active] == ["preserve tera for miraidon"]
    assert active[0].turn == 4


# --- LLM extraction -----------------------------------------------------------


def test_llm_extractor_parses_fenced_json():
    payload = """```json
{"intent": "double into the fire slot",
 "claims": ["incineroar lacks protect", "amoonguss is slower"],
 "correction": "torterra is not the ground immune mon",
 "directives": ["keep trick room denied"],
 "open_questions": ["what is the fourth move?"]}
```"""
    provider = ScriptedProvider([LLMResponse(text=payload, stop_reason="end_turn")])
    extractor = LLMNoteExtractor(provider, max_tokens=123, input_chars=1000)

    note = asyncio.run(extractor.extract(_record(cot="c" * 5000)))

    assert note is not None
    assert note.source == "llm"
    assert note.intent == "double into the fire slot"
    assert note.claims == ["incineroar lacks protect", "amoonguss is slower"]
    assert note.directives == ["keep trick room denied"]
    call = provider.calls[0]
    assert call["tools"] == []
    assert call["max_tokens"] == 123
    # input is bounded: only the configured CoT prefix reaches the model
    user = call["messages"][0].content
    assert "c" * 1000 in user and "c" * 1001 not in user


def test_llm_extractor_returns_none_on_error_or_garbage():
    error = ScriptedProvider([LLMResponse(stop_reason="error", error="boom")])
    assert asyncio.run(LLMNoteExtractor(error).extract(_record())) is None

    garbage = ScriptedProvider([LLMResponse(text="incineroar is fast, I think.")])
    assert asyncio.run(LLMNoteExtractor(garbage).extract(_record())) is None

    empty = ScriptedProvider([])
    assert asyncio.run(LLMNoteExtractor(empty).extract(_record())) is None


def test_background_extraction_stores_note(tmp_path):
    payload = '{"intent": "set up trick room", "claims": ["amoonguss is bulky"], "directives": ["focus incineroar"]}'
    provider = ScriptedProvider([LLMResponse(text=payload)])
    cm = _cm(tmp_path)
    cm.note_extractor = LLMNoteExtractor(provider)

    async def run():
        cm.record_turn(_record())
        for _ in range(20):
            await asyncio.sleep(0)

    asyncio.run(run())
    assert len(cm.ledger.notes) == 1
    note = cm.ledger.notes[0]
    assert note.source == "llm" and note.seq == 1
    assert note.claims == ["amoonguss is bulky"]
    assert cm.ledger.note_for_seq(1) is note
    assert "focus incineroar" in cm._reflection_text()


def test_extraction_failure_is_logged_not_raised(tmp_path):
    class Boom(Exception):
        pass

    class BoomExtractor:
        def extract_now(self, record, state=None):
            return None

        async def extract(self, record, state=None):
            raise Boom("no provider")

    cm = _cm(tmp_path)
    cm.note_extractor = BoomExtractor()

    async def run():
        cm.record_turn(_record())
        for _ in range(10):
            await asyncio.sleep(0)

    asyncio.run(run())
    assert cm.ledger.notes == []
    events = (cm.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert '"reasoning_note_error"' in events
    assert "Boom: no provider" in events