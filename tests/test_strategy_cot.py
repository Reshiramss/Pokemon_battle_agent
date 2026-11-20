"""Battle-level COT: helpers, the maintainer sub-agent, and [STRATEGY] rendering.

The COT is the plan the agent re-reads every decision: rewritten after each
turn by a background sub-agent on the main model from the previous text, the
board, the turn's hidden thinking and what actually happened. These tests pin
the bounded text handling (thinking head+tail, conversation sanitizing), the
thinking-only fallback this gateway needs, the seq fence that stops a slow
answer from regressing the plan, and the prompt order the section must keep.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from helpers import ScriptedProvider, make_battle, make_kb

from pokeagent.config import AgentConfig
from pokeagent.context.cot import (
    MAX_COT_CHARS,
    build_cot_input,
    cap_thinking,
    sanitize_cot,
    thinking_tail,
)
from pokeagent.context.manager import ContextManager
from pokeagent.context.reasoning import DeterministicNoteExtractor
from pokeagent.context.records import TurnRecord
from pokeagent.providers.base import LLMResponse
from pokeagent.state import BattleState, GlobalCOT
from pokeagent.tasks.base import SubAgentContext
from pokeagent.tasks.strategy_cot import StrategyCOTAgent


def _cm(tmp_path) -> ContextManager:
    config = AgentConfig(runs_dir=tmp_path / "runs")
    cm = ContextManager(battle_tag="battle-1", config=config, kb=None)
    cm.state = BattleState.create(make_battle(), make_kb(), ledger=cm.ledger)
    cm.ledger = cm.state.ledger  # the player shares one ledger object
    return cm


def _events(cm):
    path = cm.run_dir / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- text helpers -------------------------------------------------------------


def test_sanitize_cot_drops_fences_and_headings_and_caps():
    raw = "```text\n# battle plan\nPLAN: Keep Tornadus\nhealthy\n```\n" + "x" * 900
    out = sanitize_cot(raw)
    assert "`" not in out and "#" not in out and "PLAN:" not in out
    assert out.startswith("Keep Tornadus healthy")
    assert len(out) <= MAX_COT_CHARS


def test_sanitize_cot_prefers_sentence_boundary():
    out = sanitize_cot("Keep the fast slot healthy through the early game. " + "y" * 600)
    assert out.endswith(".")
    assert len(out) <= MAX_COT_CHARS


def test_cap_thinking_keeps_head_and_tail():
    text = "A" * 4000 + "MIDDLE" + "Z" * 4000
    out = cap_thinking(text, head=1500, tail=3500)
    assert out.startswith("A" * 1500)
    assert out.endswith("Z" * 3500)
    assert "elided" in out


def test_thinking_tail_prefers_plan_marker_then_last_paragraph():
    marked = (
        "We need decide. Lots of internal arithmetic about damage rolls...\n\n"
        "PLAN: Keep Farigiraf healthy and Taunt the Trick Room setter before T4."
    )
    assert thinking_tail(marked).startswith("Keep Farigiraf")

    paragraphs = "First idea about speed tiers and some more text here.\n\n" \
        "Second idea is the conclusion that should be kept."
    assert thinking_tail(paragraphs).startswith("Second idea")

    assert thinking_tail("too short") == ""


def test_build_cot_input_budgets_a_huge_thinking_dump():
    record = SimpleNamespace(
        turn=7,
        seq=9,
        decision="move:taunt->opp1",
        reason="shut down Trick Room",
        events=["opp1 used Trick Room"] * 20,
        thinking="T" * 30000,
    )
    out = build_cot_input(
        turn=7,
        seq=9,
        previous="old plan",
        board="B" * 20000,
        record=record,
        limit=8000,
    )
    assert len(out) <= 8000
    assert "PREVIOUS COT" in out
    assert "PRIVATE THINKING" in out and "elided" in out
    assert "T" * 30000 not in out


# --- the maintainer agent -----------------------------------------------------


def _sctx(record=None):
    record = record or TurnRecord(seq=3, turn=3, request="board", decision="move:x")
    return SubAgentContext(
        battle=make_battle(),
        inputs={"record": record, "turn": record.turn, "seq": record.seq},
    )


def test_agent_returns_sanitized_visible_text():
    provider = ScriptedProvider(
        [LLMResponse(text="```\nPLAN: trade Calyrex for their Trick Room.\n```")]
    )
    agent = StrategyCOTAgent(provider, max_tokens=3000)
    sctx = _sctx()

    out = asyncio.run(agent.run(sctx))

    assert out.startswith("trade Calyrex")
    assert sctx.inputs["cot_source"] == "llm"
    # one call, no tools, generous budget (thinking is spent from it)
    call = provider.calls[0]
    assert call["tools"] == []
    assert call["max_tokens"] == 3000


def test_agent_falls_back_to_thinking_when_text_is_empty():
    provider = ScriptedProvider(
        [
            LLMResponse(
                text=" \n",
                thinking=(
                    "Weighing options at length, damage roll after damage roll...\n\n"
                    "PLAN: Preserve Farigiraf and stall Trick Room until T6."
                ),
                stop_reason="max_tokens",
            )
        ]
    )
    sctx = _sctx()

    out = asyncio.run(StrategyCOTAgent(provider).run(sctx))

    assert "Preserve Farigiraf" in out
    assert sctx.inputs["cot_source"] == "thinking"


def test_agent_returns_empty_on_provider_error():
    provider = ScriptedProvider([LLMResponse(stop_reason="error", error="boom")])
    assert asyncio.run(StrategyCOTAgent(provider).run(_sctx())) == ""


def test_agent_separates_the_last_outcome_from_the_fresh_commit():
    """``record.events`` belong to the *previous* decision, not this one."""
    provider = ScriptedProvider([LLMResponse(text="Hold mid.")])
    prev = TurnRecord(
        seq=1, turn=2, request="board", decision="move:taunt->opp1",
        reason="deny Trick Room",
    )
    record = TurnRecord(
        seq=2, turn=3, request="board", decision="move:electrodrift->opp1",
        reason="finish it", events=["opp1 used Trick Room"],
    )
    sctx = SubAgentContext(
        battle=make_battle(), inputs={"record": record, "previous_record": prev}
    )

    asyncio.run(StrategyCOTAgent(provider).run(sctx))

    sent = provider.calls[0]["messages"][0].content
    assert "YOUR PREVIOUS DECISION: move:taunt->opp1" in sent
    assert "WHAT CAME OF IT" in sent
    assert "JUST COMMITTED (outcome unknown at this point): move:electrodrift->opp1" in sent
    assert sent.index("WHAT CAME OF IT") < sent.index("JUST COMMITTED")


def test_build_cot_input_keeps_the_thinking_conclusion_under_pressure():
    """The final clamp must never eat the tail: it is the conclusion."""
    conclusion = "CONCLUSION: preserve Miraidon and answer Farigiraf next turn."
    record = SimpleNamespace(
        turn=9, seq=12, decision="move:x", reason="r" * 400,
        events=["e"] * 12, thinking="HEAD" + "x" * 30000 + conclusion,
    )
    plan = SimpleNamespace(
        line="L" * 200, win_conditions=["w" * 120] * 2, commitments=["c" * 120] * 3
    )

    out = build_cot_input(
        turn=9, seq=12, previous="P" * 800, board="B" * 20000,
        record=record, plan=plan,
    )

    assert len(out) <= 8000
    assert out.rstrip().endswith(conclusion)


# --- manager: storing, fencing, rendering -------------------------------------


def test_store_global_cot_writes_state_and_rejects_stale(tmp_path):
    cm = _cm(tmp_path)
    cm.store_global_cot("Plan A", turn=2, seq=2)
    assert cm.ledger.global_cot.text == "Plan A"
    saved = json.loads((cm.run_dir / "state.json").read_text())
    assert saved["ledger"]["global_cot"]["text"] == "Plan A"

    cm.store_global_cot("Plan B", turn=3, seq=3)
    cm.store_global_cot("Stale", turn=1, seq=1)
    assert cm.ledger.global_cot.text == "Plan B"
    kinds = [e["type"] for e in _events(cm)]
    assert "strategy_cot" in kinds and "strategy_cot_stale" in kinds

    cm.store_global_cot("   ", turn=4, seq=4)
    assert cm.ledger.global_cot.text == "Plan B"
    assert "strategy_cot_empty" in [e["type"] for e in _events(cm)]


def test_seeded_cot_is_used_until_the_first_rewrite(tmp_path):
    cm = _cm(tmp_path)
    cm.seed_global_cot("lead Tornadus and Calyrex, deny Trick Room", turn=0)
    assert cm.ledger.global_cot.source == "seed"
    assert "Opening line" in cm.ledger.global_cot.text

    cm.seed_global_cot("something else", turn=0)  # never overwrites
    assert "something else" not in cm.ledger.global_cot.text

    cm.store_global_cot("the real plan", turn=1, seq=1)
    assert cm.ledger.global_cot.source in {"llm", "thinking"}
    assert "real plan" in cm.ledger.global_cot.text


def test_strategy_section_prefers_cot_and_keeps_prompt_order(tmp_path):
    cm = _cm(tmp_path)
    cm.store_global_cot("Hold Tornadus; Taunt the setter.", turn=2, seq=2)
    cm.record_turn(
        TurnRecord(
            seq=0, turn=2, request="board", decision="move:taunt",
            reason="deny Trick Room", cot="visible reason",
        )
    )

    text = cm._strategy_text()
    assert text.startswith("[STRATEGY — how this game is won")
    assert "Hold Tornadus" in text
    assert len(text) <= 600

    prompt = cm.assemble_prompt(make_battle(), [])[1][0].content
    assert prompt.index("[HISTORY]") < prompt.index("[STRATEGY") < prompt.index("[REFLECTION")


def test_strategy_section_is_capped_even_with_a_long_cot(tmp_path):
    cm = _cm(tmp_path)
    cm.ledger.global_cot = GlobalCOT(text="z" * 2000, turn=3, seq=3)
    text = cm._strategy_text()
    assert len(text) <= 600 and text.endswith("…")


def test_strategy_header_names_the_plan_turn_and_flags_staleness(tmp_path):
    cm = _cm(tmp_path)
    cm.store_global_cot("Hold Tornadus.", turn=2, seq=2)

    fresh = cm._strategy_text(turn=3)
    assert "as of turn 2" in fresh and "stale" not in fresh

    stale = cm._strategy_text(turn=4)
    assert "may be stale" in stale
    assert len(stale) <= 600


def test_seed_without_a_real_reason_is_not_written(tmp_path):
    """No placeholder plan: an empty slot beats a fabricated one."""
    cm = _cm(tmp_path)
    cm.seed_global_cot(None, turn=1)
    assert cm.ledger.global_cot is None


def test_reflection_prefers_the_thinking_conclusion_over_its_opening(tmp_path):
    cm = _cm(tmp_path)
    thinking = (
        "head " * 400
        + "\n\nCONCLUSION: keep Miraidon alive and answer Farigiraf next."
    )
    cm.record_turn(
        TurnRecord(seq=0, turn=4, request="board", decision="move:x", thinking=thinking)
    )

    text = cm._reflection_text()

    assert "keep Miraidon alive" in text
    assert "head head" not in text


async def _await(predicate, timeout: float = 5.0) -> bool:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


def test_player_fires_the_background_maintenance(tmp_path):
    from types import SimpleNamespace

    from pokeagent.player import PokeAgent
    from pokeagent.tasks.runner import TaskRunner

    async def scenario():
        cm = _cm(tmp_path)
        runner = TaskRunner(AgentConfig(runs_dir=tmp_path / "runs"))
        provider = ScriptedProvider(
            [LLMResponse(text="Preserve Tornadus; Taunt the setter at T4.")]
        )
        runner.register(StrategyCOTAgent(provider, max_tokens=3000))
        # only the attributes _fire_strategy_cot touches
        fake_player = SimpleNamespace(
            runner=runner,
            kb=make_kb(),
            config=cm.config,
            _cot_in_flight={},
            _cot_pending={},
        )
        record = TurnRecord(
            seq=1, turn=2, request="board", decision="move:taunt",
            reason="deny Trick Room", events=["opp1 set Trick Room"],
            thinking="thinking about the speed order...",
        )
        cm.record_turn(record)
        cm.seed_global_cot("opening line", turn=0)

        PokeAgent._fire_strategy_cot(fake_player, make_battle(), cm, record)

        assert await _await(
            lambda: (
                cm.ledger.global_cot is not None
                and "Preserve Tornadus" in cm.ledger.global_cot.text
            )
        )
        assert cm.ledger.global_cot.source == "llm"
        assert cm.ledger.global_cot.seq == record.seq
        assert not any(fake_player._cot_in_flight.values())

    asyncio.run(scenario())


def test_cancelled_battle_does_not_refire_the_maintenance(tmp_path):
    from pokeagent.player import PokeAgent
    from pokeagent.tasks.base import SubAgent
    from pokeagent.tasks.runner import TaskRunner

    class SlowAgent(SubAgent):
        name = "strategy_cot"

        def __init__(self):
            self.started = asyncio.Event()

        async def run(self, sctx):  # pragma: no cover - cancelled mid-flight
            self.started.set()
            await asyncio.sleep(60)
            return "never"

    async def scenario():
        cm = _cm(tmp_path)
        battle = make_battle()
        runner = TaskRunner(AgentConfig(runs_dir=tmp_path / "runs"))
        agent = SlowAgent()
        runner.register(agent)
        fake_player = SimpleNamespace(
            runner=runner,
            kb=make_kb(),
            config=cm.config,
            _cot_in_flight={},
            _cot_pending={},
        )
        first = TurnRecord(seq=1, turn=2, request="board", decision="move:x")
        cm.record_turn(first)
        PokeAgent._fire_strategy_cot(fake_player, battle, cm, first)
        await asyncio.wait_for(agent.started.wait(), timeout=5)

        # a later turn arrives while the first rewrite is still in flight
        later = TurnRecord(seq=2, turn=3, request="board", decision="move:y")
        cm.record_turn(later)
        PokeAgent._fire_strategy_cot(fake_player, battle, cm, later)
        assert fake_player._cot_pending

        runner.cancel_battle(battle.battle_tag)
        await asyncio.sleep(0.05)

        assert fake_player._cot_in_flight == {}
        assert fake_player._cot_pending == {}
        assert runner.submitted_by_tag.get(battle.battle_tag) is None
        assert cm.ledger.global_cot is None

    asyncio.run(scenario())


def test_without_cot_the_brief_still_renders(tmp_path):
    cm = _cm(tmp_path)
    cm.note_extractor = DeterministicNoteExtractor()
    cm.record_turn(TurnRecord(seq=0, turn=1, request="board", reason="hold mid"))
    assert cm.ledger.brief is not None
    text = cm._strategy_text()
    assert "rolling brief" in text