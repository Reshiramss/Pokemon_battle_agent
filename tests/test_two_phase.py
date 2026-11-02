"""Two-phase turn loop, [REFLECTION] block, and reasoning retrieval.

The turn flow splits into an information phase (info tools only) and a
decision phase (submit_decision only) over ONE message list; the boundary
message, the fast path, forced transitions, phase-2 hallucinated tool
calls, and the legacy ``two_phase=False`` single loop are all pinned here,
plus the [REFLECTION] prompt block and the battle_history CoT retrieval.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from helpers import ScriptedProvider, make_battle, make_ctx, make_kb

from pokeagent.config import AgentConfig
from pokeagent.context.manager import (
    ContextManager,
    DECISION_PHASE_NUDGE,
    NUDGE,
)
from pokeagent.context.records import TurnRecord
from pokeagent.hooks.base import HookSystem
from pokeagent.loop import AgentLoop, TurnAbort
from pokeagent.providers.base import LLMResponse, ToolCallRequest
from pokeagent.tools.battle_query import BattleHistoryTool
from pokeagent.tools.base import Tool, ToolError, ToolRegistry


class _EchoTool(Tool):
    """Minimal information tool so cross-phase tool routing is observable."""

    name = "echo"
    description = "test tool"
    input_schema = {
        "type": "object",
        "properties": {"msg": {"type": "string"}},
        "required": ["msg"],
    }

    async def execute(self, ctx, msg: str = "") -> str:
        return f"echo: {msg}"


def make_loop(tmp_path, responses, *, registry=None, **overrides):
    config = AgentConfig(runs_dir=tmp_path / "runs", **overrides)
    provider = ScriptedProvider(responses)
    loop = AgentLoop(
        provider=provider,
        registry=registry if registry is not None else ToolRegistry(),
        hooks=HookSystem(),
        config=config,
        runner=None,
    )
    cm = ContextManager(battle_tag="battle-gen9randombattle-1", config=config)
    return loop, cm, provider


def _text(text="thinking it over"):
    return LLMResponse(text=text, stop_reason="end_turn")


def _submit(move="flamethrower", reason="stub"):
    return LLMResponse(
        tool_calls=[
            ToolCallRequest(
                "c1",
                "submit_decision",
                {
                    "actions": [{"slot": 1, "kind": "move", "move": move}],
                    "reason": reason,
                },
            )
        ],
        stop_reason="tool_use",
    )


def _echo_call(msg="hi"):
    return LLMResponse(
        tool_calls=[ToolCallRequest("c2", "echo", {"msg": msg})],
        stop_reason="tool_use",
    )


def _last_user(call):
    return [m for m in call["messages"] if type(m).__name__ == "UserMessage"][-1]


def _tool_names(call):
    return [t.name for t in call["tools"]]


def _llm_entries(cm):
    return [
        json.loads(line)
        for line in (cm.run_dir / "llm.jsonl").read_text(encoding="utf-8").splitlines()
    ]


# --- phase machinery ---------------------------------------------------------


def test_boundary_message_and_decision_tools(tmp_path):
    """Natural transition: the next complete() sees the boundary message and
    a submit_decision-only tool list."""
    async def scenario():
        registry = ToolRegistry()
        registry.register(_EchoTool())
        loop, cm, provider = make_loop(
            tmp_path, [_text(), _submit()], registry=registry
        )
        outcome = await loop.run_turn(make_battle(), cm)

        assert outcome.decision == "move:flamethrower"
        assert "submit_decision" not in _tool_names(provider.calls[0])
        assert "echo" in _tool_names(provider.calls[0])
        assert _tool_names(provider.calls[1]) == ["submit_decision"]
        assert _last_user(provider.calls[1]).content == DECISION_PHASE_NUDGE

        phases = [e.get("phase") for e in _llm_entries(cm)]
        assert "info" in phases and "decision" in phases

    asyncio.run(scenario())


def test_info_tools_execute_in_phase_one(tmp_path):
    async def scenario():
        registry = ToolRegistry()
        registry.register(_EchoTool())
        loop, cm, provider = make_loop(
            tmp_path, [_echo_call("check"), _text(), _submit()], registry=registry
        )
        outcome = await loop.run_turn(make_battle(), cm)
        assert outcome.rounds == 3
        # the echo result survived the transition into the decision phase
        decision_msgs = provider.calls[2]["messages"]
        assert any(
            getattr(m, "content", "") == "echo: check" for m in decision_msgs
        )
        assert any(c["name"] == "echo" for c in outcome.tool_calls)

    asyncio.run(scenario())


def test_fast_path_submit_during_info_phase(tmp_path):
    """A valid early submit commits without any boundary round trip."""
    async def scenario():
        loop, cm, provider = make_loop(tmp_path, [_submit()])
        outcome = await loop.run_turn(make_battle(), cm)
        assert outcome.decision == "move:flamethrower"
        assert len(provider.calls) == 1

    asyncio.run(scenario())


def test_invalid_fast_submit_retries_in_decision_phase(tmp_path):
    async def scenario():
        loop, cm, provider = make_loop(
            tmp_path, [_submit(move="notamove"), _submit()]
        )
        outcome = await loop.run_turn(make_battle(), cm)
        assert len(provider.calls) == 2
        assert _tool_names(provider.calls[1]) == ["submit_decision"]
        error = [m for m in provider.calls[1]["messages"] if getattr(m, "is_error", False)]
        assert "Invalid move 'notamove'" in error[-1].content
        assert outcome.decision == "move:flamethrower"
        assert outcome.rounds == 2

    asyncio.run(scenario())


def test_info_rounds_exhausted_forces_transition(tmp_path):
    async def scenario():
        registry = ToolRegistry()
        registry.register(_EchoTool())
        loop, cm, provider = make_loop(
            tmp_path, [_echo_call("a"), _submit()],
            registry=registry, info_rounds=1,
        )
        outcome = await loop.run_turn(make_battle(), cm)
        boundary = _last_user(provider.calls[1])
        assert "information-gathering limit" in boundary.content
        assert outcome.decision == "move:flamethrower"

    asyncio.run(scenario())


def test_time_budget_forces_transition(tmp_path):
    """Past 0.6 x turn_budget in the info phase, the handoff is forced."""
    async def scenario():
        registry = ToolRegistry()
        registry.register(_EchoTool())

        class _ClockAdvancer(ScriptedProvider):
            async def complete(self, **kw):
                self.calls.append(kw)
                clock["t"] += 40  # default deadline is 45 * 0.6 = 27s
                if not self.responses:
                    return LLMResponse(stop_reason="end_turn")
                item = self.responses.pop(0)
                return item

        clock = {"t": 0.0}
        provider = _ClockAdvancer([_echo_call("a"), _submit()])
        config = AgentConfig(runs_dir=tmp_path / "runs")
        loop = AgentLoop(
            provider=provider,
            registry=registry,
            hooks=HookSystem(),
            config=config,
            runner=None,
        )
        cm = ContextManager(battle_tag="battle-1", config=config)

        import pokeagent.loop as loop_mod
        real_time = loop_mod.time
        loop_mod.time = type("T", (), {"monotonic": staticmethod(lambda: clock["t"])})
        try:
            outcome = await loop.run_turn(make_battle(), cm)
        finally:
            loop_mod.time = real_time

        boundary = _last_user(provider.calls[1])
        assert "information-gathering limit" in boundary.content
        assert outcome.decision == "move:flamethrower"

    asyncio.run(scenario())


def test_decision_phase_hallucinated_tool_gets_error(tmp_path):
    """Info tools called in the decision phase get errors, not crashes."""
    async def scenario():
        registry = ToolRegistry()
        registry.register(_EchoTool())
        loop, cm, provider = make_loop(
            tmp_path,
            [_text(), _echo_call("sneaky"), _submit()],
            registry=registry,
        )
        outcome = await loop.run_turn(make_battle(), cm)
        errors = [
            m
            for m in provider.calls[2]["messages"]
            if getattr(m, "is_error", False)
        ]
        assert "not available in the decision phase" in errors[-1].content
        assert outcome.decision == "move:flamethrower"

    asyncio.run(scenario())


def test_nudge_on_plain_text_in_decision_phase(tmp_path):
    async def scenario():
        loop, cm, provider = make_loop(
            tmp_path, [_text(), _text("still deciding"), _submit()]
        )
        outcome = await loop.run_turn(make_battle(), cm)
        assert _last_user(provider.calls[2]).content == NUDGE
        assert outcome.decision == "move:flamethrower"

    asyncio.run(scenario())


def test_decision_rounds_exhausted_aborts(tmp_path):
    async def scenario():
        loop, cm, provider = make_loop(
            tmp_path, [_text()] + [_text()] * 3, decision_rounds=2
        )
        with pytest.raises(TurnAbort):
            await loop.run_turn(make_battle(), cm)
        # one info round (natural transition) + two empty decision rounds
        assert len(provider.calls) == 3

    asyncio.run(scenario())


def test_provider_error_in_info_phase_aborts(tmp_path):
    async def scenario():
        loop, cm, _provider = make_loop(tmp_path, [RuntimeError("api down")])
        with pytest.raises(TurnAbort):
            await loop.run_turn(make_battle(), cm)

    asyncio.run(scenario())


def test_legacy_single_loop_when_two_phase_off(tmp_path):
    """two_phase=False reproduces the old flow: everything in one tool list,
    no boundary message, NUDGE on plain text."""
    async def scenario():
        registry = ToolRegistry()
        registry.register(_EchoTool())
        loop, cm, provider = make_loop(
            tmp_path, [_text(), _submit()],
            registry=registry, two_phase=False, max_tool_rounds=6,
        )
        outcome = await loop.run_turn(make_battle(), cm)
        assert "submit_decision" in _tool_names(provider.calls[0])
        assert "echo" in _tool_names(provider.calls[0])
        contents = [
            getattr(m, "content", "") or ""
            for call in provider.calls
            for m in call["messages"]
        ]
        assert not any("[DECISION PHASE]" in c for c in contents)
        assert _last_user(provider.calls[1]).content == NUDGE
        assert outcome.decision == "move:flamethrower"

    asyncio.run(scenario())


# --- [REFLECTION] block --------------------------------------------------------


def _record(**kwargs):
    base = dict(
        seq=0, turn=1, request="[NOW] something", events=["|damage|p1a: 50"],
        decision="move:flamethrower", reason="super effective",
        cot="I checked the type chart first.", elapsed_s=1.0,
    )
    base.update(kwargs)
    return TurnRecord(**base)


def make_cm(tmp_path, **overrides):
    config = AgentConfig(runs_dir=tmp_path / "runs", **overrides)
    return ContextManager(battle_tag="battle-1", config=config)


def test_reflection_absent_on_first_decision(tmp_path):
    cm = make_cm(tmp_path)
    prompt = cm.assemble_prompt(make_battle(), [])[1][0].content
    assert "[REFLECTION" not in prompt


def test_reflection_present_from_second_decision(tmp_path):
    cm = make_cm(tmp_path)
    cm.record_turn(_record())
    prompt = cm.assemble_prompt(make_battle(), [])[1][0].content
    assert "[REFLECTION — your last decision and its outcome]" in prompt
    assert "move:flamethrower — super effective" in prompt
    assert "I checked the type chart first." in prompt
    assert "What happened: |damage|p1a: 50" in prompt
    # reflection sits between history and the current decision (the plain
    # "[NOW" would match the verbatim request text inside [HISTORY])
    assert (
        prompt.index("[HISTORY]")
        < prompt.index("[REFLECTION")
        < prompt.index("[NOW — your current decision]")
    )


def test_reflection_fallback_variant(tmp_path):
    cm = make_cm(tmp_path)
    cm.record_turn(
        _record(decision="random_fallback", reason=None, cot=None,
                fell_back=True, fallback_reason="turn_budget_exceeded")
    )
    prompt = cm.assemble_prompt(make_battle(), [])[1][0].content
    assert "FAILED" in prompt
    assert "turn_budget_exceeded" in prompt
    assert "Do not repeat that failure" in prompt


def test_reflection_cot_is_excerpted(tmp_path):
    cm = make_cm(tmp_path)
    cm.record_turn(_record(cot="x" * 2000))
    prompt = cm.assemble_prompt(make_battle(), [])[1][0].content
    assert "x" * 600 in prompt
    assert "x" * 601 not in prompt


# --- battle_history CoT retrieval (memory management) ---------------------------


def test_battle_history_seq_and_cot(tmp_path):
    cm = make_cm(tmp_path)
    cm.record_turn(_record(cot="y" * 3000))
    ctx = make_ctx(make_battle(), make_kb(), cm=cm)

    # default: no CoT in the full record
    plain = asyncio.run(BattleHistoryTool().execute(ctx, turn=1))
    assert "Your reasoning (CoT)" not in plain

    with_cot = asyncio.run(
        BattleHistoryTool().execute(ctx, turn=1, include_cot=True)
    )
    assert "Your reasoning (CoT): " + "y" * 2000 in with_cot
    assert "y" * 2001 not in with_cot  # clipped at 2000 chars

    # seq lookup matches the same record (and takes precedence)
    by_seq = asyncio.run(BattleHistoryTool().execute(ctx, seq=1))
    assert "record #1" in by_seq

    with pytest.raises(ToolError):
        asyncio.run(BattleHistoryTool().execute(ctx, seq=99))


def test_battle_history_log_mentions_fallbacks(tmp_path):
    cm = make_cm(tmp_path)
    cm.record_turn(_record())
    cm.record_turn(
        _record(turn=2, decision="random_fallback", reason=None, fell_back=True,
                fallback_reason="turn_budget_exceeded")
    )
    ctx = make_ctx(make_battle(), make_kb(), cm=cm)
    log = asyncio.run(BattleHistoryTool().execute(ctx))
    assert "FALLBACK (turn_budget_exceeded)" in log
