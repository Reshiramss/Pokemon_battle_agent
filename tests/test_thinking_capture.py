"""Reasoning-model thinking: captured, logged, and kept out of the prompt.

The gateway in use returns Anthropic ``thinking`` blocks (and OpenAI-style
servers return ``reasoning_content``). That text is not assistant content —
it must not be replayed — but it is the only visible reasoning a reasoning
model gives us, so llm.jsonl records it and the turn record counts its
tokens. Without this the log shows an empty answer next to 4000 output
tokens and there is nothing to debug.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from helpers import ScriptedProvider, make_battle, make_kb  # noqa: F401

from pokeagent.config import AgentConfig
from pokeagent.context.manager import ContextManager
from pokeagent.context.records import TurnRecord
from pokeagent.hooks.base import HookSystem
from pokeagent.loop import AgentLoop
from pokeagent.providers.anthropic_provider import AnthropicProvider
from pokeagent.providers.base import LLMResponse, ToolCallRequest
from pokeagent.providers.openai_provider import OpenAIProvider
from pokeagent.tools.base import ToolRegistry


def _usage(**kw):
    fields = {"input_tokens": 11, "output_tokens": 22}
    fields.update(kw)
    return SimpleNamespace(**fields)


# --- provider parsing ---------------------------------------------------------


def _anthropic(response):
    provider = object.__new__(AnthropicProvider)
    provider.model = "test-model"

    async def create(**kwargs):
        return response

    provider._client = SimpleNamespace(
        messages=SimpleNamespace(create=create)
    )
    return provider


def test_anthropic_thinking_is_separate_from_text():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="Tornadus is faster."),
            SimpleNamespace(type="thinking", thinking="Armor Tail blocks Taunt."),
            SimpleNamespace(type="text", text="Taunt fails; use Knock Off."),
        ],
        stop_reason="end_turn",
        usage=_usage(
            output_tokens_details=SimpleNamespace(thinking_tokens=15)
        ),
    )

    resp = asyncio.run(
        _anthropic(response).complete(system="", messages=[], tools=[], max_tokens=100)
    )

    assert resp.text == "Taunt fails; use Knock Off."
    assert resp.thinking == "Tornadus is faster.\n\nArmor Tail blocks Taunt."
    assert resp.thinking_tokens == 15


def test_anthropic_thinking_only_response_still_reports_thinking():
    """Truncated at max_tokens: text is empty, but the reasoning is not."""
    response = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="Let me compute...")],
        stop_reason="max_tokens",
        usage=_usage(output_tokens=4000, output_tokens_details=SimpleNamespace(thinking_tokens=4000)),
    )

    resp = asyncio.run(
        _anthropic(response).complete(system="", messages=[], tools=[], max_tokens=4000)
    )

    assert resp.text is None
    assert resp.thinking == "Let me compute..."
    assert resp.stop_reason == "max_tokens"
    assert resp.thinking_tokens == 4000


def test_openai_reasoning_content_is_captured():
    message = SimpleNamespace(
        content=None,
        tool_calls=[],
        reasoning_content="Consider the speed tiers...",
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(
            prompt_tokens=5,
            completion_tokens=30,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=28),
        ),
    )
    provider = object.__new__(OpenAIProvider)
    provider.model = "test-model"
    provider.reasoning_effort = None

    async def create(**kwargs):
        return response

    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    resp = asyncio.run(
        provider.complete(system="", messages=[], tools=[], max_tokens=100)
    )

    assert resp.text is None
    assert resp.thinking == "Consider the speed tiers..."
    assert resp.thinking_tokens == 28


# --- loop / logs --------------------------------------------------------------


def _submit():
    return LLMResponse(
        text="Go.",
        thinking="Weighing options before committing.",
        thinking_tokens=40,
        tool_calls=[
            ToolCallRequest(
                "c1",
                "submit_decision",
                {
                    "actions": [{"slot": 1, "kind": "move", "move": "flamethrower"}],
                    "reason": "stub",
                },
            )
        ],
        stop_reason="tool_use",
    )


def _make_loop(tmp_path, responses, **overrides):
    config = AgentConfig(runs_dir=tmp_path / "runs", **overrides)
    loop = AgentLoop(
        provider=ScriptedProvider(responses),
        registry=ToolRegistry(),
        hooks=HookSystem(),
        config=config,
        runner=None,
    )
    cm = ContextManager(battle_tag="battle-gen9randombattle-1", config=config)
    return loop, cm


def _entries(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_whitespace_only_answer_is_not_cot(tmp_path):
    """A single space used to make postmortem claim "CoT logged"."""
    response = _submit()
    response.text = " \n"
    outcome = asyncio.run(_run_with(tmp_path, response))
    assert outcome.cot == ""


async def _run_with(tmp_path, response):
    loop, cm = _make_loop(tmp_path, [response])
    return await loop.run_turn(make_battle(), cm)


def test_thinking_is_logged_but_not_cot(tmp_path):
    async def scenario():
        loop, cm = _make_loop(tmp_path, [_submit()])
        outcome = await loop.run_turn(make_battle(), cm)
        return outcome, cm

    outcome, cm = asyncio.run(scenario())

    # thinking is never assistant text: [REFLECTION]/notes only see the answer
    assert outcome.cot == "Go."
    assert outcome.thinking == "Weighing options before committing."
    assert outcome.thinking_tokens == 40

    responses = [
        e for e in _entries(cm.run_dir / "llm.jsonl")
        if e["type"] == "llm_response"
    ]
    assert responses[0]["thinking"] == "Weighing options before committing."
    assert responses[0]["thinking_tokens"] == 40

    # the per-turn record carries the count (events.jsonl is the lean stream;
    # the thinking text itself stays in llm.jsonl)
    cm.record_turn(TurnRecord(seq=0, turn=1, request="board", thinking_tokens=40))
    turns = [
        e for e in _entries(cm.run_dir / "events.jsonl") if e["type"] == "turn"
    ]
    assert turns[-1]["thinking_tokens"] == 40


def test_log_thinking_can_be_disabled_but_tokens_still_count(tmp_path):
    async def scenario():
        loop, cm = _make_loop(tmp_path, [_submit()], log_thinking=False)
        await loop.run_turn(make_battle(), cm)
        return cm

    cm = asyncio.run(scenario())

    responses = [
        e for e in _entries(cm.run_dir / "llm.jsonl")
        if e["type"] == "llm_response"
    ]
    assert "thinking" not in responses[0]
    assert responses[0]["thinking_tokens"] == 40