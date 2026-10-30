"""Unit tests for the OpenAI-compatible provider translation layer (offline).

The client is stubbed with a canned-response fake, so no network or API key
is needed; these pin the wire-format contract (message/tool serialization,
finish_reason mapping, argument parsing, never-raise error path).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from pokeagent.config import AgentConfig
from pokeagent.providers.base import (
    AssistantMessage,
    ToolCallRequest,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)
from pokeagent.providers.openai_provider import OpenAIProvider


def make_provider(responses: list[Any] | None = None, exc: Exception | None = None):
    """Build a provider whose client returns canned responses / raises."""
    config = AgentConfig(base_url="http://stub/v1")
    provider = OpenAIProvider(config)
    captured: dict = {}

    class _FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            if exc is not None:
                raise exc
            item = responses.pop(0)
            return item

    class _FakeChat:
        completions = _FakeCompletions()

    provider._client = SimpleNamespace(chat=_FakeChat())
    return provider, captured


def _wire_response(
    content: str | None,
    finish_reason: str,
    tool_calls: list[dict] | None = None,
) -> SimpleNamespace:
    calls = [
        SimpleNamespace(
            id=c["id"], function=SimpleNamespace(name=c["name"], arguments=c["args"])
        )
        for c in (tool_calls or [])
    ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, tool_calls=calls),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )


# --- message translation -----------------------------------------------------


def test_translate_messages_roundtrip():
    messages = [
        UserMessage("attack?"),
        AssistantMessage(
            content="checking",
            tool_calls=[
                ToolCallRequest("call-1", "damage_calc", {"attacker": "Furret"})
            ],
        ),
        ToolResultMessage("call-1", "damage_calc", "84-100 damage"),
        AssistantMessage("final answer"),
    ]
    wire = OpenAIProvider._translate_messages(messages)
    assert [m["role"] for m in wire] == ["user", "assistant", "tool", "assistant"]
    assert wire[1]["tool_calls"][0]["function"]["name"] == "damage_calc"
    assert json.loads(wire[1]["tool_calls"][0]["function"]["arguments"]) == {
        "attacker": "Furret"
    }
    assert wire[2] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "84-100 damage",
    }
    # plain assistant message carries no tool_calls key
    assert "tool_calls" not in wire[3]


def test_translate_tools_is_openai_shape():
    spec = ToolSpec("damage_calc", "calc damage", {"type": "object", "properties": {}})
    wire = OpenAIProvider._translate_tools([spec])
    assert wire == [
        {
            "type": "function",
            "function": {
                "name": "damage_calc",
                "description": "calc damage",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


# --- complete() against canned responses --------------------------------------


def test_complete_maps_tool_call():
    provider, captured = make_provider(
        [
            _wire_response(
                content="checking",
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "call-9",
                        "name": "damage_calc",
                        "args": '{"attacker": "Furret", "defender": "Torterra"}',
                    }
                ],
            )
        ]
    )
    response = asyncio.run(
        provider.complete(
            system="sys",
            messages=[UserMessage("go")],
            tools=[ToolSpec("damage_calc", "d", {})],
            max_tokens=100,
        )
    )
    assert response.stop_reason == "tool_use"
    assert response.tool_calls[0].id == "call-9"
    assert response.tool_calls[0].arguments == {
        "attacker": "Furret",
        "defender": "Torterra",
    }
    assert (response.input_tokens, response.output_tokens) == (11, 7)
    # system message prepended; tools serialized in OpenAI shape
    assert captured["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["tools"][0]["type"] == "function"


def test_complete_maps_stop_and_length():
    provider, _ = make_provider([_wire_response("done", "stop")])
    r = asyncio.run(
        provider.complete(
            system="s", messages=[UserMessage("x")], tools=[], max_tokens=10
        )
    )
    assert r.stop_reason == "end_turn" and r.text == "done"

    provider, captured = make_provider([_wire_response(None, "length")])
    r = asyncio.run(
        provider.complete(
            system="s", messages=[UserMessage("x")], tools=[], max_tokens=10
        )
    )
    assert r.stop_reason == "max_tokens"
    # empty tools list must not be sent
    assert "tools" not in captured


def test_complete_error_paths_never_raise():
    provider, _ = make_provider(exc=RuntimeError("conn refused"))
    r = asyncio.run(
        provider.complete(
            system="s", messages=[UserMessage("x")], tools=[], max_tokens=10
        )
    )
    assert r.stop_reason == "error" and "conn refused" in r.error

    provider, _ = make_provider([_wire_response(None, "content_filter")])
    r = asyncio.run(
        provider.complete(
            system="s", messages=[UserMessage("x")], tools=[], max_tokens=10
        )
    )
    assert r.stop_reason == "error" and "content_filter" in r.error


def test_complete_malformed_json_arguments_become_empty():
    provider, _ = make_provider(
        [
            _wire_response(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[{"id": "c1", "name": "damage_calc", "args": "{oops"}],
            )
        ]
    )
    r = asyncio.run(
        provider.complete(
            system="s",
            messages=[UserMessage("x")],
            tools=[ToolSpec("damage_calc", "d", {})],
            max_tokens=10,
        )
    )
    # empty args -> tool returns a validation error the model can retry on
    assert r.stop_reason == "tool_use"
    assert r.tool_calls[0].arguments == {}
