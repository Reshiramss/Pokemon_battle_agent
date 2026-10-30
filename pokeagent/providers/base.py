"""Provider-agnostic LLM interface.

The framework speaks ONLY in normalized messages (UserMessage /
AssistantMessage / ToolResultMessage); each provider translates them to and
from its native wire format. Adding a backend means implementing
``LLMProvider.complete`` and registering it in the factory — nothing else in
the framework changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    """A tool as announced to the LLM (JSON-Schema input)."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCallRequest:
    """A single tool call requested by the model in one assistant turn."""

    id: str  # provider call id, echoed in the matching ToolResultMessage
    name: str
    arguments: dict[str, Any]  # provider already parsed the JSON


@dataclass
class LLMResponse:
    text: str | None = None  # assistant reasoning / plain text (the CoT)
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    # normalized stop reasons: end_turn | tool_use | max_tokens | error
    stop_reason: str = "end_turn"
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class UserMessage:
    content: str


@dataclass
class AssistantMessage:
    content: str | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)


@dataclass
class ToolResultMessage:
    call_id: str
    tool_name: str
    content: str
    is_error: bool = False


LLMMessage = UserMessage | AssistantMessage | ToolResultMessage


def _message_size(msg: LLMMessage) -> int:
    if isinstance(msg, (UserMessage, ToolResultMessage)):
        return len(msg.content)
    size = len(msg.content or "")
    for call in msg.tool_calls:
        size += len(str(call.arguments))
    return size


class LLMProvider(ABC):
    """Abstract chat backend with tool calling."""

    @abstractmethod
    async def complete(
        self,
        *,
        system: str,
        messages: list[LLMMessage],
        tools: list[ToolSpec],
        max_tokens: int,
    ) -> LLMResponse:
        """Run one model turn. Must never raise: return stop_reason="error" instead."""

    async def count_tokens(self, messages: list[LLMMessage], system: str = "") -> int:
        """Cheap estimate; providers may override with exact counting."""
        total = len(system) + sum(_message_size(m) for m in messages)
        return total // 4
