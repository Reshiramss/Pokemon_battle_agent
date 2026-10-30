"""Anthropic Claude provider (default backend).

Credentials come from the environment: ANTHROPIC_API_KEY (required) and
ANTHROPIC_BASE_URL (optional, e.g. an internal gateway).
"""

from __future__ import annotations

import logging
import os

from ..config import AgentConfig
from .base import (
    AssistantMessage,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        config: AgentConfig,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        import anthropic

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it (or ANTHROPIC_BASE_URL "
                "if you use a gateway) before starting the agent."
            )
        client_kwargs: dict = {"api_key": api_key, "timeout": timeout}
        base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**client_kwargs)
        self.model = config.model

    @staticmethod
    def _translate_messages(messages: list[LLMMessage]) -> list[dict]:
        """Normalized messages -> Anthropic wire format.

        Consecutive same-role messages (e.g. several tool results, or a user
        nudge right after tool results) are merged into one message, as the
        Anthropic API requires role alternation.
        """
        out: list[dict] = []

        def _push(role: str, blocks: list):
            if not blocks:
                return
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": blocks})

        for msg in messages:
            if isinstance(msg, UserMessage):
                _push("user", [{"type": "text", "text": msg.content}])
            elif isinstance(msg, AssistantMessage):
                blocks = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                for call in msg.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                if not blocks:
                    blocks.append({"type": "text", "text": ""})
                _push("assistant", blocks)
            elif isinstance(msg, ToolResultMessage):
                _push(
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.call_id,
                            "content": msg.content,
                            "is_error": msg.is_error,
                        }
                    ],
                )
            else:  # pragma: no cover - defensive
                raise TypeError(f"Unknown message type: {type(msg)!r}")
        return out

    async def complete(
        self,
        *,
        system: str,
        messages: list[LLMMessage],
        tools: list[ToolSpec],
        max_tokens: int,
    ) -> LLMResponse:
        try:
            response = await self._client.messages.create(
                model=self.model,
                system=system,
                messages=self._translate_messages(messages),
                tools=[
                    {
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema,
                    }
                    for t in tools
                ],
                max_tokens=max_tokens,
            )
        except Exception as e:  # network / auth / rate limits — never raise
            logger.warning("Anthropic API error: %s: %s", type(e).__name__, e)
            return LLMResponse(
                stop_reason="error", error=f"{type(e).__name__}: {e}"
            )

        text = "\n".join(b.text for b in response.content if b.type == "text") or None
        tool_calls = [
            ToolCallRequest(id=b.id, name=b.name, arguments=dict(b.input))
            for b in response.content
            if b.type == "tool_use"
        ]
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )
