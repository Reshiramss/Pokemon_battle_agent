"""OpenAI-compatible provider — chat.completions wire format with tools.

Works with any OpenAI-compatible endpoint (vLLM, OpenRouter, OpenAI itself).
Endpoint and credentials resolve in order: explicit kwargs -> config fields
-> OPENAI_API_KEY / OPENAI_BASE_URL env vars. Local vLLM endpoints don't
check the key, so a placeholder is used when none is configured.
"""

from __future__ import annotations

import json
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

# OpenAI finish_reason -> framework stop_reason
_FINISH_REASONS = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}


def _reasoning_text(message) -> str | None:
    """Reasoning-model thinking from an OpenAI-shaped message, if any.

    Different servers park it in different fields: DeepSeek uses
    ``reasoning_content``, vLLM/OpenRouter use ``reasoning``, some gateways
    send a ``reasoning_details`` list. The OpenAI SDK keeps unmodelled fields
    in ``model_extra``, so both places are checked.
    """
    extra = getattr(message, "model_extra", None) or {}
    for key in ("reasoning_content", "reasoning"):
        value = getattr(message, key, None) or extra.get(key)
        if isinstance(value, str) and value.strip():
            return value
    details = extra.get("reasoning_details") or getattr(message, "reasoning_details", None)
    if isinstance(details, list):
        parts = [
            d.get("text") or d.get("summary")
            for d in details
            if isinstance(d, dict)
        ]
        joined = "\n\n".join(p for p in parts if p)
        if joined:
            return joined
    return None


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        config: AgentConfig,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        import openai

        api_key = (
            api_key
            or config.api_key
            or os.environ.get("OPENAI_API_KEY")
            or "EMPTY"  # local endpoints (vLLM) don't check the key
        )
        base_url = (
            base_url
            or config.base_url
            or os.environ.get("OPENAI_BASE_URL")
            or None  # unset -> api.openai.com
        )
        client_kwargs: dict = {"api_key": api_key, "timeout": timeout}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**client_kwargs)
        self.model = config.model
        # e.g. deepseek-flash "low"|"high"|"max"; None -> param not sent
        self.reasoning_effort = getattr(config, "reasoning_effort", None)

    @staticmethod
    def _translate_messages(messages: list[LLMMessage]) -> list[dict]:
        """Normalized messages -> OpenAI wire format.

        Tool results become ``role: "tool"`` messages that must follow the
        assistant message holding the matching ``tool_calls`` — the agent
        loop already guarantees that ordering.
        """
        out: list[dict] = []
        for msg in messages:
            if isinstance(msg, UserMessage):
                out.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AssistantMessage):
                entry: dict = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in msg.tool_calls
                    ]
                elif not msg.content:
                    entry["content"] = ""  # neither text nor calls: keep the key non-null
                out.append(entry)
            elif isinstance(msg, ToolResultMessage):
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.call_id,
                        "content": msg.content,
                    }
                )
            else:  # pragma: no cover - defensive
                raise TypeError(f"Unknown message type: {type(msg)!r}")
        return out

    @staticmethod
    def _translate_tools(tools: list[ToolSpec]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    async def complete(
        self,
        *,
        system: str,
        messages: list[LLMMessage],
        tools: list[ToolSpec],
        max_tokens: int,
    ) -> LLMResponse:
        wire_messages = [{"role": "system", "content": system}]
        wire_messages.extend(self._translate_messages(messages))
        kwargs: dict = {
            "model": self.model,
            "messages": wire_messages,
            "max_tokens": max_tokens,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if tools:
            kwargs["tools"] = self._translate_tools(tools)
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as e:  # network / auth / rate limits — never raise
            logger.warning("OpenAI API error: %s: %s", type(e).__name__, e)
            return LLMResponse(
                stop_reason="error", error=f"{type(e).__name__}: {e}"
            )

        if not response.choices:  # pragma: no cover - defensive
            return LLMResponse(stop_reason="error", error="empty response.choices")
        choice = response.choices[0]
        finish = choice.finish_reason or "stop"
        if finish not in _FINISH_REASONS:  # content_filter, function_call legacy...
            return LLMResponse(
                stop_reason="error", error=f"unusable finish_reason: {finish}"
            )

        message = choice.message
        thinking = _reasoning_text(message)
        tool_calls: list[ToolCallRequest] = []
        for call in getattr(message, "tool_calls", None) or []:
            raw = getattr(call.function, "arguments", "") or ""
            try:
                arguments = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                # Malformed JSON: emit empty args so the tool returns a
                # validation error result and the model re-emits the call.
                logger.warning("unparseable tool arguments from %s: %r", self.model, raw)
                arguments = {}
            tool_calls.append(
                ToolCallRequest(
                    id=call.id, name=call.function.name, arguments=arguments
                )
            )

        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        return LLMResponse(
            text=message.content or None,
            thinking=thinking,
            tool_calls=tool_calls,
            stop_reason=_FINISH_REASONS[finish],
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            thinking_tokens=getattr(details, "reasoning_tokens", 0) or 0,
        )
