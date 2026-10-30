"""OpenAI-compatible provider — placeholder for round 2.

The normalized message types in ``providers.base`` already absorb the two
API-shape differences (message encoding, tool-argument format); implementing
this class is a matter of translating them to
``openai.AsyncOpenAI().chat.completions.create``.
"""

from __future__ import annotations

from ..config import AgentConfig
from .base import LLMMessage, LLMProvider, LLMResponse, ToolSpec


class OpenAIProvider(LLMProvider):
    def __init__(self, config: AgentConfig, **_kwargs):
        self.model = config.model
        raise NotImplementedError(
            "OpenAI-compatible provider lands in round 2; "
            "set provider='anthropic' or register your own implementation "
            "in pokeagent.providers.PROVIDERS."
        )

    async def complete(
        self,
        *,
        system: str,
        messages: list[LLMMessage],
        tools: list[ToolSpec],
        max_tokens: int,
    ) -> LLMResponse:
        raise NotImplementedError
