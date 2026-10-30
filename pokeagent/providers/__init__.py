"""Provider registry and factory."""

from __future__ import annotations

from ..config import AgentConfig
from .anthropic_provider import AnthropicProvider
from .base import LLMProvider
from .openai_provider import OpenAIProvider

PROVIDERS: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}


def get_provider(config: AgentConfig, **kwargs) -> LLMProvider:
    """Instantiate the configured provider. Register new backends in PROVIDERS."""
    cls = PROVIDERS.get(config.provider)
    if cls is None:
        raise ValueError(
            f"Unknown provider '{config.provider}'. Available: {sorted(PROVIDERS)}"
        )
    return cls(config, **kwargs)
