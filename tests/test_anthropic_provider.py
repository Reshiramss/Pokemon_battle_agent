"""Credential / endpoint resolution for the Anthropic provider (offline).

The SDK client is replaced with a capturing stub, so these pin how env vars
and config map onto AsyncAnthropic kwargs — in particular the Bearer-token
path (ANTHROPIC_AUTH_TOKEN) that some gateways require instead of x-api-key.
"""

from __future__ import annotations

import anthropic
import pytest

from pokeagent.config import AgentConfig
from pokeagent.providers.anthropic_provider import AnthropicProvider


@pytest.fixture
def capture(monkeypatch):
    captured: dict = {}

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _FakeAnthropic)
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    return captured


def test_auth_token_wins_over_api_key(capture, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "bearer-tok")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x-key")
    AnthropicProvider(AgentConfig())
    assert capture["auth_token"] == "bearer-tok"
    assert "api_key" not in capture


def test_api_key_env_used_when_no_auth_token(capture, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x-key")
    AnthropicProvider(AgentConfig())
    assert capture["api_key"] == "x-key"
    assert "auth_token" not in capture


def test_base_url_from_config_then_env(capture, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "t")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://env-gw")
    AnthropicProvider(AgentConfig(base_url="http://config-gw"))
    assert capture["base_url"] == "http://config-gw"

    AnthropicProvider(AgentConfig())
    assert capture["base_url"] == "http://env-gw"


def test_missing_credentials_raise(capture):
    with pytest.raises(RuntimeError, match="ANTHROPIC_AUTH_TOKEN"):
        AnthropicProvider(AgentConfig())