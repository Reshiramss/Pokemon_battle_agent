"""Shared test helpers: hand-built battles, scripted LLM stub, no server needed."""

from __future__ import annotations

import logging

from poke_env.battle import AbstractBattle, Battle, Pokemon

from pokeagent.config import AgentConfig
from pokeagent.knowledge.static import StaticKB
from pokeagent.providers.base import (
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)
from pokeagent.tools.base import ToolContext

LOGGER = logging.getLogger("test_battle")


class ScriptedProvider(LLMProvider):
    """Returns canned LLMResponses in order; records every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, *, system, messages, tools, max_tokens):
        self.calls.append(
            {
                "system": system,
                "messages": list(messages),
                "tools": list(tools),
                "max_tokens": max_tokens,
            }
        )
        if not self.responses:
            return LLMResponse(stop_reason="end_turn")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            return LLMResponse(stop_reason="error", error=str(item))
        return item


class DecisionStubProvider(LLMProvider):
    """Offline decision-maker: parses the [NOW] section and submits move #1.

    Drives a full real battle through the whole agent loop with zero LLM
    cost — used by the end-to-end test to verify plumbing, records, and
    postmortem without an API key.
    """

    def __init__(self):
        self.calls: list[str] = []  # one entry per turn: "turn:move" or "turn:switch"

    async def complete(self, *, system, messages, tools, max_tokens):
        latest = None
        for msg in messages:
            text = getattr(msg, "content", None)
            if text:
                latest = text
        # only parse the CURRENT turn's options — the prompt also carries
        # verbatim history whose [NOW] sections would resolve to stale moves
        if latest and "[NOW" in latest:
            latest = latest.rsplit("[NOW", 1)[-1]
        move_id = None
        switch = None
        if latest and "Moves:" in latest:
            for line in latest.splitlines():
                line = line.strip()
                if line.startswith("- ") and "|" in line and "|" in line[2:]:
                    move_id = line[2:].split("|")[0].strip()
                    break
        if move_id is None and latest and "Switches:" in latest:
            for line in latest.splitlines():
                line = line.strip()
                if line.startswith("- "):
                    switch = line[2:].split()[0]
                    break
        if move_id:
            self.calls.append(f"move:{move_id}")
            arguments = {"action": "move", "target": move_id,
                         "reason": "stub chose first available move"}
        elif switch:
            self.calls.append(f"switch:{switch}")
            arguments = {"action": "switch", "target": switch,
                         "reason": "stub forced to switch"}
        else:
            return LLMResponse(stop_reason="end_turn", text="no options parsed")
        return LLMResponse(
            text=f"Reasoning for {self.calls[-1]}.",
            tool_calls=[ToolCallRequest("stub-1", "submit_decision", arguments)],
            stop_reason="tool_use",
        )


def make_battle(*, with_moves: bool = True) -> AbstractBattle:
    """A minimal gen9 battle: active Furret (known stats) vs Torterra (unknown stats)."""
    battle = Battle(
        battle_tag="battle-gen9randombattle-1",
        username="AgentUser",
        logger=LOGGER,
        save_replays=False,
        gen=9,
    )
    battle._format = "gen9randombattle"
    battle._player_role = "p1"

    attacker = Pokemon(9, species="furret", name="Furret")
    attacker._level = 50
    attacker._max_hp = 170
    attacker._current_hp = 170
    attacker._stats = {
        "hp": 170, "atk": 96, "def": 82, "spa": 62, "spd": 72, "spe": 110,
    }
    attacker._add_move("flamethrower")
    if with_moves:
        attacker._add_move("quickattack")
        battle._available_moves = [
            attacker.moves["flamethrower"],
            attacker.moves["quickattack"],
        ]
    attacker._active = True

    defender = Pokemon(9, species="torterra", name="Torterra")
    defender._level = 50
    defender._add_move("stoneedge")
    defender._active = True

    battle._team = {"p1: Furret": attacker}
    battle._opponent_team = {"p2: Torterra": defender}
    return battle


def make_kb() -> StaticKB:
    return StaticKB("gen9randombattle")


def make_ctx(battle, kb, cm=None, config=None) -> ToolContext:
    return ToolContext(
        battle=battle, kb=kb, context=cm, config=config or AgentConfig()
    )
