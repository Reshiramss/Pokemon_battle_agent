"""Phase-1 decision loop tests: team preview lead choice (submit_lead).

Two decision kinds now exist: run_preview (choose the lead) and run_turn
(in-battle). These tests cover the preview loop offline: success, invalid
retry, provider failure, round exhaustion, and the full player path.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from helpers import LOGGER, ScriptedProvider
from poke_env.battle import Battle, Pokemon

from pokeagent.config import AgentConfig
from pokeagent.context.manager import ContextManager
from pokeagent.hooks.base import HookSystem
from pokeagent.loop import AgentLoop, PreviewAbort, PreviewOutcome
from pokeagent.player import PokeAgent
from pokeagent.providers.base import LLMResponse, ToolCallRequest
from pokeagent.tasks.runner import TaskRunner
from pokeagent.tasks.team_preview import TeamPreviewAnalysisAgent
from pokeagent.tools.base import ToolRegistry


def make_preview_battle():
    """A minimal gen9ou battle with full 3v3 teams (preview format)."""
    battle = Battle(
        battle_tag="battle-gen9ou-1",
        username="AgentUser",
        logger=LOGGER,
        save_replays=False,
        gen=9,
    )
    battle._format = "gen9ou"
    battle._player_role = "p1"

    team, opp_team = {}, {}
    for species in ("furret", "torterra", "gyarados"):
        mon = Pokemon(9, species=species, name=species.capitalize())
        mon._level = 50
        mon._add_move("tackle")
        team[f"p1: {species.capitalize()}"] = mon

        foe = Pokemon(9, species=species, name=species.capitalize())
        foe._level = 50
        foe._add_move("tackle")
        opp_team[f"p2: {species.capitalize()}"] = foe
    battle._team = team
    battle._opponent_team = opp_team
    return battle


def make_loop(tmp_path, responses):
    config = AgentConfig(runs_dir=tmp_path / "runs")
    provider = ScriptedProvider(responses)
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry(),
        hooks=HookSystem(),
        config=config,
        runner=None,
    )
    cm = ContextManager(battle_tag="battle-gen9ou-1", config=config)
    return loop, cm, provider


def _lead_response(slot: int, reason: str) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCallRequest("c1", "submit_lead", {"slot": slot, "reason": reason})],
        stop_reason="tool_use",
    )


def test_preview_lead_submitted_and_traced(tmp_path):
    async def scenario():
        loop, cm, _provider = make_loop(
            tmp_path,
            [_lead_response(3, "Gyarados sets up and threatens everything.")],
        )
        outcome = await loop.run_preview(make_preview_battle(), cm)
        assert isinstance(outcome, PreviewOutcome)
        assert outcome.lead == 3
        assert outcome.reason == "Gyarados sets up and threatens everything."
        assert outcome.rounds == 1

        # the llm.jsonl trace carries the full preview conversation
        entries = [
            json.loads(line)
            for line in (cm.run_dir / "llm.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        prompt = next(e for e in entries if e["type"] == "turn_prompt")
        assert prompt["phase"] == "preview"
        assert "[TEAM PREVIEW" in prompt["prompt"]
        response = next(e for e in entries if e["type"] == "llm_response")
        assert response["tool_calls"][0]["name"] == "submit_lead"

    asyncio.run(scenario())


def test_preview_invalid_slot_then_valid(tmp_path):
    async def scenario():
        loop, cm, provider = make_loop(
            tmp_path,
            [
                _lead_response(9, "out of range"),  # invalid: 3-mon team
                _lead_response(2, "settled on Torterra."),
            ],
        )
        outcome = await loop.run_preview(make_preview_battle(), cm)
        assert outcome.lead == 2
        assert outcome.rounds == 2
        # the retry round saw the validation error as a tool result
        second_call = provider.calls[1]
        errors = [
            m for m in second_call["messages"] if getattr(m, "is_error", False)
        ]
        assert errors and "between 1 and" in errors[-1].content

    asyncio.run(scenario())


def test_preview_provider_error_aborts(tmp_path):
    async def scenario():
        loop, cm, _provider = make_loop(tmp_path, [RuntimeError("provider down")])
        with pytest.raises(PreviewAbort):
            await loop.run_preview(make_preview_battle(), cm)

    asyncio.run(scenario())


def test_preview_exhausts_rounds_after_nudges(tmp_path):
    async def scenario():
        loop, cm, provider = make_loop(tmp_path, [])  # always plain text
        with pytest.raises(PreviewAbort):
            await loop.run_preview(make_preview_battle(), cm)
        assert len(provider.calls) == loop.config.max_tool_rounds

    asyncio.run(scenario())


def test_player_teampreview_decides_lead(tmp_path):
    async def scenario():
        config = AgentConfig(runs_dir=tmp_path / "runs", battle_format="gen9ou")
        stub = ScriptedProvider(
            [_lead_response(2, "Torterra tanks hits and sets Stealth Rock.")]
        )
        sub = ScriptedProvider([LLMResponse(text="plan", stop_reason="end_turn")])
        runner = TaskRunner(config, provider=sub)
        runner.register(TeamPreviewAnalysisAgent(sub))
        agent = PokeAgent(config, provider=stub, runner=runner, max_concurrent_battles=1)

        battle = make_preview_battle()
        order = await agent.teampreview(battle)
        assert order.startswith("/team ")

        cm = agent._contexts[battle.battle_tag]
        events = (cm.run_dir / "events.jsonl").read_text(encoding="utf-8")
        assert '"team_preview"' in events
        assert '"lead": 2' in events
        assert order == "/team 213"

    asyncio.run(scenario())


def test_player_teampreview_falls_back_on_error(tmp_path):
    async def scenario():
        config = AgentConfig(runs_dir=tmp_path / "runs", battle_format="gen9ou")
        stub = ScriptedProvider([RuntimeError("provider down")])
        sub = ScriptedProvider([LLMResponse(text="plan", stop_reason="end_turn")])
        runner = TaskRunner(config, provider=sub)
        runner.register(TeamPreviewAnalysisAgent(sub))
        agent = PokeAgent(config, provider=stub, runner=runner, max_concurrent_battles=1)

        battle = make_preview_battle()
        order = await agent.teampreview(battle)
        assert order.startswith("/team ")  # base class random order

        cm = agent._contexts[battle.battle_tag]
        events = (cm.run_dir / "events.jsonl").read_text(encoding="utf-8")
        assert '"preview_fallback"' in events

    asyncio.run(scenario())
