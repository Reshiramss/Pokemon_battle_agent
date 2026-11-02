"""Phase-1 decision loop tests: team preview selection (submit_team).

Two decision kinds now exist: run_preview (pick the team/leads) and
run_turn (in-battle). These tests cover the preview loop offline: success
(singles + doubles pick-4), invalid retries, provider failure, round
exhaustion, and the full player path.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from helpers import LOGGER, ScriptedProvider
from poke_env.battle import Battle, DoubleBattle, Pokemon

from pokeagent.config import AgentConfig
from pokeagent.context.manager import ContextManager
from pokeagent.hooks.base import HookSystem
from pokeagent.loop import AgentLoop, PreviewAbort, PreviewOutcome
from pokeagent.player import PokeAgent
from pokeagent.providers.base import LLMResponse, ToolCallRequest
from pokeagent.tasks.runner import TaskRunner
from pokeagent.tasks.team_preview import TeamPreviewAnalysisAgent
from pokeagent.tools.base import ToolRegistry


def _preview_mon(gen, species: str) -> Pokemon:
    mon = Pokemon(gen, species=species, name=species.capitalize())
    mon._level = 50
    mon._add_move("tackle")
    return mon


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

    species = ("furret", "torterra", "gyarados")
    battle._team = {
        f"p1: {s.capitalize()}": _preview_mon(9, s) for s in species
    }
    battle._opponent_team = {
        f"p2: {s.capitalize()}": _preview_mon(9, s) for s in species
    }
    return battle


def make_preview_double_battle():
    """A minimal VGC-format battle: 6v6 shown, pick 4 (max_team_size=4)."""
    battle = DoubleBattle(
        battle_tag="battle-gen9vgc2025regihidden-1",
        username="AgentUser",
        logger=LOGGER,
        gen=9,
        save_replays=False,
    )
    battle._format = "gen9vgc2025regihidden"
    battle._player_role = "p1"
    battle._max_team_size = 4

    species = ("furret", "torterra", "gyarados", "incineroar", "amoonguss", "miraidon")
    battle._team = {
        f"p1: {s.capitalize()}": _preview_mon(9, s) for s in species
    }
    battle._opponent_team = {
        f"p2: {s.capitalize()}": _preview_mon(9, s) for s in species
    }
    return battle


def make_loop(tmp_path, responses, battle_tag="battle-gen9ou-1"):
    config = AgentConfig(runs_dir=tmp_path / "runs")
    provider = ScriptedProvider(responses)
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry(),
        hooks=HookSystem(),
        config=config,
        runner=None,
    )
    cm = ContextManager(battle_tag=battle_tag, config=config)
    return loop, cm, provider


def _team_response(order: list[int], reason: str) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCallRequest("c1", "submit_team", {"order": order, "reason": reason})],
        stop_reason="tool_use",
    )


def test_preview_team_submitted_and_traced(tmp_path):
    async def scenario():
        loop, cm, _provider = make_loop(
            tmp_path,
            [_team_response([3, 2, 1], "Gyarados sets up and threatens everything.")],
        )
        outcome = await loop.run_preview(make_preview_battle(), cm)
        assert isinstance(outcome, PreviewOutcome)
        # singles: only the lead matters; the rest keeps team index order
        assert outcome.order_list == [3, 1, 2]
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
        assert response["tool_calls"][0]["name"] == "submit_team"

    asyncio.run(scenario())


def test_preview_doubles_picks_four_leads_first(tmp_path):
    async def scenario():
        loop, cm, _provider = make_loop(
            tmp_path,
            [
                _team_response([5, 2], "too short"),  # invalid: needs exactly 4
                _team_response([5, 2, 6, 1], "Miraidon+Incineroar lead."),
            ],
            battle_tag="battle-gen9vgc2025regihidden-1",
        )
        outcome = await loop.run_preview(make_preview_double_battle(), cm)
        # doubles: the picked order is preserved verbatim, leads first
        assert outcome.order_list == [5, 2, 6, 1]
        assert outcome.rounds == 2

    asyncio.run(scenario())


def test_preview_invalid_slot_then_valid(tmp_path):
    async def scenario():
        loop, cm, provider = make_loop(
            tmp_path,
            [
                _team_response([9], "out of range"),  # invalid: 3-mon team
                _team_response([2], "settled on Torterra."),
            ],
        )
        outcome = await loop.run_preview(make_preview_battle(), cm)
        assert outcome.order_list == [2, 1, 3]
        assert outcome.rounds == 2
        # the retry round saw the validation error as a tool result
        second_call = provider.calls[1]
        errors = [
            m for m in second_call["messages"] if getattr(m, "is_error", False)
        ]
        assert errors and "between 1 and" in errors[-1].content

    asyncio.run(scenario())


def test_preview_duplicate_slots_rejected(tmp_path):
    async def scenario():
        loop, cm, provider = make_loop(
            tmp_path,
            [
                _team_response([1, 1, 2], "dupes"),
                _team_response([2], "fixed."),
            ],
        )
        outcome = await loop.run_preview(make_preview_battle(), cm)
        assert outcome.order_list == [2, 1, 3]
        second_call = provider.calls[1]
        errors = [
            m for m in second_call["messages"] if getattr(m, "is_error", False)
        ]
        assert errors and "duplicate" in errors[-1].content

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
            [_team_response([2], "Torterra tanks hits and sets Stealth Rock.")]
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
        assert '"order_list": [2, 1, 3]' in events
        assert order == "/team 213"

    asyncio.run(scenario())


def test_player_teampreview_doubles_picks_four(tmp_path):
    async def scenario():
        config = AgentConfig(runs_dir=tmp_path / "runs")
        stub = ScriptedProvider(
            [_team_response([4, 2, 6, 1], "Incineroar pivot lead with Gyarados backup.")]
        )
        sub = ScriptedProvider([LLMResponse(text="plan", stop_reason="end_turn")])
        runner = TaskRunner(config, provider=sub)
        runner.register(TeamPreviewAnalysisAgent(sub))
        agent = PokeAgent(config, provider=stub, runner=runner, max_concurrent_battles=1)

        battle = make_preview_double_battle()
        order = await agent.teampreview(battle)
        assert order == "/team 4261"

        cm = agent._contexts[battle.battle_tag]
        events = (cm.run_dir / "events.jsonl").read_text(encoding="utf-8")
        assert '"leads": [4, 2]' in events
        # the picked 4 are flagged, the benched 2 are not
        team_list = list(battle.team.values())
        assert team_list[0]._selected_in_teampreview  # furret (slot 1)
        assert team_list[1]._selected_in_teampreview  # torterra (slot 2)
        assert team_list[3]._selected_in_teampreview  # incineroar (slot 4)
        assert team_list[5]._selected_in_teampreview  # miraidon (slot 6)
        assert not team_list[2]._selected_in_teampreview  # gyarados benched
        assert not team_list[4]._selected_in_teampreview  # amoonguss benched

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
