"""Mechanics knowledge module: fuzzy topic matching, the lookup tool,
and the doubles hook with its situational weather/field lines."""

from __future__ import annotations

import asyncio

import pytest
from helpers import make_battle, make_ctx, make_double_battle, make_kb
from poke_env.battle import Field, Weather

from pokeagent.knowledge.mechanics import get_mechanics_kb
from pokeagent.tools.base import ToolError
from pokeagent.tools.mechanics_info import MechanicsInfoTool


# --- fuzzy matching -----------------------------------------------------------


def test_match_by_alias_and_fragments():
    kb = get_mechanics_kb()
    assert kb.match("speed").id == "turn_order"
    assert kb.match("Trick Room!").id == "turn_order"
    assert kb.match("boost").id == "stat_stages"
    assert kb.match("terablast").id == "tera"
    assert kb.match("spread").id == "spread_moves"
    assert kb.match("nonsense-topic") is None
    assert kb.match("") is None


def test_core_doubles_lines_are_summaries():
    kb = get_mechanics_kb()
    lines = kb.core_doubles_lines()
    assert len(lines) == 3
    assert all(len(line) < 300 for line in lines)
    assert any("Turn order" in line for line in lines)
    assert any("0.75" in line for line in lines)


# --- tool ----------------------------------------------------------------------


def test_tool_lists_and_looks_up():
    async def scenario():
        ctx = make_ctx(make_battle(), make_kb())
        tool = MechanicsInfoTool()
        listing = await tool.execute(ctx)
        assert "turn_order" in listing and "tera" in listing

        full = await tool.execute(ctx, topic="spread moves")
        assert "0.75x" in full
        # details beyond the summary are included
        assert "Wide Guard" in full

        with pytest.raises(ToolError) as exc:
            await tool.execute(ctx, topic="digivolution")
        assert "'tera'" in str(exc.value)

    asyncio.run(scenario())


# --- hook ----------------------------------------------------------------------


def _hook_output(battle):
    from pokeagent.hooks.mechanics_hook import MechanicsHook

    async def scenario():
        ctx = make_ctx(battle, make_kb())
        return await MechanicsHook().run(ctx)

    return asyncio.run(scenario())


def test_hook_core_lines_for_doubles():
    out = _hook_output(make_double_battle())
    assert out is not None
    assert "Turn order and speed" in out
    assert "0.75x" in out
    assert "Protect" in out


def test_hook_contextual_weather_and_field_lines():
    battle = make_double_battle()
    battle._weather = {Weather.RAINDANCE: battle.turn}
    battle._fields = {Field.TRICK_ROOM: battle.turn}
    out = _hook_output(battle)
    assert "RAIN is up" in out
    assert "SLOWER Pokemon moves first" in out


def test_hook_silent_for_singles():
    assert _hook_output(make_battle()) is None
