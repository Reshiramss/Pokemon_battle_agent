"""Team-composition tool: archetype matching against revealed species."""

from __future__ import annotations

import asyncio

import pytest
from helpers import make_ctx, make_double_battle, make_kb
from poke_env.battle import Pokemon

from pokeagent.tools.base import ToolError
from pokeagent.tools.team_comp import TeamCompTool


def _battle_with_foes(*species):
    battle = make_double_battle()
    battle._opponent_team = {
        f"p2: {s.capitalize()}": Pokemon(9, species=s, name=s.capitalize())
        for s in species
    }
    return battle


def _run(battle, **kwargs):
    async def scenario():
        ctx = make_ctx(battle, make_kb())
        return await TeamCompTool().execute(ctx, **kwargs)

    return asyncio.run(scenario())


def test_matches_rain_core():
    out = _run(_battle_with_foes("kyogre", "archaludon", "floatzel"))
    assert "Best fit: Rain offense (rain)" in out
    assert "matched ['archaludon', 'floatzel', 'kyogre']" in out
    assert "Plan: " in out
    assert "Play against it: " in out


def test_partial_reveal_hedges_with_alternative():
    # a partial reveal ties several archetypes — the best fit is named and
    # the runner-ups surface as alternatives
    out = _run(_battle_with_foes("whimsicott", "kyogre"))
    assert "Best fit:" in out
    assert "Possible alternative" in out
    assert "(rain)" in out


def test_unknown_species_reports_no_match():
    out = _run(_battle_with_foes("furret"))
    assert "No known archetype matches" in out
    assert "'trick_room'" in out  # known ids are still listed


def test_archetype_param_reads_full_entry():
    out = _run(make_double_battle(), archetype="trick_room")
    assert "Trick Room" in out
    assert "Core mons: calyrexice" in out
    assert "Weak to: " in out


def test_archetype_param_rejects_unknown_id():
    with pytest.raises(ToolError) as exc:
        _run(make_double_battle(), archetype="digivolve")
    assert "'rain'" in str(exc.value)
