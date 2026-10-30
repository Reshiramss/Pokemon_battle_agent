"""Unit tests for the Step-3 domain tools and static knowledge base.

Battles are hand-constructed from poke-env's Battle/Pokemon classes (no
server needed); the damage-calc test covers the opponent-unknown-stats
estimation path used in random battles.
"""

from __future__ import annotations

import asyncio

import pytest
from helpers import make_battle, make_ctx

from pokeagent.config import AgentConfig
from pokeagent.context.manager import ContextManager
from pokeagent.context.records import TurnRecord
from pokeagent.knowledge.static import StaticKB
from pokeagent.providers.base import ToolCallRequest
from pokeagent.tools.base import Tool, ToolContext, ToolRegistry
from pokeagent.tools.battle_query import BattleHistoryTool
from pokeagent.tools.damage_calc import DamageCalcTool
from pokeagent.tools.static_kb import (
    MoveInfoTool,
    SpeciesInfoTool,
    TypeInfoTool,
)


# --- static knowledge base --------------------------------------------------


@pytest.fixture(scope="module")
def kb() -> StaticKB:
    return StaticKB("gen9randombattle")


def test_type_effectiveness_direction(kb: StaticKB):
    # chart is indexed [defender][attacker] — the wrapper must get this right
    assert kb.type_effectiveness("water", ["fire"]) == 2.0
    assert kb.type_effectiveness("fire", ["water"]) == 0.5
    assert kb.type_effectiveness("water", ["ground", "rock"]) == 4.0
    assert kb.type_effectiveness("normal", ["ghost"]) == 0.0


def test_estimate_stats_returns_dict(kb: StaticKB):
    stats = kb.estimate_stats("furret", level=50)
    assert set(stats) == {"hp", "atk", "def", "spa", "spd", "spe"}
    assert all(isinstance(v, int) and v > 0 for v in stats.values())
    # Furret base HP 85 with even spread at L50: well above its base atk
    assert stats["hp"] > stats["atk"] > 0
    assert kb.estimate_stats("notapokemonnope", level=50) == {}


def test_learnset_unwraps_nested_entry(kb: StaticKB):
    moves = kb.learnset("furret")
    assert "agility" in moves
    assert all(isinstance(m, str) for m in moves)


# --- static-kb tools --------------------------------------------------------


def test_move_info(kb: StaticKB):
    result = asyncio.run(
        MoveInfoTool(kb).execute(
            make_ctx(make_battle(), kb), move="Flamethrower"
        )
    )
    assert "Fire" in result and "Special" in result
    assert "power 90" in result
    assert "burn" in result.lower() or "brn" in result.lower()


def test_move_info_unknown(kb: StaticKB):
    with pytest.raises(Exception):
        asyncio.run(
            MoveInfoTool(kb).execute(make_ctx(make_battle(), kb), move="Nope")
        )


def test_type_matchup_tool(kb: StaticKB):
    result = asyncio.run(
        TypeInfoTool(kb).execute(
            make_ctx(make_battle(), kb),
            attacking_type="Water",
            defending_types=["Ground", "Rock"],
        )
    )
    assert "x4" in result and "super effective" in result


def test_species_info(kb: StaticKB):
    result = asyncio.run(
        SpeciesInfoTool(kb).execute(make_ctx(make_battle(), kb), species="Furret")
    )
    assert "Normal" in result
    assert "HP 85" in result and "SPE 90" in result
    assert "Frisk" in result  # hidden ability listed


# --- damage calc ------------------------------------------------------------


def test_damage_calc_estimates_opponent_stats(kb: StaticKB):
    battle = make_battle()
    ctx = make_ctx(battle, kb)
    tool = DamageCalcTool()
    result = asyncio.run(
        tool.execute(ctx, attacker="Furret", defender="Torterra", move="Flamethrower")
    )
    # damage range present and estimation flagged
    assert "damage" in result
    assert "defender stats estimated" in result
    # the estimate must have been written onto the Pokemon object
    defender = battle.opponent_team["p2: Torterra"]
    assert defender.stats["atk"] is not None
    # Fire vs Grass/Ground is 2x super effective: non-trivial damage
    lo, hi = _parse_range(result)
    assert hi > 0 and hi >= lo


def _parse_range(result: str) -> tuple[int, int]:
    import re

    m = re.search(r"(\d+)-(\d+) damage", result)
    assert m, result
    return int(m.group(1)), int(m.group(2))


def test_damage_calc_unknown_pokemon(kb: StaticKB):
    ctx = make_ctx(make_battle(), kb)
    with pytest.raises(Exception):
        asyncio.run(
            DamageCalcTool().execute(
                ctx, attacker="Missingno", defender="Torterra", move="Flamethrower"
            )
        )


def test_damage_calc_unknown_move(kb: StaticKB):
    ctx = make_ctx(make_battle(), kb)
    with pytest.raises(Exception):
        asyncio.run(
            DamageCalcTool().execute(
                ctx, attacker="Furret", defender="Torterra", move="Splash Dance"
            )
        )


# --- battle history tool ----------------------------------------------------


def _recorded_cm(tmp_path) -> tuple[ContextManager, ToolContext]:
    config = AgentConfig(runs_dir=tmp_path / "runs")
    cm = ContextManager(battle_tag="battle-1", config=config, kb=None)
    for turn, (decision, reason) in enumerate(
        [("move:flamethrower", "super effective"), ("switch:Furret", "preserve HP")],
        start=1,
    ):
        cm.record_turn(
            TurnRecord(
                seq=0,
                turn=turn,
                request=f"[NOW] turn {turn} state",
                events=[f"|move|p1a: Furret|Flamethrower|p2a: Torterra"] if turn == 1 else [],
                decision=decision,
                reason=reason,
            )
        )
    ctx = make_ctx(make_battle(), kb=None, cm=cm, config=config)
    return cm, ctx


def test_battle_history_log_and_turn(tmp_path):
    cm, ctx = _recorded_cm(tmp_path)
    tool = BattleHistoryTool()

    log = asyncio.run(tool.execute(ctx))
    assert "move:flamethrower" in log and "switch:Furret" in log

    detail = asyncio.run(tool.execute(ctx, turn=1))
    assert "super effective" in detail
    assert "Flamethrower" in detail  # battle events included

    with pytest.raises(Exception):
        asyncio.run(tool.execute(ctx, turn=99))


def test_battle_history_requires_records(tmp_path):
    config = AgentConfig(runs_dir=tmp_path / "runs")
    cm = ContextManager(battle_tag="battle-2", config=config, kb=None)
    ctx = make_ctx(make_battle(), kb=None, cm=cm, config=config)
    with pytest.raises(Exception):
        asyncio.run(BattleHistoryTool().execute(ctx))


# --- registry plumbing ------------------------------------------------------


class _BoomTool(Tool):
    name = "boom"
    description = "raises"
    input_schema = {"type": "object", "properties": {}}

    async def execute(self, ctx: ToolContext, **kwargs) -> str:
        raise ValueError("boom")


class _SlowTool(Tool):
    name = "slow"
    description = "sleeps"
    input_schema = {"type": "object", "properties": {}}

    async def execute(self, ctx: ToolContext, **kwargs) -> str:
        await asyncio.sleep(5)
        return "done"


class _OkTool(Tool):
    name = "ok"
    description = "works"
    input_schema = {"type": "object", "properties": {}}

    async def execute(self, ctx: ToolContext, **kwargs) -> str:
        return "fine"


def test_registry_parallel_execution_and_errors():
    registry = ToolRegistry()
    for tool in (_OkTool(), _BoomTool(), _SlowTool()):
        registry.register(tool)

    config = AgentConfig(tool_call_timeout=0.2)
    ctx = make_ctx(make_battle(), kb=None, config=config)
    results = asyncio.run(
        registry.execute_parallel(
            ctx,
            [
                ToolCallRequest("1", "ok", {}),
                ToolCallRequest("2", "boom", {}),
                ToolCallRequest("3", "slow", {}),
                ToolCallRequest("4", "nope", {}),
            ],
        )
    )
    by_name = {r.tool_name: r for r in results}
    assert by_name["ok"].content == "fine"
    assert not by_name["ok"].is_error

    assert by_name["boom"].is_error and "Unexpected error" in by_name["boom"].content
    assert by_name["slow"].is_error and "timed out" in by_name["slow"].content
    assert by_name["nope"].is_error and "Unknown tool" in by_name["nope"].content
