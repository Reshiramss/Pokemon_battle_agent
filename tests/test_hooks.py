"""Tests for pre-decision hooks and the HookSystem timeboxing."""

from __future__ import annotations

import asyncio

from helpers import make_battle, make_ctx, make_kb

from pokeagent.config import AgentConfig
from pokeagent.hooks.base import HookSystem, PostDecisionHook, PreDecisionHook
from pokeagent.hooks.smogon_usage_hook import SmogonUsageHook
from pokeagent.hooks.static_kb_hook import TypeMatchupHook
from pokeagent.tools.base import ToolContext


# --- TypeMatchupHook --------------------------------------------------------


def test_type_matchup_hook_fragment():
    battle = make_battle()
    ctx = make_ctx(battle, make_kb())

    fragment = asyncio.run(TypeMatchupHook().run(ctx))
    assert fragment is not None

    # offense section: Flamethrower (Fire) vs Torterra (Grass/Ground) = 2x
    assert "Offense" in fragment
    assert "flamethrower (FIRE): x2 SUPER EFFECTIVE" in fragment
    # Quick Attack (Normal) vs Grass/Ground = neutral
    assert "quickattack (NORMAL): x1 neutral" in fragment
    # defense section: Stone Edge (Rock) vs Furret (Normal) = neutral
    assert "Defense" in fragment
    assert "stoneedge (ROCK): x1 neutral" in fragment
    # revealed moves listing
    assert "torterra: stoneedge" in fragment


def test_type_matchup_hook_without_active():
    battle = make_battle()
    battle._team["p1: Furret"]._active = False
    ctx = make_ctx(battle, make_kb())
    assert asyncio.run(TypeMatchupHook().run(ctx)) is None


# --- SmogonUsageHook (dormant) -----------------------------------------------


def test_smogon_hook_dormant_by_default():
    config = AgentConfig(enable_smogon_hook=False)
    hook = SmogonUsageHook(config)
    ctx = make_ctx(make_battle(), make_kb(), config=config)
    assert asyncio.run(hook.run(ctx)) is None


def test_smogon_hook_failed_fetch_is_sticky():
    """A failed fetch must not retry every turn (would burn the hook budget)."""
    config = AgentConfig(enable_smogon_hook=True, runs_dir="/nonexistent-run-dir")

    async def scenario():
        hook = SmogonUsageHook(config)
        ctx = make_ctx(make_battle(), make_kb(), config=config)
        await hook.run(ctx)  # fetch fails (no network / bad path)
        assert hook._fetch_failed
        return await hook.run(ctx)  # second turn: no retry

    asyncio.run(scenario())


# --- HookSystem timeboxing --------------------------------------------------


class _SlowHook(PreDecisionHook):
    name = "slow"

    async def run(self, ctx: ToolContext) -> str | None:
        await asyncio.sleep(5)
        return "late"


class _BadHook(PreDecisionHook):
    name = "bad"

    async def run(self, ctx: ToolContext) -> str | None:
        raise RuntimeError("boom")


class _GoodHook(PreDecisionHook):
    name = "good"

    async def run(self, ctx: ToolContext) -> str | None:
        return "knowledge-fragment"


class _ObservingPostHook(PostDecisionHook):
    name = "observing"

    def __init__(self):
        self.seen = []

    async def run(self, ctx, decision, elapsed_s, fell_back):
        self.seen.append((decision, fell_back))


def test_hook_system_pre_parallel_and_failure_isolation():
    system = HookSystem(fragment_char_cap=100)
    for hook in (_GoodHook(), _BadHook(), _SlowHook()):
        system.register_pre(hook)

    fragments = asyncio.run(
        system.run_pre(make_ctx(make_battle(), make_kb()), budget=0.5)
    )
    # good hook survives even though the slow hook burns its own time slice;
    # the bad hook raised and was skipped
    assert fragments == ["knowledge-fragment"]


def test_hook_system_post_observes_and_never_raises():
    post = _ObservingPostHook()
    system = HookSystem()
    system.register_post(post)
    system.register_post(_BadHook())  # type: ignore[arg-type]

    async def scenario():
        await system.run_post(
            make_ctx(make_battle(), make_kb()),
            decision="move:flamethrower",
            elapsed_s=1.0,
            fell_back=False,
        )

    asyncio.run(scenario())
    assert post.seen == [("move:flamethrower", False)]


def test_hook_system_truncates_fragments():
    system = HookSystem(fragment_char_cap=10)
    system.register_pre(_GoodHook())
    fragments = asyncio.run(
        system.run_pre(make_ctx(make_battle(), make_kb()), budget=1.0)
    )
    assert fragments == ["knowledge-"]
