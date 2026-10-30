"""Tests for the TaskRunner and the background sub-agents (Step 5)."""

from __future__ import annotations

import asyncio
import json
import time

from helpers import ScriptedProvider, make_battle, make_kb

from pokeagent.config import AgentConfig
from pokeagent.context.manager import ContextManager
from pokeagent.context.records import TurnRecord
from pokeagent.loop import AgentLoop
from pokeagent.providers.base import LLMResponse
from pokeagent.tasks.base import IntelStore, SubAgent, SubAgentContext
from pokeagent.tasks.post_battle import PostBattleReviewAgent
from pokeagent.tasks.runner import TaskRunner
from pokeagent.tasks.threat_analysis import NewMonThreatAgent


def make_cm(tmp_path) -> ContextManager:
    config = AgentConfig(runs_dir=tmp_path / "runs")
    return ContextManager(battle_tag="battle-1", config=config, kb=None)


def cm_records(cm: ContextManager, n: int = 3) -> None:
    for turn in range(1, n + 1):
        cm.record_turn(
            TurnRecord(
                seq=0, turn=turn, request=f"turn {turn}",
                decision=f"move:m{turn}", reason=f"reason {turn}",
            )
        )


async def _until(predicate, timeout: float = 5.0) -> bool:
    """Poll a sync predicate until true; returns False on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


# --- TaskRunner --------------------------------------------------------------


class _OkAgent(SubAgent):
    name = "ok_agent"

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.start_time: float | None = None
        self.end_time: float | None = None

    async def run(self, sctx: SubAgentContext) -> str:
        self.start_time = time.monotonic()
        try:
            await asyncio.sleep(self.delay)
            return f"analysis of {sctx.inputs.get('species', '?')}"
        finally:
            self.end_time = time.monotonic()


class _BoomAgent(SubAgent):
    name = "boom_agent"

    async def run(self, sctx: SubAgentContext) -> str:
        raise RuntimeError("sub-agent exploded")


def test_runner_stores_summary_and_logs(tmp_path):
    async def scenario():
        cm = make_cm(tmp_path)
        runner = TaskRunner(AgentConfig(runs_dir=tmp_path / "runs"))
        runner.register(_OkAgent())
        sctx = SubAgentContext(battle=make_battle(), intel=cm.intel,
                               inputs={"species": "torterra"})
        runner.submit("battle-1", "ok_agent", sctx, cm=cm,
                      intel_key="threat:torterra")
        assert await _until(lambda: cm.intel.get("threat:torterra"))

        events = [json.loads(l) for l in
                  (cm.run_dir / "events.jsonl").read_text().splitlines()]
        done = [e for e in events if e["type"] == "subagent_done"]
        assert done and done[0]["agent"] == "ok_agent"
        assert "analysis of torterra" in done[0]["summary"]

    asyncio.run(scenario())


def test_runner_swallows_agent_failures(tmp_path):
    async def scenario():
        cm = make_cm(tmp_path)
        runner = TaskRunner(AgentConfig(runs_dir=tmp_path / "runs"))
        runner.register(_BoomAgent())
        runner.submit("battle-1", "boom_agent",
                      SubAgentContext(battle=make_battle(), intel=cm.intel), cm=cm)
        ok = await _until(
            lambda: "subagent_error" in (cm.run_dir / "events.jsonl").read_text()
        )
        assert ok
        assert len(cm.intel) == 0

    asyncio.run(scenario())


def test_runner_timeout_kills_slow_agent(tmp_path):
    async def scenario():
        cm = make_cm(tmp_path)
        config = AgentConfig(runs_dir=tmp_path / "runs", subagent_timeout=0.1)
        runner = TaskRunner(config)
        runner.register(_OkAgent(delay=5.0))
        runner.submit("battle-1", "ok_agent",
                      SubAgentContext(battle=make_battle(), intel=cm.intel), cm=cm)
        started = time.monotonic()
        ok = await _until(
            lambda: "subagent_timeout" in (cm.run_dir / "events.jsonl").read_text()
        )
        assert ok
        assert time.monotonic() - started < 3  # did not wait the full 5s
        assert len(cm.intel) == 0

    asyncio.run(scenario())


def test_runner_semaphore_limits_concurrency(tmp_path):
    async def scenario():
        config = AgentConfig(runs_dir=tmp_path / "runs",
                             max_concurrent_subagents=1)
        runner = TaskRunner(config)
        slow1, slow2 = _OkAgent(delay=0.15), _OkAgent(delay=0.15)
        slow2.name = "ok_agent_2"
        runner.register(slow1)
        runner.register(slow2)
        battle = make_battle()
        runner.submit("b", "ok_agent", SubAgentContext(battle=battle, intel=IntelStore()))
        runner.submit("b", "ok_agent_2", SubAgentContext(battle=battle, intel=IntelStore()))
        await _until(lambda: slow1.end_time is not None and slow2.end_time is not None)

        assert slow1.start_time is not None and slow1.end_time is not None
        # serialized: the second agent started only after the first finished
        assert slow2.start_time >= slow1.end_time - 0.02

    asyncio.run(scenario())


def test_runner_cancel_battle(tmp_path):
    async def scenario():
        runner = TaskRunner(AgentConfig(runs_dir=tmp_path / "runs"))
        agent = _OkAgent(delay=2.0)
        runner.register(agent)
        task = runner.submit("b", "ok_agent",
                             SubAgentContext(battle=make_battle(), intel=IntelStore()))
        assert runner.pending("b") == 1

        runner.cancel_battle("b")
        assert runner.pending("b") == 0
        await asyncio.sleep(0.1)  # let the cancellation settle
        assert task is not None and task.cancelled()

    asyncio.run(scenario())


def test_runner_unknown_agent_is_noop(tmp_path):
    async def scenario():
        runner = TaskRunner(AgentConfig(runs_dir=tmp_path / "runs"))
        assert runner.submit("b", "nope",
                             SubAgentContext(battle=make_battle())) is None

    asyncio.run(scenario())


# --- NewMonThreatAgent ---------------------------------------------------------


def test_threat_agent_prompt_and_summary():
    async def scenario():
        provider = ScriptedProvider(
            [LLMResponse(text="Torterra threatens Furret; keep it burned.",
                         stop_reason="end_turn")]
        )
        agent = NewMonThreatAgent(provider)
        battle = make_battle()
        sctx = SubAgentContext(
            battle=battle, kb=make_kb(), intel=IntelStore(),
            inputs={"species": "torterra"},
        )
        summary = await agent.run(sctx)
        assert summary.startswith("Torterra threatens")

        prompt_text = str(provider.calls[0]["messages"])
        assert "torterra" in prompt_text
        assert "Grass" in prompt_text  # static types included
        assert "furret" in prompt_text  # our team listed

    asyncio.run(scenario())


def test_threat_agent_provider_error_returns_empty():
    async def scenario():
        provider = ScriptedProvider([RuntimeError("api down")])
        agent = NewMonThreatAgent(provider)
        sctx = SubAgentContext(battle=make_battle(), kb=make_kb(),
                               inputs={"species": "furret"})
        assert await agent.run(sctx) == ""

    asyncio.run(scenario())


# --- PostBattleReviewAgent -----------------------------------------------------


def test_post_battle_review_builds_decision_log(tmp_path):
    async def scenario():
        cm = make_cm(tmp_path)
        cm_records(cm, n=3)
        provider = ScriptedProvider(
            [LLMResponse(text="Switch less.", stop_reason="end_turn")]
        )
        agent = PostBattleReviewAgent(provider)
        battle = make_battle()
        summary = await agent.run(
            SubAgentContext(battle=battle, inputs={"context_manager": cm})
        )
        assert summary == "Switch less."
        prompt_text = str(provider.calls[0]["messages"])
        assert "move:m1" in prompt_text and "move:m3" in prompt_text
        assert "reason 2" in prompt_text

    asyncio.run(scenario())


# --- loop wiring: threat trigger dedup -----------------------------------------


def test_loop_fires_threat_analysis_once_per_species(tmp_path):
    async def scenario():
        cm = make_cm(tmp_path)
        loop = AgentLoop(
            provider=ScriptedProvider([]),
            registry=None,  # type: ignore[arg-type]
            hooks=None,  # type: ignore[arg-type]
            config=AgentConfig(runs_dir=tmp_path / "runs"),
        )
        fired: list[str] = []
        loop.on_new_opponent_mon = lambda battle, species: fired.append(species)

        battle = make_battle()  # opponent team contains only Torterra
        loop._maybe_fire_threat_analyses(battle, cm)
        loop._maybe_fire_threat_analyses(battle, cm)
        assert fired == ["torterra"]
        assert cm.seen_species == {"torterra"}

    asyncio.run(scenario())
