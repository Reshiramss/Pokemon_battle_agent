"""End-to-end offline battle: full agent loop on the real local server.

The DecisionStubProvider parses the [NOW] section and submits the first
legal move, so the battle is real (Showdown server, protocol, records) but
costs zero LLM tokens. Verifies the Step-2 acceptance criteria: a battle
completes, every turn is logged with decision + reason + CoT, and the
postmortem can read it all back.
"""

from __future__ import annotations

import json
import socket

import pytest
from helpers import DecisionStubProvider

from pokeagent.config import AgentConfig
from pokeagent.player import PokeAgent, build_server_configuration
from pokeagent.tasks.post_battle import PostBattleReviewAgent
from pokeagent.tasks.runner import TaskRunner
from pokeagent.tasks.team_preview import TeamPreviewAnalysisAgent
from pokeagent.tasks.threat_analysis import NewMonThreatAgent


def _server_up(port: int) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=2):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _server_up(8964), reason="local Showdown server not running on :8964"
)
def test_full_battle_offline_stub(tmp_path):
    from poke_env.player import RandomPlayer

    async def run_battle():
        config = AgentConfig(
            runs_dir=tmp_path / "runs",
            server_port=8964,
            # keep the battle short for CI friendliness
            max_tool_rounds=3,
        )
        stub = DecisionStubProvider()
        # background sub-agents on the same stub (they return inert text)
        runner = TaskRunner(config, provider=DecisionStubProvider())
        runner.register(NewMonThreatAgent(stub))
        runner.register(TeamPreviewAnalysisAgent(stub))
        runner.register(PostBattleReviewAgent(stub))
        agent = PokeAgent(
            config,
            provider=stub,
            runner=runner,
            max_concurrent_battles=1,
        )
        # the opponent must point at the same local server — poke-env's
        # default is port 8000, which is somebody else's service here
        opponent = RandomPlayer(
            max_concurrent_battles=1,
            server_configuration=build_server_configuration(config),
        )
        await agent.battle_against(opponent, n_battles=1)
        return agent

    agent = __import__("asyncio").run(run_battle())
    assert agent.n_finished_battles == 1
    assert agent.n_won_battles in (0, 1)

    # every decision was logged with decision + reason + CoT, no fallbacks
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    events = [
        json.loads(l)
        for l in (run_dirs[0] / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    turns = [e for e in events if e["type"] == "turn"]
    assert len(turns) >= 3  # a real battle takes multiple turns
    assert all(t["decision"] and t["reason"] for t in turns)
    assert all(t.get("cot") for t in turns)
    assert not any(t["fell_back"] for t in turns)
    assert any(e["type"] == "battle_finished" for e in events)
    # threat-analysis sub-agents fired for revealed opponent species
    assert any(e["type"] == "subagent_done" for e in events)

    # debugging artifacts: config snapshot + full conversation trace
    assert (run_dirs[0] / "meta.json").exists()
    llm_entries = [
        json.loads(line)
        for line in (run_dirs[0] / "llm.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    kinds = {e["type"] for e in llm_entries}
    assert "turn_prompt" in kinds
    assert "llm_response" in kinds
    # every prompt trace carries the system prompt and the assembled user text
    prompt_entry = next(e for e in llm_entries if e["type"] == "turn_prompt")
    assert "submit_decision" in prompt_entry["system"]
    assert "[NOW" in prompt_entry["prompt"]
