"""End-to-end doubles battle: two full PokeAgents on the real local server.

The default hidden VGC format (6-pick-4, team preview, level 50) with both
sides driven by DecisionStubProvider — zero tokens, but the complete
protocol: preview picks, per-slot targeted moves, reveals, kb tree. This
is the Step-6 acceptance harness: a doubles battle must finish with no
fallback turns, and both run dirs must stay separate and postmortem-
readable.
"""

from __future__ import annotations

import importlib.util
import json
import socket
from dataclasses import replace
from pathlib import Path

import pytest
from helpers import DecisionStubProvider
from poke_env.ps_client.account_configuration import AccountConfiguration

from pokeagent.config import AgentConfig
from pokeagent.player import PokeAgent


def _server_up(port: int) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=2):
            return True
    except OSError:
        return False


def _load_postmortem():
    path = Path(__file__).resolve().parent.parent / "scripts" / "postmortem.py"
    spec = importlib.util.spec_from_file_location("postmortem", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not _server_up(8964), reason="local Showdown server not running on :8964"
)
def test_full_doubles_battle_two_agents(tmp_path):
    import asyncio

    async def run_battle():
        config = AgentConfig(
            runs_dir=tmp_path / "runs",
            server_port=8964,
            team="Shadow Bolt",  # default battle format = hidden VGC doubles
            enable_subagents=False,  # keep the e2e light and synchronous
        )
        agent = PokeAgent(
            config, provider=DecisionStubProvider(), max_concurrent_battles=1
        )
        opp_config = replace(config, team="Slow Sun")
        opponent = PokeAgent(
            opp_config,
            provider=DecisionStubProvider(),
            account_configuration=AccountConfiguration("PokeAgentOpp", None),
            max_concurrent_battles=1,
        )
        await agent.battle_against(opponent, n_battles=1)
        return agent, opponent

    agent, opponent = asyncio.run(run_battle())
    assert agent.n_finished_battles == 1
    assert agent.n_won_battles + opponent.n_won_battles == 1

    # two separate run dirs (same battle tag, different usernames)
    run_dirs = sorted((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 2
    events = {
        d.name: [
            json.loads(line)
            for line in (d / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for d in run_dirs
    }

    for name, evts in events.items():
        turns = [e for e in evts if e["type"] == "turn"]
        previews = [e for e in evts if e["type"] == "team_preview"]
        assert previews, f"{name}: no team preview logged"
        # legal 6-pick-4: exactly 4 slots, leads first
        assert len(previews[0]["order_list"]) == 4
        assert previews[0]["order"].startswith("/team ")
        assert len(turns) >= 3, f"{name}: battle too short ({len(turns)} turns)"
        # zero fallback turns — every decision came from the loop
        assert not any(t["fell_back"] for t in turns)
        assert all(t["decision"] and t["reason"] for t in turns)
        # doubles decisions carry slot annotations and targets
        assert any("slot1:" in t["decision"] for t in turns)
        assert any("->opp" in t["decision"] for t in turns)
        assert any(e["type"] == "battle_finished" for e in evts)

    # the knowledge tree snapshot exists and holds both sides' species
    run_dir = run_dirs[0]
    assert (run_dir / "kb_tree.json").exists()
    kb = json.loads((run_dir / "kb_tree.json").read_text(encoding="utf-8"))
    assert kb.get("us") and kb.get("them")

    # postmortem reads both runs back from disk
    postmortem = _load_postmortem()
    for d in run_dirs:
        summary = postmortem.read_run(d)
        assert summary is not None
        assert summary["turns"] >= 3
        assert summary["fallbacks"] == 0
