"""End-to-end Champions battle: two full PokeAgents, zero LLM tokens.

Uses the official ``gen9championsvgc2026regmc`` format (the same one the
custom hidden champion format mirrors) with both sides driven by
``DecisionStubProvider``. This exercises the mod-aware plumbing end to end:
stat points, Champion data bundle, PP cap, no Tera, 6-pick-4 preview, state
persistence and postmortem readability. Skipped when the local server is not
running.
"""

from __future__ import annotations

import importlib.util
import json
import re
import socket
from dataclasses import replace
from pathlib import Path

import pytest
from helpers import DecisionStubProvider
from poke_env.ps_client.account_configuration import AccountConfiguration

from pokeagent.config import AgentConfig
from pokeagent.player import PokeAgent

FORMAT = "gen9championsvgc2026regmc"


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
def test_full_champions_battle_two_agents(tmp_path):
    import asyncio

    async def run_battle():
        config = AgentConfig(
            battle_format=FORMAT,
            runs_dir=tmp_path / "runs",
            server_port=8964,
            team="Champions A",
            enable_subagents=False,
        )
        agent = PokeAgent(
            config, provider=DecisionStubProvider(), max_concurrent_battles=1
        )
        opp_config = replace(config, team="Champions B")
        opponent = PokeAgent(
            opp_config,
            provider=DecisionStubProvider(),
            account_configuration=AccountConfiguration("PokeAgentChampOpp", None),
            max_concurrent_battles=1,
        )
        await agent.battle_against(opponent, n_battles=1)
        return agent, opponent

    agent, opponent = asyncio.run(run_battle())
    assert agent.n_finished_battles == 1

    own = [cm.run_dir for cm in agent._contexts.values()]
    foe = [cm.run_dir for cm in opponent._contexts.values()]
    assert own and foe, f"expected separate run dirs, got {own + foe}"
    assert own[0] != foe[0]

    for run_dir in (own[0], foe[0]):
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["format"] == FORMAT
        assert meta["mod"] == "champions"
        assert meta["stat_model"] == "champions"

        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        assert state["meta"]["mod"] == "champions"
        assert state["meta"]["stat_model"] == "champions"
        # stat points, not mainline EVs: total <= 66
        assert "resources" in state and "mega_used" in state["resources"]
        # our own stats must use the Champions formula, not the level formula
        furret = state["kb"]["us"].get("farfetchd")
        assert furret and furret["stats"]["hp"] == 52 + 11 + 75
        assert furret["stats"]["spe"] == 60 + 11 + 20

        events = [
            json.loads(line)
            for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        turns = [e for e in events if e.get("type") == "turn"]
        assert turns, "no turn records"
        assert all(not t.get("fell_back") for t in turns), "unexpected fallback turn"

        preview = [e for e in events if e.get("type") == "team_preview"]
        assert preview and len(preview[0]["order_list"]) == 4

        # Champions: no Tera note; PP display never exceeds the cap
        for turn in turns:
            request = turn.get("request") or ""
            assert "terastallization available" not in request
            for current, maximum in re.findall(r"PP (\d+)/(\d+)", request):
                assert int(maximum) <= 20, request

    pm = _load_postmortem()
    result = pm.read_run(own[0])
    assert result is not None
    assert result["turns"] > 0