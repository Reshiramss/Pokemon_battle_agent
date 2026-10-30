"""Diagnose which player's websocket client fails to connect/login.

Independent usernames, short timeouts, no battles — safe to run next to a
stuck test. Prints each player's logged_in state and connection.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from helpers import DecisionStubProvider

from pokeagent.config import AgentConfig
from pokeagent.player import PokeAgent
from pokeagent.tasks.runner import TaskRunner
from pokeagent.tasks.threat_analysis import NewMonThreatAgent


def report(name: str, player) -> None:
    client = player.ps_client
    logged = client.logged_in.is_set()
    has_ws = getattr(client, "websocket", None) is not None
    print(f"{name}: logged_in={logged} has_websocket={has_ws}", flush=True)


async def main() -> None:
    from poke_env.player import RandomPlayer
    from poke_env.ps_client.account_configuration import AccountConfiguration

    config = AgentConfig(
        runs_dir=Path("/tmp/diag_runs"),
        server_port=8964,
        max_tool_rounds=3,
    )
    stub = DecisionStubProvider()
    runner = TaskRunner(config, provider=DecisionStubProvider())
    runner.register(NewMonThreatAgent(stub))
    agent = PokeAgent(
        config,
        provider=stub,
        runner=runner,
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration("DiagAgent", None),
    )
    opponent = RandomPlayer(
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration("DiagOpp", None),
    )

    t0 = time.monotonic()
    while time.monotonic() - t0 < 8:
        report("agent", agent)
        report("opponent", opponent)
        if agent.ps_client.logged_in.is_set() and opponent.ps_client.logged_in.is_set():
            print("both logged in", flush=True)
            break
        await asyncio.sleep(2)

    await agent.ps_client.stop_listening()
    await opponent.ps_client.stop_listening()


if __name__ == "__main__":
    asyncio.run(main())
