"""Run one or more battles: PokeAgent vs a poke-env baseline on the local server.

Usage:
    .venv/bin/python scripts/run_battle.py [--opponent random|heuristic] [--battles 1]
        [--config config.toml]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer

from pokeagent.config import load_config
from pokeagent.player import PokeAgent, build_server_configuration

OPPONENTS = {
    "random": RandomPlayer,
    "heuristic": SimpleHeuristicsPlayer,
}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a config.toml")
    parser.add_argument(
        "--opponent", choices=sorted(OPPONENTS), default="random"
    )
    parser.add_argument("--battles", type=int, default=1)
    args = parser.parse_args()

    config = load_config(args.config)
    agent = PokeAgent(config, max_concurrent_battles=1)
    # the opponent must point at the same server — poke-env's default is
    # port 8000, which is not our Showdown instance
    opponent = OPPONENTS[args.opponent](
        max_concurrent_battles=1,
        server_configuration=build_server_configuration(config),
    )

    await agent.battle_against(opponent, n_battles=args.battles)

    print(f"Finished battles: {agent.n_finished_battles}/{args.battles}")
    print(f"Agent wins: {agent.n_won_battles}")
    print(f"Opponent wins: {opponent.n_won_battles}")
    print(f"Runs logged under: {Path(config.runs_dir).resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
