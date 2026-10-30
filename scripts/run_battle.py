"""Run one or more battles: PokeAgent vs a poke-env baseline on the local server.

Usage:
    python scripts/run_battle.py [--opponent random|heuristic] [--battles 1]
        [--config config.toml]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer

from pokeagent.config import load_config
from pokeagent.logging_setup import setup_logging
from pokeagent.player import PokeAgent, build_server_configuration

logger = logging.getLogger("run_battle")

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
    log_path = setup_logging(config)  # every run leaves runs/logs/agent-<ts>.log
    logger.info(
        "run_battle start: provider=%s model=%s opponent=%s battles=%d "
        "server=%s:%d log=%s",
        config.provider, config.model, args.opponent, args.battles,
        config.server_host, config.server_port, log_path,
    )
    print(f"Run log: {log_path}")

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
    runs_root = Path(config.runs_dir).resolve()
    print(f"Runs logged under: {runs_root}")
    if agent._contexts:
        print(f"This run's battle dirs: {', '.join(cm.run_dir.name for cm in agent._contexts.values())}")
    logger.info(
        "run_battle done: finished=%d won=%d lost=%d",
        agent.n_finished_battles, agent.n_won_battles, opponent.n_won_battles,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("interrupted by user")
        raise
    except Exception:
        logger.exception("run_battle failed with an unhandled error")
        raise
