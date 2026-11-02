"""Run one or more battles: PokeAgent vs a baseline or another PokeAgent.

Usage:
    python scripts/run_battle.py [--opponent random|heuristic|agent]
        [--battles 1] [--config config.toml] [--team NAME] [--opponent-team NAME]

Team-preview formats (the default VGC one) require BOTH sides to bring a
team: --team picks the agent's team from the library (or a raw packed string
containing "|"); --opponent-team picks the opponent's, defaulting to a
different library team when only --team is given. --opponent agent pits the
PokeAgent against a second full PokeAgent (same model config, its own team,
background sub-agents off — half the resource footprint on shared machines).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration

from pokeagent.config import load_config
from pokeagent.logging_setup import setup_logging
from pokeagent.player import PokeAgent, build_server_configuration
from pokeagent.teams import RawTeambuilder, TeamLibrary, resolve_team

logger = logging.getLogger("run_battle")

OPPONENTS = {
    "random": RandomPlayer,
    "heuristic": SimpleHeuristicsPlayer,
}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a config.toml")
    parser.add_argument(
        "--opponent", choices=["agent", *sorted(OPPONENTS)], default="random"
    )
    parser.add_argument("--battles", type=int, default=1)
    parser.add_argument(
        "--team",
        default=None,
        help="agent team: library name (see teams/) or raw packed team string",
    )
    parser.add_argument(
        "--opponent-team",
        default=None,
        help="opponent team; defaults to a different library team",
    )
    args = parser.parse_args()

    overrides = {}
    if args.team:
        overrides["team"] = args.team
    config = load_config(args.config, overrides=overrides or None)
    log_path = setup_logging(config)  # every run leaves runs/logs/agent-<ts>.log
    logger.info(
        "run_battle start: provider=%s model=%s opponent=%s battles=%d "
        "server=%s:%d format=%s team=%s log=%s",
        config.provider, config.model, args.opponent, args.battles,
        config.server_host, config.server_port, config.battle_format,
        args.team or "-", log_path,
    )
    print(f"Run log: {log_path}")

    agent = PokeAgent(config, max_concurrent_battles=1)

    # the opponent must point at the same server — poke-env's default is
    # port 8000, which is not our Showdown instance
    opponent_team = None
    if config.team is not None:
        library = TeamLibrary.from_config(config)
        opponent_spec = args.opponent_team or library.random_other(
            resolve_team(config.team, library)
        ).name
        opponent_team = resolve_team(opponent_spec, library)
        logger.info("opponent team: %s", opponent_spec)
    if args.opponent == "agent":
        # a second full PokeAgent: same model config, own account/team,
        # sub-agents off to halve the resource footprint
        opp_config = replace(config, enable_subagents=False)
        opponent = PokeAgent(
            opp_config,
            account_configuration=AccountConfiguration("PokeAgentOpp", None),
            max_concurrent_battles=1,
            team=RawTeambuilder(opponent_team) if opponent_team else None,
        )
    else:
        opponent = OPPONENTS[args.opponent](
            max_concurrent_battles=1,
            battle_format=config.battle_format,
            server_configuration=build_server_configuration(config),
            team=RawTeambuilder(opponent_team) if opponent_team else None,
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
