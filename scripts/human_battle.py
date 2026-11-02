"""Human vs AI: the agent idles on the server until a human challenges it.

UNTESTED PATH — written but not yet verified in a live game. If anything
misbehaves (hangs, illegal orders, no response), check runs/logs/ for the
battle's events.jsonl / llm.jsonl and file what you saw.

Usage:
    python scripts/human_battle.py [--team "Shadow Bolt"] [--count 1]
        [--wait-for YOUR_USERNAME] [--config config.toml]

Steps:
    1. Start the local Showdown server on port 8964 (see README).
    2. Run this script — the agent logs in as PokeAgent and waits.
    3. Open http://localhost:8964 in a browser, pick any username, then
       challenge the user PokeAgent to "[Gen 9] VGC 2025 Reg I (Hidden)"
       (it is listed in the challenge dropdown). With --wait-for set, only
       that username is accepted.
    4. Play your 6-pick-4 doubles game; the agent picks leads and moves
       like in any other battle.
    5. After each game the agent waits for the next challenge, up to
       --count games, then exits.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pokeagent.config import load_config
from pokeagent.logging_setup import setup_logging
from pokeagent.player import PokeAgent

logger = logging.getLogger("human_battle")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a config.toml")
    parser.add_argument(
        "--team",
        default=None,
        help="agent team: library name (see teams/) or raw packed team string",
    )
    parser.add_argument(
        "--count", type=int, default=1, help="number of games to accept before exiting"
    )
    parser.add_argument(
        "--wait-for",
        default=None,
        help="only accept challenges from this username (default: anyone)",
    )
    args = parser.parse_args()

    overrides = {"team": args.team} if args.team else None
    config = load_config(args.config, overrides=overrides)
    log_path = setup_logging(config)
    logger.info(
        "human_battle start: provider=%s model=%s server=%s:%d format=%s team=%s",
        config.provider, config.model, config.server_host, config.server_port,
        config.battle_format, args.team or "-",
    )
    print(f"Run log: {log_path}")

    agent = PokeAgent(config, max_concurrent_battles=1)
    print(f"Logged in as {agent.username} and waiting on "
          f"{config.server_host}:{config.server_port}")
    print(f"Format: {config.battle_format}")
    print(f"Accepting {args.count} challenge(s) from "
          f"{args.wait_for or 'anyone'} — open the server in a browser and "
          "challenge this username.")

    await agent.accept_challenges(args.wait_for, args.count)

    print(f"Finished battles: {agent.n_finished_battles}")
    print(f"Agent wins: {agent.n_won_battles}")
    runs_root = Path(config.runs_dir).resolve()
    print(f"Runs logged under: {runs_root}")
    if agent._contexts:
        print(f"This run's battle dirs: "
              f"{', '.join(cm.run_dir.name for cm in agent._contexts.values())}")
    logger.info(
        "human_battle done: finished=%d won=%d",
        agent.n_finished_battles, agent.n_won_battles,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("interrupted by user")
        raise
