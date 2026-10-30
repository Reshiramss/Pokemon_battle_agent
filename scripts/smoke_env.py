"""Step-1 smoke test: two RandomPlayers fight on the local Showdown server.

Verifies the environment (poke-env install + local server on port 8964)
before any framework code runs:

    python scripts/smoke_env.py [--port 8964]
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import ServerConfiguration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8964)
    args = parser.parse_args()

    server = ServerConfiguration(
        f"ws://localhost:{args.port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )

    async def run() -> None:
        player_1 = RandomPlayer(
            max_concurrent_battles=1, server_configuration=server
        )
        player_2 = RandomPlayer(
            max_concurrent_battles=1, server_configuration=server
        )
        await player_1.battle_against(player_2, n_battles=1)

        print(f"Finished battles: {player_1.n_finished_battles}")
        print(f"Player 1 wins: {player_1.n_won_battles}")
        print(f"Player 2 wins: {player_2.n_won_battles}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
