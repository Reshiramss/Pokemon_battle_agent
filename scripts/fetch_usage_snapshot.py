"""One-time download of a Smogon usage-stats snapshot for offline use.

Run this ONCE on a machine with internet access; it downloads the chaos
JSON for a real ladder format into the repository data dir and prints
the snapshot path + the config line to enable it:

    python scripts/fetch_usage_snapshot.py [--format gen9vgc2025regi] [--month latest]

    -> smogon_stats_snapshot = "pokeagent/data/usage/2026-10/gen9vgc2025regi-0.json.gz"

The agent then loads stats from that file and never touches the network
(see pokeagent/hooks/smogon_usage_hook.py). The format id must be the
REAL ladder format — never the custom hidden battle format.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poke_env.data.smogon import SmogonStats

DEFAULT_FORMAT = "gen9vgc2025regi"
# fetch's cache layout is <dir>/<month>/<format>-<cutoff>.json.gz — the
# month subdirectory is what SmogonStats.from_file later reads back
OUT_DIR = Path(__file__).resolve().parents[1] / "pokeagent" / "data" / "usage"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        default=DEFAULT_FORMAT,
        help=f"real ladder format id (default {DEFAULT_FORMAT})",
    )
    parser.add_argument("--month", default="latest", help="e.g. 2026-10 or 'latest'")
    parser.add_argument("--cutoff", type=int, default=0, help="rating cutoff")
    args = parser.parse_args()

    if args.format.endswith("hidden"):
        print(
            f"refusing to fetch '{args.format}': hidden battle formats have no "
            "Smogon ladder. Pass the real format id, e.g. gen9vgc2025regi.",
            file=sys.stderr,
        )
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = SmogonStats.fetch(args.format, month=args.month, cutoff=args.cutoff,
                              cache_dir=OUT_DIR)
    # "latest" resolves to the real month both on disk and in stats
    path = OUT_DIR / stats.month / f"{args.format}-{args.cutoff}.json.gz"
    print(f"Downloaded usage snapshot for {args.format} ({stats.month}).")
    print(f"Snapshot file: {path}")
    print("Add to config.toml [agent]:")
    print(f'  smogon_stats_snapshot = "{path}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
