#!/usr/bin/env python3
"""Seed cold-start memory (archetypes, our teams, species skeletons).

Examples:
    python scripts/seed_memory.py --format gen9championsvgc2026regmchidden
    python scripts/seed_memory.py --format gen9championsvgc2026regmchidden --all-species
    python scripts/seed_memory.py --coverage --format gen9championsvgc2026regmchidden
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokeagent.config import load_config  # noqa: E402
from pokeagent.formats import profile_for  # noqa: E402
from pokeagent.knowledge.static import StaticKB  # noqa: E402
from pokeagent.memory import seed, scope as memory_scope  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.toml")
    parser.add_argument("--format", default=None, help="battle format id")
    parser.add_argument("--team", default=None, help="our team library name (optional)")
    parser.add_argument(
        "--all-species",
        action="store_true",
        help="also prepare a skeleton for every legal species (slow heat-up)",
    )
    parser.add_argument("--coverage", action="store_true", help="print coverage only")
    args = parser.parse_args()

    config = load_config(args.config)
    overrides = {}
    if args.format:
        overrides["battle_format"] = args.format
    if args.team:
        overrides["team"] = args.team
    if overrides:
        config = replace(config, **overrides)

    profile = profile_for(config.battle_format)
    scope = memory_scope.scope_for_config(config)
    root = memory_scope.ensure_scope_root(config, scope, config.battle_format)
    kb = StaticKB(config.battle_format, profile=profile)

    if not args.coverage:
        counts = seed.seed_scope(
            root, scope, kb, profile, config, all_species=args.all_species
        )
        print(f"seeded {scope} -> {root}")
        print(json.dumps(counts, ensure_ascii=False))

    coverage = seed.coverage(root, kb, profile, scope, config)
    print(json.dumps(coverage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()