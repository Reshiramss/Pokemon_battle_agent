#!/usr/bin/env python3
"""Inspect the cross-battle memory: scopes, counts, coverage, cold keys.

    python scripts/memory_report.py
    python scripts/memory_report.py --format gen9championsvgc2026regmchidden --coverage
    python scripts/memory_report.py --cold-keys
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokeagent.config import load_config  # noqa: E402
from pokeagent.formats import profile_for  # noqa: E402
from pokeagent.knowledge.static import StaticKB  # noqa: E402
from pokeagent.memory import facts as memory_facts  # noqa: E402
from pokeagent.memory import seed  # noqa: E402
from pokeagent.memory.scope import ensure_scope_root, scope_for_config  # noqa: E402


def _scope_roots(memory_root: Path) -> list[Path]:
    if not memory_root.is_dir():
        return []
    return sorted(
        path
        for path in memory_root.glob("*/*/*")
        if path.is_dir() and (path / "scenes").is_dir()
    )


def _counts(root: Path) -> dict:
    scenes = {
        kind: len(list((root / "scenes" / kind).glob("*.json")))
        if (root / "scenes" / kind).is_dir()
        else 0
        for kind in ("species", "teams", "archetypes", "our")
    }
    by_kind: dict[str, int] = {}
    by_maturity: dict[str, int] = {}
    by_source: dict[str, int] = {}
    total = 0
    for path in memory_facts.iter_fact_files(root):
        for row in memory_facts.read_fact_file(path):
            total += 1
            by_kind[row.kind] = by_kind.get(row.kind, 0) + 1
            by_maturity[row.maturity] = by_maturity.get(row.maturity, 0) + 1
            by_source[row.source] = by_source.get(row.source, 0) + 1
    return {
        "scenes": scenes,
        "facts": total,
        "by_kind": by_kind,
        "by_maturity": by_maturity,
        "by_source": by_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--format", default=None, help="coverage for this format")
    parser.add_argument("--coverage", action="store_true", help="print cold-start coverage")
    parser.add_argument("--cold-keys", action="store_true", help="print unprepared legal species")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.format:
        config.battle_format = args.format
    memory_root = Path(config.memory_path)
    roots = _scope_roots(memory_root)
    if not roots:
        print(f"memory is empty: {memory_root}")
    for root in roots:
        print(f"scope {root.relative_to(memory_root)}  {json.dumps(_counts(root), ensure_ascii=False)}")

    if args.coverage or args.cold_keys:
        profile = profile_for(config.battle_format)
        scope = scope_for_config(config)
        root = ensure_scope_root(config, scope, config.battle_format)
        kb = StaticKB(config.battle_format, profile=profile)
        coverage = seed.coverage(root, kb, profile, scope, config)
        if not args.cold_keys:
            coverage.pop("cold_keys", None)
            coverage.pop("cold_key_count", None)
        print("coverage:", json.dumps(coverage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()