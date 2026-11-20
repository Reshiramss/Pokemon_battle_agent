#!/usr/bin/env python3
"""Backfill memory atoms from existing battle run directories.

Scans ``runs/*/state.json`` + ``meta.json`` + ``events.jsonl``, atomises each
battle (no model calls) and merges it into the scoped stores. Idempotent:
an existing atom file is left alone unless ``--force`` is passed.

    python scripts/rebuild_memory.py --dry-run
    python scripts/rebuild_memory.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokeagent.config import AgentConfig  # noqa: E402
from pokeagent.formats import profile_for  # noqa: E402
from pokeagent.memory import atoms, scenes  # noqa: E402
from pokeagent.memory.scope import (  # noqa: E402
    ensure_scope_root,
    scope_from,
)
from pokeagent.state.model import BattleState  # noqa: E402


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _event_time(run_dir: Path) -> str:
    stamp = run_dir.name.split("_", 1)[0]
    try:
        dt = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return dt.isoformat(timespec="seconds")
    except ValueError:
        try:
            return datetime.fromtimestamp(run_dir.stat().st_mtime, timezone.utc).isoformat(
                timespec="seconds"
            )
        except OSError:
            return ""


def _result_and_turns(events_path: Path) -> tuple[str, int]:
    result, turns = "finished", 0
    try:
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") == "battle_finished":
                    turns = int(entry.get("turn") or 0)
                    if entry.get("won"):
                        result = "win"
                    elif entry.get("lost"):
                        result = "loss"
    except OSError:
        pass
    return result, turns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs", help="runs directory (default: runs)")
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    parser.add_argument("--force", action="store_true", help="rewrite existing atom files")
    parser.add_argument("--limit", type=int, default=0, help="max run dirs to process")
    args = parser.parse_args()

    runs = Path(args.runs)
    if not runs.is_dir():
        raise SystemExit(f"runs directory not found: {runs}")
    dirs = [
        path
        for path in sorted(runs.iterdir())
        if path.is_dir() and not path.name.startswith("_") and path.name != "logs"
    ]
    if args.limit:
        dirs = dirs[-args.limit :]

    processed = skipped = 0
    for run_dir in dirs:
        state_data = _read_json(run_dir / "state.json")
        tree_data = _read_json(run_dir / "kb_tree.json")
        meta_data = _read_json(run_dir / "meta.json") or {}
        cfg = meta_data.get("config") or {}
        fmt = meta_data.get("format") or cfg.get("battle_format") or ""
        tag = str(meta_data.get("battle_tag") or "")
        player = ""
        parts = run_dir.name.split("_")
        if len(parts) >= 3:
            player = parts[1]
        if isinstance(state_data, dict):
            try:
                state = BattleState.from_json(state_data)
            except Exception as exc:
                skipped += 1
                print(f"skip {run_dir.name}: bad state ({exc})")
                continue
            fmt = state.meta.format or fmt
            tag = state.meta.battle_tag or tag
            player = state.meta.player_username or player
        elif isinstance(tree_data, dict):
            # Older runs predate state.json; the KB tree still has every
            # revealed set, which is all the deterministic atomiser needs.
            try:
                from pokeagent.knowledge.tree import BattleKB

                state = SimpleNamespace(
                    meta=SimpleNamespace(
                        format=fmt, battle_tag=tag, player_username=player
                    ),
                    kb=BattleKB.from_json(tree_data),
                )
            except Exception as exc:
                skipped += 1
                print(f"skip {run_dir.name}: bad kb_tree ({exc})")
                continue
        else:
            skipped += 1
            print(f"skip {run_dir.name}: no state.json/kb_tree.json")
            continue
        if not fmt:
            skipped += 1
            print(f"skip {run_dir.name}: unknown format")
            continue
        profile = profile_for(fmt)
        config = AgentConfig(
            battle_format=fmt,
            team=cfg.get("team"),
            opponent_class="unknown",
        )
        scope = scope_from(profile, fmt, config.default_regulation)
        root = ensure_scope_root(config, scope, fmt)
        result, turns = _result_and_turns(run_dir / "events.jsonl")
        event_time = _event_time(run_dir)
        _meta, atom_rows = atoms.atomize_serialized(
            state,
            result=result,
            turns=turns,
            config=config,
            profile=profile,
            event_time=event_time,
        )
        target = atoms.atoms_path(
            root,
            state.meta.battle_tag,
            event_time[:10],
            str(state.meta.player_username or ""),
        )
        if target.exists() and not args.force:
            skipped += 1
            print(f"skip {run_dir.name}: atoms already exist")
            continue
        if args.dry_run:
            print(
                f"would write {run_dir.name}: {len(atom_rows)} atoms "
                f"-> {target} ({scope})"
            )
            processed += 1
            continue
        meta, atom_rows = atoms.atomize_serialized(
            state,
            result=result,
            turns=turns,
            config=config,
            profile=profile,
            event_time=event_time,
        )
        atoms.write_atoms(root, meta, atom_rows)
        scenes.merge_atoms(root, atom_rows, scope)
        scenes.rebuild_index(root)
        processed += 1
        print(f"wrote {run_dir.name}: {len(atom_rows)} atoms -> {target}")

    print(f"done: processed={processed} skipped={skipped}")
    for root in sorted({  # report the scope layout that was touched
        str(path.parent) for path in runs.glob("_memory/*/*/*/atoms/*.ndjson")
    }):
        print("scope:", root)


if __name__ == "__main__":
    main()