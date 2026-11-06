"""Tactic tagging from revealed moves / abilities / items.

The vocabulary lives in ``pokeagent/data/tactics.json`` so it can be edited
without touching code. A tactic matches when the union of every observed
mon's reveals hits the listed triggers:

- ``mode="any"`` (default): one trigger group is enough (a Perish Song user
  alone is already worth remembering);
- ``mode="all"``: every non-empty group must match (Perish Song *plus* a
  trapping ability — the actual trap core).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..knowledge.static import normalize_name

TACTICS_PATH = Path(__file__).resolve().parents[1] / "data" / "tactics.json"


@dataclass(frozen=True)
class TacticSpec:
    id: str
    label: str
    moves: tuple[str, ...] = ()
    abilities: tuple[str, ...] = ()
    items: tuple[str, ...] = ()
    mode: str = "any"


def _strings(raw) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(normalize_name(x) for x in raw if isinstance(x, str) and x)


@lru_cache(maxsize=1)
def load_tactics() -> dict[str, TacticSpec]:
    """Load the vocabulary; a missing/corrupt file degrades to no tactics."""
    try:
        data = json.loads(TACTICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    specs: dict[str, TacticSpec] = {}
    for tactic_id, raw in data.items():
        if not isinstance(raw, dict):
            continue
        spec = TacticSpec(
            id=str(tactic_id),
            label=str(raw.get("label") or tactic_id),
            moves=_strings(raw.get("moves")),
            abilities=_strings(raw.get("abilities")),
            items=_strings(raw.get("items")),
            mode="all" if raw.get("mode") == "all" else "any",
        )
        if spec.moves or spec.abilities or spec.items:
            specs[spec.id] = spec
    return specs


def _observed(mons) -> tuple[set[str], set[str], set[str]]:
    moves: set[str] = set()
    abilities: set[str] = set()
    items: set[str] = set()
    for mon in mons or []:
        if isinstance(mon, dict):
            m, a, i = mon.get("moves"), mon.get("ability"), mon.get("item")
        else:
            m = getattr(mon, "moves", None)
            a = getattr(mon, "ability", None)
            i = getattr(mon, "item", None)
        if isinstance(m, dict):
            m = list(m.keys())
        for move in m or []:
            moves.add(normalize_name(move))
        if a:
            abilities.add(normalize_name(a))
        if i:
            items.add(normalize_name(i))
    moves.discard("")
    abilities.discard("")
    items.discard("")
    return moves, abilities, items


def detect_tactics(mons) -> list[TacticSpec]:
    """Tactics matched by the union of ``mons``' revealed information."""
    moves, abilities, items = _observed(mons)
    hits: list[TacticSpec] = []
    for spec in load_tactics().values():
        groups = []
        if spec.moves:
            groups.append(bool(moves & set(spec.moves)))
        if spec.abilities:
            groups.append(bool(abilities & set(spec.abilities)))
        if spec.items:
            groups.append(bool(items & set(spec.items)))
        if not groups:
            continue
        matched = all(groups) if spec.mode == "all" else any(groups)
        if matched:
            hits.append(spec)
    return hits


def tactic_ids(mons) -> list[str]:
    return [spec.id for spec in detect_tactics(mons)]