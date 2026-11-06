"""Write finished battles into the layered cross-battle memory.

Two entry points, both failure-safe by construction (never raise into the
battle loop):

- ``write_battle`` — deterministic rows derived from the finished battle
  and the harness state: what each opponent mon revealed, the matchup
  history of both rosters, which tactics showed up, and one battle summary.
- ``write_digest`` — routes the cheap-model post-battle review's insights
  into the same layers (species first, tactics second, generic lessons to
  the battle log), capped at three lines per key.

Keys come from ``keys.py`` so the in-battle ``MemoryHook`` can look rows up
by the same derivation while the battle is still running.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..knowledge.static import normalize_name
from .keys import opponent_key, our_team_key, sanitize_key
from .store import MemoryStore
from .tactics import TacticSpec, detect_tactics

if TYPE_CHECKING:
    from poke_env.battle import AbstractBattle

    from ..context.manager import ContextManager

logger = logging.getLogger(__name__)

MAX_DIGEST_LINES = 12
MAX_LINES_PER_KEY = 3
_TURN_RE = re.compile(r"\b(?:turn|t)\s*(\d+)\b", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s*")


@dataclass
class MonView:
    """The subset of a Pokemon's knowledge that memory keeps."""

    species: str
    types: list[str] = field(default_factory=list)
    moves: list[str] = field(default_factory=list)
    ability: str | None = None
    item: str | None = None

    def summary(self) -> str:
        parts = [self.species]
        if self.types:
            parts.append("/".join(self.types))
        if self.moves:
            parts.append("moves: " + ", ".join(self.moves))
        if self.ability:
            parts.append("ability: " + self.ability)
        if self.item:
            parts.append("item: " + self.item)
        return "; ".join(parts)


def _norm_moves(raw) -> list[str]:
    if isinstance(raw, dict):
        raw = list(raw.keys())
    out: list[str] = []
    for move in raw or []:
        key = normalize_name(move)
        if key and key not in out:
            out.append(key)
    return out


def _item(raw) -> str | None:
    if not raw or not isinstance(raw, str):
        return None
    if raw.lower() in ("unknown_item", "unknownitem", "none", ""):
        return None
    return normalize_name(raw) or None


def _mon_view(mon) -> MonView | None:
    """From a ``MonNode`` (has ``revealed_moves``) or a poke-env Pokemon."""
    species = getattr(mon, "species", None)
    if not species:
        return None
    node_moves = getattr(mon, "revealed_moves", None)
    moves = _norm_moves(node_moves if node_moves is not None else getattr(mon, "moves", None))
    types = []
    for t in getattr(mon, "types", None) or []:
        name = getattr(t, "name", str(t))
        types.append(str(name))
    known = getattr(mon, "known_ability", None)
    return MonView(
        species=normalize_name(str(species)),
        types=types,
        moves=moves,
        ability=normalize_name(str(known)) if known else _ability(mon),
        item=_item(getattr(mon, "item", None)),
    )


def _ability(mon) -> str | None:
    ability = getattr(mon, "ability", None)
    return normalize_name(str(ability)) if ability else None


def _result(battle: "AbstractBattle") -> str:
    if getattr(battle, "won", False):
        return "win"
    if getattr(battle, "lost", False):
        return "loss"
    return "finished"


def _opp_views(battle: "AbstractBattle", cm: "ContextManager | None") -> list[MonView]:
    """Opponent roster: harness KB first, live objects second."""
    nodes = None
    if cm is not None and getattr(cm, "state", None) is not None:
        nodes = list(cm.state.kb.them.values())
    source = nodes if nodes else list(getattr(battle, "opponent_team", {}).values())
    views = [v for v in (_mon_view(m) for m in source) if v is not None]
    seen: set[str] = set()
    unique: list[MonView] = []
    for view in views:
        if view.species not in seen:
            seen.add(view.species)
            unique.append(view)
    return unique


def _our_species(battle: "AbstractBattle") -> list[str]:
    return [normalize_name(str(p.species)) for p in getattr(battle, "team", {}).values()]


class MemoryWriter:
    """Derives memory rows from finished battles and post-battle reviews."""

    def __init__(self, config, store: MemoryStore | None = None):
        self.config = config
        self.store = store or MemoryStore(
            getattr(config, "memory_path", None) or "runs/_memory",
            lines_cap=getattr(config, "memory_lines_cap", 500),
        )

    # --- deterministic battle rows ----------------------------------------

    def write_battle(
        self,
        battle: "AbstractBattle",
        cm: "ContextManager | None" = None,
    ) -> dict[str, int]:
        """Write species / teams / tactics / battle rows; returns counts."""
        counts = {"species": 0, "teams": 0, "tactics": 0, "battles": 0}
        try:
            counts = self._write_battle(battle, cm)
            if sum(counts.values()):
                self.store.rebuild_index()
        except Exception as exc:  # never let memory break the callback
            logger.warning("memory write failed: %s", exc)
        return counts

    def _write_battle(self, battle, cm) -> dict[str, int]:
        counts = {"species": 0, "teams": 0, "tactics": 0, "battles": 0}
        tag = getattr(battle, "battle_tag", "") or ""
        result = _result(battle)
        turns = int(getattr(battle, "turn", 0) or 0)
        foes = _opp_views(battle, cm)
        ours = _our_species(battle)
        if not foes and not ours:
            return counts

        opp_key = opponent_key([v.species for v in foes])
        our_key = our_team_key(self.config, ours)
        tactics = detect_tactics([_view_dict(v) for v in foes])

        def add(level: str, key: str, text: str, turn: int | None = None) -> None:
            if self.store.append(
                level, key, text, battle=tag, turn=turn, source="deterministic"
            ):
                counts[level] += 1

        for view in foes:
            add(
                "species",
                view.species,
                f"opponent {view.summary()}; vs {our_key}; {result} "
                f"in {turns} turns",
            )

        foe_roster = ", ".join(v.species for v in foes)
        add(
            "teams",
            opp_key,
            f"opponent roster {foe_roster or 'unknown'}; vs {our_key}; "
            f"{result} in {turns} turns",
        )
        add(
            "teams",
            our_key,
            f"our roster {', '.join(ours)}; vs {opp_key} "
            f"({foe_roster or 'unknown'}); {result} in {turns} turns"
            + (f"; tactics: {', '.join(t.id for t in tactics)}" if tactics else ""),
        )

        for spec in tactics:
            add(
                "tactics",
                spec.id,
                f"{spec.label}: seen vs {our_key} "
                f"({', '.join(_triggers(spec, foes)) or 'revealed'}); "
                f"{result} in {turns} turns",
            )

        add(
            "battles",
            f"{_date_key()}_{sanitize_key(tag) or 'battle'}",
            f"{result} in {turns} turns; our {our_key} vs {opp_key} "
            f"({foe_roster or 'unknown'})"
            + (f"; tactics: {', '.join(t.id for t in tactics)}" if tactics else ""),
        )
        return counts

    # --- post-battle digest ------------------------------------------------

    def write_digest(
        self,
        battle: "AbstractBattle",
        cm: "ContextManager | None",
        digest: str,
    ) -> dict[str, int]:
        """Route review insights into the layers (species, then tactics)."""
        counts = {"species": 0, "teams": 0, "tactics": 0, "battles": 0}
        try:
            counts = self._write_digest(battle, cm, digest)
            if sum(counts.values()):
                self.store.rebuild_index()
        except Exception as exc:
            logger.warning("memory digest write failed: %s", exc)
        return counts

    def _write_digest(self, battle, cm, digest) -> dict[str, int]:
        counts = {"species": 0, "teams": 0, "tactics": 0, "battles": 0}
        text = (digest or "").strip()
        if not text:
            return counts
        tag = getattr(battle, "battle_tag", "") or ""
        foes = _opp_views(battle, cm)
        foe_species = {v.species for v in foes}
        tactics = detect_tactics([_view_dict(v) for v in foes])
        tactic_index: dict[str, TacticSpec] = {}
        for spec in tactics:
            # id, label, and every trigger term — a review often names the
            # tactic by its signature move ("Perish Song") rather than the
            # tactic id
            for name in (spec.id, spec.label, *spec.moves, *spec.abilities, *spec.items):
                key = normalize_name(name)
                if key:
                    tactic_index.setdefault(key, spec)
        battle_key = f"{_date_key()}_{sanitize_key(tag) or 'battle'}"
        battle_level = "battles"

        per_key: dict[tuple[str, str], int] = {}
        for raw_line in _digest_lines(text):
            turn_match = _TURN_RE.search(raw_line)
            turn = int(turn_match.group(1)) if turn_match else None
            line_norm = normalize_name(raw_line)
            targets: list[tuple[str, str]] = []
            for species in sorted(foe_species, key=len, reverse=True):
                if species in line_norm:
                    targets.append(("species", species))
            if not targets:
                for name, spec in tactic_index.items():
                    if name and name in line_norm:
                        targets.append(("tactics", spec.id))
                        break
            if not targets:
                targets = [(battle_level, battle_key)]
            for level, key in targets[:2]:
                slot = (level, key)
                if per_key.get(slot, 0) >= MAX_LINES_PER_KEY:
                    continue
                if self.store.append(
                    level,
                    key,
                    raw_line,
                    battle=tag,
                    turn=turn,
                    source="post_battle",
                    dedupe="text",
                ):
                    per_key[slot] = per_key.get(slot, 0) + 1
                    counts[level] += 1
        return counts


def _view_dict(view: MonView) -> dict[str, Any]:
    return {
        "moves": view.moves,
        "ability": view.ability,
        "item": view.item,
    }


def _triggers(spec: TacticSpec, foes: list[MonView]) -> list[str]:
    hits: list[str] = []
    for view in foes:
        for move in view.moves:
            if move in spec.moves and move not in hits:
                hits.append(move)
        if view.ability and view.ability in spec.abilities:
            hits.append(view.ability)
        if view.item and view.item in spec.items:
            hits.append(view.item)
    return hits


def _date_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _digest_lines(text: str) -> list[str]:
    """Bullets when present, otherwise sentences; trimmed and deduped."""
    raw: list[str] = []
    for line in text.splitlines():
        was_bullet = bool(_BULLET_RE.match(line))
        line = _BULLET_RE.sub("", line).strip()
        if not line:
            continue
        if len(line) > 60 and not was_bullet:
            # a wrapped paragraph: split into sentences for routing
            parts = re.split(r"(?<=[.!?])\s+", line)
            raw.extend(p.strip() for p in parts if p.strip())
        else:
            raw.append(line)
    out: list[str] = []
    seen: set[str] = set()
    for line in raw:
        if len(line) < 8:
            continue
        fingerprint = line.lower()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(line)
        if len(out) >= MAX_DIGEST_LINES:
            break
    return out