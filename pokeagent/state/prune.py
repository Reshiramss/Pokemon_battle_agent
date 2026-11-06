"""Deterministic claim pruning from observed protocol events.

No model calls here. Two kinds of hard evidence are folded into the
inferred layer (``ReasoningLedger.claims``):

1. **Turn order → speed bounds.** When two priority-0 actions happen in the
   same turn under neutral conditions, their order fixes a strict bound on
   the opponent's speed relative to *our* known speed stat. Bounds only get
   tighter (max of lower bounds, min of upper bounds).
2. **Reveals → confirm/refute sets.** A confirmed opponent ability or item
   settles any belief of the same kind: matching beliefs are confirmed,
   contradicting ones are refuted. Revealed moves confirm move beliefs (a
   non-revealed move is *not* refuted — it may simply be unclicked).

Speed inference is deliberately conservative: it skips a turn when Trick
Room, Tailwind, weather, or a speed/paralysis change is in play. A wrong
bound is worse than no bound, because the model treats these as facts.

Damage → EV inference is not here: a damage percentage alone does not pin
an EV spread without assuming the opponent's item/ability/nature, so it
stays a model belief rather than a deterministic claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..knowledge.static import normalize_name
from .model import BattleState, Claim

if TYPE_CHECKING:
    from ..knowledge.static import StaticKB

CLAIM_LIMIT = 12
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass
class PruneResult:
    """What changed, for the ``claims_pruned`` event and tests."""

    added: list[str] = field(default_factory=list)
    confirmed: list[str] = field(default_factory=list)
    refuted: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    speed_bounds: dict[str, dict[str, int | None]] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(
            self.added or self.confirmed or self.refuted
            or self.dropped or self.speed_bounds
        )


# --- event row parsing -------------------------------------------------------


def _cmd_args(row: list[Any]) -> tuple[str, list[str]]:
    """Split a poke-env replay row into (command, arguments).

    Rows normally look like ``["", "move", "p1a: X", "Move", "p2a: Y"]``;
    the tests also pass pre-joined single-cell rows, so both forms work.
    """
    cells = [str(c).strip() for c in row]
    if len(cells) >= 2 and cells[0] == "":
        return cells[1], [c for c in cells[2:]]
    joined = "|".join(cells)
    parts = joined.split("|")
    if parts and parts[0] == "":
        parts = parts[1:]
    if not parts:
        return "", []
    return parts[0].strip(), [p.strip() for p in parts[1:]]


def _species_of(cell: str) -> str:
    """'p2a: Incineroar' -> 'incineroar'; '' when the cell is not a mon."""
    if not cell:
        return ""
    name = cell.split(":", 1)[1] if ":" in cell else cell
    return normalize_name(name)


def _side_of(cell: str) -> str | None:
    head = cell.split(":", 1)[0].strip()
    if head.startswith("p1"):
        return "p1"
    if head.startswith("p2"):
        return "p2"
    return None


def _speed_of(state: BattleState, species: str) -> int | None:
    node = state.kb.us.get(species)
    if node is None or not node.stats:
        return None
    speed = node.stats.get("spe")
    return int(speed) if isinstance(speed, int) else None


# --- speed bounds ------------------------------------------------------------


def speed_bounds(
    state: BattleState,
    rows: list[list[Any]],
    *,
    static_kb: "StaticKB | None" = None,
) -> dict[str, tuple[int | None, int | None]]:
    """Per-opponent-species (lower, upper) speed bounds from turn order.

    ``state.kb.us`` provides our exact speed stats (we know our own EVs).
    Only priority-0, same-turn, neutral-condition actions are compared.
    """
    if state.field.weather:
        # weather can double speed (Swift Swim / Chlorophyll etc.)
        return {}
    ours = {normalize_name(s) for s in state.kb.us}
    if not ours:
        return {}

    trick_room = any("TRICK" in name for name in state.field.fields)
    our_tw = any("TAILWIND" in name for name in state.field.side_conditions)
    foe_tw = any("TAILWIND" in name for name in state.field.opponent_side_conditions)

    bounds: dict[str, tuple[int | None, int | None]] = {}
    actions: list[dict[str, Any]] = []
    skip_turn = False

    def flush() -> None:
        nonlocal actions, skip_turn
        if not skip_turn and not trick_room and not our_tw and not foe_tw:
            for i, foe in enumerate(actions):
                if foe["ours"]:
                    continue
                for j, other in enumerate(actions):
                    if not other["ours"]:
                        continue
                    our_spe = other["spe"]
                    if our_spe is None:
                        continue
                    low, high = bounds.get(foe["species"], (None, None))
                    if j < i:
                        # our mon acted first: the foe is strictly slower
                        high = our_spe - 1 if high is None else min(high, our_spe - 1)
                    elif j > i:
                        # the foe acted first: the foe is strictly faster
                        low = our_spe + 1 if low is None else max(low, our_spe + 1)
                    else:
                        continue
                    bounds[foe["species"]] = (low, high)
        actions = []
        skip_turn = False

    for row in rows:
        cmd, args = _cmd_args(row)
        if cmd == "turn":
            flush()
            continue
        if not args:
            continue
        if cmd in ("-weather",):
            # any weather change during the turn can alter speed (Swift Swim
            # etc.); the post-turn state alone cannot tell us what it was
            flush()
            skip_turn = True
            continue
        if cmd in ("-fieldstart", "-fieldend"):
            if "trick" in args[0].lower() or "room" in args[0].lower():
                flush()
                skip_turn = True
            continue
        if cmd in ("-sidestart", "-sideend"):
            if any("tailwind" in a.lower() for a in args):
                flush()
                skip_turn = True
            continue
        if cmd in ("-boost", "-unboost", "setboost"):
            if len(args) >= 2 and args[1].lower() == "spe":
                skip_turn = True
            continue
        if cmd in ("status", "-status", "curestatus", "-curestatus"):
            if any(a.lower() == "par" for a in args[1:]):
                skip_turn = True
            continue

        if cmd not in ("move", "switch"):
            continue
        actor = args[0] if args else ""
        species = _species_of(actor)
        side = _side_of(actor)
        if not species or side is None:
            continue
        if cmd == "move":
            move = args[1] if len(args) > 1 else ""
            priority = 0
            if static_kb is not None and move:
                data = static_kb.move_data(move) or {}
                try:
                    priority = int(data.get("priority", 0) or 0)
                except (TypeError, ValueError):
                    priority = 0
            if priority != 0:
                continue
        is_ours = species in ours
        if is_ours:
            node = state.kb.us.get(species)
            if node is not None and (
                node.boosts.get("spe") or node.status == "PAR"
            ):
                continue
        else:
            node = state.kb.them.get(species)
            if node is not None and (
                node.boosts.get("spe") or node.status == "PAR"
            ):
                continue
        actions.append(
            {
                "ours": is_ours,
                "species": species,
                "spe": _speed_of(state, species) if is_ours else None,
            }
        )
    flush()
    return bounds


# --- reveal pruning ----------------------------------------------------------


def prune_reveals(state: BattleState) -> tuple[list[str], list[str]]:
    """Confirm/refute beliefs using confirmed ability/item/move reveals."""
    confirmed: list[str] = []
    refuted: list[str] = []
    claims = state.ledger.claims
    for species, node in state.kb.them.items():
        key = normalize_name(species)
        for claim in claims:
            if claim.status != "active":
                continue
            if normalize_name(claim.subject) != key:
                continue
            text = normalize_name(claim.text)
            if claim.kind == "ability" and node.known_ability:
                if normalize_name(node.known_ability) in text:
                    claim.status = "confirmed"
                    claim.confidence = "high"
                    confirmed.append(claim.text)
                else:
                    claim.status = "refuted"
                    refuted.append(claim.text)
            elif claim.kind == "item" and node.item:
                if normalize_name(node.item) in text:
                    claim.status = "confirmed"
                    claim.confidence = "high"
                    confirmed.append(claim.text)
                else:
                    claim.status = "refuted"
                    refuted.append(claim.text)
            elif claim.kind == "move":
                if any(normalize_name(m) in text for m in node.revealed_moves):
                    claim.status = "confirmed"
                    claim.confidence = "high"
                    confirmed.append(claim.text)
    return confirmed, refuted


def cap_claims(state: BattleState, limit: int = CLAIM_LIMIT) -> list[str]:
    """Keep the ``limit`` most useful active claims (refuted ones go first)."""

    def keep_rank(claim: Claim) -> tuple[int, int, int]:
        return (
            1 if claim.status == "active" else 0,
            _CONFIDENCE_RANK.get(claim.confidence, 1),
            claim.turn,
        )

    active = [c for c in state.ledger.claims if c.status == "active"]
    if len(active) <= limit:
        return []
    ranked = sorted(active, key=keep_rank, reverse=True)
    keep = {id(c) for c in ranked[:limit]}
    dropped = [c.text for c in active if id(c) not in keep]
    state.ledger.claims = [
        c for c in state.ledger.claims if c.status != "active" or id(c) in keep
    ]
    return dropped


# --- orchestration -----------------------------------------------------------


def prune_state(
    state: BattleState,
    rows: list[list[Any]],
    *,
    static_kb: "StaticKB | None" = None,
) -> PruneResult:
    """Apply all deterministic pruning for one finished turn."""
    result = PruneResult()
    ledger = state.ledger

    for species, (low, high) in speed_bounds(
        state, rows, static_kb=static_kb
    ).items():
        low, high = _merge_speed_claim(
            state, species, low, high, turn=state.field.turn, result=result
        )
        result.speed_bounds[species] = {"lower": low, "upper": high}

    confirmed, refuted = prune_reveals(state)
    result.confirmed.extend(confirmed)
    result.refuted.extend(refuted)
    result.dropped.extend(cap_claims(state))
    return result


def _parse_speed_text(text: str) -> tuple[int | None, int | None]:
    """Recover (lower, upper) from a claim we rendered ourselves."""
    low = high = None
    match = re.search(r">\s*(\d+)", text)
    if match:
        low = int(match.group(1))
    match = re.search(r"<\s*(\d+)", text)
    if match:
        high = int(match.group(1))
    return low, high


def _merge_speed_claim(
    state: BattleState,
    species: str,
    low: int | None,
    high: int | None,
    *,
    turn: int,
    result: PruneResult,
) -> tuple[int | None, int | None]:
    """One active speed claim per species; bounds only tighten.

    Returns the effective (lower, upper) after merging with the claim that
    is already stored, so the caller logs the accumulated bound rather than
    just this turn's contribution.
    """
    existing = next(
        (
            c
            for c in state.ledger.claims
            if c.status == "active"
            and c.kind == "speed"
            and normalize_name(c.subject) == normalize_name(species)
        ),
        None,
    )
    if existing is not None:
        old_low, old_high = _parse_speed_text(existing.text)
        if old_low is not None:
            low = old_low if low is None else max(low, old_low)
        if old_high is not None:
            high = old_high if high is None else min(high, old_high)
    parts = []
    if low is not None:
        parts.append(f"> {low}")
    if high is not None:
        parts.append(f"< {high}")
    if not parts:
        return low, high
    text = f"speed {' and '.join(parts)} (level-50 stat; from turn order)"
    if existing is None:
        claim = Claim(
            subject=species,
            kind="speed",
            text=text,
            confidence="high",
            source="inference",
            turn=turn,
        )
        state.ledger.claims.append(claim)
        result.added.append(text)
    elif existing.text != text:
        existing.text = text
        existing.turn = turn
        existing.confidence = "high"
        result.added.append(text)
    return low, high