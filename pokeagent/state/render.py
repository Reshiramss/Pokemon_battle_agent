"""Render the ``[NOW]`` board text from :class:`BattleState`.

The output is byte-compatible with ``context.records.snapshot(battle)`` for
every battle that was *observed* normally (singles and doubles). That
equivalence is the contract that lets the harness render the prompt from
persisted state instead of re-reading the live object: ``tests/test_state.py``
asserts it, and any future snapshot change must keep it.

One intentional divergence: opponent team-preview stubs have no revealed HP,
so they render ``HP?`` instead of Showdown's internal ``0%`` (the stub is
not a fainted mon).
"""

from __future__ import annotations

from ..knowledge.tree import MonNode
from .model import BattleState


def _describe(node: MonNode, *, revealed_moves: bool = False) -> str:
    """One compact line describing a Pokemon (mirrors records.describe_pokemon)."""
    parts = [f"{node.species} L{node.level}"]
    try:
        parts.append(f"{node.hp_fraction * 100:.0f}%")
    except Exception:
        parts.append("HP?")
    if node.fainted:
        parts.append("FAINTED")
    if node.status:
        parts.append(f"[{node.status.lower()}]")
    if node.types:
        parts.append("(" + "/".join(node.types) + ")")
    if node.terastallized and node.tera_type:
        parts.append(f"tera:{node.tera_type}")
    if node.boosts:
        parts.append(
            "boosts{"
            + ",".join(f"{key}{value:+d}" for key, value in node.boosts.items())
            + "}"
        )
    if node.item_raw:
        parts.append(f"item:{node.item_raw}")
    if revealed_moves and node.revealed_moves:
        parts.append("known-moves:" + ",".join(sorted(node.revealed_moves)))
    return " ".join(parts)


def _ordered(state: BattleState, *, ours: bool) -> list[str]:
    """Species in team order, plus any node that arrived outside the order."""
    side = state.ours if ours else state.theirs
    nodes = state.kb.us if ours else state.kb.them
    seen: set[str] = set()
    ordered: list[str] = []
    for species in side.order:
        if species in nodes and species not in seen:
            ordered.append(species)
            seen.add(species)
    for species in nodes:
        if species not in seen:
            ordered.append(species)
            seen.add(species)
    return ordered


def _speed_estimate(node: MonNode) -> int:
    """Level-50 base-stat speed estimate (31 IV, 0 EV, neutral nature)."""
    base = node.base_stats.get("spe", 0)
    return int((2 * base + 31) * (node.level or 50) / 100 + 5)


def render_observed(state: BattleState) -> str:
    """The dense board text for ``[NOW]``, rendered from persisted state."""
    lines = [f"Turn {state.field.turn} | {state.meta.format}"]
    if state.field.weather:
        lines.append("Weather: " + ", ".join(state.field.weather))
    if state.field.fields:
        lines.append("Fields: " + ", ".join(state.field.fields))
    if state.field.side_conditions:
        lines.append("Your side: " + ", ".join(state.field.side_conditions))
    if state.field.opponent_side_conditions:
        lines.append(
            "Opponent side: " + ", ".join(state.field.opponent_side_conditions)
        )

    if state.meta.doubles:
        lines.extend(_doubles_lines(state))
    else:
        lines.append(f"Your active: {_active_line(state, ours=True, slot=0)}")
        lines.append(f"Opponent active: {_active_line(state, ours=False, slot=0)}")

    ours_order = _ordered(state, ours=True)
    if ours_order:
        selection_known = state.selected_known
        actives = set(state.ours.active_species)
        team_bits = []
        for species in ours_order:
            node = state.kb.us[species]
            selected = bool(node.selected)
            if species in actives:
                marker = "*"
            elif selection_known and selected:
                marker = "+"
            else:
                marker = "-"
            line = f"{marker} {_describe(node)}"
            if selection_known and not selected:
                line += " [NOT PICKED — unavailable this battle]"
            team_bits.append(line)
        header = (
            "Your team (* active, + picked bench, - not picked): "
            if selection_known
            else "Your team: "
        )
        lines.append(header + " | ".join(team_bits))

    theirs_order = _ordered(state, ours=False)
    if theirs_order:
        foe_bits = [
            _describe(state.kb.them[species], revealed_moves=True)
            for species in theirs_order
        ]
        lines.append("Opponent revealed so far: " + " | ".join(foe_bits))
    return "\n".join(lines)


def _active_line(state: BattleState, *, ours: bool, slot: int) -> str:
    active = state.ours.active if ours else state.theirs.active
    species = active[slot] if slot < len(active) else None
    nodes = state.kb.us if ours else state.kb.them
    node = nodes.get(species) if species else None
    if node is None:
        return "none" if ours else "unknown"
    return _describe(node, revealed_moves=not ours)


def _doubles_lines(state: BattleState) -> list[str]:
    """Slot-labelled field state plus the speed-order line (doubles only)."""
    lines: list[str] = []
    for i in range(len(state.ours.active)):
        lines.append(
            f"Your slot {i + 1}: {_slot_line(state, ours=True, slot=i)}"
        )
    for i in range(len(state.theirs.active)):
        label = f"opp{i + 1}"
        lines.append(
            f"Opponent slot {i + 1} ({label}): "
            f"{_slot_line(state, ours=False, slot=i)}"
        )

    entries: list[tuple[int, str, str]] = []
    for species in state.ours.active_species:
        node = state.kb.us.get(species)
        if node is None or node.fainted or not node.stats:
            continue
        spe = node.stats.get("spe")
        if isinstance(spe, int):
            entries.append((spe, species, "yours"))
    for species in state.theirs.active_species:
        node = state.kb.them.get(species)
        if node is None or node.fainted:
            continue
        entries.append((_speed_estimate(node), species, "foe~"))
    if entries:
        entries.sort(key=lambda entry: -entry[0])
        bits = [
            f"{name} {spe}" if who == "yours" else f"{name} ~{spe} (foe est.)"
            for spe, name, who in entries
        ]
        lines.append("Speed order (fastest first): " + " > ".join(bits))
    return lines


def _slot_line(state: BattleState, *, ours: bool, slot: int) -> str:
    active = state.ours.active if ours else state.theirs.active
    species = active[slot] if slot < len(active) else None
    nodes = state.kb.us if ours else state.kb.them
    node = nodes.get(species) if species else None
    if node is None:
        return "empty"
    return _describe(node, revealed_moves=not ours)