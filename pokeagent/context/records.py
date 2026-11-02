"""Per-turn records and the compact battle snapshot builders.

``snapshot(battle)`` / ``available_actions(battle)`` turn the live poke-env
battle object into dense text suitable for an LLM prompt — one line per
entity, levels included (they vary wildly in random battles).

Both builders branch on battle class: singles stays byte-compatible with
the original format; doubles renders per-slot state with the self/ally/
opp1/opp2 target vocabulary that ``submit_decision`` validates against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from poke_env.battle import DoubleBattle

if TYPE_CHECKING:
    from poke_env.battle import AbstractBattle

# Showdown target positions (DoubleBattle class attributes, mirrored for the
# symbolic<->int mapping used by loop.py validation)
POKEMON_POSITIONS = (-1, -2)
OPPONENT_POSITIONS = (1, 2)


def _name(value: Any) -> str:
    """Enum -> short name."""
    if value is None:
        return "None"
    return getattr(value, "name", str(value))


def is_doubles(battle: "AbstractBattle") -> bool:
    return isinstance(battle, DoubleBattle)


def target_symbol(position: int, slot_index: int) -> str:
    """Showdown target int -> the symbolic target vocabulary.

    ``slot_index`` is the 0-based slot of the mon USING the move: position
    -(slot_index+1) is "self", the other negative position is "ally".
    """
    if position == 0:
        return "auto"
    if position in OPPONENT_POSITIONS:
        return f"opp{position}"
    self_position = -(slot_index + 1)
    if position == self_position:
        return "self"
    if position in POKEMON_POSITIONS:
        return "ally"
    return "auto"


def symbol_position(symbol: str, slot_index: int) -> int:
    """Inverse of target_symbol (symbolic -> showdown target int)."""
    if symbol == "self":
        return -(slot_index + 1)
    if symbol == "ally":
        return -2 if slot_index == 0 else -1
    if symbol == "opp1":
        return 1
    if symbol == "opp2":
        return 2
    return 0  # auto / anything unrecognized


def move_target_symbols(battle: DoubleBattle, move, slot_index: int) -> list[str]:
    """Legal symbolic targets of ``move`` as used by the slot's active mon."""
    mon = battle.active_pokemon[slot_index]
    if mon is None:
        return []
    return [
        target_symbol(t, slot_index)
        for t in battle.get_possible_showdown_targets(move, mon)
    ]


def _speed_estimate(p) -> int:
    """Level-50 speed estimate from base stats (31 IV, 0 EV, neutral nature).

    Only used for OPPONENT mons whose actual EVs/IVs/nature are unknown;
    our own team reports exact ``stats['spe']``.
    """
    base = p.base_stats.get("spe", 0)
    return int((2 * base + 31) * p.level / 100 + 5)


def describe_pokemon(p, *, revealed_moves: bool = False) -> str:
    """One compact line describing a Pokemon."""
    parts = [f"{p.species} L{p.level}"]
    try:
        parts.append(f"{p.current_hp_fraction * 100:.0f}%")
    except Exception:
        parts.append("HP?")
    if p.fainted:
        parts.append("FAINTED")
    if p.status:
        parts.append(f"[{_name(p.status).lower()}]")
    try:
        types = "/".join(_name(t) for t in p.types)
        parts.append(f"({types})")
    except Exception:
        pass
    if getattr(p, "is_terastallized", False) and p.tera_type:
        parts.append(f"tera:{_name(p.tera_type)}")
    try:
        boosts = {k: v for k, v in dict(p.boosts).items() if v}
        if boosts:
            parts.append("boosts{" + ",".join(f"{_name(k)}{v:+d}" for k, v in boosts.items()) + "}")
    except Exception:
        pass
    if p.item:
        parts.append(f"item:{p.item}")
    if revealed_moves and p.moves:
        parts.append("known-moves:" + ",".join(sorted(p.moves)))
    return " ".join(parts)


def snapshot(battle: "AbstractBattle") -> str:
    """Compact full-battle state as text (what the model should know now)."""
    lines = [f"Turn {battle.turn} | {battle.format}"]
    if battle.weather:
        lines.append("Weather: " + ", ".join(_name(w) for w in battle.weather))
    if battle.fields:
        lines.append("Fields: " + ", ".join(_name(f) for f in battle.fields))
    if battle.side_conditions:
        lines.append("Your side: " + ", ".join(_name(s) for s in battle.side_conditions))
    if battle.opponent_side_conditions:
        lines.append("Opponent side: " + ", ".join(_name(s) for s in battle.opponent_side_conditions))

    if is_doubles(battle):
        lines.extend(_snapshot_doubles(battle))
    else:
        active = battle.active_pokemon
        lines.append(f"Your active: {describe_pokemon(active) if active else 'none'}")
        foe = battle.opponent_active_pokemon
        lines.append(
            f"Opponent active: {describe_pokemon(foe, revealed_moves=True) if foe else 'unknown'}"
        )

    if battle.team:
        actives = (
            set(battle.active_pokemon) if is_doubles(battle) else {battle.active_pokemon}
        )
        team_bits = []
        for p in battle.team.values():
            marker = "*" if p in actives else "-"
            team_bits.append(f"{marker} {describe_pokemon(p)}")
        lines.append("Your team: " + " | ".join(team_bits))
    if battle.opponent_team:
        foe_bits = [describe_pokemon(p, revealed_moves=True) for p in battle.opponent_team.values()]
        lines.append("Opponent revealed so far: " + " | ".join(foe_bits))
    return "\n".join(lines)


def _snapshot_doubles(battle: "DoubleBattle") -> list[str]:
    """Slot-labelled field state plus the speed-order line (doubles only)."""
    lines: list[str] = []
    for i, p in enumerate(battle.active_pokemon):
        lines.append(
            f"Your slot {i + 1}: {describe_pokemon(p) if p else 'empty'}"
        )
    for i, p in enumerate(battle.opponent_active_pokemon):
        label = f"opp{i + 1}"
        lines.append(
            f"Opponent slot {i + 1} ({label}): "
            f"{describe_pokemon(p, revealed_moves=True) if p else 'empty'}"
        )

    # speed order across the four field mons: our side is exact, theirs is a
    # base-stat estimate (their EVs/nature are hidden) — ties are genuinely
    # unknown in doubles, so equal numbers are reported as a tie
    entries = []
    for p in battle.active_pokemon:
        if p is not None and not p.fainted:
            try:
                entries.append((p.stats["spe"], p.species, "yours"))
            except Exception:
                pass
    for p in battle.opponent_active_pokemon:
        if p is not None and not p.fainted:
            entries.append((_speed_estimate(p), p.species, "foe~"))
    if entries:
        entries.sort(key=lambda e: -e[0])
        bits = [
            f"{name} {spe}" if who == "yours" else f"{name} ~{spe} (foe est.)"
            for spe, name, who in entries
        ]
        lines.append("Speed order (fastest first): " + " > ".join(bits))
    return lines


def available_actions(battle: "AbstractBattle") -> str:
    """Enumerate the legal actions of this decision point."""
    if is_doubles(battle):
        return _available_actions_doubles(battle)
    lines: list[str] = []
    if battle.available_moves:
        lines.append("Moves:")
        for m in battle.available_moves:
            damage = f"fixed {m.damage}" if m.damage else f"BP {m.base_power}"
            lines.append(
                f"  - {m.id} | type {_name(m.type)} | {damage}"
                f" | {_name(m.category)} | acc {m.accuracy} | PP {m.current_pp}/{m.max_pp}"
                f" | prio {m.priority}"
            )
    if battle.available_switches:
        lines.append("Switches:")
        for p in battle.available_switches:
            lines.append(f"  - {describe_pokemon(p)}")
    if not battle.available_moves and not battle.available_switches:
        lines.append("No moves/switches available — default (Struggle) will be used.")
    notes = []
    if getattr(battle, "can_tera", False):
        notes.append("terastallization available (pass terastallize=true with a move)")
    if battle.trapped:
        notes.append("you are trapped — switching is not possible")
    if notes:
        lines.append("Notes: " + "; ".join(notes))
    return "\n".join(lines)


def _available_actions_doubles(battle: "DoubleBattle") -> str:
    """Per-slot legal actions with symbolic targets (doubles only)."""
    lines: list[str] = ["Spread moves hit both foes at 0.75x power."]
    forced = list(battle.force_switch)
    if any(forced):
        forced_bits = [f"slot {i + 1}" for i, f in enumerate(forced) if f]
        lines.append(
            f"FORCED SWITCH for {' and '.join(forced_bits)}: those slots must "
            "choose a switch (or pass if none); other slots pass this round."
        )
    for i in range(2):
        mon = battle.active_pokemon[i]
        lines.append(f"Slot {i + 1} ({mon.species if mon else 'no active'}):")
        if any(forced) and not forced[i]:
            lines.append("  - pass (this slot sits out the forced-switch decision)")
            continue
        if forced[i]:
            switches = battle.available_switches[i]
            if switches:
                lines.append("  Switches:")
                for p in switches:
                    lines.append(f"    - {describe_pokemon(p)}")
            else:
                lines.append("  - pass (nothing to switch to)")
            continue
        if mon is None or mon.fainted:
            lines.append("  - pass (no active Pokemon)")
            continue
        moves = battle.available_moves[i]
        if moves:
            lines.append("  Moves:")
            for m in moves:
                damage = f"fixed {m.damage}" if m.damage else f"BP {m.base_power}"
                targets = move_target_symbols(battle, m, i)
                lines.append(
                    f"    - {m.id} | type {_name(m.type)} | {damage}"
                    f" | {_name(m.category)} | acc {m.accuracy} | PP {m.current_pp}/{m.max_pp}"
                    f" | prio {m.priority} | targets: {','.join(targets)}"
                )
        switches = battle.available_switches[i]
        if switches:
            lines.append("  Switches:")
            for p in switches:
                lines.append(f"    - {describe_pokemon(p)}")
        if not moves and not switches:
            lines.append("  - pass (no moves or switches available)")
        notes = []
        if battle.can_tera[i]:
            notes.append("terastallization available (terastallize=true with a move)")
        if battle.trapped[i]:
            notes.append("trapped — switching is not possible")
        if notes:
            lines.append("  Notes: " + "; ".join(notes))
    return "\n".join(lines)


@dataclass
class TurnRecord:
    """Everything about one decision point, persisted verbatim to disk."""

    seq: int  # monotonically increasing per battle (decision sequence)
    turn: int  # battle turn number (may repeat: multiple requests per turn)
    request: str  # the state + actions the model saw
    events: list[str] = field(default_factory=list)  # protocol events since previous decision
    decision: str | None = None  # e.g. "move:earthquake"
    reason: str | None = None
    cot: str | None = None  # assistant reasoning, verbatim
    tool_calls: list[dict] = field(default_factory=list)  # {round, name, arguments}
    input_tokens: int = 0
    output_tokens: int = 0
    fell_back: bool = False
    fallback_reason: str | None = None
    elapsed_s: float = 0.0
