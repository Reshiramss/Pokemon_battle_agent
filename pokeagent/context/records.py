"""Per-turn records and the compact battle snapshot builders.

``snapshot(battle)`` / ``available_actions(battle)`` turn the live poke-env
battle object into dense text suitable for an LLM prompt — one line per
entity, levels included (they vary wildly in random battles).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poke_env.battle import AbstractBattle


def _name(value: Any) -> str:
    """Enum -> short name."""
    if value is None:
        return "None"
    return getattr(value, "name", str(value))


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

    active = battle.active_pokemon
    lines.append(f"Your active: {describe_pokemon(active) if active else 'none'}")
    foe = battle.opponent_active_pokemon
    lines.append(
        f"Opponent active: {describe_pokemon(foe, revealed_moves=True) if foe else 'unknown'}"
    )

    if battle.team:
        team_bits = []
        for p in battle.team.values():
            marker = "*" if p is active else "-"
            team_bits.append(f"{marker} {describe_pokemon(p)}")
        lines.append("Your team: " + " | ".join(team_bits))
    if battle.opponent_team:
        foe_bits = [describe_pokemon(p, revealed_moves=True) for p in battle.opponent_team.values()]
        lines.append("Opponent revealed so far: " + " | ".join(foe_bits))
    return "\n".join(lines)


def available_actions(battle: "AbstractBattle") -> str:
    """Enumerate the legal actions of this decision point."""
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
