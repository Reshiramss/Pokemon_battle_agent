"""Pre-decision hook: plain-language effects of the moves in play.

The per-turn prompt lists move ids with type/power/accuracy/PP, but not what
a move *does* — "taunt", "protect" or "astralbarrage" assume the reader knows
the effect. In practice the model rarely spent a tool call on ``move_info``
(one lookup across 125 decisions in an observed battle), so the knowledge
that changes a play was usually absent.

This hook injects the Showdown ``shortDesc`` for:

- our currently selectable moves (per slot in doubles),
- the opponent's revealed moves,

plus a one-shot marker for newly revealed opponent moves / abilities /
items on the turn they first appear. Descriptions are shortDesc-only,
per-line and total capped, so the fragment stays a few hundred characters.

State is keyed by battle tag so a long-lived agent does not repeat the
"NEW" markers every turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..context.records import is_doubles
from ..knowledge.showdown_data import get_showdown_data
from .base import PreDecisionHook

if TYPE_CHECKING:
    from ..tools.base import ToolContext

MAX_CHARS = 1200
MAX_DESC_CHARS = 110
MAX_AVAILABLE = 8
MAX_REVEALED = 10

_UNKNOWN_ITEMS = {"unknown_item", "unknownitem", "none", ""}


def _data(ctx: "ToolContext"):
    return getattr(getattr(ctx, "kb", None), "data", None) or get_showdown_data()


def _desc(data, move_id: str) -> str | None:
    text = data.move_text(move_id) or {}
    desc = text.get("shortDesc") or text.get("desc")
    if not desc:
        return None
    desc = " ".join(str(desc).split())
    if len(desc) > MAX_DESC_CHARS:
        desc = desc[: MAX_DESC_CHARS - 1].rstrip() + "…"
    return desc


def _ability_desc(data, ability: str) -> tuple[str, str | None]:
    text = data.ability_text(ability) or {}
    entry = data.ability(ability) or {}
    name = text.get("name") or entry.get("name") or ability
    desc = text.get("shortDesc") or text.get("desc")
    if desc:
        desc = " ".join(str(desc).split())
        if len(desc) > MAX_DESC_CHARS:
            desc = desc[: MAX_DESC_CHARS - 1].rstrip() + "…"
    return str(name), (str(desc) if desc else None)


@dataclass
class _Seen:
    """What this battle has already shown us (for one-shot reveal lines)."""

    initialized: bool = False
    moves: set[str] = field(default_factory=set)
    abilities: set[str] = field(default_factory=set)
    items: set[str] = field(default_factory=set)


class MoveEffectsHook(PreDecisionHook):
    """ShortDesc for available and revealed moves + one-shot reveals."""

    name = "move_effects"

    def __init__(self, max_chars: int = MAX_CHARS):
        self.max_chars = max_chars
        self._seen: dict[str, _Seen] = {}

    async def run(self, ctx: "ToolContext") -> str | None:
        data = _data(ctx)
        if data is None:
            return None
        battle = ctx.battle

        # --- opponent reveals (ordered: active foes first) ------------------
        foes = list(getattr(battle, "opponent_team", {}).values())
        active = getattr(battle, "opponent_active_pokemon", None)
        active = active if isinstance(active, list) else ([active] if active else [])
        active_species = {getattr(m, "species", None) for m in active}
        ordered = sorted(
            foes, key=lambda m: getattr(m, "species", None) not in active_species
        )

        foe_moves: list[str] = []
        for mon in ordered:
            for move_id in (getattr(mon, "moves", None) or {}):
                key = str(move_id)
                if key and key not in foe_moves:
                    foe_moves.append(key)
        foe_abilities = {
            str(a).lower()
            for mon in foes
            if (a := getattr(mon, "ability", None))
        }
        foe_items = {
            str(i).lower()
            for mon in foes
            if (i := getattr(mon, "item", None))
            and str(i).lower() not in _UNKNOWN_ITEMS
        }

        tag = getattr(battle, "battle_tag", "") or "battle"
        seen = self._seen.setdefault(tag, _Seen())
        if seen.initialized:
            new_moves = [m for m in foe_moves if m not in seen.moves]
            new_abilities = sorted(foe_abilities - seen.abilities)
            new_items = sorted(foe_items - seen.items)
        else:
            # first prompt of the battle: record the baseline without
            # flooding it with "NEW" markers for what the preview/request
            # already showed
            new_moves, new_abilities, new_items = [], [], []
        seen.initialized = True
        seen.moves.update(foe_moves)
        seen.abilities.update(foe_abilities)
        seen.items.update(foe_items)

        # --- our available moves --------------------------------------------
        available: list[str] = []
        if is_doubles(battle):
            sources = list(getattr(battle, "available_moves", []) or [])
        else:
            sources = [list(getattr(battle, "available_moves", []) or [])]
        for moves in sources:
            for move in moves or []:
                move_id = str(getattr(move, "id", move))
                if move_id and move_id not in available:
                    available.append(move_id)

        # --- assemble, most salient first -----------------------------------
        lines: list[str] = []
        used = 0
        emitted_moves: set[str] = set()

        def add(line: str | None) -> bool:
            nonlocal used
            if not line or used + len(line) + 1 > self.max_chars:
                return False
            lines.append(line)
            used += len(line) + 1
            return True

        for move_id in new_moves:
            desc = _desc(data, move_id)
            if add(f"NEW foe move {move_id}: {desc}" if desc else None):
                emitted_moves.add(move_id)
        for ability in new_abilities:
            name, desc = _ability_desc(data, ability)
            add(f"NEW foe ability {name}: {desc}" if desc else f"NEW foe ability {name}")
        for item in new_items:
            add(f"NEW foe item {item}")

        for move_id in available[:MAX_AVAILABLE]:
            if move_id in emitted_moves:
                continue
            desc = _desc(data, move_id)
            if add(f"yours {move_id}: {desc}" if desc else None):
                emitted_moves.add(move_id)

        for move_id in foe_moves[:MAX_REVEALED]:
            if move_id in emitted_moves:
                continue
            desc = _desc(data, move_id)
            if add(f"foe {move_id}: {desc}" if desc else None):
                emitted_moves.add(move_id)

        return "\n".join(lines) if lines else None