"""Pre-decision hook: plain-language effects of the abilities in play.

``[NOW]`` and the knowledge tree name each Pokemon's ability, but a bare id
("prankster", "armortail") does not tell the model what it does. The classic
failure is a status move boosted by Prankster being blocked by Armor Tail:
both names appeared in the prompt, neither effect did.

This hook injects one short line per unique *confirmed* ability — our whole
team (bench included, since switch decisions are made now) and the
opponent's revealed abilities. Unknown opponent abilities stay unknown;
BattleKBHook already lists the possible ones without effects, so nothing is
invented here. Output is shortDesc-only and self-capped well below
HookSystem's fragment cap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..knowledge.showdown_data import get_showdown_data
from .base import PreDecisionHook

if TYPE_CHECKING:
    from ..tools.base import ToolContext

MAX_CHARS = 1000
MAX_DESC_CHARS = 180


def _data(ctx: "ToolContext"):
    """The battle's format bundle, or the base bundle as a fallback."""
    return getattr(getattr(ctx, "kb", None), "data", None) or get_showdown_data()


def _ability_line(
    data, ability: str, *, owner: str, species: str
) -> str | None:
    """One ``owner species has Ability: shortDesc`` line, or None."""
    text = data.ability_text(ability) or {}
    entry = data.ability(ability) or {}
    desc = text.get("shortDesc") or text.get("desc")
    if not desc:
        return None
    name = text.get("name") or entry.get("name") or ability
    desc = " ".join(str(desc).split())
    if len(desc) > MAX_DESC_CHARS:
        desc = desc[: MAX_DESC_CHARS - 1].rstrip() + "…"
    return f"{owner} {species} has {name}: {desc}"


class AbilityEffectsHook(PreDecisionHook):
    """One line per confirmed ability effect in this battle."""

    name = "ability_effects"

    def __init__(self, max_chars: int = MAX_CHARS):
        self.max_chars = max_chars

    async def run(self, ctx: "ToolContext") -> str | None:
        data = _data(ctx)
        if data is None:
            return None

        lines: list[str] = []
        seen: set[str] = set()
        used = 0

        def add(mon, owner: str) -> None:
            nonlocal used
            ability = getattr(mon, "ability", None)
            if not ability:
                return
            key = str(ability).lower()
            if key in seen:
                return
            line = _ability_line(
                data, key, owner=owner, species=str(mon.species)
            )
            if line is None:
                return
            if used + len(line) + 1 > self.max_chars:
                return
            seen.add(key)
            lines.append(line)
            used += len(line) + 1

        # after a pick-N preview the unpicked mons can never act; their
        # abilities are still on the [NOW] team line, but their effect text
        # would only spend prompt budget
        ours = list(getattr(ctx.battle, "team", {}).values())
        selection_known = any(
            getattr(mon, "_selected_in_teampreview", False) for mon in ours
        )
        for mon in ours:
            if getattr(mon, "fainted", False):
                continue
            if selection_known and not getattr(
                mon, "_selected_in_teampreview", False
            ):
                continue
            add(mon, "our")
        for mon in getattr(ctx.battle, "opponent_team", {}).values():
            if getattr(mon, "fainted", False):
                continue
            add(mon, "foe")

        return "\n".join(lines) if lines else None