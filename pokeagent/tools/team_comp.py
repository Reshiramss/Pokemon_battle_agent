"""Team composition tool — recognize the opponent's archetype and its plan.

Backed by the curated ``data/team_comps.json`` (VGC Reg I archetypes).
With no arguments it matches the opponent's revealed species against each
archetype's core mons and reports the best fits with their game plan;
with ``archetype`` it returns that full entry. Knowing the plan ("rain
team keeps rain up at all costs", "Trick Room wants to redirect your
hits while it sets up") is the fastest way to pick the right lead and
play against the strategy, not just the six individuals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .base import Tool, ToolContext, ToolError

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "team_comps.json"
MAX_RESULT_CHARS = 3000


@dataclass(frozen=True)
class Archetype:
    id: str
    name: str
    core_mons: tuple[str, ...]
    plan: str
    weak_to: str


@lru_cache(maxsize=1)
def _archetypes() -> list[Archetype]:
    try:
        with DATA_PATH.open(encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return []
    return [
        Archetype(
            id=a["id"],
            name=a["name"],
            core_mons=tuple(a["core_mons"]),
            plan=a["plan"],
            weak_to=a["weak_to"],
        )
        for a in raw.get("archetypes", [])
    ]


def _clip(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + " …[truncated]"


class TeamCompTool(Tool):
    name = "team_comp"
    description = (
        "Identify the opponent's team archetype and how to play against it. "
        "No arguments: match their revealed Pokemon against known VGC "
        "archetypes (rain, Trick Room, hyper offense, ...) and get the game "
        "plan plus weaknesses of the best fit. archetype=<id>: read one "
        "archetype entry in full."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "archetype": {
                "type": "string",
                "description": "A specific archetype id (e.g. 'rain', 'trick_room').",
            },
        },
        "required": [],
    }

    async def execute(self, ctx: ToolContext, archetype: str = "") -> str:
        entries = _archetypes()
        if not entries:
            raise ToolError("team composition data is not available")
        if archetype:
            match = next((a for a in entries if a.id == archetype), None)
            if match is None:
                valid = ", ".join(f"'{a.id}'" for a in entries)
                raise ToolError(
                    f"unknown archetype '{archetype}'. Valid ids: {valid}"
                )
            return _clip(self._format_full(match))
        return _clip(self._format_matches(ctx, entries))

    # --- rendering ---------------------------------------------------------

    @staticmethod
    def _format_full(a: Archetype) -> str:
        return (
            f"{a.name} ({a.id})\n"
            f"Core mons: {', '.join(a.core_mons)}\n"
            f"Plan: {a.plan}\n"
            f"Weak to: {a.weak_to}"
        )

    @staticmethod
    def _format_matches(ctx: ToolContext, entries: "list[Archetype]") -> str:
        revealed = {
            mon.species for mon in ctx.battle.opponent_team.values()
        }
        scored = sorted(
            (
                (len(set(a.core_mons) & revealed), a)
                for a in entries
            ),
            key=lambda kv: -kv[0],
        )
        best_score, best = scored[0]
        if best_score == 0:
            return (
                "No known archetype matches the opponent's revealed "
                "Pokemon so far. Known archetypes: "
                + ", ".join(f"'{a.id}' ({a.name})" for a in entries)
            )
        matched = set(best.core_mons) & revealed
        lines = [
            f"Best fit: {best.name} ({best.id}) — matched {sorted(matched)}.",
            f"Plan: {best.plan}",
            f"Play against it: {best.weak_to}",
        ]
        # one follow-up candidate, so the model can hedge on a partial reveal
        for score, other in scored[1:3]:
            if 0 < score >= best_score - 1:
                lines.append(
                    f"Possible alternative: {other.name} ({other.id}) — "
                    f"shared {len(set(other.core_mons) & revealed)} core mons."
                )
        return "\n".join(lines)
