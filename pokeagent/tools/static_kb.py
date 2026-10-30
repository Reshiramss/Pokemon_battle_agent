"""Static knowledge lookup tools: move info, type matchups, species info.

Thin, CPU-cheap wrappers over ``StaticKB`` — the agent calls these when it
needs details the pre-decision hook did not already inject.
"""

from __future__ import annotations

from typing import Any

from ..knowledge.static import normalize_name
from .base import Tool, ToolContext, ToolError

MAX_RESULT_CHARS = 1200


def _clip(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + " …[truncated]"


class MoveInfoTool(Tool):
    name = "move_info"
    description = (
        "Look up static data for a move: type, category (Physical/Special/"
        "Status), base power, accuracy, PP, priority, target, and secondary "
        "effects (e.g. burn chance)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "move": {"type": "string", "description": "Move name (e.g. 'Knock Off')."},
        },
        "required": ["move"],
    }

    def __init__(self, kb):
        self.kb = kb

    async def execute(self, ctx: ToolContext, move: str = "") -> str:
        if not move:
            raise ToolError("move name is required")
        data = ctx.kb.move_data(move) if ctx.kb else None
        if data is None and self.kb is not None:
            data = self.kb.move_data(move)
        if data is None:
            raise ToolError(f"unknown move '{move}'")
        return _clip(self._format(data))

    @staticmethod
    def _format(m: dict[str, Any]) -> str:
        parts = [
            f"{m.get('name')}",
            f"type {m.get('type')}",
            f"{m.get('category', '?')}",
            f"power {m.get('basePower')}",
            f"accuracy {m.get('accuracy')}",
            f"pp {m.get('pp')}",
            f"priority {m.get('priority')}",
            f"target {m.get('target')}",
        ]
        secondary = m.get("secondary")
        if isinstance(secondary, dict):
            bits = [
                f"{k}={v}"
                for k, v in secondary.items()
                if v not in (None, 0, False)
            ]
            if bits:
                parts.append("secondary: " + ", ".join(bits))
        return " | ".join(str(p) for p in parts)


class TypeInfoTool(Tool):
    name = "type_matchup"
    description = (
        "Compute the type-effectiveness multiplier of one attacking type "
        "against one or two defending types (e.g. 'Water' vs ['Ground', "
        "'Rock'] -> 4.0). Result labeled 0 (immune), not very effective, "
        "effective, super effective, or double super effective."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "attacking_type": {"type": "string", "description": "e.g. 'Water'."},
            "defending_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The defender's type(s), e.g. ['Ground', 'Rock'].",
            },
        },
        "required": ["attacking_type", "defending_types"],
    }

    def __init__(self, kb):
        self.kb = kb

    async def execute(
        self,
        ctx: ToolContext,
        attacking_type: str = "",
        defending_types: list[str] | None = None,
    ) -> str:
        if not attacking_type or not defending_types:
            raise ToolError("attacking_type and defending_types are required")
        kb = ctx.kb or self.kb
        if kb is None:
            raise ToolError("no knowledge base available")
        multiplier = kb.type_effectiveness(attacking_type, list(defending_types))
        label = {
            0.0: "no effect (immune)",
            0.25: "double not-very-effective",
            0.5: "not very effective",
            1.0: "normally effective",
            2.0: "super effective",
            4.0: "double super effective",
        }.get(multiplier, "custom multiplier")
        return (
            f"{attacking_type} vs {'/'.join(defending_types)}: "
            f"x{multiplier:g} ({label})"
        )


class SpeciesInfoTool(Tool):
    name = "species_info"
    description = (
        "Look up static data for a species: types, base stats, abilities "
        "(H = hidden), and evolution line basics. Useful for judging an "
        "opponent you have not battled yet."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "species": {"type": "string", "description": "e.g. 'Garchomp'."},
        },
        "required": ["species"],
    }

    def __init__(self, kb):
        self.kb = kb

    async def execute(self, ctx: ToolContext, species: str = "") -> str:
        if not species:
            raise ToolError("species name is required")
        kb = ctx.kb or self.kb
        if kb is None:
            raise ToolError("no knowledge base available")
        data = kb.species_data(species)
        if data is None:
            raise ToolError(f"unknown species '{species}'")
        return _clip(self._format(data))

    @staticmethod
    def _format(s: dict[str, Any]) -> str:
        name = s.get("name") or "?"
        types = "/".join(s.get("types") or []) or "?"
        stats = s.get("baseStats") or {}
        stat_line = " ".join(
            f"{k.upper()} {stats.get(k)}" for k in ("hp", "atk", "def", "spa", "spd", "spe")
        )
        abilities = s.get("abilities") or {}
        ability_line = ", ".join(
            f"{label}: {name_}" for label, name_ in abilities.items()
        )
        line = [f"{name} ({types}) — {stat_line}"]
        if ability_line:
            line.append(f"Abilities: {ability_line}")
        if s.get("prevo"):
            line.append(f"Evolves from {s.get('prevo')} (at level {s.get('evoLevel')})")
        return "; ".join(line)


def static_kb_tools(kb) -> list[Tool]:
    """The default static-knowledge tool set for a given StaticKB."""
    return [MoveInfoTool(kb), TypeInfoTool(kb), SpeciesInfoTool(kb)]
