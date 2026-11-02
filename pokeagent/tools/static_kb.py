"""Static knowledge lookup tools: move info, type matchups, species info.

Thin, CPU-cheap wrappers over the static knowledge sources. The primary
source is the Showdown data bundle (see ``knowledge/showdown_data.py``),
which is fresher and richer than poke-env's GenData — move descriptions,
competitive tiers, evolution lines, flavor text. When that bundle is absent
the tools fall back to the GenData-based ``StaticKB`` path unchanged.
"""

from __future__ import annotations

from typing import Any

from ..knowledge.showdown_data import get_showdown_data
from .base import Tool, ToolContext, ToolError

MAX_RESULT_CHARS = 1200

# Showdown status ids -> verb, for readable secondary-effect lines.
_STATUS_NAMES = {
    "brn": "burn",
    "par": "paralyze",
    "slp": "make sleep",
    "frz": "freeze",
    "psn": "poison",
    "tox": "badly poison",
}


def _clip(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + " …[truncated]"


class MoveInfoTool(Tool):
    name = "move_info"
    description = (
        "Look up static data for a move: type, category (Physical/Special/"
        "Status), base power, accuracy, PP, priority, target, contact, and "
        "secondary effects with a plain-language description."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "move": {"type": "string", "description": "Move name (e.g. 'Knock Off')."},
        },
        "required": ["move"],
    }

    def __init__(self, kb=None):
        self.kb = kb

    async def execute(self, ctx: ToolContext, move: str = "") -> str:
        if not move:
            raise ToolError("move name is required")
        sd = get_showdown_data()
        if sd is not None:
            data = sd.move(move)
            if data is None:
                raise ToolError(f"unknown move '{move}'")
            return _clip(self._format(data, sd.move_text(move)))

        # fallback: GenData-based path (no Showdown bundle on disk)
        data = ctx.kb.move_data(move) if ctx.kb else None
        if data is None and self.kb is not None:
            data = self.kb.move_data(move)
        if data is None:
            raise ToolError(f"unknown move '{move}'")
        return _clip(self._format(data))

    @staticmethod
    def _format(m: dict[str, Any], text: dict[str, Any] | None = None) -> str:
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
        flags = m.get("flags") or {}
        if flags.get("contact"):
            parts.append("makes contact")
        secondary = m.get("secondary")
        if isinstance(secondary, dict):
            bits: list[str] = []
            chance = secondary.get("chance")
            pct = f"{chance}% chance to " if chance else ""
            if secondary.get("status"):
                bits.append(pct + _STATUS_NAMES.get(secondary["status"], secondary["status"]))
            else:
                bits.extend(
                    f"{k}={v}"
                    for k, v in secondary.items()
                    if k != "chance" and v not in (None, 0, False)
                )
            if bits:
                parts.append("secondary: " + ", ".join(bits))
        if text:
            desc = text.get("desc") or text.get("shortDesc")
            if desc:
                parts.append(f"Effect: {desc}")
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

    def __init__(self, kb=None):
        self.kb = kb

    async def execute(
        self,
        ctx: ToolContext,
        attacking_type: str = "",
        defending_types: list[str] | None = None,
    ) -> str:
        if not attacking_type or not defending_types:
            raise ToolError("attacking_type and defending_types are required")
        sd = get_showdown_data()
        if sd is not None:
            try:
                multiplier = sd.type_multiplier(attacking_type, list(defending_types))
            except ValueError as e:
                raise ToolError(
                    f"{e}. Valid types: {', '.join(sd.types)}"
                ) from e
        else:
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
        "Look up static data for a species: types, base stats (+ total), "
        "abilities (hidden marked), competitive tier, size/weight, and the "
        "evolution line. Useful for judging an opponent you have not "
        "battled yet."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "species": {"type": "string", "description": "e.g. 'Garchomp'."},
        },
        "required": ["species"],
    }

    def __init__(self, kb=None):
        self.kb = kb

    async def execute(self, ctx: ToolContext, species: str = "") -> str:
        if not species:
            raise ToolError("species name is required")
        sd = get_showdown_data()
        if sd is not None:
            data = sd.species(species)
            if data is None:
                raise ToolError(f"unknown species '{species}'")
            return _clip(
                self._format(
                    data, tier=sd.tier(species), text=sd.species_text(species)
                )
            )
        kb = ctx.kb or self.kb
        if kb is None:
            raise ToolError("no knowledge base available")
        data = kb.species_data(species)
        if data is None:
            raise ToolError(f"unknown species '{species}'")
        return _clip(self._format(data))

    @staticmethod
    def _format(
        s: dict[str, Any],
        tier: str | None = None,
        text: dict[str, Any] | None = None,
    ) -> str:
        name = s.get("name") or "?"
        types = "/".join(s.get("types") or []) or "?"
        stats = s.get("baseStats") or {}
        stat_line = " ".join(
            f"{k.upper()} {stats.get(k)}" for k in ("hp", "atk", "def", "spa", "spd", "spe")
        )
        bst = sum(v for v in stats.values() if isinstance(v, int))
        line = [f"{name} ({types}) — {stat_line} (BST {bst})"]
        abilities = s.get("abilities") or {}
        ability_bits = [
            f"{ability} (hidden)" if slot == "H" else str(ability)
            for slot, ability in abilities.items()
        ]
        if ability_bits:
            line.append(f"Abilities: {', '.join(ability_bits)}")
        if tier:
            line.append(f"Tier: {tier}")
        height, weight = s.get("heightm"), s.get("weightkg")
        if height and weight:
            line.append(f"Size: {height}m, {weight}kg")
        if s.get("prevo"):
            level = s.get("evoLevel")
            line.append(
                f"Evolves from {s.get('prevo')}"
                + (f" (at level {level})" if level else "")
            )
        if s.get("evos"):
            line.append("Evolves into " + ", ".join(s["evos"]))
        elif s.get("prevo"):
            line.append("Fully evolved")
        if s.get("otherFormes"):
            line.append(f"Other formes: {', '.join(s['otherFormes'])}")
        if text:
            desc = text.get("desc") or text.get("shortDesc")
            if desc:
                line.append(desc)
        return "; ".join(line)


def static_kb_tools(kb=None) -> list[Tool]:
    """The default static-knowledge tool set for a given StaticKB."""
    return [MoveInfoTool(kb), TypeInfoTool(kb), SpeciesInfoTool(kb)]
