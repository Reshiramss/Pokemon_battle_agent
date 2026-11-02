"""Mechanics lookup tool — curated VGC/doubles rules knowledge.

Backed by ``data/mechanics.json`` (see ``knowledge/mechanics.py``).
Without a topic it lists what it knows; with a fuzzy topic match it
returns the full entry. Use it to check a rule before committing to a
play (e.g. "does Protect decay?", "how do spread moves work?").
"""

from __future__ import annotations

from ..knowledge.mechanics import get_mechanics_kb
from .base import Tool, ToolContext, ToolError

MAX_RESULT_CHARS = 2500


class MechanicsInfoTool(Tool):
    name = "mechanics_info"
    description = (
        "Look up Pokemon battle mechanics — the rules of the game, not a "
        "specific move or species: turn order and speed, spread-move damage, "
        "Protect decay, redirection, weather, terrain, status conditions, "
        "stat stages, terastallization, doubles-only tools. Without a topic "
        "it lists the available topics."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": (
                    "Topic to look up, fuzzy-matched (e.g. 'speed', "
                    "'spread moves', 'protect', 'tera')."
                ),
            },
        },
        "required": [],
    }

    async def execute(self, ctx: ToolContext, topic: str = "") -> str:
        kb = get_mechanics_kb()
        if not kb.topics:
            raise ToolError("mechanics data is not available")
        if not topic:
            return "\n".join(
                f"- {t.title} (id '{t.id}'): {t.summary}"
                for t in kb.topics.values()
            )
        matched = kb.match(topic)
        if matched is None:
            valid = ", ".join(
                f"'{t.id}' ({t.title})" for t in kb.topics.values()
            )
            raise ToolError(f"unknown topic '{topic}'. Valid topics: {valid}")
        result = matched.render()
        if len(result) <= MAX_RESULT_CHARS:
            return result
        return result[:MAX_RESULT_CHARS] + " …[truncated]"
