"""Ability lookup tool — static data plus a plain-language description.

Descriptions come from the Showdown text bundle (the zh-cn ability files
only carry official Chinese names, so the description is English).
"""

from __future__ import annotations

from typing import Any

from ..knowledge.showdown_data import get_showdown_data
from .base import Tool, ToolContext, ToolError

MAX_RESULT_CHARS = 1200


class AbilityInfoTool(Tool):
    name = "ability_info"
    description = (
        "Look up an ability and what it does in plain language: the effect "
        "(desc or shortDesc) and, when known, which of the opponent's "
        "possible abilities it is. Use it to understand revealed abilities "
        "before committing to a play."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "ability": {
                "type": "string",
                "description": "Ability name (e.g. 'Intimidate', '威吓').",
            },
        },
        "required": ["ability"],
    }

    async def execute(self, ctx: ToolContext, ability: str = "") -> str:
        if not ability:
            raise ToolError("ability name is required")
        sd = get_showdown_data()
        if sd is None:
            raise ToolError("ability data is not available (no Showdown bundle)")
        data = sd.ability(ability)
        if data is None:
            raise ToolError(f"unknown ability '{ability}'")
        text = sd.ability_text(ability) or {}
        return self._format(data, text)

    @staticmethod
    def _format(a: dict[str, Any], text: dict[str, Any]) -> str:
        name = a.get("name") or "?"
        parts = [str(name)]
        # shortDesc first: competitive one-liners keep prompts small; the
        # long desc is only used when no short form exists
        desc = text.get("shortDesc") or text.get("desc")
        if desc:
            parts.append(f"Effect: {desc}")
        if a.get("isBreakable") is False:
            parts.append("cannot be suppressed (e.g. by Mold Breaker)")
        result = " | ".join(parts)
        return result if len(result) <= MAX_RESULT_CHARS else (
            result[:MAX_RESULT_CHARS] + " …[truncated]"
        )
