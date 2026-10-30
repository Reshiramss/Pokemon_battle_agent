"""Battle history query tool.

The assembled prompt already carries a bounded history window; this tool
lets the agent dig into the full per-turn records (decisions, reasons,
battle events) when it needs detail that dropped out of the window. Raw
protocol lines are only returned when explicitly requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Tool, ToolContext, ToolError

if TYPE_CHECKING:
    from ..context.records import TurnRecord

MAX_RESULT_CHARS = 3000


def _clip(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + " …[truncated]"


class BattleHistoryTool(Tool):
    name = "battle_history"
    description = (
        "Query this battle's structured history. With no arguments: a one-line "
        "log of every decision so far (turn, action, reason). With turn=N: the "
        "full record of that turn — the state you saw, what you decided and "
        "why, and the battle events of that turn. Set include_raw_protocol=true "
        "to also get the raw Showdown protocol lines for that turn."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "turn": {
                "type": "integer",
                "description": "Specific turn to inspect; omit for a summary log.",
            },
            "include_raw_protocol": {
                "type": "boolean",
                "default": False,
                "description": "Include raw protocol lines (verbose).",
            },
        },
        "required": [],
    }

    async def execute(
        self,
        ctx: ToolContext,
        turn: int | None = None,
        include_raw_protocol: bool = False,
    ) -> str:
        cm = ctx.context
        if cm is None or not cm.records:
            raise ToolError("no battle history recorded yet")
        if turn is None:
            return _clip(self._decision_log(cm.records))
        record = next((r for r in cm.records if r.turn == turn), None)
        if record is None:
            known = [r.turn for r in cm.records]
            raise ToolError(
                f"no record for turn {turn}; recorded turns: {known}"
            )
        return _clip(self._full_record(record, include_raw_protocol))

    @staticmethod
    def _decision_log(records: "list[TurnRecord]") -> str:
        lines = []
        for r in records:
            if r.fell_back:
                lines.append(
                    f"turn {r.turn}: FALLBACK ({r.fallback_reason})"
                )
            else:
                lines.append(f"turn {r.turn}: {r.decision} — {r.reason}")
        return "\n".join(lines)

    @staticmethod
    def _full_record(r: "TurnRecord", include_raw_protocol: bool) -> str:
        lines = [f"=== Turn {r.turn} (record #{r.seq}) ==="]
        if r.request:
            lines.append(r.request.rstrip())
        lines.append(
            f"You decided: {r.decision} — {r.reason}"
            if not r.fell_back
            else f"FALLBACK: {r.decision} ({r.fallback_reason})"
        )
        if r.tool_calls:
            calls = "; ".join(
                f"{c.name}({', '.join(f'{k}={v}' for k, v in c.arguments.items())})"
                if hasattr(c, "arguments")
                else str(c)
                for c in r.tool_calls
            )
            lines.append(f"Tool calls: {calls}")
        if r.events:
            lines.append("Battle events: " + "; ".join(r.events))
        if include_raw_protocol:
            lines.append("Raw protocol: " + " | ".join(r.events))
        return "\n".join(lines)
