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
        "Query this battle's structured history — your own memory. With no "
        "arguments: a one-line log of every decision so far (turn, action, "
        "reason). With turn=N (or seq=N, the record number from that log): the "
        "full record — the state you saw, what you decided and why, and the "
        "battle events of that turn. Set include_cot=true to also retrieve your "
        "recorded reasoning (chain of thought) for that turn, or "
        "include_raw_protocol=true for the raw Showdown lines."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "turn": {
                "type": "integer",
                "description": "Specific turn to inspect; omit for a summary log.",
            },
            "seq": {
                "type": "integer",
                "description": (
                    "Record sequence number (takes precedence over turn)."
                ),
            },
            "include_cot": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Include your recorded reasoning (chain of thought) for "
                    "that turn."
                ),
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
        seq: int | None = None,
        include_cot: bool = False,
        include_raw_protocol: bool = False,
    ) -> str:
        cm = ctx.context
        if cm is None or not cm.records:
            raise ToolError("no battle history recorded yet")
        if seq is None and turn is None:
            return _clip(self._decision_log(cm.records))
        if seq is not None:
            record = next((r for r in cm.records if r.seq == seq), None)
            key = f"seq {seq}"
        else:
            record = next((r for r in cm.records if r.turn == turn), None)
            key = f"turn {turn}"
        if record is None:
            known = [f"#{r.seq}/turn {r.turn}" for r in cm.records]
            raise ToolError(f"no record for {key}; recorded: {known}")
        return _clip(self._full_record(record, include_raw_protocol, include_cot))

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
    def _full_record(
        r: "TurnRecord", include_raw_protocol: bool, include_cot: bool
    ) -> str:
        lines = [f"=== Turn {r.turn} (record #{r.seq}) ==="]
        if r.request:
            lines.append(r.request.rstrip())
        lines.append(
            f"You decided: {r.decision} — {r.reason}"
            if not r.fell_back
            else f"FALLBACK: {r.decision} ({r.fallback_reason})"
        )
        if include_cot and r.cot:
            cot = r.cot[:2000]
            lines.append("Your reasoning (CoT): " + cot + ("…" if len(r.cot) > 2000 else ""))
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
