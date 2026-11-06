"""Battle history query tool.

The assembled prompt already carries a bounded history window; this tool
lets the agent dig into the full per-turn records (decisions, reasons,
battle events) when it needs detail that dropped out of the window. Raw
protocol lines are only returned when explicitly requested.

With ``pattern`` set the tool switches to grep mode over this battle's
memory — records plus the reasoning ledger (notes, claims) — mirroring how
one greps a codebase for a reminder. Scope selects which fields are
searched; every hit is reported as ``#seq Tn [scope] snippet``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..memory.grep import Query, excerpt
from .base import Tool, ToolContext, ToolError

if TYPE_CHECKING:
    from ..context.records import TurnRecord

MAX_RESULT_CHARS = 3000
SCOPES = ("decision", "reason", "cot", "events", "notes", "claims")


def _clip(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + " …[truncated]"


def _note_text(note) -> str:
    """Flatten a ReasoningNote into one searchable string."""
    parts: list[str] = []
    if note.intent:
        parts.append(f"intent: {note.intent}")
    if note.claims:
        parts.append("claims: " + "; ".join(note.claims))
    if note.correction:
        parts.append(f"correction: {note.correction}")
    if note.directives:
        parts.append("directives: " + "; ".join(note.directives))
    if note.open_questions:
        parts.append("open: " + "; ".join(note.open_questions))
    return " | ".join(parts)


class BattleHistoryTool(Tool):
    name = "battle_history"
    description = (
        "Query this battle's structured history — your own memory. With no "
        "arguments: a one-line log of every decision so far (turn, action, "
        "reason). With turn=N (or seq=N, the record number from that log): the "
        "full record — the state you saw, what you decided and why, and the "
        "battle events of that turn. Set include_cot=true to also retrieve your "
        "recorded reasoning (chain of thought) for that turn, or "
        "include_raw_protocol=true for the raw Showdown lines. Alternatively "
        "set pattern to grep this battle's memory (case-insensitive; terms are "
        "ANDed; regex=true for a regular expression) across decisions, "
        "reasons, CoT, events, reasoning notes and claims — use it to find "
        "'when did I note X'."
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
            "pattern": {
                "type": "string",
                "description": "Grep this battle's memory instead of fetching a record.",
            },
            "scope": {
                "type": "string",
                "enum": ["all", *SCOPES],
                "default": "all",
                "description": "Fields searched in grep mode.",
            },
            "regex": {
                "type": "boolean",
                "default": False,
                "description": "Treat pattern as one regular expression.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "description": "Maximum grep hits (default 8).",
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
        pattern: str | None = None,
        scope: str = "all",
        regex: bool = False,
        limit: int = 8,
    ) -> str:
        cm = ctx.context
        if cm is None or not cm.records:
            raise ToolError("no battle history recorded yet")
        if pattern is not None:
            return self._grep(
                cm, pattern, scope=scope, regex=regex, limit=limit, seq=seq, turn=turn
            )
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
    def _grep(cm, pattern: str, *, scope: str, regex: bool, limit: int,
              seq: int | None, turn: int | None) -> str:
        if scope not in ("all", *SCOPES):
            raise ToolError(f"unknown scope {scope!r}; use: all, {', '.join(SCOPES)}")
        try:
            query = Query(pattern, regex=regex)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        limit = max(1, min(int(limit), 50))
        wanted = set(SCOPES) if scope == "all" else {scope}

        records = list(cm.records)
        if seq is not None:
            records = [r for r in records if r.seq == seq]
        elif turn is not None:
            records = [r for r in records if r.turn == turn]

        entries: list[tuple[int | None, int, str, str]] = []  # (seq, turn, scope, text)

        for r in records:
            if "decision" in wanted and r.decision:
                entries.append((r.seq, r.turn, "decision", r.decision))
            if "reason" in wanted and r.reason:
                entries.append((r.seq, r.turn, "reason", r.reason))
            if "cot" in wanted and r.cot:
                entries.append((r.seq, r.turn, "cot", r.cot))
            if "events" in wanted and r.events:
                entries.append((r.seq, r.turn, "events", "; ".join(r.events)))

        ledger = getattr(cm, "ledger", None)
        notes = list(getattr(ledger, "notes", []) or [])
        claims = list(getattr(ledger, "claims", []) or [])
        if seq is not None:
            notes = [n for n in notes if n.seq == seq]
        elif turn is not None:
            notes = [n for n in notes if n.turn == turn]
            claims = [c for c in claims if c.turn == turn]

        if "notes" in wanted:
            for note in notes:
                entries.append(
                    (note.seq, note.turn, "notes", _note_text(note))
                )
        if "claims" in wanted:
            for claim in claims:
                entries.append(
                    (
                        None,
                        claim.turn,
                        "claims",
                        f"{claim.subject} {claim.kind}: {claim.text} "
                        f"({claim.status})",
                    )
                )

        hits = [
            (seq_no, turn_no, scope_name, excerpt(text, query))
            for seq_no, turn_no, scope_name, text in entries
            if query.matches(text)
        ]
        hits.sort(key=lambda h: (h[1] or 0, h[0] or 0), reverse=True)
        if not hits:
            return f"no match for {pattern!r} in {scope} of this battle's memory."

        lines = [
            f"#- T{turn_no} [{scope_name}] {snippet}" if seq_no is None
            else f"#{seq_no} T{turn_no} [{scope_name}] {snippet}"
            for seq_no, turn_no, scope_name, snippet in hits[:limit]
        ]
        footer = (
            f"\n({len(hits)} match{'es' if len(hits) != 1 else ''}; showing "
            f"{len(lines)} — use turn=N / seq=N for the full record)"
        )
        return _clip("\n".join(lines) + footer)

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
