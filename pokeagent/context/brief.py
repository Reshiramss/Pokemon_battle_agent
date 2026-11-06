"""Rolling strategy brief (L2 reasoning memory).

Notes compress *one turn*; the brief compresses *the battle so far* into a
fixed schema that is rewritten every few turns. A rule-based concatenation
cannot express "the perish-song line failed at T4, so clear the fast slot
first" — only a model can rewrite the plan with that causal chain. The
schema and the rewrite-not-append rule keep the output small and stable:

    stance              one line: how we win right now
    win_conditions      <= 2 short strings
    threats             <= 3 short strings (name the Pokemon)
    commitments         <= 3 short strings (what to preserve/deny)
    beliefs             <= 4 revisable reads (speed/EV/item/ability)
    open_questions      <= 2 short strings

Everything here is bounded: the prompts cap input characters and the
sanitizer caps every field, so a runaway model output cannot bloat the
prompt. Failures return ``None`` and the caller keeps the previous brief.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ..providers.base import UserMessage
from ..state.model import PlanBoard, ReasoningLedger, ReasoningNote, StrategyBrief
from .reasoning import _parse_json

if TYPE_CHECKING:
    from ..providers.base import LLMProvider
    from .records import TurnRecord

BRIEF_SYSTEM_PROMPT = (
    "You maintain a rolling strategy brief for one Pokemon battle. Rewrite it "
    "from scratch from the evidence each time; do not append or keep resolved "
    "items. Return ONLY a JSON object with these keys:\n"
    '  "stance": one sentence — how we win right now,\n'
    '  "win_conditions": 0-2 short strings,\n'
    '  "threats": 0-3 short strings, naming the Pokemon that must be answered,\n'
    '  "commitments": 0-3 short strings — what to preserve, deny, or set up,\n'
    '  "beliefs": 0-4 short strings — revisable reads about speed/EVs/item/ability,\n'
    '  "open_questions": 0-2 short strings.\n'
    "Only keep what changes the next few decisions. Cite turn numbers like T4 "
    "as evidence. Never invent observations; when unsure, omit. Keep every "
    "string terse (no preamble)."
)

_STANCE_CHARS = 160
_ITEM_CHARS = 120
_LIMITS = {
    "win_conditions": 2,
    "threats": 3,
    "commitments": 3,
    "beliefs": 4,
    "open_questions": 2,
}


def _clip(text, limit: int) -> str | None:
    if not isinstance(text, str):
        return None
    text = " ".join(text.split()).strip()
    if not text:
        return None
    return text[:limit]


def _strings(raw, limit: int, chars: int = _ITEM_CHARS) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        text = _clip(item, chars)
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def sanitize_brief(
    payload, *, covered_seq: int, turn: int
) -> StrategyBrief | None:
    """Coerce a parsed JSON payload into a bounded ``StrategyBrief``.

    Returns ``None`` when the payload has none of the expected keys, so a
    malformed answer falls back to the previous brief instead of blanking it.
    """
    if not isinstance(payload, dict):
        return None
    if not any(key in payload for key in (*_LIMITS, "stance")):
        return None
    brief = StrategyBrief(
        covered_seq=int(covered_seq or 0),
        turn=int(turn or 0),
        stance=_clip(payload.get("stance"), _STANCE_CHARS),
        source="llm",
    )
    for key, limit in _LIMITS.items():
        setattr(brief, key, _strings(payload.get(key), limit))
    return None if brief.empty() else brief


def deterministic_brief(
    ledger: ReasoningLedger,
    *,
    plan: PlanBoard | None = None,
    records: "list[TurnRecord] | None" = None,
    covered_seq: int = 0,
    turn: int = 0,
) -> StrategyBrief:
    """No-model fallback: reuse what the agent already committed.

    Never invents analysis — it only lifts the last reason, the plan board,
    and the newest directives/claims into the brief schema.
    """
    stance = None
    for record in reversed(records or []):
        if not record.fell_back and record.reason:
            stance = _clip(record.reason, _STANCE_CHARS)
            break
    if stance is None:
        note = ledger.latest_note()
        if note is not None and note.intent:
            stance = _clip(note.intent, _STANCE_CHARS)
        elif ledger.brief is not None and ledger.brief.stance:
            stance = ledger.brief.stance

    commitments = [d.text for d in ledger.active_directives(limit=3)]
    if plan is not None:
        win_conditions = list(plan.win_conditions[:2])
        threats = list(plan.primary_threats[:3])
        if plan.commitments:
            commitments = list(plan.commitments[:3])
        if plan.line and stance is None:
            stance = _clip(plan.line, _STANCE_CHARS)
    else:
        win_conditions, threats = [], []

    beliefs = [
        _clip(c.text, _ITEM_CHARS)
        for c in reversed(ledger.claims)
        if c.status == "active" and c.kind != "plan"
    ]
    bridge = StrategyBrief(
        covered_seq=int(covered_seq or 0),
        turn=int(turn or 0),
        stance=stance,
        win_conditions=[t for t in (_clip(v, _ITEM_CHARS) for v in win_conditions) if t][:2],
        threats=[t for t in (_clip(v, _ITEM_CHARS) for v in threats) if t][:3],
        commitments=[t for t in (_clip(v, _ITEM_CHARS) for v in commitments) if t][:3],
        beliefs=[t for t in beliefs if t][:4],
        source="deterministic",
    )
    return bridge


class LLMBriefCompressor:
    """Cheap-model rewrite of the rolling brief (same shape as note extraction)."""

    def __init__(
        self,
        provider: "LLMProvider",
        *,
        max_tokens: int = 400,
        input_chars: int = 4000,
    ):
        self.provider = provider
        self.max_tokens = max_tokens
        self.input_chars = input_chars

    async def compress(
        self,
        *,
        brief: StrategyBrief | None,
        notes: list[ReasoningNote],
        records: "list[TurnRecord]",
        plan: PlanBoard | None = None,
        turn: int = 0,
    ) -> StrategyBrief | None:
        prompt = self._input(brief=brief, notes=notes, records=records, plan=plan, turn=turn)
        try:
            response = await self.provider.complete(
                system=BRIEF_SYSTEM_PROMPT,
                messages=[UserMessage(prompt)],
                tools=[],
                max_tokens=self.max_tokens,
            )
        except Exception:
            return None
        if response.stop_reason == "error" or not response.text:
            return None
        payload = _parse_json(response.text)
        covered = notes[-1].seq if notes else (brief.covered_seq if brief else 0)
        return sanitize_brief(payload, covered_seq=covered, turn=turn)

    def _input(self, *, brief, notes, records, plan, turn) -> str:
        lines = [f"Turn: {turn}"]
        if plan is not None and (
            plan.win_conditions or plan.commitments or plan.line
        ):
            bits = []
            if plan.line:
                bits.append(f"line: {plan.line}")
            if plan.win_conditions:
                bits.append("win: " + "; ".join(plan.win_conditions))
            if plan.commitments:
                bits.append("commit: " + "; ".join(plan.commitments))
            if plan.primary_threats:
                bits.append("threats: " + "; ".join(plan.primary_threats))
            lines.append("COMMITTED PLAN (written by the agent, overrides the brief): " + " | ".join(bits))
        if brief is not None and not brief.empty():
            lines.append(
                "CURRENT BRIEF (JSON): " + json.dumps(brief.__dict__, ensure_ascii=False)
            )
        if notes:
            lines.append("NEW NOTES (newest last):")
            for note in notes:
                bits = [f"#{note.seq} T{note.turn}"]
                if note.intent:
                    bits.append(f"intent: {note.intent}")
                if note.claims:
                    bits.append("claims: " + "; ".join(note.claims))
                if note.correction:
                    bits.append(f"correction: {note.correction}")
                if note.directives:
                    bits.append("do: " + "; ".join(note.directives))
                if note.open_questions:
                    bits.append("unknown: " + "; ".join(note.open_questions))
                lines.append(" | ".join(bits))
        if records:
            lines.append("RECENT TURNS (newest last):")
            for record in records:
                bits = [f"T{record.turn}"]
                bits.append(
                    f"FALLBACK ({record.fallback_reason})"
                    if record.fell_back
                    else f"did {record.decision} — {record.reason}"
                )
                if record.events:
                    bits.append("saw: " + "; ".join(record.events[:4]))
                lines.append(" | ".join(bits))
        return "\n".join(lines)[: self.input_chars]