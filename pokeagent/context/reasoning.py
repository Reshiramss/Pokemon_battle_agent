"""Structured reasoning notes: compress one turn's raw CoT for later turns.

Raw chain-of-thought is persisted verbatim in ``llm.jsonl`` but is too large
to re-inject every turn. A :class:`ReasoningNote` keeps what later turns
actually need: the intent, falsifiable claims about the opponent, a
correction of any earlier belief that proved wrong, directives to carry
forward, and open questions.

Two extractors implement the :class:`NoteExtractor` protocol:

- :class:`LLMNoteExtractor` — one cheap-model call per turn, run in the
  background off the critical path.
- :class:`DeterministicNoteExtractor` — no model call; records only what is
  already structured (decision intent, fallback flag). Used when sub-agents
  (and thus the cheap-model provider) are disabled, or as the fallback when
  extraction fails.

Neither extractor invents information: a note that cannot be grounded in
the record stays empty.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Protocol

from ..providers.base import UserMessage
from ..state.model import ReasoningNote
from .records import TurnRecord

if TYPE_CHECKING:
    from ..providers.base import LLMProvider
    from ..state.model import BattleState


NOTE_SYSTEM_PROMPT = """You compress one Pokemon battle turn into a structured reasoning note. \
You will receive the decision, the agent's own reasoning, and the battle events that followed. \
Answer with ONLY a JSON object, no prose, with exactly these keys:
  "intent": one sentence (<= 25 words): what this decision tried to achieve.
  "claims": list of 0-3 falsifiable beliefs about the opponent or game state that drove it.
  "correction": one sentence naming an earlier belief that the events proved wrong, or null.
  "directives": list of 0-2 imperatives for future turns.
  "open_questions": list of 0-2 unknowns that still matter.
Be terse and do not invent information that is absent from the input."""


class NoteExtractor(Protocol):
    """A reasoning-note extractor.

    ``extract_now`` is the synchronous fast path (returns None when the
    extractor needs the event loop); ``extract`` is the async form.
    """

    def extract_now(
        self, record: TurnRecord, state: "BattleState | None" = None
    ) -> ReasoningNote | None: ...

    async def extract(
        self, record: TurnRecord, state: "BattleState | None" = None
    ) -> ReasoningNote | None: ...


# --- deterministic fallback -------------------------------------------------


def deterministic_note(record: TurnRecord) -> ReasoningNote:
    """No-model note: keep the intent and the outcome flag, nothing else."""
    intent = (record.reason or "").strip()[:300] or None
    return ReasoningNote(
        seq=record.seq,
        turn=record.turn,
        intent=intent,
        fallback=bool(record.fell_back),
    )


class DeterministicNoteExtractor:
    """Synchronous extractor used without a cheap-model provider."""

    def extract_now(
        self, record: TurnRecord, state: "BattleState | None" = None
    ) -> ReasoningNote | None:
        return deterministic_note(record)

    async def extract(
        self, record: TurnRecord, state: "BattleState | None" = None
    ) -> ReasoningNote | None:
        return deterministic_note(record)


# --- LLM extraction ---------------------------------------------------------


class LLMNoteExtractor:
    """One cheap-model call per turn; returns None on any failure."""

    def __init__(
        self,
        provider: "LLMProvider",
        *,
        max_tokens: int = 350,
        input_chars: int = 6000,
    ):
        self.provider = provider
        self.max_tokens = max_tokens
        self.input_chars = input_chars

    def extract_now(
        self, record: TurnRecord, state: "BattleState | None" = None
    ) -> ReasoningNote | None:
        return None  # needs the event loop

    async def extract(
        self, record: TurnRecord, state: "BattleState | None" = None
    ) -> ReasoningNote | None:
        try:
            response = await self.provider.complete(
                system=NOTE_SYSTEM_PROMPT,
                messages=[UserMessage(self._input(record, state))],
                tools=[],
                max_tokens=self.max_tokens,
            )
        except Exception:
            return None
        if response.error or not response.text:
            return None
        data = _parse_json(response.text)
        if data is None:
            return None
        return ReasoningNote(
            seq=record.seq,
            turn=record.turn,
            intent=_one_line(data.get("intent")),
            claims=_string_list(data.get("claims"), limit=3),
            correction=_one_line(data.get("correction")),
            directives=_string_list(data.get("directives"), limit=2),
            open_questions=_string_list(data.get("open_questions"), limit=2),
            source="llm",
            fallback=bool(record.fell_back),
        )

    def _input(self, record: TurnRecord, state: "BattleState | None") -> str:
        lines = [
            f"Turn {record.turn}",
            f"Decision: {record.decision or 'none'}",
        ]
        if record.fell_back:
            lines.append(f"NOTE: the decision loop failed ({record.fallback_reason})")
        if record.reason:
            lines.append(f"Reason given: {record.reason}")
        if record.cot:
            lines.append("Reasoning:\n" + record.cot[: self.input_chars])
        if record.events:
            events = "; ".join(record.events[:12])
            lines.append(f"Events after the decision: {events}")
        return "\n".join(lines)


def _one_line(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip()
    if not text or text.lower() in {"null", "none"}:
        return None
    return text[:300]


def _string_list(value: Any, *, limit: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _one_line(item)
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _parse_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from a model response."""
    candidates: list[str] = []
    match = _FENCE.search(text)
    if match:
        candidates.append(match.group(1))
    candidates.append(text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate.strip())
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return None