"""Context compression strategies.

``TruncatingCompressor`` is the always-available, no-LLM fallback: it keeps
decisions and reasons but drops verbose detail. ``LLMCompressor`` summarizes
old turns with the cheap background model — it runs off the critical path
and the swap is race-free: a summary covers records up to a monotonic
sequence number, records only ever append, and an in-flight summary can
never clobber newer state (it is simply dropped in on completion).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..providers.base import UserMessage
from .records import TurnRecord

if TYPE_CHECKING:
    from ..providers.base import LLMProvider


def _compact_record(record: TurnRecord) -> str:
    bits = [f"T{record.turn}"]
    if record.decision:
        bits.append(f"did {record.decision}")
    if record.reason:
        bits.append(f"({record.reason})")
    if record.fell_back:
        bits.append(f"[fallback: {record.fallback_reason}]")
    if record.events:
        bits.append("saw: " + "; ".join(record.events[:3]))
    return " ".join(bits)


class ContextCompressor(Protocol):
    async def compress(self, records: list[TurnRecord], budget_tokens: int) -> str:
        """Render ``records`` into at most ~budget_tokens (chars/4 estimate)."""
        ...


class TruncatingCompressor:
    """Cheap, deterministic: keep every record as one line, cap the total."""

    async def compress(self, records: list[TurnRecord], budget_tokens: int) -> str:
        lines = [_compact_record(r) for r in records]
        text = "\n".join(lines)
        # leave room for the omission note the overflow path prepends
        max_chars = max(budget_tokens * 4 - 64, 0)
        if len(text) > max_chars:
            # drop oldest lines until it fits, note the cut
            dropped = 0
            while lines and len("\n".join(lines)) > max_chars:
                lines.pop(0)
                dropped += 1
            text = (f"[{dropped} oldest decisions omitted]\n" if dropped else "") + "\n".join(lines)
        return text


class LLMCompressor:
    """Background summarization of old turns with the cheap model."""

    def __init__(self, provider: "LLMProvider", max_input_chars: int = 12_000):
        self.provider = provider
        self.max_input_chars = max_input_chars

    async def compress(self, records: list[TurnRecord], budget_tokens: int) -> str:
        if not records:
            return ""
        lines = [_compact_record(r) for r in records]
        text = "\n".join(lines)[: self.max_input_chars]
        prompt = (
            "Summarize this Pokemon battle's decision history so far. Keep: "
            "key momentum shifts, which Pokemon are healthy/fainted on both "
            "sides, revealed opponent moves of note, and strategic takeaways "
            f"(traps to avoid, win condition). Max ~{budget_tokens} tokens, "
            "plain text, no preamble.\n\n" + text
        )
        response = await self.provider.complete(
            system="You are a concise battle-log summarizer.",
            messages=[UserMessage(prompt)],
            tools=[],
            max_tokens=budget_tokens,
        )
        if response.stop_reason == "error" or not response.text:
            raise RuntimeError(
                f"compression LLM call failed: {response.error or 'empty output'}"
            )
        return response.text.strip()
