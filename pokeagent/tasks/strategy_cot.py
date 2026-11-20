"""Battle-level COT maintainer: rewrite "how this game is won" after a turn.

Runs as a background sub-agent on the **main** model: the battle plan is the
one piece of reasoning worth the good model. The call never blocks a turn —
a late or failed rewrite keeps the previous text.

Input: the previous COT, the board the decision was made from, the last
decision and what came of it, the decision just committed (its outcome is
unknown), the model's hidden thinking for that turn, and the committed
``PlanBoard``. Output: plain text, one short paragraph, rewritten rather than
appended.

Fallback: on this gateway reasoning models often return thinking only (the
visible answer is empty or cut at max_tokens). In that case the last finished
paragraph of the thinking is used and marked ``thinking`` so the log shows
where the plan text came from.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..context.cot import (
    MAX_COT_CHARS,
    build_cot_input,
    sanitize_cot,
    thinking_tail,
)
from ..providers.base import UserMessage
from .base import SubAgent, SubAgentContext

if TYPE_CHECKING:
    from ..providers.base import LLMProvider

logger = logging.getLogger(__name__)

COT_SYSTEM_PROMPT = """You maintain the battle plan for one Pokemon battle — the single \
"how we win this game" note the playing agent re-reads before every decision.

You get: the previous plan, the board the decision was made from, the last \
decision and what came of it, the decision just committed (that turn has not \
resolved yet, so its outcome is unknown), the agent's private thinking for \
that turn, and the plan it has explicitly committed to.

Rewrite the plan from scratch using the newest evidence. Do not append, do not \
keep resolved items, do not narrate the turn. State:
1. the current winning line (what sequence actually wins, not a vague goal),
2. which of your Pokemon must be preserved and which opponent must be answered,
3. whether the previous line still holds — if the events refuted it, say what changed,
4. the concrete next step.

Rules: at most 90 words; plain text only (no JSON, headings, bullets or preamble); \
cite turn numbers as evidence; never invent observations — when unsure, omit. \
If the private thinking contradicts the previous plan, trust the thinking and \
the events, and correct the plan."""


class StrategyCOTAgent(SubAgent):
    """One background call per turn that rewrites the global COT."""

    name = "strategy_cot"

    def __init__(
        self,
        provider: "LLMProvider",
        *,
        max_tokens: int = 3000,
        input_chars: int = 8000,
    ):
        self.provider = provider
        self.max_tokens = max_tokens
        self.input_chars = input_chars

    async def run(self, sctx: SubAgentContext) -> str:
        record = sctx.inputs.get("record")
        if record is None:
            return ""
        prompt = build_cot_input(
            turn=int(sctx.inputs.get("turn") or getattr(record, "turn", 0)),
            seq=int(sctx.inputs.get("seq") or getattr(record, "seq", 0)),
            previous=sctx.inputs.get("previous"),
            board=sctx.inputs.get("board"),
            record=record,
            prev_record=sctx.inputs.get("previous_record"),
            plan=sctx.inputs.get("plan"),
            limit=self.input_chars,
        )
        try:
            response = await self.provider.complete(
                system=COT_SYSTEM_PROMPT,
                messages=[UserMessage(prompt)],
                tools=[],
                max_tokens=self.max_tokens,
            )
        except Exception as e:  # a sub-agent must never raise
            logger.warning("strategy COT call failed: %s: %s", type(e).__name__, e)
            return ""

        if response.stop_reason == "error":
            return ""
        text = sanitize_cot(response.text, limit=MAX_COT_CHARS)
        if text:
            sctx.inputs["cot_source"] = "llm"
            return text
        # thinking-only answer: keep the conclusion, mark the source
        tail = sanitize_cot(thinking_tail(response.thinking), limit=MAX_COT_CHARS)
        if tail:
            sctx.inputs["cot_source"] = "thinking"
            return tail
        return ""