"""Post-battle review sub-agent: cheap-model summary after the battle ends.

Fired from ``_battle_finished_callback`` (sync context → create_task via
the runner). The review is logged into the run's events.jsonl for the
postmortem tooling; it never touches the battle loop.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..providers.base import UserMessage
from .base import SubAgent, SubAgentContext

if TYPE_CHECKING:
    from ..context.manager import ContextManager
    from ..providers.base import LLMProvider

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are reviewing a finished {battle_format} Pokemon battle \
({result}).

Decision log (turn: decision — reason; FALLBACK marks degraded turns):
{decisions}

Write a short review (max ~150 words): the 2-3 pivotal moments, recurring \
mistakes (bad switches, missed KOs, wasted terastallization), and the single \
most important improvement for the next battle. Plain text."""


class PostBattleReviewAgent(SubAgent):
    name = "post_battle"

    def __init__(self, provider: "LLMProvider", max_tokens: int = 800):
        self.provider = provider
        self.max_tokens = max_tokens

    async def run(self, sctx: SubAgentContext) -> str:
        cm: "ContextManager | None" = sctx.inputs.get("context_manager")
        if cm is None or not cm.records:
            return ""
        battle = sctx.battle
        result = "won" if getattr(battle, "won", False) else (
            "lost" if getattr(battle, "lost", False) else "finished"
        )
        decisions = "\n".join(
            f"turn {r.turn}: {r.decision}"
            + (f" — {r.reason}" if r.reason else "")
            + (f"  [FALLBACK: {r.fallback_reason}]" if r.fell_back else "")
            for r in cm.records
        )
        response = await self.provider.complete(
            system="You are a concise, expert competitive Pokemon coach.",
            messages=[
                UserMessage(
                    PROMPT_TEMPLATE.format(
                        battle_format=battle.format,
                        result=result,
                        decisions=decisions[:4000],
                    )
                )
            ],
            tools=[],
            max_tokens=self.max_tokens,
        )
        if response.stop_reason == "error":
            logger.warning("post-battle review provider error: %s", response.error)
            return ""
        return response.text or ""
