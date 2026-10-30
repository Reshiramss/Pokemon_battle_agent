"""Team-preview analysis sub-agent.

Only formats with team preview (e.g. standard formats) trigger it; in
gen9randombattle ``Player.teampreview`` is never called by the server, so
this agent stays dormant automatically. When it runs, it fires in the
background AFTER the (heuristic) preview order has been returned — the
critical path never waits on it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..context.records import describe_pokemon
from ..providers.base import UserMessage
from .base import SubAgent, SubAgentContext

if TYPE_CHECKING:
    from ..providers.base import LLMProvider

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a competitive Pokemon analyst preparing for a \
{battle_format} battle with team preview.

Your team:
{my_team}

Opponent's team:
{opp_team}

Write a compact battle plan (max ~150 words, plain text): your likely lead, \
which opposing Pokemon is the biggest threat and your answer to it, your \
general win condition. Be concrete."""


class TeamPreviewAnalysisAgent(SubAgent):
    name = "team_preview"

    def __init__(self, provider: "LLMProvider", max_tokens: int = 800):
        self.provider = provider
        self.max_tokens = max_tokens

    async def run(self, sctx: SubAgentContext) -> str:
        battle = sctx.battle
        my_team = "\n".join(
            f"- {line}" for line in (
                describe_pokemon(p, revealed_moves=True) for p in battle.team.values()
            )
        )
        opp_team = "\n".join(
            f"- {line}" for line in (
                describe_pokemon(p, revealed_moves=True) for p in battle.opponent_team.values()
            )
        )
        response = await self.provider.complete(
            system="You are a concise, expert competitive Pokemon analyst.",
            messages=[
                UserMessage(
                    PROMPT_TEMPLATE.format(
                        battle_format=battle.format,
                        my_team=my_team or "(unknown)",
                        opp_team=opp_team or "(unknown)",
                    )
                )
            ],
            tools=[],
            max_tokens=self.max_tokens,
        )
        if response.stop_reason == "error":
            logger.warning("team preview analysis provider error: %s", response.error)
            return ""
        return response.text or ""
