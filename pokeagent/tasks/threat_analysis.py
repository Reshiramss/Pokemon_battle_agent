"""Threat analysis sub-agent: fired when a new opponent Pokemon is revealed.

Runs one cheap-model LLM call off the critical path; the capped summary
lands in the IntelStore under ``threat:<species>`` and appears in the next
prompt's [INTEL] section.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..context.records import describe_pokemon
from ..providers.base import UserMessage
from .base import SubAgent, SubAgentContext

if TYPE_CHECKING:
    from poke_env.battle import Pokemon

    from ..providers.base import LLMProvider

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a competitive Pokemon analyst. An opponent's {species} \
just got revealed in a {battle_format} battle. Assess the threat for your team.

Opponent {species} static info:
{species_info}

Your team:
{team}

Write a compact tactical briefing (max ~120 words, plain text):
1. Its likely role and most dangerous moves/abilities for this team.
2. Which of YOUR Pokemon it most threatens, and which of yours best answers it.
3. One concrete playing tip (e.g. "keep X healthy for it", "watch for setup move Y").
Be concrete; no generic filler."""


class NewMonThreatAgent(SubAgent):
    """One background LLM call per newly revealed opponent species."""

    name = "threat_analysis"

    def __init__(self, provider: "LLMProvider", max_tokens: int = 700):
        self.provider = provider
        self.max_tokens = max_tokens

    async def run(self, sctx: SubAgentContext) -> str:
        species = str(sctx.inputs.get("species", ""))
        mon = sctx.inputs.get("pokemon")  # optional live object
        if not species:
            return ""

        species_info = "unknown"
        if sctx.kb is not None:
            data = sctx.kb.species_data(species)
            if data:
                types = "/".join(data.get("types") or [])
                stats = data.get("baseStats") or {}
                abilities = ", ".join((data.get("abilities") or {}).values())
                species_info = (
                    f"types {types}; base stats "
                    + " ".join(f"{k.upper()} {stats.get(k, '?')}"
                               for k in ("hp", "atk", "def", "spa", "spd", "spe"))
                    + f"; abilities {abilities}"
                )
        if mon is not None and getattr(mon, "moves", None):
            revealed = ", ".join(m.id for m in mon.moves.values())
            species_info += f"; revealed moves so far: {revealed}"

        team_lines = [
            describe_pokemon(p, revealed_moves=True) for p in sctx.battle.team.values()
        ] or ["(team not yet visible)"]

        prompt = PROMPT_TEMPLATE.format(
            species=species,
            battle_format=sctx.battle.format,
            species_info=species_info,
            team="\n".join(f"- {line}" for line in team_lines),
        )
        response = await self.provider.complete(
            system="You are a concise, expert competitive Pokemon analyst.",
            messages=[UserMessage(prompt)],
            tools=[],
            max_tokens=self.max_tokens,
        )
        if response.stop_reason == "error":
            logger.warning("threat analysis provider error: %s", response.error)
            return ""
        return response.text or ""
