"""The two decision loops: reason -> parallel tool calls -> commit.

The loop is the ONLY place that talks to the LLM provider inside
``choose_move`` / ``teampreview``. Both phases share one skeleton
(hooks -> prompt -> tool loop -> terminal tool) and differ only in prompt
content and terminal tool:

- ``run_turn`` (in-battle): ends when the model calls ``submit_decision``
  with a valid choice; otherwise (invalid choice, exhausted rounds) raises
  ``TurnAbort`` and the player falls back to a random order.
- ``run_preview`` (team preview): ends when the model calls ``submit_lead``
  with a valid team slot; otherwise raises ``PreviewAbort`` and the player
  falls back to the base class's random lead order.

The player wraps both in ``asyncio.wait_for``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from poke_env.player import Player

from .context.manager import ContextManager, NUDGE
from .context.records import describe_pokemon
from .hooks.base import HookSystem
from .knowledge.static import normalize_name
from .providers.base import (
    AssistantMessage,
    LLMProvider,
    ToolCallRequest,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)
from .tools.base import ToolContext, ToolRegistry

if TYPE_CHECKING:
    from .config import AgentConfig
    from .tasks.runner import TaskRunner

logger = logging.getLogger(__name__)

SUBMIT_DECISION = ToolSpec(
    name="submit_decision",
    description=(
        "Commit your chosen action for this turn. This ends the turn. "
        "Plain text is never interpreted as a move."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["move", "switch"],
                "description": (
                    "action=move: the exact move id; "
                    "action=switch: the exact species name from the switch list"
                ),
            },
            "target": {"type": "string"},
            "terastallize": {"type": "boolean"},
            "reason": {
                "type": "string",
                "description": "One short sentence explaining the choice.",
            },
        },
        "required": ["action", "target", "reason"],
    },
)

SUBMIT_LEAD = ToolSpec(
    name="submit_lead",
    description=(
        "Commit your lead Pokemon for the upcoming battle (team-preview "
        "decision). This is the phase-1 decision before the battle starts."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "slot": {
                "type": "integer",
                "description": "1-based slot of the lead Pokemon (1-6).",
            },
            "reason": {
                "type": "string",
                "description": "One short sentence explaining the lead choice.",
            },
        },
        "required": ["slot", "reason"],
    },
)

PREVIEW_SYSTEM_PROMPT = """You are an expert competitive Pokemon battle agent in the \
team-preview phase of a {battle_format} battle on Pokemon Showdown.

Your team:
{my_team}

Opponent's team:
{opponent_team}

Choose your lead Pokemon. Consider type matchups against the likely opposing \
lead, speed tiers, entry-hazard control, and team synergy. In best-of-one \
play, favor leads that are immediately useful. You MUST call the \
`submit_lead` tool with the chosen slot number (1-6) and a one-sentence \
reason."""


class TurnAbort(Exception):
    """Raised when the loop cannot produce a decision; caller falls back."""


class PreviewAbort(Exception):
    """Raised when the loop cannot produce a lead choice; caller falls back."""


def _dump_results(results: list[ToolResultMessage]) -> list[dict]:
    """Tool results in a JSON-friendly form for the llm.jsonl trace."""
    return [
        {
            "call_id": r.call_id,
            "tool": r.tool_name,
            "content": r.content,
            "is_error": r.is_error,
        }
        for r in results
    ]


@dataclass
class TurnOutcome:
    order: Any  # BattleOrder
    action: str
    target: str
    reason: str
    rounds: int
    cot: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class PreviewOutcome:
    """What run_preview returned: lead slot + reason."""
    lead: int
    reason: str
    rounds: int
    cot: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class AgentLoop:
    """Hosts the per-turn reasoning cycle. Fully stateless between turns."""

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        hooks: HookSystem,
        config: "AgentConfig",
        runner: "TaskRunner | None" = None,
    ):
        self.provider = provider
        self.registry = registry
        self.hooks = hooks
        self.config = config
        self.runner = runner
        # callback(battle, species) fired when a new opponent mon is revealed
        self.on_new_opponent_mon = None

    # --- phase 1: team preview ---------------------------------------------

    async def run_preview(self, battle, cm: ContextManager) -> PreviewOutcome:
        """Phase-1 decision: choose the lead Pokemon (team preview).

        Mirrors run_turn's shape: hooks -> prompt -> tool loop with the
        submit_lead terminal tool, bounded by the same budgets.
        """
        config = self.config
        ctx = ToolContext(battle=battle, kb=cm.kb, context=cm, config=config)

        fragments = await self.hooks.run_pre(ctx, config.hook_budget)

        team_lines = "\n".join(
            f"- {describe_pokemon(p, revealed_moves=True)}"
            for p in battle.team.values()
        )
        opp_lines = "\n".join(
            f"- {describe_pokemon(p, revealed_moves=True)}"
            for p in battle.opponent_team.values()
        )
        system = PREVIEW_SYSTEM_PROMPT.format(
            battle_format=battle.format,
            my_team=team_lines or "(unknown)",
            opponent_team=opp_lines or "(unknown)",
        )
        user = (
            "[TEAM PREVIEW — your current decision]\n"
            f"Format: {battle.format}\n"
            "Choose your lead Pokemon by calling the `submit_lead` tool.\n"
        )

        tool_specs = self.registry.specs() + [SUBMIT_LEAD]
        cot_parts: list[str] = []
        tool_log: list[dict] = []
        in_tokens = out_tokens = 0
        trace = config.log_llm_conversation
        messages = [UserMessage(user)]

        if trace:
            cm.log_llm(
                "turn_prompt",
                turn=0,
                phase="preview",
                system=system,
                prompt=user,
                tools=[t.name for t in tool_specs],
            )

        for round_ in range(config.max_tool_rounds):
            resp = await self.provider.complete(
                system=system,
                messages=messages,
                tools=tool_specs,
                max_tokens=config.max_tokens_per_call,
            )
            in_tokens += resp.input_tokens
            out_tokens += resp.output_tokens
            if resp.text:
                cot_parts.append(resp.text)

            if resp.stop_reason == "error":
                if trace:
                    cm.log_llm(
                        "llm_error", turn=0, phase="preview",
                        round=round_, error=resp.error,
                    )
                raise PreviewAbort(f"provider error: {resp.error}")

            messages.append(
                AssistantMessage(content=resp.text, tool_calls=list(resp.tool_calls))
            )
            if trace:
                cm.log_llm(
                    "llm_response",
                    turn=0,
                    phase="preview",
                    round=round_,
                    text=resp.text,
                    tool_calls=[
                        {"id": c.id, "name": c.name, "arguments": c.arguments}
                        for c in resp.tool_calls
                    ],
                    stop_reason=resp.stop_reason,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                )
            for call in resp.tool_calls:
                tool_log.append(
                    {"round": round_, "name": call.name, "arguments": call.arguments}
                )

            submit = next(
                (c for c in resp.tool_calls if c.name == "submit_lead"), None
            )
            if submit is not None:
                lead, reason, error = self._validate_lead(battle, submit)
                if lead is not None:
                    return PreviewOutcome(
                        lead=lead,
                        reason=reason,
                        rounds=round_ + 1,
                        cot="\n\n".join(cot_parts),
                        tool_calls=tool_log,
                        input_tokens=in_tokens,
                        output_tokens=out_tokens,
                    )
                # invalid slot: let the model retry with a clear error
                other_calls = [c for c in resp.tool_calls if c.name != "submit_lead"]
                results = await self.registry.execute_parallel(ctx, other_calls)
                results.append(
                    ToolResultMessage(submit.id, "submit_lead", error, True)
                )
                messages.extend(results)
                if trace:
                    cm.log_llm(
                        "tool_results",
                        turn=0,
                        phase="preview",
                        round=round_,
                        results=_dump_results(results),
                    )
                continue

            if resp.tool_calls:
                results = await self.registry.execute_parallel(ctx, resp.tool_calls)
                messages.extend(results)
                if trace:
                    cm.log_llm(
                        "tool_results",
                        turn=0,
                        phase="preview",
                        round=round_,
                        results=_dump_results(results),
                    )
            else:
                messages.append(UserMessage(NUDGE))

        raise PreviewAbort(
            f"no valid lead after {config.max_tool_rounds} rounds"
        )

    def _validate_lead(self, battle, call: ToolCallRequest):
        """Validate a submit_lead call against the team.

        Returns (lead, reason, error) — lead is None and error explains why
        when the choice is invalid, so the model can retry.
        """
        args = call.arguments or {}
        reason = str(args.get("reason", "")).strip()
        try:
            slot = int(args.get("slot"))
        except (TypeError, ValueError):
            return (
                None,
                reason,
                "'slot' must be an integer between 1 and "
                f"{len(battle.team)}.",
            )
        team_list = list(battle.team.values())
        if not 1 <= slot <= len(team_list):
            return (
                None,
                reason,
                f"'slot' must be an integer between 1 and {len(team_list)}.",
            )
        return slot, reason, None

    # --- phase 2: in-battle decision -----------------------------------------

    async def run_turn(self, battle, cm: ContextManager) -> TurnOutcome:
        config = self.config
        ctx = ToolContext(battle=battle, kb=cm.kb, context=cm, config=config)

        self._maybe_fire_threat_analyses(battle, cm)

        # no meaningful choice to make — struggle / forced default
        if not battle.available_moves and not battle.available_switches:
            return TurnOutcome(
                order=Player.choose_default_move(),
                action="default",
                target="struggle/default",
                reason="no moves or switches available",
                rounds=0,
            )

        fragments = await self.hooks.run_pre(ctx, config.hook_budget)
        system, messages = cm.assemble_prompt(battle, fragments)

        tool_specs = self.registry.specs() + [SUBMIT_DECISION]
        cot_parts: list[str] = []
        tool_log: list[dict] = []
        in_tokens = out_tokens = 0
        trace = config.log_llm_conversation

        if trace:
            # everything the model saw this turn, verbatim — replayable
            cm.log_llm(
                "turn_prompt",
                turn=battle.turn,
                system=system,
                prompt=messages[0].content,
                tools=[t.name for t in tool_specs],
            )

        for round_ in range(config.max_tool_rounds):
            resp = await self.provider.complete(
                system=system,
                messages=messages,
                tools=tool_specs,
                max_tokens=config.max_tokens_per_call,
            )
            in_tokens += resp.input_tokens
            out_tokens += resp.output_tokens
            if resp.text:
                cot_parts.append(resp.text)

            if resp.stop_reason == "error":
                if trace:
                    cm.log_llm(
                        "llm_error", turn=battle.turn, round=round_, error=resp.error
                    )
                raise TurnAbort(f"provider error: {resp.error}")

            messages.append(
                AssistantMessage(content=resp.text, tool_calls=list(resp.tool_calls))
            )
            if trace:
                cm.log_llm(
                    "llm_response",
                    turn=battle.turn,
                    round=round_,
                    text=resp.text,
                    tool_calls=[
                        {"id": c.id, "name": c.name, "arguments": c.arguments}
                        for c in resp.tool_calls
                    ],
                    stop_reason=resp.stop_reason,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                )
            for call in resp.tool_calls:
                tool_log.append(
                    {"round": round_, "name": call.name, "arguments": call.arguments}
                )

            submit = next(
                (c for c in resp.tool_calls if c.name == "submit_decision"), None
            )
            if submit is not None:
                order, action, target, reason, error = self._validate_submit(battle, submit)
                if order is not None:
                    return TurnOutcome(
                        order=order,
                        action=action,
                        target=target,
                        reason=reason,
                        rounds=round_ + 1,
                        cot="\n\n".join(cot_parts),
                        tool_calls=tool_log,
                        input_tokens=in_tokens,
                        output_tokens=out_tokens,
                    )
                # invalid choice: let the model retry with a clear error
                other_calls = [c for c in resp.tool_calls if c.name != "submit_decision"]
                results = await self.registry.execute_parallel(ctx, other_calls)
                results.append(
                    ToolResultMessage(submit.id, "submit_decision", error, True)
                )
                messages.extend(results)
                if trace:
                    cm.log_llm(
                        "tool_results", turn=battle.turn, round=round_,
                        results=_dump_results(results),
                    )
                continue

            if resp.tool_calls:
                results = await self.registry.execute_parallel(ctx, resp.tool_calls)
                messages.extend(results)
                if trace:
                    cm.log_llm(
                        "tool_results", turn=battle.turn, round=round_,
                        results=_dump_results(results),
                    )
            else:
                messages.append(UserMessage(NUDGE))

        raise TurnAbort(
            f"no valid decision after {config.max_tool_rounds} rounds "
            f"(turn {battle.turn})"
        )

    # --- internals -----------------------------------------------------------

    def _maybe_fire_threat_analyses(self, battle, cm: ContextManager) -> None:
        """Detect newly revealed opponent species; fire the threat callback."""
        if self.on_new_opponent_mon is None:
            return
        for p in battle.opponent_team.values():
            species = getattr(p, "species", None)
            if species and species not in cm.seen_species:
                cm.seen_species.add(species)
                try:
                    self.on_new_opponent_mon(battle, species)
                except Exception as e:
                    logger.warning("threat-analysis trigger failed: %s", e)

    def _validate_submit(self, battle, call: ToolCallRequest):
        """Validate a submit_decision call against the legal actions.

        Returns (order, action, target, reason, error) — order is None and
        error explains why when the choice is invalid, so the model can retry.
        """
        args = call.arguments or {}
        action = str(args.get("action", "")).lower()
        target = str(args.get("target", "")).strip()
        reason = str(args.get("reason", "")).strip()
        tera = bool(args.get("terastallize", False))

        if action == "move":
            norm = normalize_name(target)
            for m in battle.available_moves:
                if m.id == norm or normalize_name(m.id) == norm:
                    return (
                        Player.create_order(m, terastallize=tera),
                        "move",
                        m.id,
                        reason,
                        None,
                    )
            available = ", ".join(m.id for m in battle.available_moves) or "none"
            return (
                None,
                "move",
                target,
                reason,
                f"Invalid move '{target}'. Available moves: {available}.",
            )

        if action == "switch":
            norm = normalize_name(target)
            for p in battle.available_switches:
                if normalize_name(p.species) == norm:
                    return (Player.create_order(p), "switch", p.species, reason, None)
            available = ", ".join(p.species for p in battle.available_switches) or "none"
            return (
                None,
                "switch",
                target,
                reason,
                f"Invalid switch '{target}'. Available switches: {available}.",
            )

        return None, action, target, reason, "action must be 'move' or 'switch'."
