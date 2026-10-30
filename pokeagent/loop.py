"""The per-turn agentic loop: reason -> parallel tool calls -> commit.

The loop is the ONLY place that talks to the LLM provider inside
``choose_move``. A turn ends when the model calls the terminal
``submit_decision`` tool with a valid choice; anything else (invalid choice,
exhausted rounds) raises ``TurnAbort`` and the player falls back to a random
order. The player wraps the whole ``run_turn`` in ``asyncio.wait_for``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from poke_env.player import Player

from .context.manager import NUDGE, ContextManager
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
        "Choose ONLY from the options listed in the [NOW] section: an exact move id, "
        "or an exact species from the available switches."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["move", "switch"]},
            "target": {
                "type": "string",
                "description": (
                    "action=move: the exact move id; "
                    "action=switch: the exact species name from the switch list"
                ),
            },
            "terastallize": {
                "type": "boolean",
                "default": False,
                "description": "Only pass true if the notes say terastallization is available.",
            },
            "reason": {
                "type": "string",
                "description": "One short sentence explaining the choice.",
            },
        },
        "required": ["action", "target", "reason"],
    },
)


class TurnAbort(Exception):
    """Raised when the loop cannot produce a decision; caller falls back."""


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

    # --- public ------------------------------------------------------------

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
                raise TurnAbort(f"provider error: {resp.error}")

            messages.append(
                AssistantMessage(content=resp.text, tool_calls=list(resp.tool_calls))
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
                continue

            if resp.tool_calls:
                results = await self.registry.execute_parallel(ctx, resp.tool_calls)
                messages.extend(results)
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
