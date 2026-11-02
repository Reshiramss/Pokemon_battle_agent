"""The two decision loops: reason -> parallel tool calls -> commit.

The loop is the ONLY place that talks to the LLM provider inside
``choose_move`` / ``teampreview``. Both phases share one skeleton
(hooks -> prompt -> tool loop -> terminal tool) and differ only in prompt
content and terminal tool:

- ``run_turn`` (in-battle): ends when the model calls ``submit_decision``
  with a valid choice; otherwise (invalid choice, exhausted rounds) raises
  ``TurnAbort`` and the player falls back to a random order. Singles and
  doubles share the v2 schema — in doubles ``actions`` holds one entry per
  active slot with symbolic move targets.
- ``run_preview`` (team preview): ends when the model calls ``submit_team``
  with a valid slot order; otherwise raises ``PreviewAbort`` and the player
  falls back to the base class's order.

The player wraps both in ``asyncio.wait_for``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from poke_env.player import DoubleBattleOrder, PassBattleOrder, Player

from .context.manager import ContextManager, NUDGE
from .context.records import (
    describe_pokemon,
    is_doubles,
    move_target_symbols,
    symbol_position,
)
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
        "Commit your chosen action(s) for this turn. This ends the turn. "
        "Plain text is never interpreted as a move. Singles: exactly one "
        "action (slot is ignored). Doubles: one action per active slot that "
        "has choices — slots without choices auto-pass."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "slot": {
                            "type": "integer",
                            "description": "1-based active slot (doubles: 1 or 2).",
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["move", "switch", "pass"],
                            "description": (
                                "move: pick 'move'; switch: pick 'switch_to'; "
                                "pass: only where the slot list says pass is legal"
                            ),
                        },
                        "move": {
                            "type": "string",
                            "description": "exact move id (kind=move).",
                        },
                        "switch_to": {
                            "type": "string",
                            "description": (
                                "species name from the switch list (kind=switch)."
                            ),
                        },
                        "target": {
                            "type": "string",
                            "enum": ["auto", "self", "ally", "opp1", "opp2"],
                            "description": (
                                "doubles move target — one of the move's listed "
                                "targets (kind=move; ignored in singles)"
                            ),
                        },
                        "terastallize": {"type": "boolean"},
                    },
                    "required": ["slot", "kind"],
                },
            },
            "reason": {
                "type": "string",
                "description": "One short sentence explaining the choice.",
            },
        },
        "required": ["actions", "reason"],
    },
)

SUBMIT_TEAM = ToolSpec(
    name="submit_team",
    description=(
        "Commit your team selection for the upcoming battle (team-preview "
        "decision). In pick-N doubles formats the order list holds exactly N "
        "slots: the FIRST TWO are your leads, the rest your back row. In "
        "singles the first entry is your lead."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "order": {
                "type": "array",
                "items": {
                    "type": "integer",
                    "description": "1-based team slot of a picked Pokemon.",
                },
                "description": "Picked slots, leads first.",
            },
            "reason": {
                "type": "string",
                "description": "One short sentence explaining the selection.",
            },
        },
        "required": ["order", "reason"],
    },
)

PREVIEW_SYSTEM_PROMPT = """You are an expert competitive Pokemon battle agent in the \
team-preview phase of a {battle_format} battle on Pokemon Showdown.

Your team:
{my_team}

Opponent's team:
{opponent_team}

Select your Pokemon for this battle by calling the `submit_team` tool with an \
`order` list of 1-based team slots. In pick-N doubles formats (e.g. VGC \
6-pick-4) the list holds exactly N slots and the FIRST TWO are your leads. \
In singles the first entry is your lead. Consider type matchups against the \
likely opposing leads, speed tiers, and team synergy. In best-of-one play, \
favor leads that are immediately useful. Finish with a one-sentence reason."""


class TurnAbort(Exception):
    """Raised when the loop cannot produce a decision; caller falls back."""


class PreviewAbort(Exception):
    """Raised when the loop cannot produce a team order; caller falls back."""


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
    decision: str  # e.g. "move:earthquake" or "slot1:move:icebeam->opp1 | slot2:pass"
    reason: str
    rounds: int
    cot: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class PreviewOutcome:
    """What run_preview returned: team slots, leads first."""

    order_list: list[int]
    reason: str
    rounds: int
    cot: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


def _slot_plans(battle) -> list[str]:
    """Per-slot 'act' / 'pass' plan for a doubles decision request.

    A slot plans to 'pass' when it has no decision to make: it is being
    force-switched with no replacement available, it sits out a mid-turn
    forced-switch decision (only the forced slot chooses), or it simply has
    no moves or switches. Everything else must receive an action.
    """
    forced = list(battle.force_switch)
    plans: list[str] = []
    for i in range(2):
        if forced[i]:
            plans.append("act" if battle.available_switches[i] else "pass")
        elif any(forced):
            plans.append("pass")
        else:
            has_choice = bool(battle.available_moves[i]) or bool(
                battle.available_switches[i]
            )
            plans.append("act" if has_choice else "pass")
    return plans


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
        """Phase-1 decision: pick the team and leads (team preview).

        Mirrors run_turn's shape: hooks -> prompt -> tool loop with the
        submit_team terminal tool, bounded by the same budgets.
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
            "Pick your team for this battle by calling the `submit_team` tool.\n"
        )

        tool_specs = self.registry.specs() + [SUBMIT_TEAM]
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
                (c for c in resp.tool_calls if c.name == "submit_team"), None
            )
            if submit is not None:
                order_list, reason, error = self._validate_team(battle, submit)
                if order_list is not None:
                    return PreviewOutcome(
                        order_list=order_list,
                        reason=reason,
                        rounds=round_ + 1,
                        cot="\n\n".join(cot_parts),
                        tool_calls=tool_log,
                        input_tokens=in_tokens,
                        output_tokens=out_tokens,
                    )
                # invalid order: let the model retry with a clear error
                other_calls = [c for c in resp.tool_calls if c.name != "submit_team"]
                results = await self.registry.execute_parallel(ctx, other_calls)
                results.append(
                    ToolResultMessage(submit.id, "submit_team", error, True)
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
            f"no valid team order after {config.max_tool_rounds} rounds"
        )

    def _validate_team(self, battle, call: ToolCallRequest):
        """Validate a submit_team call against the team.

        Returns (order_list, reason, error) — order_list is None and error
        explains why when the choice is invalid, so the model can retry.
        """
        args = call.arguments or {}
        reason = str(args.get("reason", "")).strip()
        team_size = len(battle.team)
        raw = args.get("order")
        if not isinstance(raw, list) or not raw:
            return (
                None,
                reason,
                f"'order' must be a list of team slots (1-based, up to {team_size}).",
            )
        slots: list[int] = []
        for v in raw:
            if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= team_size:
                return (
                    None,
                    reason,
                    f"'order' entries must be integers between 1 and {team_size}.",
                )
            slots.append(v)
        if len(set(slots)) != len(slots):
            return None, reason, "'order' contains duplicate slots."

        if is_doubles(battle):
            need = battle.max_team_size or 4
            if len(slots) != need:
                return (
                    None,
                    reason,
                    f"this format picks {need} slots (first two are the leads) — "
                    f"you submitted {len(slots)}.",
                )
        else:
            # singles: only the lead matters; the back row keeps team order
            lead = slots[0]
            slots = [lead] + [i + 1 for i in range(team_size) if i + 1 != lead]
        return slots, reason, None

    # --- phase 2: in-battle decision -----------------------------------------

    async def run_turn(self, battle, cm: ContextManager) -> TurnOutcome:
        config = self.config
        ctx = ToolContext(battle=battle, kb=cm.kb, context=cm, config=config)

        self._maybe_fire_threat_analyses(battle, cm)

        if is_doubles(battle):
            # poke-env guards against wait'd battles upstream; this is insurance
            if battle.wait:
                return TurnOutcome(
                    order=DoubleBattleOrder(
                        Player.choose_default_move(), Player.choose_default_move()
                    ),
                    decision="default:wait",
                    reason="battle is waiting — nothing to decide",
                    rounds=0,
                )
            # no meaningful choice in either slot — struggle / forced default
            if not any(battle.available_moves) and not any(battle.available_switches):
                return TurnOutcome(
                    order=DoubleBattleOrder(
                        Player.choose_default_move(), Player.choose_default_move()
                    ),
                    decision="default:struggle",
                    reason="no moves or switches available",
                    rounds=0,
                )
        elif not battle.available_moves and not battle.available_switches:
            # no meaningful choice to make — struggle / forced default
            return TurnOutcome(
                order=Player.choose_default_move(),
                decision="default:struggle/default",
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
                order, decision, reason, error = self._validate_submit(battle, submit)
                if order is not None:
                    return TurnOutcome(
                        order=order,
                        decision=decision,
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

        Returns (order, decision, reason, error) — order is None and error
        explains why when the choice is invalid, so the model can retry.
        """
        if is_doubles(battle):
            return self._validate_doubles(battle, call)
        return self._validate_singles(battle, call)

    def _validate_singles(self, battle, call: ToolCallRequest):
        """Singles: read actions[0]; the target field is ignored."""
        args = call.arguments or {}
        reason = str(args.get("reason", "")).strip()
        actions = args.get("actions")
        if not isinstance(actions, list) or not actions:
            return (
                None,
                "",
                reason,
                "'actions' must be a non-empty list with one action object.",
            )
        a = actions[0]
        if not isinstance(a, dict):
            return None, "", reason, "actions[0] must be an object with slot/kind fields."
        kind = str(a.get("kind", "")).lower()
        tera = bool(a.get("terastallize", False))

        if kind == "move":
            name = str(a.get("move", "")).strip()
            norm = normalize_name(name)
            for m in battle.available_moves:
                if m.id == norm or normalize_name(m.id) == norm:
                    if tera and not battle.can_tera:
                        return (
                            None,
                            f"move:{m.id}",
                            reason,
                            "terastallization is not available right now.",
                        )
                    return (
                        Player.create_order(m, terastallize=tera),
                        f"move:{m.id}",
                        reason,
                        None,
                    )
            available = ", ".join(m.id for m in battle.available_moves) or "none"
            return (
                None,
                f"move:{name}",
                reason,
                f"Invalid move '{name}'. Available moves: {available}.",
            )

        if kind == "switch":
            if tera:
                return (
                    None,
                    "switch",
                    reason,
                    "terastallize=true requires kind='move'.",
                )
            name = str(a.get("switch_to", "")).strip()
            norm = normalize_name(name)
            for p in battle.available_switches:
                if normalize_name(p.species) == norm:
                    return (Player.create_order(p), f"switch:{p.species}", reason, None)
            available = (
                ", ".join(p.species for p in battle.available_switches) or "none"
            )
            return (
                None,
                f"switch:{name}",
                reason,
                f"Invalid switch '{name}'. Available switches: {available}.",
            )

        return None, kind, reason, "kind must be 'move' or 'switch' in singles."

    def _validate_doubles(self, battle, call: ToolCallRequest):
        """Doubles: one action per deciding slot, symbolic move targets.

        Slots without a decision auto-pass; submitting an action for such a
        slot, missing a deciding slot, or duplicating a slot is an error the
        model can retry on. Switches claim bench mons in order, so two slots
        can each switch into a different same-species mon.
        """
        args = call.arguments or {}
        reason = str(args.get("reason", "")).strip()
        raw = args.get("actions")
        if not isinstance(raw, list) or not raw:
            return (
                None,
                "",
                reason,
                "'actions' must be a non-empty list of action objects.",
            )
        if len(raw) > 2:
            return None, "", reason, "at most 2 actions (one per active slot)."

        plans = _slot_plans(battle)
        # where the protocol allows an explicit pass (mirrors valid_orders:
        # non-deciding slots, and both-forced-with-a-single-bench-mon)
        passable = [self._pass_legal(battle, i) for i in range(2)]

        parsed: dict[int, dict] = {}
        claimed: list = []  # bench Pokemon objects already taken by a switch
        tera_slot: int | None = None
        for a in raw:
            if not isinstance(a, dict):
                return None, "", reason, "each action must be an object."
            slot = a.get("slot")
            if not isinstance(slot, int) or isinstance(slot, bool) or slot not in (1, 2):
                return None, "", reason, "action 'slot' must be 1 or 2."
            i = slot - 1
            if i in parsed:
                return None, "", reason, f"duplicate action for slot {slot}."
            kind = str(a.get("kind", "")).lower()

            if kind == "pass":
                if not passable[i]:
                    return (
                        None,
                        "",
                        reason,
                        f"slot {slot} cannot pass — it must choose a move or switch.",
                    )
                parsed[i] = {"kind": "pass"}
                continue

            if plans[i] == "pass":
                return (
                    None,
                    "",
                    reason,
                    f"slot {slot} has no decision this round (it auto-passes) — "
                    "remove its action.",
                )

            tera = bool(a.get("terastallize", False))
            if kind == "move":
                name = str(a.get("move", "")).strip()
                norm = normalize_name(name)
                move = next(
                    (
                        m
                        for m in battle.available_moves[i]
                        if m.id == norm or normalize_name(m.id) == norm
                    ),
                    None,
                )
                if move is None:
                    available = (
                        ", ".join(m.id for m in battle.available_moves[i]) or "none"
                    )
                    return (
                        None,
                        "",
                        reason,
                        f"slot {slot}: invalid move '{name}'. Available: {available}.",
                    )
                if tera:
                    if not battle.can_tera[i]:
                        return (
                            None,
                            "",
                            reason,
                            f"slot {slot}: terastallization is not available "
                            "for this Pokemon.",
                        )
                    if tera_slot is not None:
                        return (
                            None,
                            "",
                            reason,
                            "only one Pokemon can terastallize per turn.",
                        )
                    tera_slot = i
                symbols = move_target_symbols(battle, move, i)
                target = str(a.get("target", "")).strip().lower()
                if target and target not in symbols:
                    return (
                        None,
                        "",
                        reason,
                        f"slot {slot}: invalid target '{target}' for {move.id}. "
                        f"Legal targets: {', '.join(symbols)}.",
                    )
                if not target:
                    if len(symbols) == 1:
                        target = symbols[0]
                    else:
                        return (
                            None,
                            "",
                            reason,
                            f"slot {slot}: {move.id} needs an explicit target. "
                            f"Legal targets: {', '.join(symbols)}.",
                        )
                parsed[i] = {
                    "kind": "move",
                    "order": Player.create_order(
                        move,
                        terastallize=tera,
                        move_target=symbol_position(target, i),
                    ),
                    "move": move.id,
                    "target": target,
                    "tera": tera,
                }
            elif kind == "switch":
                if tera:
                    return (
                        None,
                        "",
                        reason,
                        f"slot {slot}: terastallize=true requires kind='move'.",
                    )
                name = str(a.get("switch_to", "")).strip()
                norm = normalize_name(name)
                options = [
                    p
                    for p in battle.available_switches[i]
                    if normalize_name(p.species) == norm and p not in claimed
                ]
                if not options:
                    available = (
                        ", ".join(p.species for p in battle.available_switches[i])
                        or "none"
                    )
                    taken = ", ".join(
                        p.species for p in claimed
                    ) or "none"
                    return (
                        None,
                        "",
                        reason,
                        f"slot {slot}: invalid switch '{name}' (already claimed "
                        f"by another slot: {taken}). Available: {available}.",
                    )
                mon = options[0]
                claimed.append(mon)
                parsed[i] = {
                    "kind": "switch",
                    "order": Player.create_order(mon),
                    "species": mon.species,
                }
            else:
                return None, "", reason, f"slot {slot}: kind must be 'move', 'switch' or 'pass'."

        # missing slots: auto-pass pass-slots; a deciding slot may be omitted
        # only where passing is legal (both-forced with a single bench mon)
        for i in range(2):
            if i in parsed:
                continue
            if plans[i] == "pass" or passable[i]:
                parsed[i] = {"kind": "pass"}
            else:
                return (
                    None,
                    "",
                    reason,
                    f"missing action for slot {i + 1}, which has choices this round.",
                )

        if all(v["kind"] == "pass" for v in parsed.values()):
            if any(battle.force_switch):
                return (
                    None,
                    "",
                    reason,
                    "a forced-switch slot must switch to something — pass only "
                    "the other slot.",
                )
            # nothing to decide in either slot
            return (
                DoubleBattleOrder(
                    Player.choose_default_move(), Player.choose_default_move()
                ),
                "default:struggle",
                reason,
                None,
            )

        order = DoubleBattleOrder(
            parsed[0].get("order", PassBattleOrder()),
            parsed[1].get("order", PassBattleOrder()),
        )
        parts: list[str] = []
        for i in range(2):
            v = parsed[i]
            if v["kind"] == "pass":
                parts.append(f"slot{i + 1}:pass")
            elif v["kind"] == "move":
                bits = f"slot{i + 1}:move:{v['move']}"
                if v["target"] != "auto":
                    bits += f"->{v['target']}"
                if v["tera"]:
                    bits += "+tera"
                parts.append(bits)
            else:
                parts.append(f"slot{i + 1}:switch:{v['species']}")
        return order, " | ".join(parts), reason, None

    def _pass_legal(self, battle, slot_index: int) -> bool:
        """Whether the protocol accepts an explicit pass for this slot."""
        return any(
            isinstance(o, PassBattleOrder) for o in battle.valid_orders[slot_index]
        )
