"""PokeAgent — the poke-env Player that hosts the agent framework.

This is the ONLY module coupled to poke-env's Player lifecycle: it wraps the
AgentLoop in a hard time budget inside ``choose_move`` and turns any failure
into a random fallback order, so the agent can degrade in quality but can
never stall or forfeit on time.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING

from poke_env.player import Player
from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ServerConfiguration,
    ShowdownServerConfiguration,
)

from .config import AgentConfig
from .context.compressor import LLMCompressor
from .context.manager import ContextManager
from .context.records import TurnRecord
from .hooks.base import HookSystem
from .knowledge.static import StaticKB
from .knowledge.tree import BattleKB
from .loop import AgentLoop, TurnAbort, TurnOutcome
from .providers import get_provider
from .providers.base import LLMProvider
from .tasks.base import SubAgentContext
from .tasks.runner import TaskRunner
from .teams import RawTeambuilder, TeamLibrary, resolve_team
from .tools.base import Tool, ToolContext, ToolRegistry

if TYPE_CHECKING:
    from poke_env.battle import AbstractBattle, BattleOrder

logger = logging.getLogger(__name__)

SERVERS = {
    "localhost": LocalhostServerConfiguration,
    "showdown": ShowdownServerConfiguration,
}


def build_server_configuration(config: AgentConfig) -> ServerConfiguration:
    """Server config honoring a custom host/port.

    LocalhostServerConfiguration hardcodes port 8000, which is occupied by
    another user's service on this shared machine, so the local server runs
    on config.server_port (default 8964) instead.
    """
    if config.server == "showdown":
        return ShowdownServerConfiguration
    return ServerConfiguration(
        f"ws://{config.server_host}:{config.server_port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )


class PokeAgent(Player):
    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        provider: LLMProvider | None = None,
        registry: ToolRegistry | None = None,
        hooks: HookSystem | None = None,
        kb: StaticKB | None = None,
        runner: TaskRunner | None = None,
        **kwargs,
    ):
        config = config or AgentConfig()
        self.config = config
        server = build_server_configuration(config)
        log_level = getattr(logging, config.poke_env_log_level.upper(), logging.WARNING)
        # config.team is a library name or raw packed string; an explicit team=
        # kwarg wins over it. Raw strings are wrapped in RawTeambuilder to
        # avoid ConstantTeambuilder's lossy re-serialization (it mangles
        # ability slot codes).
        team = kwargs.pop("team", None)
        if team is None and config.team is not None:
            self.team_library = TeamLibrary.from_config(config)
            team = resolve_team(config.team, self.team_library)
        if isinstance(team, str):
            team = RawTeambuilder(team)
        if team is not None:
            kwargs["team"] = team
        super().__init__(
            battle_format=config.battle_format,
            server_configuration=server,
            log_level=log_level,
            **kwargs,
        )

        self.kb = kb or StaticKB(config.battle_format)
        self.provider = provider or get_provider(config)
        self.registry = registry if registry is not None else self._default_registry()
        self.hooks = hooks if hooks is not None else self._default_hooks()
        self.runner = runner if runner is not None else self._default_runner()
        self.loop = AgentLoop(
            provider=self.provider,
            registry=self.registry,
            hooks=self.hooks,
            config=config,
            runner=self.runner,
        )
        self.loop.on_new_opponent_mon = self._fire_threat_analysis
        self._contexts: dict[str, ContextManager] = {}
        self._threat_counts: dict[str, int] = {}

    def _default_runner(self) -> TaskRunner:
        """Background sub-agents on the cheap model, bounded by config."""
        from .tasks.post_battle import PostBattleReviewAgent
        from .tasks.team_preview import TeamPreviewAnalysisAgent
        from .tasks.threat_analysis import NewMonThreatAgent

        sub_config = replace(self.config, model=self.config.effective_subagent_model)
        provider = get_provider(sub_config)
        runner = TaskRunner(self.config, provider=provider)
        for agent in (
            NewMonThreatAgent(provider),
            TeamPreviewAnalysisAgent(provider),
            PostBattleReviewAgent(provider),
        ):
            runner.register(agent)
        return runner

    def _default_registry(self) -> ToolRegistry:
        """Domain tools; the submit_decision terminal tool is owned by the loop."""
        registry = ToolRegistry()
        for tool in self._default_tools():
            registry.register(tool)
        return registry

    def _default_tools(self) -> list[Tool]:
        """Domain tools; the submit_decision terminal tool is owned by the loop."""
        from .tools.battle_query import BattleHistoryTool
        from .tools.damage_calc import DamageCalcTool
        from .tools.mechanics_info import MechanicsInfoTool
        from .tools.static_kb import static_kb_tools
        from .tools.team_comp import TeamCompTool

        return [
            DamageCalcTool(),
            *static_kb_tools(self.kb),
            BattleHistoryTool(),
            MechanicsInfoTool(),
            TeamCompTool(),
        ]

    def _default_hooks(self) -> HookSystem:
        from .hooks.battle_kb_hook import BattleKBHook
        from .hooks.mechanics_hook import MechanicsHook
        from .hooks.smogon_usage_hook import SmogonUsageHook
        from .hooks.static_kb_hook import TypeMatchupHook

        hooks = HookSystem()
        hooks.register_pre(BattleKBHook())
        hooks.register_pre(TypeMatchupHook())
        hooks.register_pre(MechanicsHook())
        hooks.register_pre(SmogonUsageHook(self.config))
        return hooks

    # --- poke-env lifecycle ---------------------------------------------------

    def _context_for(self, battle: "AbstractBattle") -> ContextManager:
        cm = self._contexts.get(battle.battle_tag)
        if cm is None:
            cm = ContextManager(
                battle_tag=battle.battle_tag, config=self.config, kb=self.kb
            )
            # background history compression on the cheap model
            if self.runner is not None and self.runner.provider is not None:
                cm.compressor = LLMCompressor(self.runner.provider)
            self._contexts[battle.battle_tag] = cm
        return cm

    async def choose_move(self, battle: "AbstractBattle") -> "BattleOrder":
        cm = self._context_for(battle)
        self._refresh_battle_kb(battle, cm)
        ctx = ToolContext(battle=battle, kb=cm.kb, context=cm, config=self.config)
        started = time.monotonic()

        try:
            outcome = await asyncio.wait_for(
                self.loop.run_turn(battle, cm), timeout=self.config.turn_budget
            )
        except asyncio.TimeoutError:
            await self._record_fallback(battle, cm, ctx, "turn_budget_exceeded", started)
            return self.choose_random_move(battle)
        except TurnAbort as e:
            await self._record_fallback(battle, cm, ctx, str(e), started)
            return self.choose_random_move(battle)
        except Exception as e:
            logger.exception("unexpected error in turn %s: %s", battle.turn, e)
            await self._record_fallback(
                battle, cm, ctx, f"{type(e).__name__}: {e}", started
            )
            return self.choose_random_move(battle)

        elapsed = time.monotonic() - started
        record = TurnRecord(
            seq=0,  # assigned by record_turn
            turn=battle.turn,
            request=cm.last_now_text,
            events=cm.new_events(battle),
            decision=outcome.decision,
            reason=outcome.reason,
            cot=outcome.cot or None,
            tool_calls=outcome.tool_calls,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            elapsed_s=round(elapsed, 2),
        )
        cm.record_turn(record)
        await self.hooks.run_post(
            ctx, decision=record.decision, elapsed_s=elapsed, fell_back=False
        )
        return outcome.order

    def _refresh_battle_kb(self, battle: "AbstractBattle", cm: ContextManager) -> None:
        """Keep the per-battle knowledge tree current; snapshot it to disk.

        Built lazily (team preview, or the first turn of formats without
        preview) and refreshed from the live objects every decision point.
        Never blocks the battle: the tree is an aid, not a dependency.
        """
        try:
            if cm.battle_kb is None:
                cm.battle_kb = BattleKB()
                cm.battle_kb.build(battle, self.kb)
            else:
                cm.battle_kb.refresh(battle, self.kb)
            cm.save_battle_kb()
        except Exception:
            logger.debug("battle_kb refresh failed", exc_info=True)

    async def _record_fallback(
        self, battle, cm: ContextManager, ctx: ToolContext, reason: str, started: float
    ) -> None:
        cm.log_event(
            "fallback",
            turn=battle.turn,
            reason=reason,
            elapsed_s=round(time.monotonic() - started, 2),
        )
        record = TurnRecord(
            seq=0,
            turn=battle.turn,
            request=cm.last_now_text or "n/a",
            events=cm.new_events(battle),
            decision="random_fallback",
            reason=None,
            cot=None,
            fell_back=True,
            fallback_reason=reason,
            elapsed_s=round(time.monotonic() - started, 2),
        )
        cm.record_turn(record)
        await self.hooks.run_post(
            ctx, decision="random_fallback", elapsed_s=record.elapsed_s, fell_back=True
        )

    # --- background sub-agents -------------------------------------------------

    def _fire_threat_analysis(self, battle: "AbstractBattle", species: str) -> None:
        """Loop callback: a new opponent species was revealed."""
        if self.runner is None:
            return
        tag = battle.battle_tag
        fired = self._threat_counts.get(tag, 0)
        if fired >= self.config.max_threat_analyses_per_battle:
            return
        self._threat_counts[tag] = fired + 1

        cm = self._context_for(battle)
        mon = next(
            (p for p in battle.opponent_team.values() if p.species == species), None
        )
        sctx = SubAgentContext(
            battle=battle, kb=self.kb, intel=cm.intel,
            inputs={"species": species, "pokemon": mon},
        )
        self.runner.submit(
            tag, "threat_analysis", sctx, cm=cm, intel_key=f"threat:{species}"
        )

    async def teampreview(self, battle: "AbstractBattle") -> str:
        """Phase-1 decision: the LLM picks the team and leads (team preview).

        gen9randombattle never requests team preview, so this stays dormant
        there; in preview formats the LLM picks the order (in doubles, the
        first two entries are the leads) within preview_budget and falls back
        to the base class order on failure. The background
        TeamPreviewAnalysisAgent still runs — its plan informs early turns,
        not the selection itself.
        """
        cm = self._context_for(battle)
        self._refresh_battle_kb(battle, cm)

        # background analysis first — its plan is useful from turn 1, even if
        # the lead itself uses the fast heuristic path
        if self.runner is not None:
            sctx = SubAgentContext(battle=battle, kb=self.kb, intel=cm.intel)
            self.runner.submit(
                battle.battle_tag, "team_preview", sctx, cm=cm,
                intel_key="team_preview",
            )

        started = time.monotonic()
        try:
            outcome = await asyncio.wait_for(
                self.loop.run_preview(battle, cm), timeout=self.config.preview_budget
            )
        except Exception as e:  # PreviewAbort, timeout, provider failure...
            reason = (
                "preview_budget_exceeded"
                if isinstance(e, asyncio.TimeoutError)
                else f"{type(e).__name__}: {e}"
            )
            logger.warning("team preview fell back: %s", reason)
            cm.log_event(
                "preview_fallback",
                reason=reason,
                elapsed_s=round(time.monotonic() - started, 2),
            )
            order = super().teampreview(battle)
            if inspect.isawaitable(order):
                order = await order
            return order

        # order_list is leads-first and, in doubles, already the picked length
        members = list(outcome.order_list)
        # insurance: never send more slots than the format allows
        if battle.max_team_size and len(members) > battle.max_team_size:
            members = members[: battle.max_team_size]
        team_list = list(battle.team.values())
        for i in members:
            try:
                team_list[i - 1]._selected_in_teampreview = True
            except Exception:
                pass
        cm.log_event(
            "team_preview",
            leads=members[:2],
            order_list=members,
            reason=outcome.reason,
            order="/team " + "".join(str(m) for m in members),
            elapsed_s=round(time.monotonic() - started, 2),
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
        )
        return "/team " + "".join(str(m) for m in members)

    def _battle_finished_callback(self, battle: "AbstractBattle") -> None:
        cm = self._contexts.get(battle.battle_tag)
        if cm is None:
            return
        cm.log_event(
            "battle_finished",
            turn=battle.turn,
            won=bool(getattr(battle, "won", False)),
            lost=bool(getattr(battle, "lost", False)),
        )
        if self.runner is not None:
            # cancel stale in-battle analyses first, THEN submit the review
            self.runner.cancel_battle(battle.battle_tag)
            sctx = SubAgentContext(
                battle=battle, kb=self.kb, intel=cm.intel,
                inputs={"context_manager": cm},
            )
            self.runner.submit(
                battle.battle_tag, "post_battle", sctx, cm=cm,
                intel_key="post_battle_review",
            )
        self._threat_counts.pop(battle.battle_tag, None)
