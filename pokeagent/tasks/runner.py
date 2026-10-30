"""TaskRunner: bounded background execution of sub-agents.

Sub-agents never run on the critical path. The runner owns a semaphore
(global concurrency cap), a per-task timeout, per-battle task tracking, and
guarantees that failures are logged and swallowed — a crashed sub-agent can
never take the battle loop down with it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from .base import SubAgent, SubAgentContext

if TYPE_CHECKING:
    from ..config import AgentConfig
    from ..context.manager import ContextManager

logger = logging.getLogger(__name__)


class TaskRunner:
    """Fire-and-forget scheduling of SubAgent runs, capped by config."""

    def __init__(self, config: "AgentConfig", provider=None):
        self.config = config
        self.provider = provider  # the cheap/background LLM (already model-swapped)
        self._agents: dict[str, SubAgent] = {}
        self._semaphore = asyncio.Semaphore(config.max_concurrent_subagents)
        self._tasks: dict[str, list[asyncio.Task]] = {}  # battle_tag -> tasks
        self.submitted_by_tag: dict[str, int] = {}  # simple budget counters

    # --- registration -------------------------------------------------------

    def register(self, agent: SubAgent) -> None:
        self._agents[agent.name] = agent

    def get(self, name: str) -> SubAgent | None:
        return self._agents.get(name)

    # --- execution ------------------------------------------------------------

    def submit(
        self,
        battle_tag: str,
        agent_name: str,
        sctx: SubAgentContext,
        cm: "ContextManager | None" = None,
        intel_key: str | None = None,
    ) -> asyncio.Task | None:
        """Schedule one sub-agent run. Returns the task (fire-and-forget).

        The summary lands in ``sctx.intel[intel_key]`` (default: agent name);
        a ContextManager, if given, receives lifecycle events for the log.
        """
        agent = self._agents.get(agent_name)
        if agent is None:
            logger.warning("submit: unknown sub-agent '%s'", agent_name)
            return None
        if sctx.intel is None:
            logger.warning("submit: no IntelStore on context; result will be dropped")
        key = intel_key or agent_name

        async def _run() -> None:
            try:
                async with self._semaphore:
                    summary = await asyncio.wait_for(
                        agent.run(sctx), timeout=self.config.subagent_timeout
                    )
                summary = (summary or "").strip()
                if sctx.intel is not None and summary:
                    sctx.intel.put(key, summary)
                if cm is not None:
                    cm.log_event(
                        "subagent_done",
                        agent=agent_name,
                        key=key,
                        chars=len(summary),
                        summary=summary[:1500],
                    )
            except asyncio.TimeoutError:
                if cm is not None:
                    cm.log_event("subagent_timeout", agent=agent_name, key=key)
                logger.warning("sub-agent '%s' timed out", agent_name)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if cm is not None:
                    cm.log_event("subagent_error", agent=agent_name, key=key,
                                 error=f"{type(e).__name__}: {e}")
                logger.warning("sub-agent '%s' failed: %s: %s",
                               agent_name, type(e).__name__, e)

        task = asyncio.create_task(_run())
        self.submitted_by_tag[battle_tag] = self.submitted_by_tag.get(battle_tag, 0) + 1
        self._tasks.setdefault(battle_tag, []).append(task)

        def _cleanup(t: asyncio.Task, tag: str = battle_tag) -> None:
            tasks = self._tasks.get(tag)
            if tasks and t in tasks:
                tasks.remove(t)

        task.add_done_callback(_cleanup)
        return task

    # --- lifecycle ------------------------------------------------------------

    def cancel_battle(self, battle_tag: str) -> None:
        """Cancel all in-flight sub-agents for a finished battle."""
        tasks = self._tasks.pop(battle_tag, [])
        for task in tasks:
            if not task.done():
                task.cancel()
        self.submitted_by_tag.pop(battle_tag, None)

    def pending(self, battle_tag: str) -> int:
        return sum(1 for t in self._tasks.get(battle_tag, []) if not t.done())
