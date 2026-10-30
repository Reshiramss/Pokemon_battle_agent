"""Decision hooks.

Pre-decision hooks enrich the prompt with knowledge fragments (static KB,
usage stats, ...) right before the LLM loop. They run concurrently under a
time budget, and any failure is logged and skipped — a broken hook must
never cost the agent a turn. Post-decision hooks observe/logging only.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..tools.base import ToolContext

if TYPE_CHECKING:
    from ..config import AgentConfig

logger = logging.getLogger(__name__)


class PreDecisionHook(ABC):
    """Return a markdown knowledge fragment to inject into this turn's prompt."""

    name: str = "hook"

    @abstractmethod
    async def run(self, ctx: ToolContext) -> str | None: ...


class PostDecisionHook(ABC):
    """Observe a committed decision (logging / metrics only)."""

    name: str = "post-hook"

    @abstractmethod
    async def run(
        self, ctx: ToolContext, decision: str, elapsed_s: float, fell_back: bool
    ) -> None: ...


class HookSystem:
    def __init__(self, fragment_char_cap: int = 1500):
        self.fragment_char_cap = fragment_char_cap
        self._pre: list[PreDecisionHook] = []
        self._post: list[PostDecisionHook] = []

    def register_pre(self, hook: PreDecisionHook) -> None:
        self._pre.append(hook)

    def register_post(self, hook: PostDecisionHook) -> None:
        self._post.append(hook)

    async def run_pre(self, ctx: ToolContext, budget: float) -> list[str]:
        """Run all pre-decision hooks in parallel; returns usable fragments.

        Each hook gets its own wait_for slice of the shared budget, so one
        slow or hung hook cannot starve the others.
        """
        if not self._pre:
            return []

        loop = asyncio.get_event_loop()
        deadline = loop.time() + budget

        async def _safe(hook: PreDecisionHook) -> str | None:
            remaining = max(deadline - loop.time(), 0.01)
            try:
                return await asyncio.wait_for(hook.run(ctx), timeout=remaining)
            except asyncio.TimeoutError:
                logger.warning("Hook '%s' timed out; skipped", hook.name)
                return None
            except Exception as e:
                logger.warning("Hook '%s' failed: %s: %s", hook.name, type(e).__name__, e)
                return None

        results = await asyncio.gather(*[_safe(h) for h in self._pre])
        return [
            frag[: self.fragment_char_cap]
            for frag in results
            if frag and frag.strip()
        ]

    async def run_post(
        self, ctx: ToolContext, decision: str, elapsed_s: float, fell_back: bool
    ) -> None:
        for hook in self._post:
            try:
                await asyncio.wait_for(
                    hook.run(ctx, decision, elapsed_s, fell_back), timeout=5.0
                )
            except Exception as e:
                logger.warning("Post hook '%s' failed: %s", hook.name, e)
