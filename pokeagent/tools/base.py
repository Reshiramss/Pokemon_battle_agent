"""Tool abstraction and registry with parallel execution.

A Tool is one function-callable capability announced to the LLM via JSON
Schema. Tools return strings (what the model sees); they raise ToolError for
invalid arguments, which the registry turns into an error tool-result so the
model can retry. Multiple tool calls requested in one model turn execute
concurrently via asyncio.gather.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..providers.base import ToolCallRequest, ToolResultMessage, ToolSpec

if TYPE_CHECKING:
    from poke_env.battle import AbstractBattle

    from ..config import AgentConfig
    from ..context.manager import ContextManager
    from ..knowledge.static import StaticKB


class ToolError(Exception):
    """Invalid arguments / unusable state — surfaces to the LLM as an error result."""


@dataclass
class ToolContext:
    """Everything a tool may need at execution time."""

    battle: "AbstractBattle"
    kb: "StaticKB | None" = None
    context: "ContextManager | None" = None
    config: "AgentConfig | None" = None


class Tool(ABC):
    """A single LLM-callable capability."""

    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = {}

    @abstractmethod
    async def execute(self, ctx: ToolContext, **kwargs) -> str:
        """Run the tool and return a string result for the LLM."""

    def spec(self) -> ToolSpec:
        if not self.name:
            raise ValueError(f"Tool {type(self).__name__} has no name")
        return ToolSpec(self.name, self.description, self.input_schema)


class ToolRegistry:
    """Pluggable tool set. ``register`` is last-wins so setups can override."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools.values()]

    async def execute_parallel(
        self,
        ctx: ToolContext,
        calls: list[ToolCallRequest],
        timeout: float | None = None,
    ) -> list[ToolResultMessage]:
        """Execute a batch of tool calls concurrently. Never raises."""

        if timeout is None:
            timeout = ctx.config.tool_call_timeout if ctx.config else 8.0

        async def _run_one(call: ToolCallRequest) -> ToolResultMessage:
            tool = self._tools.get(call.name)
            if tool is None:
                return ToolResultMessage(
                    call.id,
                    call.name,
                    f"Unknown tool '{call.name}'. Available tools: {sorted(self._tools)}",
                    is_error=True,
                )
            try:
                result = await asyncio.wait_for(
                    tool.execute(ctx, **call.arguments), timeout=timeout
                )
                return ToolResultMessage(call.id, call.name, str(result))
            except ToolError as e:
                return ToolResultMessage(call.id, call.name, f"Error: {e}", True)
            except asyncio.TimeoutError:
                return ToolResultMessage(
                    call.id, call.name, f"Tool '{call.name}' timed out after {timeout}s.", True
                )
            except TypeError as e:
                return ToolResultMessage(
                    call.id, call.name, f"Invalid arguments for '{call.name}': {e}", True
                )
            except Exception as e:  # tool bugs must never break the turn
                return ToolResultMessage(
                    call.id,
                    call.name,
                    f"Unexpected error in '{call.name}': {type(e).__name__}: {e}",
                    True,
                )

        if not calls:
            return []
        return list(await asyncio.gather(*[_run_one(c) for c in calls]))
