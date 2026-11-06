"""Cross-battle memory tools: grep-style search and corpus stats.

The corpus is the layered JSONL store written at battle end
(``species`` / ``teams`` / ``tactics`` / ``battles``). Search is
deliberately codebase-like: case-insensitive terms ANDed, ``regex=true``
for a regular expression, newest rows first, every hit carrying the battle
it came from.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..memory.store import LEVELS, MemoryStore
from .base import Tool, ToolContext, ToolError

if TYPE_CHECKING:
    from ..config import AgentConfig

MAX_RESULT_CHARS = 3000


def _clip(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + " …[truncated]"


def _store(config: "AgentConfig | None") -> MemoryStore:
    if config is None:
        raise ToolError("memory is unavailable: no agent config")
    if not getattr(config, "enable_memory", False):
        raise ToolError("cross-battle memory is disabled in this agent config")
    root = Path(getattr(config, "memory_path", None) or "runs/_memory")
    return MemoryStore(root, lines_cap=getattr(config, "memory_lines_cap", 500))


class MemorySearchTool(Tool):
    name = "memory_search"
    description = (
        "Search cross-battle memory — lessons from previous battles against "
        "these opponents, with these Pokemon, or with these tactics. Memory is "
        "layered: species (per-Pokemon observations), teams (matchup history of "
        "a roster), tactics (e.g. Perish Song trap, Trick Room, rain), battles "
        "(whole-battle summaries and post-battle reviews). Syntax mirrors "
        "grepping a codebase: case-insensitive, whitespace-separated terms must "
        "all appear, or set regex=true. Omit query to list the newest rows. "
        "Every hit shows its layer/key and source battle."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Terms to search for (all must appear). E.g. "
                    "'incineroar fake out' or 'perish'. Omit for newest rows."
                ),
            },
            "level": {
                "type": "string",
                "enum": ["all", *LEVELS],
                "default": "all",
                "description": "Restrict to one memory layer.",
            },
            "key": {
                "type": "string",
                "description": (
                    "Restrict to one file key, e.g. a species ('incineroar'), "
                    "team key ('team-fairytail'), or tactic id ('perish_trap')."
                ),
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "description": "Maximum rows to return (default from config).",
            },
            "regex": {
                "type": "boolean",
                "default": False,
                "description": "Treat query as one regular expression.",
            },
        },
        "required": [],
    }

    async def execute(
        self,
        ctx: ToolContext,
        query: str | None = None,
        level: str = "all",
        key: str | None = None,
        limit: int | None = None,
        regex: bool = False,
    ) -> str:
        config = ctx.config
        if limit is None:
            limit = getattr(config, "memory_search_limit", 8)
        limit = max(1, min(int(limit), 50))
        if level not in ("all", *LEVELS):
            raise ToolError(f"unknown level {level!r}; use: all, {', '.join(LEVELS)}")
        store = _store(config)
        try:
            lines, total = store.search(
                query, level=level, key=key, limit=limit, regex=regex
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        if not lines:
            what = f"matching {query!r}" if query else ""
            return f"no memory rows {what}".strip() + f" (searched {level})."
        body = "\n".join(line.render() for line in lines)
        footer = (
            f"\n({total} match{'es' if total != 1 else ''}; showing {len(lines)} "
            f"— narrow with level= or key=)"
        )
        return _clip(body + footer)


class MemoryStatsTool(Tool):
    name = "memory_stats"
    description = (
        "Summarize the cross-battle memory corpus: how many rows and keys each "
        "layer holds (species / teams / tactics / battles) and how much is "
        "remembered overall. Use before searching to see whether anything "
        "relevant exists."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def execute(self, ctx: ToolContext) -> str:
        store = _store(ctx.config)
        stats = store.stats()
        grand = sum(v["total"] for v in stats.values())
        if grand == 0:
            return f"memory is empty ({store.root}) — no finished battles recorded yet."
        lines = [f"memory root: {store.root}", f"total rows: {grand}"]
        for level in LEVELS:
            entry = stats[level]
            keys = entry["keys"]
            if not keys:
                continue
            top = sorted(keys.items(), key=lambda kv: -kv[1])[:6]
            lines.append(
                f"- {level}: {entry['total']} rows / {len(keys)} keys "
                f"(top: {', '.join(f'{k}×{n}' for k, n in top)})"
            )
        return "\n".join(lines)