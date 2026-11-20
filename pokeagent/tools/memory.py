"""Memory tools: grep the fact projection, read a scene, inspect the scope.

These are the drill-down surface behind ``[MEMORY INDEX]``. They never call a
model and never mutate memory; each call counts against a small per-turn
budget so a model cannot spend the whole context on history.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from ..knowledge.static import normalize_name
from ..memory import facts as memory_facts
from ..memory.grep import Query
from ..memory.keys import sanitize_key
from ..memory.schema import normalize_subject
from ..memory.scenes import load_scene, scene_for_key
from ..memory.scope import previous_season_roots, scope_from, scope_root
from .base import Tool, ToolContext, ToolError

if TYPE_CHECKING:
    from ..formats import FormatProfile
    from ..memory.schema import FactLine

MAX_RESULT_CHARS = 4000
_SEPARATOR = " | "


def _clip(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 8].rstrip() + " …[truncated]"


def _profile(ctx: ToolContext) -> "FormatProfile":
    profile = getattr(ctx.kb, "profile", None)
    if profile is not None:
        return profile
    from ..formats import profile_for

    return profile_for(ctx.config.battle_format)  # type: ignore[union-attr]


def _context(ctx: ToolContext) -> tuple[Any, Path, Any]:
    config = ctx.config
    if config is None:
        raise ToolError("memory is unavailable: no agent config")
    if not getattr(config, "enable_memory", False):
        raise ToolError("cross-battle memory is disabled in this agent config")
    profile = _profile(ctx)
    scope = scope_from(profile, config.battle_format, config.default_regulation)
    root = scope_root(config, scope)
    return scope, root, config


def _check_budget(ctx: ToolContext) -> None:
    cm = ctx.context
    if cm is None or not hasattr(cm, "_memory_tool_calls"):
        return
    limit = int(getattr(ctx.config, "memory_tool_calls_per_turn", 3))
    if cm._memory_tool_calls >= limit:
        raise ToolError(
            f"memory tool budget exceeded ({limit} calls this turn); decide with what you have"
        )
    cm._memory_tool_calls += 1


def _roots(config: Any, scope: Any, root: Path) -> list[Path]:
    roots = [root]
    if getattr(config, "allow_previous_seasons", True):
        roots.extend(previous_season_roots(config, scope, limit=2))
    return [path for path in roots if path.is_dir()]


def _normalize_subject(value: str | None) -> str:
    return normalize_subject(value)


def _matches(row: "FactLine", *, subject: str, kind: str, archetype: str, min_priority: int, since: str, query: Query) -> bool:
    if subject and row.subject != subject:
        return False
    if kind and row.kind != kind:
        return False
    if archetype and archetype not in row.archetypes:
        return False
    if min_priority and row.priority < min_priority:
        return False
    if since and row.date and row.date < since:
        return False
    text = " ".join((row.subject, row.kind, " ".join(row.archetypes), " ".join(row.roles), row.content))
    return query.matches(text)


def _iter_rows(roots: Iterable[Path], layer: str) -> Iterable[tuple["FactLine", Path]]:
    for root in roots:
        if layer in ("facts", "all"):
            for path in memory_facts.iter_fact_files(root):
                for row in memory_facts.read_fact_file(path):
                    row.origin = root.name
                    yield row, root
        if layer in ("atoms", "all"):
            for path in memory_facts.iter_atom_files(root):
                _meta, rows = memory_facts.read_atoms_file(path)
                for row in rows:
                    row.origin = root.name
                    yield row, root


def _render_row(row: "FactLine") -> str:
    marker = ""
    if row.maturity == "prior":
        marker = f"[PRIOR:{row.source}] "
    if row.conflict:
        marker += "[CONFLICT] "
    return (
        f"{row.id} {row.subject}/{row.kind} m={row.maturity} n={row.evidence} "
        f"p={row.priority} d={row.date or '-'} {marker}{row.content}"
    )


class MemoryGrepTool(Tool):
    name = "memory_grep"
    description = (
        "Search this ruleset/season's memory (facts and immutable atoms). "
        "Field filters: subject (species/roster key), kind (set/event/trait/lesson/matchup), "
        "archetype tag, min_priority, since (YYYY-MM-DD). A free-text query uses grep "
        "semantics: whitespace-separated terms are ANDed; regex=true switches to one regex. "
        "Returns at most `limit` rows with provenance. Use memory_read for a full profile."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Free-text grep pattern (optional)."},
            "subject": {"type": "string", "description": "Exact subject key (e.g. incineroar, opp-3f9a)."},
            "kind": {
                "type": "string",
                "enum": ["set", "event", "trait", "lesson", "matchup"],
                "description": "Optional kind filter.",
            },
            "archetype": {"type": "string", "description": "Archetype/tactic tag to require."},
            "min_priority": {"type": "integer", "minimum": 0, "maximum": 100},
            "since": {"type": "string", "description": "Earliest event date, YYYY-MM-DD."},
            "scope": {
                "type": "string",
                "enum": ["facts", "atoms", "all"],
                "description": "Which layer to search (default: facts).",
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            "regex": {"type": "boolean", "description": "Treat query as one regular expression."},
        },
        "required": [],
    }

    async def execute(
        self,
        ctx: ToolContext,
        query: str | None = None,
        subject: str | None = None,
        kind: str | None = None,
        archetype: str | None = None,
        min_priority: int | None = None,
        since: str | None = None,
        scope: str = "facts",
        limit: int | None = None,
        regex: bool = False,
    ) -> str:
        _check_budget(ctx)
        mem_scope, root, config = _context(ctx)
        layer = (scope or "facts").lower()
        if layer not in ("facts", "atoms", "all"):
            raise ToolError(f"unknown scope {scope!r}; use facts, atoms or all")
        try:
            q = Query(query, regex=regex)
        except ValueError as exc:
            raise ToolError(str(exc))
        limit = max(1, min(50, int(limit or 8)))
        subject_key = _normalize_subject(subject)
        archetype_key = sanitize_key(archetype) if archetype else ""
        since_text = (since or "").strip()
        min_p = max(0, min(100, int(min_priority or 0)))
        hits: list["FactLine"] = []
        for row, _root in _iter_rows(_roots(config, mem_scope, root), layer):
            if _matches(
                row,
                subject=subject_key,
                kind=(kind or "").strip(),
                archetype=archetype_key,
                min_priority=min_p,
                since=since_text,
                query=q,
            ):
                hits.append(row)
        hits.sort(
            key=lambda row: (-row.priority, -row.evidence, row.date, row.id)
        )
        if not hits:
            return f"no memory rows matched in {root}"
        shown = hits[:limit]
        lines = [_render_row(row) for row in shown]
        footer = f"\n({len(hits)} match(es); showing {len(shown)} — narrow with subject/kind/archetype)"
        return _clip("\n".join(lines) + footer)


class MemoryReadTool(Tool):
    name = "memory_read"
    description = (
        "Read one full memory profile by key (species id, opp- roster key, archetype id, "
        "or team-/our- key). Returns a deterministic summary, the narrative, and the "
        "observed/prior fact lines, capped to the configured budget."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Scene key to read."},
        },
        "required": ["key"],
    }

    async def execute(self, ctx: ToolContext, key: str | None = None) -> str:
        _check_budget(ctx)
        mem_scope, root, config = _context(ctx)
        if not key:
            raise ToolError("key is required")
        found = scene_for_key(root, key)
        if found is None:
            return f"no memory profile for {key!r} in {root}"
        kind, scene = found
        meta = scene.get("meta") or {}
        parts = [
            f"[{kind}] {scene.get('key')} (season {meta.get('season') or '-'}, "
            f"source {meta.get('source') or '-'})",
            f"summary: {meta.get('summary') or ''}",
        ]
        narrative = str(scene.get("narrative") or "").strip()
        if narrative:
            parts.append("narrative: " + narrative)
        facts_path = root / "facts" / {
            "species": "species",
            "team": "teams",
            "archetype": "archetypes",
            "our": "our",
        }.get(kind, kind) / f"{scene.get('key')}.ndjson"
        rows = memory_facts.read_fact_file(facts_path)
        prior = [row for row in rows if row.maturity == "prior"]
        observed = [row for row in rows if row.maturity != "prior"]
        if prior:
            parts.append("[PRIOR]\n" + "\n".join(_render_row(row) for row in prior))
        if observed:
            parts.append("[OBSERVED]\n" + "\n".join(_render_row(row) for row in observed))
        limit = int(getattr(config, "memory_read_chars", 2000))
        return _clip("\n".join(parts), limit=limit)


class MemoryStatsTool(Tool):
    name = "memory_stats"
    description = (
        "Summarise what this ruleset/season remembers: scene counts by kind, fact counts "
        "by maturity/source, and the memory root."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def execute(self, ctx: ToolContext) -> str:
        _check_budget(ctx)
        mem_scope, root, config = _context(ctx)
        if not root.is_dir():
            return f"memory is empty ({root}) — no battles recorded yet."
        scene_counts: dict[str, int] = {}
        for kind in ("species", "teams", "archetypes", "our"):
            directory = root / "scenes" / kind
            scene_counts[kind] = len(list(directory.glob("*.json"))) if directory.is_dir() else 0
        maturity: dict[str, int] = {}
        sources: dict[str, int] = {}
        total = 0
        for path in memory_facts.iter_fact_files(root):
            for row in memory_facts.read_fact_file(path):
                total += 1
                maturity[row.maturity] = maturity.get(row.maturity, 0) + 1
                sources[row.source] = sources.get(row.source, 0) + 1
        lines = [
            f"scope: {mem_scope}",
            f"root: {root}",
            "scenes: " + ", ".join(f"{k}={v}" for k, v in scene_counts.items()),
            f"facts: {total} (" + ", ".join(f"{k}={v}" for k, v in sorted(maturity.items())) + ")",
            "sources: " + ", ".join(f"{k}={v}" for k, v in sorted(sources.items())),
        ]
        return _clip("\n".join(lines))