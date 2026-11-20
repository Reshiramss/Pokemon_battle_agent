"""Budgeted, deterministic recall: the memory index and per-turn deltas.

Preview gets one stable index (system prompt, cached for the whole battle).
Each turn gets only the facts triggered by new reveals, capped at a few lines.
Ranking is a pure function of the row fields — no model, no embeddings, and
the same inputs always produce byte-identical output.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from ..knowledge.static import normalize_name
from . import classify, facts, scenes, seed
from .keys import opponent_key, our_team_key
from .scope import (
    MemoryScope,
    ensure_scope_root,
    previous_season_roots,
    scope_from,
)

if TYPE_CHECKING:
    from ..config import AgentConfig
    from ..formats import FormatProfile

PHASE_KINDS = {
    "preview": ("trait", "set", "lesson", "matchup"),
    "turn": ("trait", "set", "lesson", "matchup"),
}
PHASE_WEIGHT = {
    "preview": {"matchup": 0.95, "lesson": 0.9, "trait": 0.8, "set": 0.7, "event": 0.4},
    "turn": {"set": 0.95, "trait": 0.9, "lesson": 0.85, "matchup": 0.7, "event": 0.4},
}
DIMENSION_WEIGHT = {
    "species": {"preview": 0.75, "turn": 1.0},
    "team": {"preview": 1.0, "turn": 0.55},
    "archetype": {"preview": 1.0, "turn": 0.5},
    "our": {"preview": 0.85, "turn": 0.6},
}
HALF_LIFE_DAYS = 30.0

MEMORY_TOOLS_GUIDE = """[MEMORY TOOLS]
- memory_grep(query, subject, kind, archetype, min_priority, since, scope=facts|atoms|all, limit)
  searches the current ruleset/season's immutable facts and atoms (grep semantics:
  whitespace = AND, regex=true for one regex).
- memory_read(key) reads one full scene profile (pokedex-style summary + observed facts).
- memory_stats() shows what this scope remembers.
At most 3 memory tool calls per turn. Injected memory is historical priors, not
current facts: a live reveal always overrides it."""


def _category(key: str) -> str:
    if key.startswith("opp-"):
        return "team"
    if key.startswith("team-") or key.startswith("our-"):
        return "our"
    return "species"


def _roster(battle: Any, ours: bool = False) -> list[str]:
    team = getattr(battle, "team", None) if ours else getattr(battle, "opponent_team", None)
    out = []
    for mon in (team or {}).values() if isinstance(team, dict) else team or []:
        species = normalize_name(str(getattr(mon, "species", "") or ""))
        if species:
            out.append(species)
    return sorted(set(out))


def _foe_mons(battle: Any) -> list[Any]:
    team = getattr(battle, "opponent_team", None)
    return list(team.values()) if isinstance(team, dict) else list(team or [])


def _tactics(battle: Any) -> list[str]:
    try:
        from .tactics import tactic_ids

        return tactic_ids(_foe_mons(battle))
    except Exception:
        return []


def _reveal_snapshot(battle: Any) -> dict[str, tuple]:
    snap: dict[str, tuple] = {}
    for mon in _foe_mons(battle):
        species = normalize_name(str(getattr(mon, "species", "") or ""))
        if not species:
            continue
        raw_moves = getattr(mon, "moves", None)
        moves = tuple(sorted(normalize_name(m) for m in (raw_moves.keys() if isinstance(raw_moves, dict) else raw_moves or []) if m))
        item = normalize_name(str(getattr(mon, "item", "") or ""))
        ability = normalize_name(str(getattr(mon, "ability", "") or ""))
        tera = normalize_name(str(getattr(mon, "tera_type", "") or ""))
        snap[species] = (moves, item, ability, tera)
    return snap


def _ensure_team_scene(
    root: Path,
    scope: MemoryScope,
    key: str,
    *,
    roster: list[str],
    archetype: str = "",
    archetype_score: float = 0.0,
    source: str = "obs",
) -> dict[str, Any]:
    scene = scenes.ensure_scene(root, "team", key, scope, source=source)
    changed = False
    if roster and not scene.get("roster"):
        scene["roster"] = roster
        changed = True
    if archetype and not scene.get("archetype"):
        scene["archetype"] = archetype
        scene["archetype_score"] = round(float(archetype_score or 0.0), 3)
        changed = True
    if changed or not scene["meta"].get("summary"):
        scene["meta"]["summary"] = scenes.summarize(scene)
        scenes.save_scene(root, scene)
    return scene


def _ensure_our_scene(
    root: Path,
    scope: MemoryScope,
    key: str,
    roster: list[str],
) -> dict[str, Any]:
    scene = scenes.ensure_scene(root, "our", key, scope, source="library")
    if roster and not scene.get("roster"):
        scene["roster"] = roster
        scene["meta"]["summary"] = scenes.summarize(scene)
        scenes.save_scene(root, scene)
    return scene


def _preview_keys(
    battle: Any,
    config: "AgentConfig",
    root: Path,
) -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    opp = _roster(battle)
    ours = _roster(battle, ours=True)
    opp_key = opponent_key(opp)
    our_key = our_team_key(config, ours)
    candidates = classify.archetype_candidates(opp, root)
    arch = candidates[0][0] if candidates else ""
    arch_score = candidates[0][1] if candidates else 0.0
    keys = [opp_key, our_key, *opp, *(candidate for candidate, _score in candidates)]
    return keys, {
        "opp": opp_key,
        "our": our_key,
        "archetype": arch,
        "archetype_score": arch_score,
    }, {
        "opp_roster": opp,
        "our_roster": ours,
    }


def _scene_line(kind: str, key: str, scene: dict[str, Any], *, limit: int = 120) -> str:
    meta = scene.get("meta") or {}
    summary = str(meta.get("summary") or scenes.summarize(scene) or "").strip()
    season = str(meta.get("season") or "")
    tag = {
        "team": "TEAM",
        "teams": "TEAM",
        "archetype": "ARCH",
        "archetypes": "ARCH",
        "species": "MON",
        "our": "OUR",
    }.get(kind, kind.upper())
    extra = f" [{season}]" if season else ""
    if len(summary) > limit:
        cut = summary[: max(1, limit - 1)]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        summary = cut.rstrip(" ·,;-") + "…"
    return f"[{tag}] {key}{extra}: {summary}"


def build_index(
    battle: Any,
    cm: Any,
    config: "AgentConfig",
    profile: "FormatProfile | None" = None,
) -> str:
    """Build and cache the stable [MEMORY INDEX] for this battle."""
    if not getattr(config, "enable_memory", False):
        return ""
    if getattr(cm, "_memory_index_built", False):
        return getattr(cm, "memory_index", "")
    cm._memory_index_built = True
    cm.memory_index = ""
    cm._memory_reveal_state = _reveal_snapshot(battle)
    cm._memory_recalled_ids = set()
    try:
        profile = profile or getattr(cm, "profile", None)
        if profile is None:
            from ..formats import profile_for

            profile = profile_for(config.battle_format)
        scope = scope_from(profile, getattr(battle, "format", ""), config.default_regulation)
        root = ensure_scope_root(config, scope, getattr(battle, "format", ""))
        kb = getattr(cm, "kb", None)
        if kb is not None:
            seed.ensure_scope_seeded(root, scope, kb, profile, config)
        keys, ids, rosters = _preview_keys(battle, config, root)
        for species in rosters["opp_roster"]:
            if kb is not None:
                seed.ensure_species_scene(root, scope, species, kb, profile, config)
        _ensure_team_scene(
            root,
            scope,
            ids["opp"],
            roster=rosters["opp_roster"],
            archetype=ids["archetype"],
            archetype_score=ids.get("archetype_score", 0.0),
        )
        _ensure_our_scene(root, scope, ids["our"], rosters["our_roster"])

        groups: dict[str, list[str]] = {
            "team": [],
            "archetype": [],
            "species": [],
            "our": [],
        }
        limits = {"team": 90, "archetype": 90, "species": 70, "our": 80}
        for key in keys:
            found = scenes.scene_for_key(root, key)
            if found is None:
                continue
            kind, scene = found
            bucket = kind.rstrip("s") if kind in ("teams", "archetypes") else kind
            if bucket in groups:
                groups[bucket].append(
                    _scene_line(kind, key, scene, limit=limits[bucket])
                )
        if not any(groups.values()):
            return ""
        header = (
            "[MEMORY INDEX — historical experience, not current facts; "
            "live reveals always win]"
        )
        footer = "[READ] full profiles: memory_read(key); free search: memory_grep(query=...)"
        budget = max(80, int(getattr(config, "memory_index_chars_preview", 1200)))
        # Reserve a slice per dimension so a long archetype summary cannot
        # starve the species/our lines (the design's interleaving rule).
        alloc = {"team": 130, "archetype": 190, "species": 600, "our": 110}
        body: list[str] = [header]
        used = len(header) + len(footer) + 4
        for kind in ("team", "archetype", "species", "our"):
            group_used = 0
            for line in groups[kind]:
                if group_used + len(line) + 1 > alloc[kind]:
                    break
                if used + len(line) + 1 > budget:
                    break
                body.append(line)
                used += len(line) + 1
                group_used += len(line) + 1
        if len(body) == 1:
            return ""
        body.append(footer)
        cm.memory_index = "\n".join(body)
        if getattr(cm, "attach_memory_index", None):
            cm.attach_memory_index(cm.memory_index)
        if getattr(cm, "log_event", None):
            cm.log_event(
                "memory_index",
                chars=len(cm.memory_index),
                keys=keys,
                archetype=ids.get("archetype"),
            )
        scenes.write_recall_log(
            root,
            phase="preview",
            keys=keys,
            facts=[],
            chars=len(cm.memory_index),
            season=scope.regulation,
        )
    except Exception:
        cm.memory_index = ""
        if getattr(cm, "attach_memory_index", None):
            cm.attach_memory_index("")
    return cm.memory_index


def _recency(row: Any) -> float:
    try:
        when = datetime.strptime(str(row.date), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return 0.5
    days = max(0, (date.today() - when).days)
    return math.exp(-days / HALF_LIFE_DAYS)


def _score(
    row: Any,
    phase: str,
    *,
    current_root_name: str,
    prior_weight: float,
    old_season_decay: float,
) -> float:
    category = _category(row.subject)
    if row.archetypes and row.subject == row.archetypes[0] and row.kind in (
        "trait",
        "lesson",
        "matchup",
    ):
        category = "archetype"
    base = 0.35 * (row.priority / 100.0)
    if row.maturity != "prior":
        base += 0.20 * math.log1p(max(0, row.evidence)) / 5.0
    base += 0.15 * _recency(row)
    base += 0.20 * PHASE_WEIGHT.get(phase, {}).get(row.kind, 0.5)
    base += 0.10 * DIMENSION_WEIGHT.get(category, {}).get(phase, 0.5)
    if row.maturity == "prior":
        base *= prior_weight
    if row.origin and row.origin != current_root_name:
        base *= old_season_decay
    return base


def _render_fact(row: Any, *, current_root_name: str) -> str:
    marker = ""
    if row.maturity == "prior":
        marker = f"[PRIOR:{row.source}] "
    elif row.origin and row.origin != current_root_name:
        marker = "[OLD SEASON] "
    if row.conflict:
        marker += "[CONFLICT] "
    return (
        f"- ({row.kind}, {row.subject}, m={row.maturity}, n={row.evidence}, p={row.priority}) "
        f"{marker}{row.content}"
    )


def recall_turn(battle: Any, cm: Any, config: "AgentConfig") -> str | None:
    """The dynamic [MEMORY RECALL] block for this decision."""
    if not getattr(config, "enable_memory", False):
        return None
    try:
        profile = getattr(cm, "profile", None)
        if profile is None:
            from ..formats import profile_for

            profile = profile_for(config.battle_format)
        scope = scope_from(profile, getattr(battle, "format", ""), config.default_regulation)
        root = ensure_scope_root(config, scope, getattr(battle, "format", ""))
        current = _reveal_snapshot(battle)
        previous = getattr(cm, "_memory_reveal_state", {}) or {}
        changed = [
            species for species, reveal in current.items()
            if previous.get(species) != reveal
        ]
        tactics = _tactics(battle)
        keys = [*changed, *tactics]
        if not keys:
            return None
        kb = getattr(cm, "kb", None)
        if kb is not None:
            for species in changed:
                seed.ensure_species_scene(root, scope, species, kb, profile, config)
        roots = [root]
        if getattr(config, "allow_previous_seasons", True):
            roots.extend(previous_season_roots(config, scope, limit=2))
        rows = facts.read_facts_for_keys(roots, keys, season=scope.regulation)
        if not rows:
            return None
        current_name = root.name
        recalled = getattr(cm, "_memory_recalled_ids", set())
        tag = str(getattr(battle, "battle_tag", "") or "")
        filtered = [
            row for row in rows
            if row.maturity != "ref"
            and row.kind in PHASE_KINDS["turn"]
            and row.battle != tag
            and row.id not in recalled
        ]
        if not filtered:
            return None
        filtered.sort(
            key=lambda row: (
                -_score(
                    row,
                    "turn",
                    current_root_name=current_name,
                    prior_weight=float(getattr(config, "prior_rank_weight", 0.6)),
                    old_season_decay=float(getattr(config, "previous_season_decay", 0.6)),
                ),
                row.id,
            )
        )
        budget = max(80, int(getattr(config, "memory_recall_chars_turn", 400)))
        header = "[MEMORY RECALL — for this turn; live reveals win]"
        lines = [header]
        used = len(header) + 1
        chosen = []
        for row in filtered:
            text = _render_fact(row, current_root_name=current_name)
            if used + len(text) + 1 > budget:
                continue
            lines.append(text)
            chosen.append(row)
            used += len(text) + 1
            if len(chosen) >= 6:
                break
        if not chosen:
            return None
        cm._memory_reveal_state = current
        recalled.update(row.id for row in chosen)
        cm._memory_recalled_ids = recalled
        if getattr(cm, "log_event", None):
            cm.log_event("memory_recall", phase="turn", keys=keys, chars=used)
        scenes.write_recall_log(
            root,
            phase="turn",
            keys=keys,
            facts=chosen,
            chars=used,
            season=scope.regulation,
        )
        return "\n".join(lines)
    except Exception:
        return None


def index_context(config: "AgentConfig", profile: "FormatProfile | None") -> str:
    """Static helper for tools/tests: the guide appended to the system prompt."""
    return MEMORY_TOOLS_GUIDE