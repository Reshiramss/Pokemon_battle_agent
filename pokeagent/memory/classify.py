"""Deterministic roster/archetype classification (no model, no embeddings).

A never-seen roster still needs a plan: score every archetype scene by the
share of its core Pokemon present in the preview, and rank existing team
scenes by roster Jaccard as a fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..knowledge.static import normalize_name
from .scenes import load_scene, scene_path


def normalize_roster(roster: Iterable[str]) -> list[str]:
    return sorted({normalize_name(s) for s in roster if s})


def roster_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    left = {normalize_name(s) for s in a if s}
    right = {normalize_name(s) for s in b if s}
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def archetype_candidates(
    roster: Iterable[str],
    root: Path,
    *,
    limit: int = 3,
    min_score: float = 0.34,
) -> list[tuple[str, float]]:
    """(archetype id, score) for the best matching seeded archetypes."""
    roster_set = {normalize_name(s) for s in roster if s}
    if not roster_set:
        return []
    directory = Path(root) / "scenes" / "archetypes"
    if not directory.is_dir():
        return []
    scored: list[tuple[str, float]] = []
    for path in sorted(directory.glob("*.json")):
        scene = load_scene(path)
        if not scene:
            continue
        core = [normalize_name(m) for m in scene.get("core_mons") or [] if m]
        if not core:
            continue
        overlap = len(roster_set & set(core))
        score = overlap / max(1, len(set(core)))
        if score >= min_score:
            scored.append((str(scene.get("key") or path.stem), round(score, 3)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[: max(1, int(limit))]


def similar_teams(
    roster: Iterable[str],
    root: Path,
    *,
    limit: int = 2,
    min_score: float = 0.5,
) -> list[tuple[str, float]]:
    """(team key, Jaccard) for previously seen rosters close to this one."""
    roster_set = {normalize_name(s) for s in roster if s}
    if not roster_set:
        return []
    directory = Path(root) / "scenes" / "teams"
    if not directory.is_dir():
        return []
    scored: list[tuple[str, float]] = []
    for path in sorted(directory.glob("*.json")):
        scene = load_scene(path)
        if not scene:
            continue
        score = roster_similarity(roster_set, scene.get("roster") or [])
        if score >= min_score:
            scored.append((str(scene.get("key") or path.stem), round(score, 3)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[: max(1, int(limit))]


def archetype_scene(root: Path, archetype_id: str) -> dict[str, Any] | None:
    return load_scene(scene_path(root, "archetype", archetype_id))