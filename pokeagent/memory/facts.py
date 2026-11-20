"""Scene JSON -> grep-able fact lines, and the read side of the projection.

Facts are a deterministic projection, never a second hand-maintained source:
the same scene JSON always produces the same lines. Writers call
``write_scene_facts`` after a merge; the whole layer can be rebuilt from
``scenes/`` with ``rebuild_facts``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from .schema import (
    FactLine,
    SCENE_KINDS,
    fact_id,
    format_fact_line,
    parse_fact_line,
)

FACT_KIND_DIRS = {
    "species": "species",
    "team": "teams",
    "archetype": "archetypes",
    "our": "our",
}
SCENE_DIR = {
    "species": "species",
    "team": "teams",
    "archetype": "archetypes",
    "our": "our",
}


def scene_dir(root: Path, scene_kind: str) -> Path:
    return Path(root) / "scenes" / SCENE_DIR.get(scene_kind, scene_kind)


def facts_dir(root: Path, scene_kind: str) -> Path:
    return Path(root) / "facts" / FACT_KIND_DIRS.get(scene_kind, scene_kind)


def _scene_kind_of(scene: dict) -> str:
    kind = str(scene.get("kind") or "")
    return kind if kind in SCENE_DIR else "species"


def _rank_set(support: int) -> int:
    return min(90, 40 + 10 * max(0, int(support or 0)))


def _fact(
    scene: dict,
    *,
    kind: str,
    content: str,
    priority: int,
    maturity: str = "obs",
    source: str | None = None,
    evidence: int = 0,
    date: str = "",
    battle: str = "",
    archetypes: list[str] | None = None,
    roles: list[str] | None = None,
    conflict: bool = False,
    signature: str | None = None,
) -> FactLine:
    key = str(scene.get("key") or "")
    meta = scene.get("meta") or {}
    source = source or str(meta.get("source") or "obs")
    if source in ("skeleton", "curated", "usage"):
        source = "prior"
    date = date or str(meta.get("last_event_time") or "")[:10]
    if not date:
        date = str(meta.get("updated") or "")[:10]
    return FactLine(
        id=fact_id(kind, key, signature or content),
        subject=key,
        kind=kind,
        maturity=maturity,
        archetypes=list(archetypes or []),
        roles=list(roles or []),
        priority=priority,
        evidence=evidence,
        date=date,
        source=source,
        season=str(meta.get("season") or ""),
        battle=battle,
        conflict=conflict,
        content=content,
    )


def _move_text(values) -> str:
    if isinstance(values, dict):
        values = list(values.keys())
    return ",".join(str(v) for v in values or [] if v)


def _record_text(record: dict) -> str:
    out = []
    for key, score in (record or {}).items():
        if not isinstance(score, dict):
            continue
        out.append(f"vs {key} {int(score.get('w', 0))}-{int(score.get('l', 0))}")
    return "; ".join(sorted(out))


def project_scene(scene: dict) -> list[FactLine]:
    """Deterministic fact lines for one scene document."""
    kind = _scene_kind_of(scene)
    key = str(scene.get("key") or "")
    meta = scene.get("meta") or {}
    roles = [str(r) for r in scene.get("roles") or []]
    archetypes = [str(a) for a in scene.get("archetype_tags") or []]
    out: list[FactLine] = []

    if kind == "species":
        for prior in scene.get("prior_sets") or []:
            moves = _move_text(prior.get("moves"))
            abilities = _move_text(prior.get("abilities"))
            role = str(prior.get("role") or "").strip()
            text = f"{role + ': ' if role else ''}moves: {moves}"
            if abilities:
                text += f"; abilities: {abilities}"
            out.append(
                _fact(
                    scene,
                    kind="set",
                    content=text,
                    priority=35,
                    maturity="prior",
                    source="prior",
                    evidence=0,
                    archetypes=archetypes,
                    signature=f"prior|{role}|{moves}|{abilities}",
                )
            )
        for cluster in scene.get("sets") or []:
            moves = _move_text(cluster.get("moves"))
            item = _move_text(cluster.get("item"))
            ability = _move_text(cluster.get("ability"))
            tera = _move_text(cluster.get("tera"))
            text = f"moves: {moves}"
            if item:
                text += f"; item: {item}"
            if ability:
                text += f"; ability: {ability}"
            if tera:
                text += f"; tera: {tera}"
            support = int(cluster.get("support") or 0)
            signature = "|".join(
                (moves, item, ability, tera, str(cluster.get("source") or ""))
            )
            out.append(
                _fact(
                    scene,
                    kind="set",
                    content=text,
                    priority=_rank_set(support),
                    evidence=support,
                    maturity="obs",
                    source=str(cluster.get("source") or "obs"),
                    date=str(cluster.get("last_seen") or ""),
                    battle=str(cluster.get("last_battle") or ""),
                    archetypes=archetypes,
                    signature=signature,
                )
            )
        if roles:
            out.append(
                _fact(
                    scene,
                    kind="trait",
                    content="roles: " + ", ".join(roles),
                    priority=50,
                    maturity="prior",
                    source=str(meta.get("source") or "prior"),
                    archetypes=archetypes,
                    roles=roles,
                )
            )
        bounds = scene.get("speed_bounds") or {}
        if bounds:
            text = "; ".join(f"{k}: {v}" for k, v in bounds.items() if v)
            if text:
                out.append(
                    _fact(
                        scene,
                        kind="trait",
                        content="speed " + text,
                        priority=60,
                        maturity="prior",
                        source=str(meta.get("source") or "prior"),
                        archetypes=archetypes,
                        roles=roles,
                    )
                )
        behavior = scene.get("behavior") or {}
        sample = int(behavior.get("sample") or 0)
        if sample:
            lead = behavior.get("lead_rate")
            text = f"lead {int(round(float(lead or 0) * sample))}/{sample}"
            if behavior.get("protect_rate") is not None:
                text += f"; protect {float(behavior['protect_rate']):.2f}"
            out.append(
                _fact(
                    scene,
                    kind="trait",
                    content=text,
                    priority=55,
                    evidence=sample,
                    archetypes=archetypes,
                    roles=roles,
                )
            )
        for hypothesis in scene.get("hypotheses") or []:
            text = str(hypothesis.get("text") or "").strip()
            if text:
                out.append(
                    _fact(
                        scene,
                        kind="trait",
                        content=text,
                        priority=min(50, int(hypothesis.get("priority") or 40)),
                        maturity=str(hypothesis.get("maturity") or "hyp"),
                        source=str(hypothesis.get("source") or "model"),
                        evidence=int(hypothesis.get("support") or 0),
                        archetypes=archetypes,
                        roles=roles,
                    )
                )
        for conflict in scene.get("conflicts") or []:
            text = str(conflict.get("text") or "").strip()
            if text:
                out.append(
                    _fact(
                        scene,
                        kind="trait",
                        content=text,
                        priority=45,
                        maturity="hyp",
                        source="model",
                        archetypes=archetypes,
                        roles=roles,
                        conflict=True,
                    )
                )
        for lesson in scene.get("lessons") or []:
            text = str(lesson.get("text") or "").strip()
            if text:
                out.append(
                    _fact(
                        scene,
                        kind="lesson",
                        content=text,
                        priority=int(lesson.get("priority") or 50),
                        maturity=str(lesson.get("maturity") or "hyp"),
                        source=str(lesson.get("source") or "model"),
                        evidence=int(lesson.get("support") or 0),
                        archetypes=archetypes,
                        roles=roles,
                    )
                )
        counters = scene.get("counters") or {}
        if counters:
            text = _record_text({k: {"w": v.get("w", 0), "l": v.get("l", 0)} for k, v in counters.items() if isinstance(v, dict)})
            if text:
                out.append(
                    _fact(
                        scene,
                        kind="matchup",
                        content=text,
                        priority=65,
                        archetypes=archetypes,
                        roles=roles,
                    )
                )

    elif kind == "team":
        archetype = str(scene.get("archetype") or "")
        modes = [str(m) for m in scene.get("modes") or []]
        if archetype or modes:
            out.append(
                _fact(
                    scene,
                    kind="trait",
                    content="; ".join(
                        part for part in (
                            f"archetype: {archetype}" if archetype else "",
                            "modes: " + ", ".join(modes) if modes else "",
                        ) if part
                    ),
                    priority=70,
                    archetypes=[archetype] if archetype else [],
                    roles=modes,
                )
            )
        leads = scene.get("leads") or {}
        if leads:
            top = sorted(leads.items(), key=lambda item: -int(item[1] or 0))[:2]
            text = "; ".join(f"{lead} x{int(n)}" for lead, n in top)
            out.append(
                _fact(
                    scene,
                    kind="trait",
                    content="leads: " + text,
                    priority=60,
                    evidence=sum(int(n or 0) for _, n in leads.items()),
                    archetypes=[archetype] if archetype else [],
                    roles=modes,
                )
            )
        pick4 = scene.get("pick4") or {}
        if pick4:
            top = sorted(pick4.items(), key=lambda item: -int(item[1] or 0))[:2]
            text = "; ".join(f"{pick} x{int(n)}" for pick, n in top)
            out.append(
                _fact(
                    scene,
                    kind="trait",
                    content="pick4: " + text,
                    priority=50,
                    archetypes=[archetype] if archetype else [],
                    roles=modes,
                )
            )
        record = scene.get("record") or {}
        if record:
            out.append(
                _fact(
                    scene,
                    kind="matchup",
                    content=_record_text(record),
                    priority=75,
                    archetypes=[archetype] if archetype else [],
                    roles=modes,
                )
            )
        counterplan = str(scene.get("counterplan") or "").strip()
        if counterplan:
            out.append(
                _fact(
                    scene,
                    kind="lesson",
                    content=counterplan,
                    priority=85,
                    maturity="pat",
                    source="model",
                    archetypes=[archetype] if archetype else [],
                )
            )

    elif kind == "archetype":
        if scene.get("core_mons"):
            out.append(
                _fact(
                    scene,
                    kind="trait",
                    content="core: " + ", ".join(str(m) for m in scene["core_mons"]),
                    priority=70,
                    maturity="prior",
                    source=str(meta.get("source") or "prior"),
                    archetypes=[key],
                )
            )
        if scene.get("tell_moves") or scene.get("tell_abilities"):
            text = []
            if scene.get("tell_moves"):
                text.append("moves: " + ", ".join(str(m) for m in scene["tell_moves"]))
            if scene.get("tell_abilities"):
                text.append("abilities: " + ", ".join(str(a) for a in scene["tell_abilities"]))
            out.append(
                _fact(
                    scene,
                    kind="trait",
                    content="tells: " + "; ".join(text),
                    priority=65,
                    maturity="prior",
                    source=str(meta.get("source") or "prior"),
                    archetypes=[key],
                )
            )
        if scene.get("win_condition"):
            out.append(
                _fact(
                    scene,
                    kind="lesson",
                    content="win: " + str(scene["win_condition"]),
                    priority=80,
                    maturity="prior",
                    source=str(meta.get("source") or "prior"),
                    archetypes=[key],
                )
            )
        if scene.get("how_to_beat"):
            out.append(
                _fact(
                    scene,
                    kind="lesson",
                    content="beat: " + str(scene["how_to_beat"]),
                    priority=85,
                    maturity="prior",
                    source=str(meta.get("source") or "prior"),
                    archetypes=[key],
                )
            )
        if scene.get("record"):
            out.append(
                _fact(
                    scene,
                    kind="matchup",
                    content=_record_text(scene["record"]),
                    priority=75,
                    archetypes=[key],
                )
            )

    elif kind == "our":
        if scene.get("roster"):
            out.append(
                _fact(
                    scene,
                    kind="trait",
                    content="roster: " + ", ".join(str(m) for m in scene["roster"]),
                    priority=70,
                    maturity="prior",
                    source=str(meta.get("source") or "library"),
                )
            )
        playstyle = scene.get("playstyle") or {}
        for field_name in ("preferred_lead", "recurring_mistakes", "reliable_patterns"):
            values = playstyle.get(field_name) or []
            if values:
                label = field_name.replace("_", " ")
                out.append(
                    _fact(
                        scene,
                        kind="trait" if field_name != "recurring_mistakes" else "lesson",
                        content=label + ": " + "; ".join(str(v) for v in values),
                        priority=60,
                        maturity="hyp" if field_name == "recurring_mistakes" else "prior",
                        source="model" if field_name == "recurring_mistakes" else "library",
                    )
                )
        if scene.get("record"):
            out.append(
                _fact(
                    scene,
                    kind="matchup",
                    content=_record_text(scene["record"]),
                    priority=75,
                )
            )
        for lesson in scene.get("lessons") or []:
            text = str(lesson.get("text") or "").strip()
            if text:
                out.append(
                    _fact(
                        scene,
                        kind="lesson",
                        content=text,
                        priority=int(lesson.get("priority") or 50),
                        maturity=str(lesson.get("maturity") or "hyp"),
                        source=str(lesson.get("source") or "model"),
                        evidence=int(lesson.get("support") or 0),
                    )
                )

    # Deduplicate by id, keeping the strongest row.
    best: dict[str, FactLine] = {}
    for row in out:
        if not row.valid():
            continue
        current = best.get(row.id)
        if current is None or (row.priority, row.evidence) > (
            current.priority,
            current.evidence,
        ):
            best[row.id] = row
    return sorted(best.values(), key=lambda row: (-row.priority, -row.evidence, row.id))


def write_scene_facts(root: Path, scene: dict) -> Path:
    kind = _scene_kind_of(scene)
    key = str(scene.get("key") or "")
    path = facts_dir(root, kind) / f"{key}.ndjson"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [format_fact_line(row) for row in project_scene(scene)]
    tmp = path.with_suffix(".ndjson.tmp")
    tmp.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_fact_file(path: Path) -> list[FactLine]:
    rows: list[FactLine] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = parse_fact_line(line)
                if row is not None:
                    rows.append(row)
    except OSError:
        return []
    return rows


def read_facts_for_keys(
    roots: Iterable[Path],
    keys: Iterable[str],
    *,
    season: str = "",
) -> list[FactLine]:
    """Read every fact file for ``keys`` across one or more scope roots."""
    wanted = {str(k) for k in keys if k}
    if not wanted:
        return []
    out: list[FactLine] = []
    for root in roots:
        root = Path(root)
        for scene_kind in SCENE_KINDS:
            directory = facts_dir(root, scene_kind)
            if not directory.is_dir():
                continue
            for key in wanted:
                path = directory / f"{key}.ndjson"
                if not path.is_file():
                    continue
                for row in read_fact_file(path):
                    row.origin = root.name
                    if season and not row.season:
                        row.season = season
                    out.append(row)
    return out


def iter_fact_files(root: Path, layer: str = "facts") -> Iterable[Path]:
    base = Path(root) / layer
    if not base.is_dir():
        return []
    return sorted(base.glob("*/*.ndjson"))


def iter_atom_files(root: Path) -> Iterable[Path]:
    base = Path(root) / "atoms"
    if not base.is_dir():
        return []
    return sorted(base.glob("*.ndjson"))


def read_atoms_file(path: Path) -> tuple[dict[str, Any] | None, list[FactLine]]:
    """First line may be a JSON ``battle_meta``; the rest are fact rows."""
    meta: dict[str, Any] | None = None
    rows: list[FactLine] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                if text.startswith("{") and meta is None:
                    try:
                        meta = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    continue
                row = parse_fact_line(text)
                if row is not None:
                    rows.append(row)
    except OSError:
        return None, []
    return meta, rows


def rebuild_facts(root: Path) -> int:
    """Re-project every scene; returns the number of fact files written."""
    written = 0
    scenes = Path(root) / "scenes"
    if not scenes.is_dir():
        return 0
    for directory in sorted(scenes.iterdir()):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                scene = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(scene, dict) and scene.get("key"):
                write_scene_facts(root, scene)
                written += 1
    return written