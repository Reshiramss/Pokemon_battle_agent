"""Consolidated scene documents (L2) and their deterministic merge rules.

Scenes are the materialised view of immutable atoms. This module owns:

- on-disk layout + atomic writes;
- set clustering (Jaccard on the move set + item/ability compatibility);
- behaviour/record counters for teams, archetypes and our own team;
- deterministic summaries used by the memory index;
- ``scene_index.json`` rebuild (heat comes from ``recall_log``).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..knowledge.static import normalize_name
from .facts import project_scene, scene_dir, write_scene_facts
from .keys import sanitize_key
from .schema import SCENE_KINDS, now_date, now_iso
from .scope import MemoryScope


def refresh_key(scene_kind: str, key: str) -> str:
    """Species keys are Showdown ids; team/our/archetype keys keep '-' and '_'."""
    return normalize_name(key) if scene_kind == "species" else sanitize_key(key)

SET_CLUSTER_MIN = 0.6  # a 4-move set sharing 3 moves is the same shape
MAX_SET_CLUSTERS = 5


def scene_path(root: Path, scene_kind: str, key: str) -> Path:
    return scene_dir(root, scene_kind) / f"{key}.json"


def load_scene(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and data.get("key"):
        return data
    return None


def save_scene(root: Path, scene: dict[str, Any]) -> Path:
    kind = str(scene.get("kind") or "species")
    key = str(scene.get("key") or "unknown")
    path = scene_path(root, kind, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(scene, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    write_scene_facts(root, scene)
    return path


def _new_meta(scope: MemoryScope, *, source: str) -> dict[str, Any]:
    stamp = now_iso()
    return {
        "created": stamp,
        "updated": stamp,
        "last_event_time": "",
        "heat": 0,
        "summary": "",
        "season": scope.regulation,
        "source": source,
    }


def new_scene(
    root: Path,
    scene_kind: str,
    key: str,
    scope: MemoryScope,
    *,
    source: str = "obs",
    insert: bool = True,
) -> dict[str, Any]:
    key = refresh_key(scene_kind, key)
    base: dict[str, Any] = {
        "schema": 1,
        "kind": scene_kind,
        "key": key,
        "meta": _new_meta(scope, source=source),
    }
    if scene_kind == "species":
        base.update(
            {
                "types": [],
                "abilities": [],
                "base_stats": {},
                "roles": [],
                "archetype_tags": [],
                "sets": [],
                "prior_sets": [],
                "speed_bounds": {},
                "behavior": {},
                "counters": {},
                "hypotheses": [],
                "conflicts": [],
                "changes": [],
                "narrative": "",
                "lessons": [],
            }
        )
    elif scene_kind == "team":
        base.update(
            {
                "roster": [],
                "archetype": "",
                "archetype_score": 0.0,
                "modes": [],
                "leads": {},
                "pick4": {},
                "record": {},
                "per_member": {},
                "counterplan": "",
                "narrative": "",
                "lessons": [],
            }
        )
    elif scene_kind == "archetype":
        base.update(
            {
                "core_mons": [],
                "tell_moves": [],
                "tell_abilities": [],
                "typical_leads": [],
                "win_condition": "",
                "how_to_beat": "",
                "record": {},
                "narrative": "",
                "lessons": [],
            }
        )
    elif scene_kind == "our":
        base.update(
            {
                "roster": [],
                "record": {},
                "playstyle": {},
                "narrative": "",
                "lessons": [],
            }
        )
    else:  # pragma: no cover - callers use a known kind
        raise ValueError(f"unknown scene kind {scene_kind!r}")
    if insert:
        save_scene(root, base)
    return base


def ensure_scene(
    root: Path,
    scene_kind: str,
    key: str,
    scope: MemoryScope,
    *,
    source: str = "obs",
) -> dict[str, Any]:
    key = refresh_key(scene_kind, key)
    path = scene_path(root, scene_kind, key)
    scene = load_scene(path)
    if scene is not None:
        return scene
    return new_scene(root, scene_kind, key, scope, source=source)


def scene_for_key(root: Path, key: str) -> tuple[str, dict[str, Any]] | None:
    for scene_kind in SCENE_KINDS:
        path = scene_path(root, scene_kind, refresh_key(scene_kind, key))
        scene = load_scene(path)
        if scene is not None:
            return str(scene.get("kind") or scene_kind), scene
    return None


# --- deterministic summary ---------------------------------------------------


def _top_moves(cluster: dict[str, Any], limit: int = 4) -> str:
    moves = cluster.get("moves") or {}
    ordered = sorted(moves.items(), key=lambda item: -int(item[1] or 0))[:limit]
    return ", ".join(name for name, _count in ordered)


def _dominant(table: dict[str, Any]) -> str:
    if not isinstance(table, dict) or not table:
        return ""
    return str(max(table.items(), key=lambda item: int(item[1] or 0))[0])


def _record_text(record: dict[str, Any]) -> str:
    parts = []
    for key, score in sorted((record or {}).items()):
        if isinstance(score, dict):
            parts.append(f"{key} {int(score.get('w', 0))}-{int(score.get('l', 0))}")
    return "; ".join(parts)


def summarize(scene: dict[str, Any]) -> str:
    kind = str(scene.get("kind") or "")
    if kind == "species":
        types = "/".join(str(t) for t in scene.get("types") or [])
        abilities = ", ".join(str(a) for a in (scene.get("abilities") or [])[:2])
        roles = ", ".join(str(r) for r in (scene.get("roles") or [])[:2])
        sets = scene.get("sets") or []
        set_text = ""
        if sets:
            best = max(sets, key=lambda s: int(s.get("support") or 0))
            moves = _top_moves(best)
            item = _dominant(best.get("item") or {})
            set_text = f" · sets: {moves}" + (f" ({item})" if item else "")
            set_text += f" x{int(best.get('support') or 0)}"
        elif scene.get("prior_sets"):
            prior = scene["prior_sets"][0]
            set_text = " · prior: " + ", ".join(str(m) for m in (prior.get("moves") or [])[:3])
        bounds = scene.get("speed_bounds") or {}
        speed = ""
        if bounds:
            speed = " · spe " + "/".join(
                str(bounds[k]) for k in ("min", "max") if bounds.get(k)
            )
        text = " · ".join(part for part in (types, abilities, roles) if part)
        return (text + speed + set_text)[:240]
    if kind == "team":
        archetype = str(scene.get("archetype") or "unknown")
        score = float(scene.get("archetype_score") or 0.0)
        if score:
            archetype += f" ({score:.2f})"
        roster = ", ".join(str(m) for m in (scene.get("roster") or [])[:5])
        leads = scene.get("leads") or {}
        lead_text = ""
        if leads:
            lead, count = max(leads.items(), key=lambda item: int(item[1] or 0))
            lead_text = f" · leads {lead} x{count}"
        record = _record_text(scene.get("record") or {})
        base = f"{archetype} · roster: {roster}" if roster else archetype
        return (base + lead_text + (f" · {record}" if record else ""))[:240]
    if kind == "archetype":
        core = ", ".join(str(m) for m in (scene.get("core_mons") or [])[:5])
        beat = str(scene.get("how_to_beat") or "")
        return (("core: " + core) if core else "") + (f" · beat: {beat}" if beat else "")
    if kind == "our":
        roster = ", ".join(str(m) for m in (scene.get("roster") or [])[:5])
        record = _record_text(scene.get("record") or {})
        return (("roster: " + roster) if roster else "") + (f" · {record}" if record else "")
    return ""


# --- merging -----------------------------------------------------------------


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _bump(table: dict[str, int], value: str | None, amount: int = 1) -> None:
    if not value:
        return
    table[value] = int(table.get(value, 0)) + amount


def _merge_set(scene: dict[str, Any], atom: Any) -> None:
    meta = atom.metadata or {}
    moves = {normalize_name(m) for m in meta.get("moves") or [] if m}
    item = normalize_name(meta.get("item") or "")
    ability = normalize_name(meta.get("ability") or "")
    tera = normalize_name(meta.get("tera") or "")
    if not moves:
        return
    date = atom.date or now_date()
    for cluster in scene.setdefault("sets", []):
        old_moves = {normalize_name(m) for m in (cluster.get("moves") or {})}
        if _jaccard(old_moves, moves) < SET_CLUSTER_MIN:
            continue
        if item and item not in (cluster.get("item") or {}) and cluster.get("item"):
            continue
        if ability and ability not in (cluster.get("ability") or {}) and cluster.get("ability"):
            continue
        battles = cluster.setdefault("battles", [])
        if atom.battle and atom.battle in battles:
            return  # AI-vs-AI: both sides may observe the same set
        if atom.battle:
            battles.append(atom.battle)
            del battles[:-200]  # bound the evidence trail
        for move in moves:
            _bump(cluster.setdefault("moves", {}), move)
        _bump(cluster.setdefault("item", {}), item)
        _bump(cluster.setdefault("ability", {}), ability)
        _bump(cluster.setdefault("tera", {}), tera)
        cluster["support"] = int(cluster.get("support") or 0) + 1
        cluster["last_seen"] = date
        cluster["last_battle"] = atom.battle
        scene["meta"]["last_event_time"] = max(
            str(scene["meta"].get("last_event_time") or ""), atom.event_time or date
        )
        return
    scene.setdefault("sets", []).append(
        {
            "moves": {move: 1 for move in moves},
            "item": {item: 1} if item else {},
            "ability": {ability: 1} if ability else {},
            "tera": {tera: 1} if tera else {},
            "support": 1,
            "first_seen": date,
            "last_seen": date,
            "last_battle": atom.battle,
            "battles": [atom.battle] if atom.battle else [],
            "source": atom.source or "obs",
        }
    )
    scene["sets"].sort(key=lambda cluster: -int(cluster.get("support") or 0))
    del scene["sets"][MAX_SET_CLUSTERS:]
    scene["meta"]["last_event_time"] = max(
        str(scene["meta"].get("last_event_time") or ""), atom.event_time or date
    )


def _record_bump(record: dict[str, Any], key: str, result: str) -> None:
    if not key or result not in ("win", "loss"):
        return
    slot = record.setdefault(key, {"w": 0, "l": 0})
    slot["w" if result == "win" else "l"] = int(slot.get("w" if result == "win" else "l", 0)) + 1


def merge_atom(root: Path, atom: Any, scope: MemoryScope) -> dict[str, Any]:
    scene_kind = atom.scene_kind
    scene = ensure_scene(root, scene_kind, atom.subject, scope)
    meta = atom.metadata or {}

    if scene_kind == "species":
        if atom.kind == "set":
            _merge_set(scene, atom)
        elif atom.kind == "event":
            behavior = scene.setdefault("behavior", {})
            battles = behavior.setdefault("battles", [])
            if atom.battle and atom.battle in battles:
                return scene
            if atom.battle:
                battles.append(atom.battle)
                del battles[:-200]
            behavior["appearances"] = int(behavior.get("appearances") or 0) + 1
            if meta.get("leads") and atom.subject in (meta.get("leads") or []):
                behavior["lead_count"] = int(behavior.get("lead_count") or 0) + 1
            behavior["sample"] = int(behavior.get("appearances") or 0)
            if behavior["sample"]:
                behavior["lead_rate"] = round(
                    float(behavior.get("lead_count") or 0) / behavior["sample"], 3
                )
    elif scene_kind == "team":
        if meta.get("roster"):
            scene["roster"] = sorted(
                {normalize_name(s) for s in meta["roster"] if s}
            )
        if meta.get("archetype"):
            scene["archetype"] = str(meta["archetype"])
        for mode in meta.get("modes") or []:
            if mode not in scene.setdefault("modes", []):
                scene["modes"].append(mode)
        if atom.kind == "event":
            leads = scene.setdefault("leads", {})
            for lead_set in meta.get("leads") or []:
                if isinstance(lead_set, (list, tuple)):
                    key = "+".join(sorted(normalize_name(x) for x in lead_set if x))
                else:
                    key = str(lead_set)
                if key:
                    leads[key] = int(leads.get(key, 0)) + 1
            pick4 = scene.setdefault("pick4", {})
            pick = meta.get("pick4") or []
            if pick:
                key = ",".join(sorted(normalize_name(x) for x in pick if x))
                if key:
                    pick4[key] = int(pick4.get(key, 0)) + 1
        if atom.kind == "matchup":
            _record_bump(
                scene.setdefault("record", {}),
                str(meta.get("our_key") or ""),
                str(meta.get("result") or ""),
            )
    elif scene_kind == "archetype":
        if meta.get("core_mons") and not scene.get("core_mons"):
            scene["core_mons"] = [normalize_name(m) for m in meta["core_mons"] if m]
        if atom.kind == "matchup":
            _record_bump(
                scene.setdefault("record", {}),
                str(meta.get("our_key") or ""),
                str(meta.get("result") or ""),
            )
    elif scene_kind == "our":
        if meta.get("roster"):
            scene["roster"] = sorted({normalize_name(s) for s in meta["roster"] if s})
        if atom.kind == "matchup":
            _record_bump(
                scene.setdefault("record", {}),
                str(meta.get("opp_key") or meta.get("archetype") or ""),
                str(meta.get("result") or ""),
            )

    scene["meta"]["updated"] = now_iso()
    scene["meta"]["summary"] = summarize(scene)
    save_scene(root, scene)
    return scene


def merge_atoms(root: Path, atoms: Iterable[Any], scope: MemoryScope) -> list[dict[str, Any]]:
    touched: dict[tuple[str, str], dict[str, Any]] = {}
    for atom in atoms:
        if not getattr(atom, "subject", None):
            continue
        scene = merge_atom(root, atom, scope)
        touched[(scene["kind"], scene["key"])] = scene
    return list(touched.values())


# --- index --------------------------------------------------------------------


def _heat(root: Path) -> dict[str, int]:
    heat: dict[str, int] = {}
    path = Path(root) / "recall_log.ndjson"
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in entry.get("keys") or []:
                    heat[str(key)] = heat.get(str(key), 0) + 1
    except OSError:
        pass
    return heat


def rebuild_index(root: Path) -> dict[str, Any]:
    root = Path(root)
    heat = _heat(root)
    entries: list[dict[str, Any]] = []
    scenes = root / "scenes"
    if scenes.is_dir():
        for directory in sorted(scenes.iterdir()):
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.json")):
                scene = load_scene(path)
                if scene is None:
                    continue
                meta = scene.get("meta") or {}
                key = str(scene.get("key") or "")
                if not key:
                    continue
                entries.append(
                    {
                        "key": key,
                        "kind": scene.get("kind"),
                        "path": str(path.relative_to(root)),
                        "summary": meta.get("summary") or summarize(scene),
                        "heat": int(heat.get(key, meta.get("heat") or 0)),
                        "updated": meta.get("updated") or "",
                        "season": meta.get("season") or "",
                    }
                )
    index = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entries": entries,
    }
    path = root / "scene_index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return index


def write_recall_log(
    root: Path,
    *,
    phase: str,
    keys: Iterable[str],
    facts: Iterable[Any],
    chars: int,
    season: str = "",
) -> None:
    entry = {
        "date": now_date(),
        "phase": phase,
        "keys": [str(k) for k in keys],
        "fact_ids": [str(getattr(f, "id", "")) for f in facts],
        "chars": int(chars),
        "season": season,
    }
    path = Path(root) / "recall_log.ndjson"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass