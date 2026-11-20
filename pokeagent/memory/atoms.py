"""Deterministic atomisation of a finished battle (L1 -> L2 evidence).

Every battle produces an immutable atom file: the first line is a JSON
``battle_meta``, the rest are one-line facts tagged with a scene kind. Atoms
never call a model. The merge into scenes is deterministic and failure-safe.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from ..knowledge.static import normalize_name
from . import scenes, seed
from .keys import opponent_key, our_team_key, sanitize_key
from .quality import quality_score
from .schema import FactLine, format_fact_line, now_date, now_iso
from .scope import MemoryScope, ensure_scope_root, scope_from

if TYPE_CHECKING:
    from ..config import AgentConfig
    from ..formats import FormatProfile

_LEAD_RE = re.compile(
    r"^(?:switch|drag|replace|send)\s+p(1|2)([ab])?:\s+(\S+)\s+(.+)$",
    re.IGNORECASE,
)


@dataclass
class Atom:
    scene_kind: str
    subject: str
    kind: str
    content: str
    priority: int = 50
    maturity: str = "obs"
    source: str = "obs"
    date: str = ""
    event_time: str = ""
    battle: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)


def _flat(text: str) -> str:
    return " ".join((text or "").split())


def _result(battle: Any) -> str:
    if getattr(battle, "won", False):
        return "win"
    if getattr(battle, "lost", False):
        return "loss"
    return "finished"


def _foe_nodes(battle: Any, cm: Any) -> list[Any]:
    # Live opponent objects accumulate reveals through the last turn, so they
    # are fresher than the KB snapshot taken at the last decision point.
    team = getattr(battle, "opponent_team", None)
    live = list(team.values()) if isinstance(team, dict) else list(team or [])
    if live:
        return live
    state = getattr(cm, "state", None) if cm is not None else None
    kb = getattr(state, "kb", None) if state is not None else None
    return list(getattr(kb, "them", {}).values()) if kb is not None else []


def _node_species(mon: Any) -> str:
    return normalize_name(str(getattr(mon, "species", "") or ""))


def _node_moves(mon: Any) -> list[str]:
    raw = getattr(mon, "revealed_moves", None)
    if raw is None:
        raw = getattr(mon, "moves", None)
    if isinstance(raw, dict):
        raw = list(raw.keys())
    return [normalize_name(m) for m in raw or [] if m][:8]


def _node_item(mon: Any) -> str:
    raw = getattr(mon, "item", None) or getattr(mon, "item_raw", None)
    if not isinstance(raw, str):
        return ""
    if raw.lower() in ("unknown_item", "unknownitem", "none", "unknownitem"):
        return ""
    return normalize_name(raw)


def _node_ability(mon: Any) -> str:
    raw = getattr(mon, "known_ability", None) or getattr(mon, "ability", None)
    return normalize_name(str(raw)) if raw else ""


def _node_tera(mon: Any) -> str:
    raw = getattr(mon, "tera_type", None)
    return normalize_name(str(raw)) if raw else ""


def _roster_map(team: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for mon in (team or {}).values() if isinstance(team, dict) else team or []:
        species = normalize_name(str(getattr(mon, "species", "") or ""))
        if not species:
            continue
        for name in (
            getattr(mon, "name", None),
            getattr(mon, "nickname", None),
            getattr(mon, "species", None),
        ):
            if name:
                mapping[normalize_name(str(name))] = species
    return mapping


def _leads_and_pick4(battle: Any, cm: Any) -> tuple[list[str], list[str], list[str], list[str]]:
    """(opp leads, opp picked, our leads, our picked) parsed from protocol text."""
    opp_map = _roster_map(getattr(battle, "opponent_team", {}))
    our_map = _roster_map(getattr(battle, "team", {}))
    opp_leads: list[str] = []
    opp_seen: list[str] = []
    our_leads: list[str] = []
    our_seen: list[str] = []
    records = list(getattr(cm, "records", None) or []) if cm is not None else []
    for record in records:
        for event in getattr(record, "events", None) or []:
            match = _LEAD_RE.match(str(event))
            if not match:
                continue
            side = match.group(1)
            nickname = normalize_name(match.group(3))
            details = str(match.group(4)).split(",", 1)[0].strip()
            detail_key = normalize_name(details)
            if side == "2":
                species = opp_map.get(nickname) or opp_map.get(detail_key) or detail_key
                if species and species not in opp_seen:
                    opp_seen.append(species)
                if species and len(opp_leads) < 2 and species not in opp_leads:
                    opp_leads.append(species)
            else:
                species = our_map.get(nickname) or our_map.get(detail_key) or detail_key
                if species and species not in our_seen:
                    our_seen.append(species)
                if species and len(our_leads) < 2 and species not in our_leads:
                    our_leads.append(species)
    return opp_leads, opp_seen, our_leads, our_seen


def _quality(battle: Any, cm: Any, config: "AgentConfig") -> dict[str, Any]:
    lesson, behavior, gate = _config_knobs(config)
    return quality_score(
        battle,
        cm,
        opponent_class=getattr(config, "opponent_class", "unknown"),
        quality_min_lesson=lesson,
        quality_min_behavior=behavior,
        fallback_hard_gate=gate,
    )


def _config_knobs(config: "AgentConfig") -> tuple[float, float, float]:
    # The v1 config keeps these inline rather than as public AgentConfig
    # fields; quality.py owns the defaults.
    return (
        float(getattr(config, "quality_min_lesson", 0.6) or 0.6),
        float(getattr(config, "quality_min_behavior", 0.5) or 0.5),
        float(getattr(config, "fallback_hard_gate", 0.25) or 0.25),
    )


def atomize_battle(
    battle: Any,
    cm: Any,
    config: "AgentConfig",
    profile: "FormatProfile",
    root: Path | None = None,
) -> tuple[dict[str, Any], list[Atom]]:
    """Build ``(battle_meta, atoms)`` for a finished battle."""
    scope = scope_from(profile, getattr(battle, "format", ""), config.default_regulation)
    tag = str(getattr(battle, "battle_tag", "") or "")
    result = _result(battle)
    turns = int(getattr(battle, "turn", 0) or 0)
    quality = _quality(battle, cm, config)
    date = now_date()
    event_time = now_iso()

    foes = [mon for mon in _foe_nodes(battle, cm) if _node_species(mon)]
    opp_roster = sorted({_node_species(mon) for mon in foes})
    our_roster = sorted(
        {
            normalize_name(str(getattr(p, "species", "") or ""))
            for p in getattr(battle, "team", {}).values()
        }
        - {""}
    )
    opp_key = opponent_key(opp_roster)
    our_key = our_team_key(config, our_roster)
    opp_leads, opp_pick, our_leads, our_pick = _leads_and_pick4(battle, cm)
    root = root or ensure_scope_root(config, scope, getattr(battle, "format", ""))
    best_arch = ""
    modes: list[str] = []
    try:
        from .classify import archetype_candidates

        candidates = archetype_candidates(opp_roster, root)
        best_arch = candidates[0][0] if candidates else ""
    except Exception:
        best_arch = ""
    try:
        from .tactics import tactic_ids

        modes = tactic_ids(foes)
    except Exception:
        modes = []

    atoms: list[Atom] = []

    for mon in foes:
        species = _node_species(mon)
        moves = _node_moves(mon)
        item = _node_item(mon)
        ability = _node_ability(mon)
        tera = _node_tera(mon)
        tags = seed._archetype_tags(species, profile, getattr(cm, "kb", None)) if cm is not None else []
        if moves or item or ability or tera:
            bits = []
            if moves:
                bits.append("moves: " + ",".join(moves))
            if item:
                bits.append("item: " + item)
            if ability:
                bits.append("ability: " + ability)
            if tera:
                bits.append("tera: " + tera)
            atoms.append(
                Atom(
                    scene_kind="species",
                    subject=species,
                    kind="set",
                    content="; ".join(bits),
                    priority=60,
                    date=date,
                    event_time=event_time,
                    battle=tag,
                    tags=tags,
                    metadata={"moves": moves, "item": item, "ability": ability, "tera": tera},
                    quality=quality,
                )
            )
        atoms.append(
            Atom(
                scene_kind="species",
                subject=species,
                kind="event",
                content=(
                    "appeared vs "
                    + (our_key)
                    + ("; lead" if species in opp_leads else "")
                    + f"; {result} in {turns} turns"
                ),
                priority=40,
                date=date,
                event_time=event_time,
                battle=tag,
                tags=tags,
                metadata={"leads": list(opp_leads), "pick4": opp_pick},
                quality=quality,
            )
        )

    atoms.append(
        Atom(
            scene_kind="team",
            subject=opp_key,
            kind="event",
            content=(
                "roster: "
                + ", ".join(opp_roster)
                + (f"; leads: {'+'.join(opp_leads)}" if opp_leads else "")
                + (f"; pick4: {','.join(opp_pick)}" if opp_pick else "")
            ),
            priority=70,
            date=date,
            event_time=event_time,
            battle=tag,
            metadata={
                "roster": opp_roster,
                "leads": [opp_leads] if opp_leads else [],
                "pick4": opp_pick,
                "archetype": best_arch,
                "modes": modes,
            },
            quality=quality,
        )
    )
    atoms.append(
        Atom(
            scene_kind="team",
            subject=opp_key,
            kind="matchup",
            content=f"vs {our_key}: {result} in {turns} turns",
            priority=75,
            date=date,
            event_time=event_time,
            battle=tag,
            metadata={"our_key": our_key, "result": result, "turns": turns},
            quality=quality,
        )
    )
    if best_arch:
        atoms.append(
            Atom(
                scene_kind="archetype",
                subject=best_arch,
                kind="matchup",
                content=f"vs {our_key}: {result} in {turns} turns",
                priority=75,
                date=date,
                event_time=event_time,
                battle=tag,
                metadata={"our_key": our_key, "result": result, "turns": turns},
                quality=quality,
            )
        )
    atoms.append(
        Atom(
            scene_kind="our",
            subject=our_key,
            kind="event",
            content=f"roster: {', '.join(our_roster)}; vs {opp_key}; {result}",
            priority=60,
            date=date,
            event_time=event_time,
            battle=tag,
            metadata={
                "roster": our_roster,
                "leads": [our_leads] if our_leads else [],
                "pick4": our_pick,
            },
            quality=quality,
        )
    )
    atoms.append(
        Atom(
            scene_kind="our",
            subject=our_key,
            kind="matchup",
            content=f"vs {opp_key}: {result}",
            priority=70,
            date=date,
            event_time=event_time,
            battle=tag,
            metadata={"opp_key": opp_key, "result": result},
            quality=quality,
        )
    )

    meta = {
        "type": "battle_meta",
        "battle": tag,
        "date": date,
        "event_time": event_time,
        "format": getattr(battle, "format", ""),
        "scope": asdict(scope),
        "player": str(getattr(battle, "player_username", "") or ""),
        **quality,
        "our_team": our_key,
        "our_roster": our_roster,
        "opp_roster": opp_roster,
        "result": result,
        "turns": turns,
    }
    return meta, atoms


def atomize_serialized(
    state: Any,
    *,
    result: str,
    turns: int,
    config: "AgentConfig",
    profile: "FormatProfile",
    event_time: str = "",
) -> tuple[dict[str, Any], list[Atom]]:
    """Backfill variant: atomise a persisted ``BattleState`` (no live battle)."""
    scope = scope_from(profile, getattr(state.meta, "format", ""), config.default_regulation)
    tag = str(getattr(state.meta, "battle_tag", "") or "")
    date = (event_time or now_iso())[:10] or now_date()
    event_time = event_time or now_iso()
    foes = list(getattr(state.kb, "them", {}).values())
    opp_roster = sorted({_node_species(mon) for mon in foes if _node_species(mon)})
    our_roster = sorted(
        {
            normalize_name(str(getattr(p, "species", "") or ""))
            for p in getattr(state.kb, "us", {}).values()
        }
        - {""}
    )
    opp_key = opponent_key(opp_roster)
    our_key = our_team_key(config, our_roster)
    quality = {
        "opponent_class": "unknown",
        "q": 0.6,
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "luck_factor": 0.0,
        "complete": True,
        "behavior_ok": True,
        "lesson_ok": False,
    }
    atoms: list[Atom] = []
    for mon in foes:
        species = _node_species(mon)
        if not species:
            continue
        moves = _node_moves(mon)
        item = _node_item(mon)
        ability = _node_ability(mon)
        tera = _node_tera(mon)
        tags = seed._archetype_tags(species, profile, getattr(state, "kb", None))
        if moves or item or ability or tera:
            bits = []
            if moves:
                bits.append("moves: " + ",".join(moves))
            if item:
                bits.append("item: " + item)
            if ability:
                bits.append("ability: " + ability)
            if tera:
                bits.append("tera: " + tera)
            atoms.append(
                Atom(
                    scene_kind="species",
                    subject=species,
                    kind="set",
                    content="; ".join(bits),
                    priority=60,
                    date=date,
                    event_time=event_time,
                    battle=tag,
                    tags=tags,
                    metadata={"moves": moves, "item": item, "ability": ability, "tera": tera},
                    quality=quality,
                )
            )
    atoms.append(
        Atom(
            scene_kind="team",
            subject=opp_key,
            kind="event",
            content="roster: " + ", ".join(opp_roster),
            priority=65,
            date=date,
            event_time=event_time,
            battle=tag,
            metadata={"roster": opp_roster, "leads": [], "pick4": []},
            quality=quality,
        )
    )
    atoms.append(
        Atom(
            scene_kind="team",
            subject=opp_key,
            kind="matchup",
            content=f"vs {our_key}: {result} in {turns} turns",
            priority=75,
            date=date,
            event_time=event_time,
            battle=tag,
            metadata={"our_key": our_key, "result": result, "turns": turns},
            quality=quality,
        )
    )
    atoms.append(
        Atom(
            scene_kind="our",
            subject=our_key,
            kind="event",
            content=f"roster: {', '.join(our_roster)}",
            priority=55,
            date=date,
            event_time=event_time,
            battle=tag,
            metadata={"roster": our_roster},
            quality=quality,
        )
    )
    atoms.append(
        Atom(
            scene_kind="our",
            subject=our_key,
            kind="matchup",
            content=f"vs {opp_key}: {result}",
            priority=70,
            date=date,
            event_time=event_time,
            battle=tag,
            metadata={"opp_key": opp_key, "result": result},
            quality=quality,
        )
    )
    meta = {
        "type": "battle_meta",
        "battle": tag,
        "date": date,
        "event_time": event_time,
        "format": getattr(state.meta, "format", ""),
        "scope": asdict(scope),
        "player": str(getattr(state.meta, "player_username", "") or ""),
        **quality,
        "our_team": our_key,
        "our_roster": our_roster,
        "opp_roster": opp_roster,
        "result": result,
        "turns": turns,
    }
    return meta, atoms


def format_atom_line(atom: Atom, season: str = "") -> str:
    row = FactLine(
        subject=atom.subject,
        kind=atom.kind,
        maturity=atom.maturity,
        priority=atom.priority,
        evidence=1 if atom.kind in ("set", "event", "matchup") else 0,
        date=atom.date,
        source=atom.source,
        season=season,
        battle=atom.battle,
        content=atom.content,
    )
    # ``scene_kind`` is not part of FactLine; append it as an ignored token so
    # a backfill can reconstruct which scene directory the row belongs to.
    line = format_fact_line(row)
    prefix, content = line.split(" | ", 1)
    return f"{prefix} sk={atom.scene_kind} | {content}"


def atoms_path(
    root: Path,
    battle_tag: str,
    date: str | None = None,
    label: str = "",
) -> Path:
    # AI-vs-AI shares the battle tag between both players; the label (player
    # username) keeps the two atom files from overwriting each other.
    parts = [date or now_date()]
    if label:
        parts.append(sanitize_key(label))
    parts.append(sanitize_key(battle_tag) or "battle")
    return Path(root) / "atoms" / f"{'_'.join(parts)}.ndjson"


def write_atoms(root: Path, meta: dict[str, Any], atoms: Iterable[Atom]) -> Path:
    path = atoms_path(
        root,
        str(meta.get("battle") or ""),
        str(meta.get("date") or ""),
        str(meta.get("player") or ""),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(meta, ensure_ascii=False, default=str)]
    season = str((meta.get("scope") or {}).get("regulation") or "")
    lines.extend(format_atom_line(atom, season) for atom in atoms)
    tmp = path.with_suffix(".ndjson.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_and_merge(
    battle: Any,
    cm: Any,
    config: "AgentConfig",
    profile: "FormatProfile",
) -> dict[str, int]:
    """Atomise, write the immutable file, then deterministically merge scenes."""
    if not getattr(config, "enable_memory", False):
        return {"atoms": 0, "scenes": 0}
    scope = scope_from(profile, getattr(battle, "format", ""), config.default_regulation)
    root = ensure_scope_root(config, scope, getattr(battle, "format", ""))
    kb = getattr(cm, "kb", None)
    if kb is not None:
        try:
            seed.ensure_scope_seeded(root, scope, kb, profile, config)
        except Exception:
            pass
    meta, atoms = atomize_battle(battle, cm, config, profile, root=root)
    path = write_atoms(root, meta, atoms)
    merged = scenes.merge_atoms(root, atoms, scope)
    scenes.rebuild_index(root)
    return {"atoms": len(atoms), "scenes": len(merged), "path": str(path)}