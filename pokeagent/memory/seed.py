"""Cold-start memory: F0 skeletons, F1 priors, F2 archetypes, our teams.

The floor is deliberate: a brand-new ruleset or a never-seen roster must still
produce useful context. Species skeletons are generated lazily for the exact
keys a battle asks about; archetypes and our teams are seeded once per scope.
Nothing here calls an LLM, and every generated scene is rebuildable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from ..knowledge.static import normalize_name
from ..stats import champion_raw_stats, standard_raw_stats
from .facts import write_scene_facts
from .keys import sanitize_key
from .scenes import (
    ensure_scene,
    load_scene,
    new_scene,
    rebuild_index,
    save_scene,
    scene_path,
    summarize,
)
from .scope import MemoryScope, ensure_scope_root

if TYPE_CHECKING:
    from ..config import AgentConfig
    from ..formats import FormatProfile
    from ..knowledge.static import StaticKB

_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
_CURATED_DIR = _DATA_ROOT / "seed" / "champions"
_CUSTOM_SEED = _DATA_ROOT / "memory_seed"

_ROLE_PRIORITY = {
    "weather_setter": 0,
    "speed_control": 1,
    "redirect": 2,
    "fake_out": 3,
    "pivot": 4,
    "setup": 5,
    "hazard": 6,
    "screen": 7,
    "recovery": 8,
    "support": 9,
    "wallbreaker": 10,
}

_MOVE_ROLES = {
    "fake_out": {"fakeout"},
    "pivot": {"uturn", "voltswitch", "partingshot", "flipturn", "teleport", "shedtail"},
    "speed_control": {
        "tailwind",
        "trickroom",
        "icywind",
        "electroweb",
        "thunderwave",
        "nuzzle",
        "glare",
        "quash",
        "afteryou",
    },
    "redirect": {"followme", "ragepowder"},
    "weather_setter": {
        "raindance",
        "sunnyday",
        "sandstorm",
        "snowscape",
        "chillyreception",
    },
    "setup": {
        "swordsdance",
        "nastyplot",
        "dragondance",
        "calmmind",
        "shellsmash",
        "bulkup",
        "irondefense",
        "quiverdance",
        "tidyup",
        "victorydance",
    },
    "recovery": {
        "recover",
        "roost",
        "synthesis",
        "moonlight",
        "morningsun",
        "shoreup",
        "slackoff",
        "softboiled",
        "strengthsap",
    },
    "hazard": {"stealthrock", "spikes", "toxicspikes", "stickyweb"},
    "screen": {"reflect", "lightscreen", "auroraveil"},
}

_ABILITY_ROLES = {
    "weather_setter": {"drizzle", "drought", "sandstream", "snowwarning"},
    "speed_control": {
        "swiftswim",
        "chlorophyll",
        "sandrush",
        "slushrush",
        "surgesurfer",
    },
    "support": {"prankster", "intimidate", "regenerator", "friendguard"},
}

_WEATHER_TACTIC = {
    "rain": {"tell_moves": ["raindance", "hurricane", "surf", "weatherball"],
             "tell_abilities": ["drizzle", "swiftswim"]},
    "sun": {"tell_moves": ["sunnyday", "solarbeam", "weatherball"],
            "tell_abilities": ["drought", "chlorophyll", "protosynthesis"]},
    "trick_room": {"tell_moves": ["trickroom"], "tell_abilities": []},
    "tailwind_offense": {"tell_moves": ["tailwind"], "tell_abilities": []},
    "hyper_offense": {"tell_moves": ["protect", "fakeout"], "tell_abilities": []},
    "balance": {"tell_moves": ["protect", "partingshot", "ragepowder"],
                "tell_abilities": []},
}

_CURATED_CACHE: dict[str, dict[str, Any]] = {}
_ARCHETYPE_CACHE: dict[str, list[dict[str, Any]]] = {}
_SEEDED: set[str] = set()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def curated_sets(profile: "FormatProfile") -> dict[str, dict[str, Any]]:
    """Champions curated random-battle sets, keyed by species; {} if absent."""
    name = "doubles-sets.json" if (profile.game_type or "doubles") == "doubles" else "sets.json"
    candidates = [
        _CURATED_DIR / name,
        _DATA_ROOT / "showdown_champions" / name.replace("-", "_"),
        _DATA_ROOT / "showdown_champions" / "random_doubles_sets.json",
        _DATA_ROOT / "showdown_champions" / "random_sets.json",
    ]
    for path in candidates:
        cache_key = str(path)
        if cache_key in _CURATED_CACHE:
            return _CURATED_CACHE[cache_key]
        data = _load_json(path)
        if isinstance(data, dict) and data:
            _CURATED_CACHE[cache_key] = data
            return data
        _CURATED_CACHE[cache_key] = {}
    return {}


def legal_species(kb: "StaticKB", profile: "FormatProfile") -> set[str]:
    """Species usable in the current scope.

    Champions flat rules ban restricted legendaries while the pokedex bundle
    still contains them; the curated random-battle pool is the better legal
    list when it is available.
    """
    curated = curated_sets(profile) if getattr(profile, "is_champions", False) else {}
    if curated:
        return {normalize_name(species) for species in curated}
    data = getattr(kb, "data", None)
    formats = getattr(data, "formats", None) or {}
    legal: set[str] = set()
    for species, entry in formats.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("tier") == "Illegal" or entry.get("isNonstandard"):
            continue
        legal.add(normalize_name(species))
    if legal:
        return legal
    gen_data = getattr(kb, "gen_data", None)
    pokedex = getattr(gen_data, "pokedex", None) or {}
    return {normalize_name(species) for species in pokedex}


def infer_roles(moves: Iterable[str], abilities: Iterable[str], base_stats: dict) -> list[str]:
    moves = {normalize_name(m) for m in moves or [] if m}
    abilities = {normalize_name(a) for a in abilities or [] if a}
    roles: set[str] = set()
    for role, triggers in _MOVE_ROLES.items():
        if moves & triggers:
            roles.add(role)
    for role, triggers in _ABILITY_ROLES.items():
        if abilities & triggers:
            roles.add(role)
    atk = int(base_stats.get("atk", 0) or 0)
    spa = int(base_stats.get("spa", 0) or 0)
    if max(atk, spa) >= 110 and not ({"speed_control", "redirect"} & roles):
        roles.add("wallbreaker")
    if ("protect" in moves or "detect" in moves) and max(atk, spa) < 100:
        roles.add("support")
    ordered = sorted(roles, key=lambda role: (_ROLE_PRIORITY.get(role, 99), role))
    return ordered[:4]


def speed_bounds(base_stats: dict, profile: "FormatProfile") -> dict[str, int]:
    try:
        if getattr(profile, "uses_stat_points", False):
            low = champion_raw_stats(base_stats, {"spe": 0}, nature="relaxed")["spe"]
            high = champion_raw_stats(base_stats, {"spe": 32}, nature="timid")["spe"]
        else:
            low = standard_raw_stats(
                base_stats, {"spe": 0}, ivs={"spe": 31}, nature="relaxed"
            )["spe"]
            high = standard_raw_stats(
                base_stats, {"spe": 252}, ivs={"spe": 31}, nature="timid"
            )["spe"]
    except Exception:
        return {}
    return {"min": int(low), "max": int(high)}


def archetype_table(profile: "FormatProfile", kb: "StaticKB") -> list[dict[str, Any]]:
    ruleset = getattr(profile, "mod", "gen9")
    if ruleset in _ARCHETYPE_CACHE:
        return _ARCHETYPE_CACHE[ruleset]
    custom = _load_json(_CUSTOM_SEED / ruleset / "archetypes.json")
    if isinstance(custom, list) and custom:
        _ARCHETYPE_CACHE[ruleset] = [entry for entry in custom if isinstance(entry, dict)]
        return _ARCHETYPE_CACHE[ruleset]
    source = _load_json(_DATA_ROOT / "team_comps.json")
    entries = source.get("archetypes") if isinstance(source, dict) else None
    if not isinstance(entries, list):
        entries = []
    _ARCHETYPE_CACHE[ruleset] = [entry for entry in entries if isinstance(entry, dict)]
    return _ARCHETYPE_CACHE[ruleset]


def _archetype_tags(species: str, profile: "FormatProfile", kb: "StaticKB") -> list[str]:
    tags = []
    for entry in archetype_table(profile, kb):
        core = {normalize_name(m) for m in entry.get("core_mons") or [] if m}
        if species in core:
            tags.append(str(entry.get("id") or ""))
    return [tag for tag in tags if tag]


def _usage_prior(config: "AgentConfig | None", species: str) -> dict[str, Any] | None:
    snapshot = getattr(config, "smogon_stats_snapshot", None) if config else None
    if not snapshot:
        return None
    raw = _load_json(Path(snapshot))
    if not isinstance(raw, dict):
        return None
    data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
    entry = data.get(species) if isinstance(data.get(species), dict) else None
    if entry is None:
        entry = data
    if not isinstance(entry, dict):
        return None
    out: dict[str, Any] = {}
    for label, field in (("Moves", "moves"), ("Items", "items"), ("Abilities", "abilities")):
        table = entry.get(label)
        if isinstance(table, dict) and table:
            ordered = sorted(table.items(), key=lambda item: -float(item[1] or 0))[:6]
            out[field] = [normalize_name(name) for name, _pct in ordered if name]
    return out or None


def ensure_species_scene(
    root: Path,
    scope: MemoryScope,
    species: str,
    kb: "StaticKB",
    profile: "FormatProfile",
    config: "AgentConfig | None" = None,
) -> dict[str, Any] | None:
    """Create the F0/F1 skeleton for a species if it does not exist yet."""
    species = normalize_name(species)
    if not species:
        return None
    path = scene_path(root, "species", species)
    existing = load_scene(path)
    if existing is not None:
        return existing
    entry = kb.species_data(species) if kb is not None else None
    if not isinstance(entry, dict) or not entry:
        return None
    base_stats = entry.get("baseStats") or {}
    types = [str(t) for t in entry.get("types") or [] if t]
    abilities_raw = entry.get("abilities") or {}
    abilities = [
        normalize_name(a) for a in abilities_raw.values() if a and normalize_name(a) != "noability"
    ]
    learnset = []
    try:
        learnset = list(kb.learnset(species) or [])
    except Exception:
        learnset = []

    curated = curated_sets(profile).get(species) if getattr(profile, "is_champions", False) else None
    prior_sets: list[dict[str, Any]] = []
    role_moves: list[str] = []
    if isinstance(curated, dict):
        for role_set in curated.get("sets") or []:
            if not isinstance(role_set, dict):
                continue
            moves = [normalize_name(m) for m in role_set.get("movepool") or [] if m]
            role_moves.extend(moves)
            prior_sets.append(
                {
                    "role": str(role_set.get("role") or ""),
                    "moves": moves[:10],
                    "abilities": [
                        normalize_name(a) for a in role_set.get("abilities") or [] if a
                    ],
                }
            )
    usage = _usage_prior(config, species)
    if usage:
        prior_sets.append(
            {
                "role": "usage",
                "moves": usage.get("moves") or [],
                "abilities": usage.get("abilities") or [],
                "items": usage.get("items") or [],
            }
        )
    move_pool = role_moves or learnset
    roles = infer_roles(move_pool, abilities, base_stats)
    bounds = speed_bounds(base_stats, profile)

    scene = new_scene(root, "species", species, scope, source="skeleton", insert=False)
    scene["types"] = types
    scene["abilities"] = abilities
    scene["base_stats"] = {k: int(v or 0) for k, v in base_stats.items()}
    scene["roles"] = roles
    scene["archetype_tags"] = _archetype_tags(species, profile, kb)
    scene["speed_bounds"] = bounds
    scene["prior_sets"] = prior_sets
    scene["narrative"] = (
        f"{str(entry.get('name') or species)}: {'/'.join(types) or 'unknown type'}; "
        f"abilities {', '.join(abilities) or '-'}; roles {', '.join(roles) or '-'}; "
        f"spe {bounds.get('min', '?')}-{bounds.get('max', '?')}"
    )
    scene["meta"]["summary"] = summarize(scene)
    save_scene(root, scene)
    return scene


def seed_archetypes(
    root: Path,
    scope: MemoryScope,
    kb: "StaticKB",
    profile: "FormatProfile",
) -> int:
    legal = legal_species(kb, profile)
    count = 0
    for entry in archetype_table(profile, kb):
        archetype_id = sanitize_key(str(entry.get("id") or entry.get("name") or ""))
        if not archetype_id or archetype_id == "unknown":
            continue
        core = [
            normalize_name(m)
            for m in entry.get("core_mons") or []
            if m and normalize_name(m) in legal
        ]
        path = scene_path(root, "archetype", archetype_id)
        scene = load_scene(path)
        if scene is None:
            if not core:
                continue
            scene = new_scene(root, "archetype", archetype_id, scope, source="prior", insert=False)
        if not scene.get("core_mons") and core:
            scene["core_mons"] = core
        if not scene.get("win_condition"):
            scene["win_condition"] = str(entry.get("plan") or "")
        if not scene.get("how_to_beat"):
            scene["how_to_beat"] = str(entry.get("weak_to") or "")
        tell = _WEATHER_TACTIC.get(archetype_id) or {}
        if tell.get("tell_moves") and not scene.get("tell_moves"):
            scene["tell_moves"] = tell["tell_moves"]
        if tell.get("tell_abilities") and not scene.get("tell_abilities"):
            scene["tell_abilities"] = tell["tell_abilities"]
        if not scene.get("narrative"):
            scene["narrative"] = str(entry.get("plan") or "")
        scene["meta"]["summary"] = summarize(scene)
        save_scene(root, scene)
        count += 1
    return count


def _our_team_specs(config: "AgentConfig", profile: "FormatProfile") -> list[tuple[str, str]]:
    """(our key, packed string) pairs from the team library or a raw config team."""
    from ..teams import TeamLibrary

    specs: list[tuple[str, str]] = []
    team = getattr(config, "team", None)
    if team and "|" in team:
        roster = _species_from_packed(team)
        from .keys import our_team_key

        return [(our_team_key(config, roster), team)]
    try:
        library = TeamLibrary.from_config(config)
    except Exception:
        return specs
    for name in library.names:
        entry = library.get(name)
        fmt = (entry.format or "").lower()
        if fmt in ("champions", "champion") and not getattr(profile, "is_champions", False):
            continue
        if fmt in ("standard", "gen9", "regular") and getattr(profile, "is_champions", False):
            continue
        if fmt and fmt not in ("champions", "champion", "standard", "gen9", "regular"):
            if fmt != getattr(profile, "mod", ""):
                continue
        specs.append(("team-" + sanitize_key(entry.name), entry.packed))
    return specs


def _species_from_packed(packed: str) -> list[str]:
    from poke_env.teambuilder import Teambuilder

    try:
        sets = Teambuilder.parse_packed_team(packed)
    except Exception:
        return []
    out = []
    for set_ in sets:
        for name in (getattr(set_, "species", None), getattr(set_, "nickname", None)):
            if name:
                out.append(normalize_name(name))
                break
    return sorted({s for s in out if s})


def seed_our_teams(
    root: Path,
    scope: MemoryScope,
    config: "AgentConfig",
    profile: "FormatProfile",
) -> int:
    count = 0
    for key, packed in _our_team_specs(config, profile):
        roster = _species_from_packed(packed)
        if not roster:
            continue
        path = scene_path(root, "our", key)
        scene = load_scene(path)
        if scene is None:
            scene = new_scene(root, "our", key, scope, source="library", insert=False)
        if not scene.get("roster"):
            scene["roster"] = roster
        scene["meta"]["source"] = "library"
        if not scene.get("narrative"):
            scene["narrative"] = "our roster: " + ", ".join(roster)
        scene["meta"]["summary"] = summarize(scene)
        save_scene(root, scene)
        count += 1
    return count


def seed_scope(
    root: Path,
    scope: MemoryScope,
    kb: "StaticKB",
    profile: "FormatProfile",
    config: "AgentConfig | None" = None,
    *,
    all_species: bool = False,
) -> dict[str, int]:
    ensure_scope_root(config, scope, getattr(profile, "format_id", "")) if config else root.mkdir(
        parents=True, exist_ok=True
    )
    counts = {"archetypes": seed_archetypes(root, scope, kb, profile), "our": 0, "species": 0}
    if config is not None:
        counts["our"] = seed_our_teams(root, scope, config, profile)
    if all_species:
        for species in sorted(legal_species(kb, profile)):
            if ensure_species_scene(root, scope, species, kb, profile, config):
                counts["species"] += 1
        write_all = True
    else:
        write_all = False
    if write_all or counts["archetypes"] or counts["our"]:
        rebuild_index(root)
    return counts


def ensure_scope_seeded(
    root: Path,
    scope: MemoryScope,
    kb: "StaticKB",
    profile: "FormatProfile",
    config: "AgentConfig | None" = None,
) -> None:
    key = str(root)
    if key in _SEEDED:
        return
    _SEEDED.add(key)
    seed_scope(root, scope, kb, profile, config, all_species=False)


def coverage(
    root: Path,
    kb: "StaticKB",
    profile: "FormatProfile",
    scope: MemoryScope,
    config: "AgentConfig | None" = None,
) -> dict[str, Any]:
    legal = legal_species(kb, profile)
    prepared = {
        path.stem
        for path in (Path(root) / "scenes" / "species").glob("*.json")
    } if (Path(root) / "scenes" / "species").is_dir() else set()
    # A species is usable at cold start if static data exists (lazy skeleton)
    # and, where the ruleset has curated sets, a prior is available.
    static_available = 0
    for name in legal:
        try:
            if kb.species_data(name):
                static_available += 1
        except Exception:
            continue
    curated_available = 0
    if getattr(profile, "is_champions", False):
        curated_available = len(set(curated_sets(profile)) & legal)
    with_prior = 0
    for name in prepared:
        scene = load_scene(Path(root) / "scenes" / "species" / f"{name}.json")
        if scene and scene.get("prior_sets"):
            with_prior += 1
    with_prior = max(with_prior, curated_available)
    archetypes_available = sum(
        1
        for entry in archetype_table(profile, kb)
        if any(
            normalize_name(m) in legal
            for m in entry.get("core_mons") or []
            if m
        )
    )
    our_available = bool(_our_team_specs(config, profile)) if config is not None else False
    total = max(1, len(legal))
    return {
        "legal_species": len(legal),
        "prepared_species": len(prepared),
        "skeleton_coverage": round(static_available / total, 3),
        "lazy_capable": 1.0,
        "prior_coverage": round(with_prior / total, 3),
        "archetype_coverage": round(archetypes_available / 6, 3),
        "our_team_coverage": 1.0 if our_available else 0.0,
        "cold_keys": sorted(legal - prepared)[:20],
        "cold_key_count": len(legal - prepared),
        "season": scope.regulation,
    }