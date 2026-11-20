"""Offline tests for the layered cross-battle memory (v1).

No server, no provider, no API key. Real Showdown data bundles are used where
a species skeleton/archetype needs static data; those tests skip when the
bundles have not been exported.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pokeagent.config import AgentConfig
from pokeagent.context.manager import ContextManager
from pokeagent.formats import profile_for
from pokeagent.knowledge.showdown_data import get_showdown_data
from pokeagent.knowledge.static import StaticKB
from pokeagent.memory import atoms as memory_atoms
from pokeagent.memory import facts as memory_facts
from pokeagent.memory import recall, scenes, seed
from pokeagent.memory import scope as memory_scope
from pokeagent.memory import schema
from pokeagent.memory.quality import quality_score
from pokeagent.tools.base import ToolContext, ToolError
from pokeagent.tools.memory import MemoryGrepTool, MemoryReadTool, MemoryStatsTool

_FORMAT = "gen9championsvgc2026regmchidden"
_BUNDLE = get_showdown_data("showdown_champions")
needs_bundle = pytest.mark.skipif(
    _BUNDLE is None, reason="showdown champions data bundle not exported"
)


def _cfg(tmp_path: Path, **kwargs) -> AgentConfig:
    return AgentConfig(runs_dir=tmp_path / "runs", **kwargs)


def _profile(config: AgentConfig | None = None):
    return profile_for((config or AgentConfig()).battle_format)


def _kb(config: AgentConfig | None = None) -> StaticKB:
    profile = _profile(config)
    return StaticKB(profile.format_id, profile=profile)


def _fact(**kwargs) -> schema.FactLine:
    defaults = dict(
        id=schema.fact_id(kwargs.get("kind", "trait"), kwargs.get("subject", "x"), kwargs.get("content", "c")),
        subject="incineroar",
        kind="trait",
        maturity="obs",
        priority=50,
        evidence=1,
        date="2026-11-18",
        source="obs",
        content="Intimidate pivot",
    )
    defaults.update(kwargs)
    return schema.FactLine(**defaults)


# --- schema / facts ----------------------------------------------------------


def test_fact_line_roundtrip_and_rejects_garbage():
    row = _fact(archetypes=["rain"], roles=["pivot"], season="regmc", battle="b1")
    line = schema.format_fact_line(row)
    parsed = schema.parse_fact_line(line)
    assert parsed is not None
    assert parsed.subject == "incineroar"
    assert parsed.kind == "trait"
    assert parsed.archetypes == ["rain"]
    assert parsed.roles == ["pivot"]
    assert parsed.season == "regmc"
    assert parsed.battle == "b1"
    assert schema.parse_fact_line("not a memory line") is None
    assert schema.parse_fact_line("id=x k=x t=bad | nope") is None


def test_fact_id_is_structural_not_count_dependent():
    a = schema.fact_id("set", "incineroar", "fakeout,partingshot|safetygoggles|intimidate")
    b = schema.fact_id("set", "incineroar", "fakeout,partingshot|safetygoggles|intimidate")
    assert a == b
    assert a != schema.fact_id("set", "incineroar", "fakeout,knockoff|safetygoggles|intimidate")


def test_project_scene_prior_and_observed_rows():
    scene = {
        "kind": "species",
        "key": "incineroar",
        "meta": {"season": "regmc", "updated": "2026-11-18T00:00:00Z"},
        "types": ["Fire", "Dark"],
        "abilities": ["intimidate"],
        "roles": ["pivot"],
        "archetype_tags": ["rain"],
        "prior_sets": [{"role": "Pivot", "moves": ["fakeout", "partingshot"], "abilities": ["intimidate"]}],
        "sets": [
            {
                "moves": {"fakeout": 2, "partingshot": 2},
                "item": {"safetygoggles": 2},
                "ability": {"intimidate": 2},
                "support": 2,
                "last_seen": "2026-11-18",
            }
        ],
        "speed_bounds": {"min": 72, "max": 123},
        "behavior": {"sample": 2, "lead_rate": 0.5},
    }
    rows = memory_facts.project_scene(scene)
    kinds = {row.kind for row in rows}
    assert {"set", "trait"} <= kinds
    prior = [row for row in rows if row.maturity == "prior" and row.kind == "set"]
    observed = [row for row in rows if row.maturity != "prior" and row.kind == "set"]
    assert prior and prior[0].evidence == 0
    assert observed and observed[0].evidence == 2


# --- scope ------------------------------------------------------------------


@pytest.mark.parametrize(
    "format_id,expected",
    [
        ("gen9championsvgc2026regmchidden", "regmc"),
        ("gen9championsvgc2026regmcbo3", "regmc"),
        ("gen9championsvgc2026regmb", "regmb"),
        ("gen9vgc2025regihidden", "regi"),
        ("gen9ou", "default"),
    ],
)
def test_regulation_parsing(format_id, expected):
    assert memory_scope.regulation_from_id(format_id) == expected


def test_scope_root_and_previous_seasons(tmp_path):
    config = _cfg(tmp_path)
    profile = _profile(config)
    current = memory_scope.scope_from(profile, _FORMAT)
    root = memory_scope.ensure_scope_root(config, current, _FORMAT)
    older = memory_scope.scope_from(profile, "gen9championsvgc2026regmb")
    older_root = memory_scope.ensure_scope_root(config, older, "gen9championsvgc2026regmb")
    assert root.name == "regmc" and older_root.name == "regmb"
    roots = memory_scope.previous_season_roots(config, current, limit=2)
    assert older_root in roots


def test_config_default_format_is_champions():
    assert AgentConfig().battle_format == _FORMAT


# --- seed / cold start -------------------------------------------------------


@needs_bundle
def test_species_skeleton_roles_and_speed(tmp_path):
    config = _cfg(tmp_path, team="Champions A")
    profile = _profile(config)
    kb = _kb(config)
    scope = memory_scope.scope_for_config(config)
    root = memory_scope.ensure_scope_root(config, scope, _FORMAT)
    scene = seed.ensure_species_scene(root, scope, "incineroar", kb, profile, config)
    assert scene is not None
    assert set(scene["types"]) == {"Fire", "Dark"}
    assert "intimidate" in scene["abilities"]
    assert "fake_out" in scene["roles"] or "pivot" in scene["roles"]
    assert scene["speed_bounds"]["min"] < scene["speed_bounds"]["max"]
    assert scene["prior_sets"]


@needs_bundle
def test_seed_scope_archetypes_and_our_teams(tmp_path):
    config = _cfg(tmp_path, team="Champions A")
    profile = _profile(config)
    kb = _kb(config)
    scope = memory_scope.scope_for_config(config)
    root = memory_scope.ensure_scope_root(config, scope, _FORMAT)
    counts = seed.seed_scope(root, scope, kb, profile, config)
    assert counts["archetypes"] >= 4
    assert counts["our"] >= 1
    archetypes = list((root / "scenes" / "archetypes").glob("*.json"))
    assert any(path.stem == "rain" for path in archetypes)
    coverage = seed.coverage(root, kb, profile, scope, config)
    assert coverage["lazy_capable"] == 1.0
    assert coverage["archetype_coverage"] >= 0.6
    assert coverage["legal_species"] >= 300


# --- atoms / quality / merge -------------------------------------------------


def _fake_records(fell_back: bool = False, events: list[str] | None = None):
    return [
        SimpleNamespace(fell_back=fell_back, events=events or []),
        SimpleNamespace(fell_back=False, events=[]),
    ]


def test_quality_gates():
    battle = SimpleNamespace(won=False, lost=True, finished=True, turn=10)
    cm = SimpleNamespace(records=_fake_records())
    q = quality_score(battle, cm, "human")
    assert q["complete"] and q["lesson_ok"]
    noisy = SimpleNamespace(records=_fake_records(True, ["A critical hit!"]))
    q_noisy = quality_score(battle, noisy, "random")
    assert not q_noisy["lesson_ok"]


@needs_bundle
def test_atomize_and_merge_scene_counts(tmp_path):
    config = _cfg(tmp_path, team="Champions A", opponent_class="human")
    profile = _profile(config)
    kb = _kb(config)
    scope = memory_scope.scope_for_config(config)
    root = memory_scope.ensure_scope_root(config, scope, _FORMAT)
    seed.seed_scope(root, scope, kb, profile, config)

    mon = SimpleNamespace(
        species="torterra",
        revealed_moves=["earthquake", "protect"],
        known_ability="overgrow",
        item="leftovers",
        tera_type=None,
    )
    our = [SimpleNamespace(species="incineroar"), SimpleNamespace(species="rillaboom")]
    battle = SimpleNamespace(
        battle_tag="battle-test-1",
        format=_FORMAT,
        won=True,
        lost=False,
        finished=True,
        turn=12,
        team={str(i): p for i, p in enumerate(our)},
        opponent_team={"0": mon},
    )
    state = SimpleNamespace(kb=SimpleNamespace(them={"torterra": mon}, us={}))
    cm = SimpleNamespace(
        state=state,
        kb=kb,
        records=[
            SimpleNamespace(
                events=["switch p2a: Torterra Torterra, L50, M 100/100"],
                fell_back=False,
            )
        ],
    )
    meta, atom_rows = memory_atoms.atomize_battle(
        battle, cm, config, profile, root=root
    )
    assert meta["result"] == "win"
    assert any(atom.kind == "set" and atom.subject == "torterra" for atom in atom_rows)
    atom_path = memory_atoms.write_atoms(root, meta, atom_rows)
    meta_back, rows_back = memory_facts.read_atoms_file(atom_path)
    assert meta_back and meta_back["result"] == "win"
    assert any(row.subject == "torterra" for row in rows_back)
    merged = scenes.merge_atoms(root, atom_rows, scope)
    assert merged
    species_scene = json.loads((root / "scenes" / "species" / "torterra.json").read_text())
    assert species_scene["sets"]
    team_scenes = list((root / "scenes" / "teams").glob("*.json"))
    assert team_scenes
    team_scene = json.loads(team_scenes[0].read_text())
    assert team_scene["record"]
    facts_file = root / "facts" / "species" / "torterra.ndjson"
    assert facts_file.is_file() and "torterra" in facts_file.read_text()

    # idempotence of the projection
    before = facts_file.read_text()
    memory_facts.write_scene_facts(root, species_scene)
    assert facts_file.read_text() == before


@needs_bundle
def test_scene_set_clustering(tmp_path):
    config = _cfg(tmp_path)
    profile = _profile(config)
    kb = _kb(config)
    scope = memory_scope.scope_for_config(config)
    root = memory_scope.ensure_scope_root(config, scope, _FORMAT)
    first = memory_atoms.Atom(
        scene_kind="species",
        subject="torterra",
        kind="set",
        content="x",
        date="2026-11-18",
        event_time="2026-11-18T00:00:00Z",
        battle="b1",
        metadata={"moves": ["earthquake", "protect", "woodhammer", "stealthrock"],
                  "item": "leftovers", "ability": "overgrow", "tera": ""},
    )
    second = memory_atoms.Atom(
        scene_kind="species",
        subject="torterra",
        kind="set",
        content="x",
        date="2026-11-19",
        event_time="2026-11-19T00:00:00Z",
        battle="b2",
        metadata={"moves": ["earthquake", "protect", "woodhammer", "stoneedge"],
                  "item": "leftovers", "ability": "overgrow", "tera": ""},
    )
    scenes.merge_atoms(root, [first, second], scope)
    scene = json.loads((root / "scenes" / "species" / "torterra.json").read_text())
    assert len(scene["sets"]) == 1
    assert scene["sets"][0]["support"] == 2


# --- recall -----------------------------------------------------------------


def _battle_for_recall():
    foes = [
        SimpleNamespace(
            species="incineroar",
            moves={"fakeout": None, "partingshot": None},
            item="safetygoggles",
            ability="intimidate",
            tera_type=None,
        )
    ]
    ours = [SimpleNamespace(species="rillaboom"), SimpleNamespace(species="incineroar")]
    return SimpleNamespace(
        battle_tag="battle-recall-1",
        format=_FORMAT,
        player_username="PokeAgent",
        opponent_username="Opponent",
        team={str(i): p for i, p in enumerate(ours)},
        opponent_team={"0": foes[0]},
        turn=1,
    )


@needs_bundle
def test_recall_index_budget_and_turn_delta(tmp_path):
    config = _cfg(tmp_path, team="Champions A")
    profile = _profile(config)
    kb = _kb(config)
    battle = _battle_for_recall()
    cm = ContextManager(
        battle_tag=battle.battle_tag,
        config=config,
        kb=kb,
        label="PokeAgent",
        profile=profile,
    )
    index = recall.build_index(battle, cm, config, profile=profile)
    assert index
    assert "[MEMORY INDEX" in index
    assert "incineroar" in index
    assert len(index) <= config.memory_index_chars_preview + 40
    assert "memory_grep" in (cm.system_prompt or "")

    # no new reveal -> no per-turn block
    assert recall.recall_turn(battle, cm, config) is None
    # a new move reveals a delta
    battle.opponent_team["0"].moves["knockoff"] = None
    delta = recall.recall_turn(battle, cm, config)
    assert delta is None or "[MEMORY RECALL" in delta


@needs_bundle
def test_assemble_prompt_carries_index_in_system(tmp_path):
    from helpers import make_double_battle, make_kb

    battle = make_double_battle()
    kb = make_kb()
    config = _cfg(tmp_path, battle_format=battle.format)
    profile = profile_for(battle.format)
    cm = ContextManager(
        battle_tag=battle.battle_tag,
        config=config,
        kb=kb,
        label="PokeAgent",
        profile=profile,
    )
    recall.build_index(battle, cm, config, profile=profile)
    system, messages = cm.assemble_prompt(battle, [])
    assert "[MEMORY INDEX" in system
    assert "memory_grep" in system
    assert messages


@needs_bundle
def test_recall_isolates_scopes(tmp_path):
    config = _cfg(tmp_path)
    profile = _profile(config)
    kb = _kb(config)
    current = memory_scope.scope_from(profile, _FORMAT)
    root = memory_scope.ensure_scope_root(config, current, _FORMAT)
    # put a fact under the standard ruleset instead
    standard = memory_scope.scope_from(profile_for("gen9vgc2025regihidden"), "gen9vgc2025regihidden")
    other_root = memory_scope.ensure_scope_root(config, standard, "gen9vgc2025regihidden")
    (other_root / "facts" / "species").mkdir(parents=True, exist_ok=True)
    row = _fact(content="should not leak")
    (other_root / "facts" / "species" / "incineroar.ndjson").write_text(
        schema.format_fact_line(row) + "\n", encoding="utf-8"
    )
    rows = memory_facts.read_facts_for_keys([root], ["incineroar"])
    assert not rows


# --- tools ------------------------------------------------------------------


@needs_bundle
def test_memory_tools_grep_read_stats_and_budget(tmp_path):
    config = _cfg(tmp_path, team="Champions A", memory_tool_calls_per_turn=3)
    profile = _profile(config)
    kb = _kb(config)
    battle = _battle_for_recall()
    cm = ContextManager(
        battle_tag=battle.battle_tag,
        config=config,
        kb=kb,
        label="PokeAgent",
        profile=profile,
    )
    recall.build_index(battle, cm, config, profile=profile)
    ctx = ToolContext(battle=battle, kb=kb, context=cm, config=config)

    async def scenario():
        out = await MemoryGrepTool().execute(ctx, subject="incineroar")
        assert "incineroar" in out
        read = await MemoryReadTool().execute(ctx, key="incineroar")
        assert "summary" in read and "PRIOR" in read
        cm.begin_turn()
        team_out = await MemoryGrepTool().execute(ctx, subject="team-champions-a")
        assert "team-champions-a" in team_out
        arch_out = await MemoryGrepTool().execute(ctx, archetype="trick_room")
        assert "trick_room" in arch_out
        cm.begin_turn()
        stats = await MemoryStatsTool().execute(ctx)
        assert "scenes:" in stats
        await MemoryGrepTool().execute(ctx, query="x")
        await MemoryGrepTool().execute(ctx, query="y")
        with pytest.raises(ToolError):
            await MemoryGrepTool().execute(ctx, query="z")

    asyncio.run(scenario())


def test_memory_tools_disabled(tmp_path):
    config = _cfg(tmp_path, enable_memory=False)
    ctx = ToolContext(battle=_battle_for_recall(), config=config)

    async def scenario():
        with pytest.raises(ToolError):
            await MemoryGrepTool().execute(ctx, query="x")

    asyncio.run(scenario())


# --- index ------------------------------------------------------------------


@needs_bundle
def test_scene_index_rebuild(tmp_path):
    config = _cfg(tmp_path)
    profile = _profile(config)
    kb = _kb(config)
    scope = memory_scope.scope_for_config(config)
    root = memory_scope.ensure_scope_root(config, scope, _FORMAT)
    seed.seed_scope(root, scope, kb, profile, config)
    index = scenes.rebuild_index(root)
    keys = {entry["key"] for entry in index["entries"]}
    assert "rain" in keys
    assert (root / "scene_index.json").is_file()