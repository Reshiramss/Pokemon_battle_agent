"""Cross-battle memory (S4): grep, layered store, writer, hook, tools.

Everything runs offline in tmp dirs — no server, no network, no model.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from helpers import make_battle

from pokeagent.config import AgentConfig
from pokeagent.context.manager import ContextManager
from pokeagent.context.records import TurnRecord
from pokeagent.hooks.memory_hook import MemoryHook
from pokeagent.memory.grep import Query, excerpt
from pokeagent.memory.keys import opponent_key, our_team_key, roster_signature
from pokeagent.memory.store import CAP_MARKER, MemoryStore
from pokeagent.memory.tactics import detect_tactics, tactic_ids
from pokeagent.memory.writer import MemoryWriter
from pokeagent.state.model import Claim, ReasoningNote
from pokeagent.tools.base import ToolContext, ToolError
from pokeagent.tools.battle_query import BattleHistoryTool
from pokeagent.tools.memory import MemorySearchTool, MemoryStatsTool


def _cfg(tmp_path, **overrides):
    return AgentConfig(runs_dir=tmp_path / "runs", **overrides)


def _battle(*, foe=None, ours=None, won=True, turn=9, tag="battle-mem-1"):
    foe = foe or ["torterra"]
    ours = ours or ["incineroar"]
    return SimpleNamespace(
        battle_tag=tag,
        turn=turn,
        won=won,
        lost=not won,
        format="gen9vgc2025regihidden",
        team={
            f"p1{i}": SimpleNamespace(
                species=s, revealed_moves=[], known_ability=None, item=None,
                types=["FIRE"],
            )
            for i, s in enumerate(ours)
        },
        opponent_team={
            f"p2{i}": (
                m
                if isinstance(m, SimpleNamespace)
                else SimpleNamespace(
                    species=m, revealed_moves=[], known_ability=None,
                    item=None, types=["GRASS"],
                )
            )
            for i, m in enumerate(foe)
        },
    )


# --- grep -------------------------------------------------------------------


def test_query_and_semantics():
    assert Query("fake out").matches("The Fake Out user moved first")
    assert not Query("fake out perish").matches("Fake Out only")
    assert not Query("FAKE").matches("quiet turn")
    assert Query("fake.*out", regex=True).matches("fake out")
    with pytest.raises(ValueError):
        Query("(", regex=True)
    assert Query("").active is False
    assert Query("").matches("anything")


def test_excerpt_centres_on_hit():
    text = "x" * 400 + " torterra underreported " + "y" * 400
    out = excerpt(text, Query("torterra"), width=100)
    assert len(out) <= 102
    assert "torterra" in out
    assert out.startswith("…") and out.endswith("…")


# --- store ------------------------------------------------------------------


def test_store_append_dedupe_modes(tmp_path):
    store = MemoryStore(tmp_path / "m")
    assert store.append("species", "torterra", "slow wall", battle="b1")
    # same battle + text: dropped
    assert not store.append("species", "torterra", "slow wall", battle="b1")
    assert store.append("species", "torterra", "slow wall", battle="b2")
    # text dedupe collapses across battles
    assert not store.append(
        "species", "torterra", "slow wall", battle="b3", dedupe="text"
    )
    assert len(store.read("species", "torterra")) == 2


def test_store_cap_compaction_keeps_marker(tmp_path):
    store = MemoryStore(tmp_path / "m", lines_cap=3)
    for i in range(4):
        assert store.append("species", "torterra", f"row {i}", battle=f"b{i}")
    lines = store.read("species", "torterra")
    assert len(lines) == 4  # marker + 3 real rows
    assert lines[0].source == "compacted"
    assert lines[0].text == CAP_MARKER
    assert [l.text for l in lines[1:]] == ["row 1", "row 2", "row 3"]
    # search never returns the marker
    hits, total = store.search("row", level="species")
    assert total == 3
    assert all(h.source != "compacted" for h in hits)


def test_store_search_filters_and_newest_first(tmp_path):
    store = MemoryStore(tmp_path / "m")
    store.append("species", "a", "fake out every turn", battle="b1", date="2026-01-01")
    store.append("species", "b", "slow and bulky", battle="b2", date="2026-01-03")
    store.append("tactics", "perish_trap", "perish song trap wins", battle="b3",
                 date="2026-01-02")
    hits, total = store.search("")
    assert total == 3 and hits[0].date == "2026-01-03"
    hits, total = store.search("FAKE", level="species")
    assert total == 1 and hits[0].key == "a"
    hits, total = store.search("", level="tactics", key="perish_trap")
    assert total == 1 and hits[0].key == "perish_trap"
    with pytest.raises(ValueError):
        store.search("", level="nope")


def test_store_skips_corrupt_lines_and_stats_index(tmp_path):
    store = MemoryStore(tmp_path / "m")
    store.append("species", "torterra", "row", battle="b1")
    path = tmp_path / "m" / "species" / "torterra.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write("{not json\n")
        f.write(json.dumps({"level": "species", "key": "torterra"}) + "\n")
    assert [l.text for l in store.read("species", "torterra")] == ["row"]
    stats = store.stats()
    assert stats["species"]["total"] == 1
    store.rebuild_index()
    index = json.loads((tmp_path / "m" / "index.json").read_text())
    assert index["levels"]["species"]["keys"] == {"torterra": 1}


def test_roster_keys_are_stable():
    a = opponent_key(["Torterra", "Incineroar"])
    b = opponent_key(["incineroar", "torterra"])
    assert a == b
    assert roster_signature(["Torterra"]) == roster_signature(["torterra", "Torterra"])
    cfg = AgentConfig(team="Shadow Bolt")
    assert our_team_key(cfg, ["x"]) == "team-shadow-bolt"
    cfg2 = AgentConfig(team="Pikachu||Static|choiceband|...")
    assert our_team_key(cfg2, ["Inc ineroar"]).startswith("our-")


# --- tactics ----------------------------------------------------------------


def test_detect_tactics_modes():
    only_song = [{"moves": ["Perish Song"], "ability": "Damp"}]
    assert "perish_trap" not in tactic_ids(only_song)
    trap = [{"moves": ["perishsong"], "ability": "Shadow Tag"}]
    assert "perish_trap" in tactic_ids(trap)
    rain = [{"moves": [], "ability": "Drizzle"}]
    assert "rain" in tactic_ids(rain)
    none = [{"moves": ["tackle"], "ability": None, "item": None}]
    assert tactic_ids(none) == []


# --- writer -----------------------------------------------------------------


def test_write_battle_layers_and_index(tmp_path):
    foe = SimpleNamespace(
        species="Torterra", revealed_moves=["Perish Song"],
        known_ability="Shadow Tag", item="leftovers", types=["GRASS", "GROUND"],
    )
    battle = _battle(foe=[foe], ours=["Incineroar"])
    cfg = _cfg(tmp_path)
    counts = MemoryWriter(cfg).write_battle(battle, cm=None)
    assert counts == {"species": 1, "teams": 2, "tactics": 1, "battles": 1}
    store = MemoryStore(cfg.memory_path)
    species = store.read("species", "torterra")
    assert "perishsong" in species[0].text and "shadowtag" in species[0].text
    assert species[0].battle == "battle-mem-1"
    assert species[0].source == "deterministic"
    assert store.read("tactics", "perish_trap")
    assert store.read("teams", opponent_key(["torterra"]))
    assert store.read("teams", our_team_key(cfg, ["incineroar"]))
    assert (cfg.memory_path / "index.json").is_file()
    # deterministic rows are battle-deduped: re-running writes nothing new
    assert MemoryWriter(cfg).write_battle(battle, cm=None) == {
        "species": 0, "teams": 0, "tactics": 0, "battles": 0
    }


def test_write_digest_routing_and_cap(tmp_path):
    foe = SimpleNamespace(
        species="Torterra", revealed_moves=["perishsong"],
        known_ability="shadowtag", item=None, types=["GRASS"],
    )
    battle = _battle(foe=[foe])
    cfg = _cfg(tmp_path)
    writer = MemoryWriter(cfg)
    digest = "\n".join(
        [
            "- Torterra was the key threat on turn 3.",
            "- Torterra undersped everything relevant.",
            "- Torterra absorbed the fire damage.",
            "- Torterra kept the trap together.",
            "- Perish Song was under-prepared for.",
            "- Lead with Incineroar pressure next time.",
        ]
    )
    counts = writer.write_digest(battle, cm=None, digest=digest)
    store = MemoryStore(cfg.memory_path)
    species = store.read("species", "torterra")
    assert len(species) == 3  # per-key cap
    assert all(l.source == "post_battle" for l in species)
    assert species[0].turn == 3  # "turn 3" parsed
    assert len(store.read("tactics", "perish_trap")) == 1
    battles = store.read("battles")
    assert len(battles) == 1
    assert "Incineroar" in battles[0].text
    assert counts["species"] == 3 and counts["tactics"] == 1 and counts["battles"] == 1
    assert writer.write_digest(battle, cm=None, digest="") == {
        "species": 0, "teams": 0, "tactics": 0, "battles": 0
    }


# --- hook -------------------------------------------------------------------


def test_memory_hook_budget_and_keys(tmp_path):
    cfg = _cfg(tmp_path)
    store = MemoryStore(cfg.memory_path)
    battle = _battle(foe=[SimpleNamespace(
        species="torterra", revealed_moves=[], known_ability=None, item=None,
        types=["GRASS"],
    )])
    opp = opponent_key(["torterra"])
    ours = our_team_key(cfg, ["incineroar"])
    store.append("species", "torterra", "slow wall, fake out weakness", battle="b1")
    store.append("species", "torterra", "second older note", battle="b0")
    store.append("teams", opp, "their roster lost to our fire core", battle="b1")
    store.append("teams", ours, "keep incineroar healthy", battle="b1")
    store.append("species", "unrelated", "not this matchup", battle="b1")

    hook = MemoryHook(cfg, store=store)
    frag = asyncio.run(hook.run(SimpleNamespace(battle=battle)))
    assert frag is not None
    lines = frag.splitlines()
    assert 1 <= len(lines) <= cfg.memory_hook_max_lines
    assert len(frag) <= cfg.memory_hook_chars
    assert "torterra" in frag and "incineroar" in frag
    assert "unrelated" not in frag


def test_memory_hook_empty_and_disabled(tmp_path):
    cfg = _cfg(tmp_path)
    battle = _battle()
    assert asyncio.run(MemoryHook(cfg).run(SimpleNamespace(battle=battle))) is None
    disabled = _cfg(tmp_path, enable_memory=False)
    store = MemoryStore(disabled.memory_path)
    store.append("species", "torterra", "row", battle="b1")
    assert asyncio.run(
        MemoryHook(disabled, store=store).run(SimpleNamespace(battle=battle))
    ) is None


# --- memory tools -----------------------------------------------------------


def test_memory_search_and_stats_tools(tmp_path):
    cfg = _cfg(tmp_path)
    store = MemoryStore(cfg.memory_path)
    store.append("species", "torterra", "slow wall", battle="b1", date="2026-01-01")
    store.append("tactics", "perish_trap", "perish trap closed us out", battle="b2")
    ctx = ToolContext(battle=_battle(), config=cfg)

    out = asyncio.run(MemorySearchTool().execute(ctx, query="perish"))
    assert "perish_trap" in out and "1 match" in out
    out = asyncio.run(MemorySearchTool().execute(ctx, query="torterra", level="species"))
    assert "slow wall" in out
    out = asyncio.run(MemorySearchTool().execute(ctx, query="no-such-thing"))
    assert out.startswith("no memory rows")
    with pytest.raises(ToolError):
        asyncio.run(MemorySearchTool().execute(ctx, query="(", regex=True))

    stats = asyncio.run(MemoryStatsTool().execute(ctx))
    assert "total rows: 2" in stats and "species" in stats
    assert asyncio.run(MemoryStatsTool().execute(
        ToolContext(battle=_battle(), config=_cfg(tmp_path, memory_dir=tmp_path / "empty"))
    )).startswith("memory is empty")


def test_memory_tools_disabled(tmp_path):
    ctx = ToolContext(battle=_battle(), config=_cfg(tmp_path, enable_memory=False))
    with pytest.raises(ToolError):
        asyncio.run(MemorySearchTool().execute(ctx, query="x"))
    with pytest.raises(ToolError):
        asyncio.run(MemoryStatsTool().execute(ctx))


# --- in-battle grep (battle_history) ---------------------------------------


def _cm(tmp_path):
    return ContextManager(battle_tag="battle-1", config=_cfg(tmp_path))


def test_battle_history_grep_scopes_and_limit(tmp_path):
    cm = _cm(tmp_path)
    cm.records.append(
        TurnRecord(
            seq=1, turn=1, request="",
            events=["move p1a fakeout"],
            decision="move:fakeout", reason="pressure the torterra",
            cot="I think torterra will protect; fake out anyway",
        )
    )
    cm.records.append(
        TurnRecord(
            seq=2, turn=2, request="",
            decision="move:flamethrower", reason="burn the wall",
        )
    )
    cm.ledger.add_note(
        ReasoningNote(seq=1, turn=1, intent="fake out the torterra lead",
                      claims=["torterra is slow"])
    )
    cm.ledger.claims.append(
        Claim(subject="torterra", kind="speed",
              text="torterra under 100 speed", turn=1)
    )
    tool = BattleHistoryTool()
    ctx = ToolContext(battle=make_battle(), context=cm, config=cm.config)

    out = asyncio.run(tool.execute(ctx, pattern="torterra", limit=2))
    assert "#1 T1" in out and "[reason]" in out
    assert "showing 2" in out

    out = asyncio.run(tool.execute(ctx, pattern="torterra", scope="claims"))
    assert "#- T1 [claims]" in out and "under 100 speed" in out
    assert "[cot]" not in out

    out = asyncio.run(tool.execute(ctx, pattern="fake out torterra", scope="notes"))
    assert "[notes]" in out

    out = asyncio.run(tool.execute(ctx, pattern="torterra", turn=2))
    assert out.startswith("no match")

    with pytest.raises(ToolError):
        asyncio.run(tool.execute(ctx, pattern="x", scope="nope"))
    with pytest.raises(ToolError):
        asyncio.run(tool.execute(ctx, pattern="(", regex=True))


def test_battle_history_without_pattern_unchanged(tmp_path):
    cm = _cm(tmp_path)
    cm.records.append(
        TurnRecord(seq=1, turn=1, request="", decision="move:fakeout", reason="chip")
    )
    tool = BattleHistoryTool()
    ctx = ToolContext(battle=make_battle(), context=cm, config=cm.config)
    assert asyncio.run(tool.execute(ctx)) == "turn 1: move:fakeout — chip"