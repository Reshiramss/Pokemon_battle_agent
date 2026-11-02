"""Smogon usage hook: offline snapshot loading, format-id derivation, and
the silent no-snapshot default (zero network in every test)."""

from __future__ import annotations

import asyncio
import json

from helpers import make_ctx, make_double_battle, make_kb

from pokeagent.config import AgentConfig
from pokeagent.hooks.smogon_usage_hook import SmogonUsageHook


def _chaos_payload() -> dict:
    """A minimal valid chaos-JSON snapshot with one known species."""
    return {
        "info": {
            "metagame": "gen9vgc2025regi",
            "cutoff": 0,
            "number of battles": 100,
        },
        "data": {
            "Torterra": {
                "Raw count": 10,
                "usage": 0.5,
                "Abilities": {"Overgrow": 5.0, "Shell Armor": 5.0},
                "Items": {"Leftovers": 10.0},
                "Moves": {"Earthquake": 8.0, "Protect": 2.0},
                "Spreads": {"Adamant:4/252/0/0/0/252": 10.0},
                "Teammates": {"Incineroar": 4.0},
                "Checks and Counters": {},
                "Tera Types": {"Grass": 10.0},
            }
        },
    }


def _write_snapshot(tmp_path) -> "object":
    # from_file takes the month from the parent dir name (fetch's layout)
    month_dir = tmp_path / "2026-10"
    month_dir.mkdir(parents=True)
    snapshot = month_dir / "gen9vgc2025regi-0.json"
    snapshot.write_text(json.dumps(_chaos_payload()), encoding="utf-8")
    return snapshot


def _run_hook(hook, battle):
    async def scenario():
        return await hook.run(make_ctx(battle, make_kb(), config=hook.config))

    return asyncio.run(scenario())


def test_snapshot_loads_and_renders(tmp_path):
    snapshot = _write_snapshot(tmp_path)
    config = AgentConfig(runs_dir=tmp_path / "runs", smogon_stats_snapshot=snapshot)
    hook = SmogonUsageHook(config)
    out = _run_hook(hook, make_double_battle())  # foes: torterra, incineroar

    # only Torterra is in the snapshot; abilities are normalized to 0.5 each
    assert out is not None and "Torterra" in out
    assert "50.0% usage" in out
    assert "earthquake 80%" in out
    assert "incineroar" not in out
    # stats loaded once — the second turn hits the in-memory copy
    assert _run_hook(hook, make_double_battle()) == out


def test_stats_format_never_uses_hidden_id():
    # default battle format is the hidden one: "hidden" is stripped
    config = AgentConfig()
    assert SmogonUsageHook(config).stats_format == "gen9vgc2025regi"
    # explicit override wins
    config = AgentConfig(smogon_stats_format="gen9vgc2025regi")
    assert SmogonUsageHook(config).stats_format == "gen9vgc2025regi"
    # non-hidden formats pass through untouched
    config = AgentConfig(battle_format="gen9randombattle")
    assert SmogonUsageHook(config).stats_format == "gen9randombattle"


def test_no_snapshot_and_no_fetch_stays_silent(tmp_path):
    config = AgentConfig(runs_dir=tmp_path / "runs")
    hook = SmogonUsageHook(config)
    assert _run_hook(hook, make_double_battle()) is None
    assert hook._fetch_failed  # sticky: no per-turn retry storm


def test_disabled_hook_is_dormant(tmp_path):
    snapshot = _write_snapshot(tmp_path)
    config = AgentConfig(
        runs_dir=tmp_path / "runs",
        enable_smogon_hook=False,
        smogon_stats_snapshot=snapshot,
    )
    assert _run_hook(SmogonUsageHook(config), make_double_battle()) is None


def test_broken_snapshot_silent_without_fetch(tmp_path):
    config = AgentConfig(
        runs_dir=tmp_path / "runs",
        smogon_stats_snapshot=tmp_path / "missing.json",
    )
    hook = SmogonUsageHook(config)
    assert _run_hook(hook, make_double_battle()) is None
    assert hook._fetch_failed
