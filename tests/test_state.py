"""Tests for the explicit harness state (pause/refresh/persist contract).

The central guarantee: ``render_observed(state)`` is byte-identical to
``records.snapshot(battle)`` for normally observed battles. That is what
makes it safe to render the prompt from persisted state — the model sees
exactly what the postmortem can replay.
"""

from __future__ import annotations

import difflib
import json

from helpers import make_battle, make_double_battle
from poke_env.battle import PokemonType, Status

from pokeagent.config import AgentConfig
from pokeagent.context.manager import ContextManager
from pokeagent.context.records import snapshot
from pokeagent.state import BattleState, ReasoningNote, render_observed
from test_preview import make_preview_double_battle


def _assert_parity(battle) -> BattleState:
    state = BattleState.create(battle)
    want, got = snapshot(battle), render_observed(state)
    diff = "\n".join(
        difflib.unified_diff(
            want.splitlines(), got.splitlines(), "snapshot", "render", lineterm=""
        )
    )
    assert got == want, diff
    return state


def test_render_parity_singles():
    state = _assert_parity(make_battle())
    assert state.meta.format == "gen9randombattle"
    assert state.meta.doubles is False


def test_render_parity_doubles():
    state = _assert_parity(make_double_battle())
    assert state.meta.doubles is True
    assert state.ours.active == ["furret", "gyarados"]
    assert state.theirs.active == ["torterra", "incineroar"]
    assert "Speed order (fastest first):" in render_observed(state)


def test_render_parity_after_refresh_binds_observations():
    battle = make_double_battle()
    state = BattleState.create(battle)

    own = battle.team["p1: Furret"]
    own._current_hp = 85  # half HP
    foe = battle.opponent_team["p2: Torterra"]
    foe._status = Status.BRN
    foe._boosts["atk"] = 2
    foe._item = "leftovers"
    state.refresh(battle)

    # parity is re-established after mutation, not just at build time
    assert render_observed(state) == snapshot(battle)
    rendered = render_observed(state)
    assert "furret L50 50%" in rendered
    assert "[brn]" in rendered
    assert "boosts{atk+2}" in rendered
    assert "item:leftovers" in rendered
    # resources mirror the same observations
    assert state.resources.our_hp["furret"] == 0.5


def test_faint_and_tera_are_tracked():
    battle = make_double_battle()
    state = BattleState.create(battle)
    foe = battle.opponent_team["p2: Incineroar"]
    foe._status = Status.FNT
    state.refresh(battle)
    assert state.kb.them["incineroar"].fainted is True
    assert "incineroar" in state.resources.fainted
    assert render_observed(state) == snapshot(battle)

    # tera use accumulates and survives later refreshes
    live = battle._opponent_active_pokemon["p2a"]
    live._terastallized = True
    live._terastallized_type = PokemonType.from_name("Fire")
    state.refresh(battle)
    assert "torterra" in state.resources.tera_used
    live._terastallized = False
    state.refresh(battle)
    assert "torterra" in state.resources.tera_used  # never forgotten


def test_json_round_trip_preserves_render():
    battle = make_double_battle()
    state = BattleState.create(battle)
    state.plan.line = "lead Gyarados, chip Incineroar"
    state.plan.win_conditions = ["Gyarados sweeps after Incineroar falls"]
    state.resources.tera_used.append("gyarados")
    state.ledger.add_note(
        ReasoningNote(
            seq=1, turn=5, intent="chip the fire slot", claims=["incineroar is faster"]
        )
    )

    clone = BattleState.from_json(json.loads(json.dumps(state.to_json())))
    assert render_observed(clone) == render_observed(state)
    assert clone.meta.doubles is True
    assert clone.plan.line == state.plan.line
    assert clone.ledger.notes[0].claims == ["incineroar is faster"]
    assert clone.resources.tera_used == ["gyarados"]


def test_preview_state_hides_foe_hp_and_records_pick():
    battle = make_preview_double_battle()
    battle._teampreview = True
    state = BattleState.create(battle)

    assert state.meta.doubles is True
    assert state.meta.preview_pick == 4
    # the opponent stubs carry identity data but no HP (nothing revealed)
    for node in state.kb.them.values():
        assert node.hp_fraction is None
    assert "HP?" in render_observed(state)
    assert len(state.theirs.order) == 6


def test_preview_selection_markers_render():
    battle = make_preview_double_battle()
    battle._teampreview = True
    state = BattleState.create(battle)
    for i in (1, 2, 3, 4):
        list(battle.team.values())[i - 1]._selected_in_teampreview = True
    state.refresh(battle)
    state.meta.preview_pick = 4
    state.meta.preview_order = [1, 2, 3, 4]

    rendered = render_observed(state)
    assert "Your team (* active, + picked bench, - not picked):" in rendered
    assert "NOT PICKED" in rendered
    assert state.selected_known is True
    assert state.meta.preview_order == [1, 2, 3, 4]


def test_manager_renders_and_saves_state(tmp_path):
    battle = make_double_battle()
    config = AgentConfig(runs_dir=tmp_path / "runs")
    cm = ContextManager(battle_tag=battle.battle_tag, config=config, kb=None)
    cm.state = BattleState.create(battle)
    cm.battle_kb = cm.state.kb

    now = cm.build_now(battle)
    assert now.startswith(render_observed(cm.state))
    assert "Available actions:" in now
    assert cm.last_now_text == now

    cm.save_state()
    saved = json.loads((cm.run_dir / "state.json").read_text(encoding="utf-8"))
    assert saved["meta"]["doubles"] is True
    assert "kb" in saved and saved["kb"]["them"]