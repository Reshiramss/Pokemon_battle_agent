"""FormatProfile: standard Gen 9 vs Pokémon Champions rules."""

from __future__ import annotations

from pokeagent.formats import profile_for


def test_standard_vgc_profile():
    p = profile_for("gen9vgc2025regihidden")
    assert p.mod == "gen9"
    assert p.data_key == "showdown"
    assert p.stat_model == "standard"
    assert p.uses_stat_points is False
    assert p.ev_limit == 510
    assert p.pp_cap is None
    assert p.allows_tera is True
    assert p.allows_mega is False
    assert p.game_type == "doubles"
    assert p.memory_key is None
    assert p.smogon_format == "gen9vgc2025regi"


def test_champions_vgc_profile():
    p = profile_for("gen9championsvgc2026regmc")
    assert p.mod == "champions"
    assert p.data_key == "showdown_champions"
    assert p.stat_model == "champions"
    assert p.uses_stat_points is True
    assert p.ev_limit == 66
    assert p.stat_point_cap == 32
    assert p.pp_cap == 20
    assert p.allows_tera is False
    assert p.allows_mega is True
    assert p.game_type == "doubles"
    assert p.memory_key == "champions"
    assert p.smogon_format is None


def test_champions_regmb_uses_its_own_bundle():
    p = profile_for("gen9championsvgc2026regmb")
    assert p.mod == "championsregmb"
    assert p.data_key == "showdown_championsregmb"
    assert p.stat_model == "champions"


def test_champions_random_uses_level_clause_stats():
    p = profile_for("gen9championsrandomdoublesbattle")
    assert p.mod == "champions"
    # the mod's random formats carry Level Clause Mod
    assert p.stat_model == "standard"
    assert p.ev_limit == 66


def test_custom_hidden_champion_falls_back_to_champion_rules():
    p = profile_for("gen9championsvgc2026regmchidden")
    assert p.mod == "champions"
    assert p.stat_model == "champions"
    assert p.allows_mega is True
    assert p.allows_tera is False


def test_display_name_and_rule_suffix_are_normalized():
    p = profile_for("[Gen 9 Champions] VGC 2026 Reg M-C@@@!teampreview")
    assert p.mod == "champions"
    assert p.stat_model == "champions"