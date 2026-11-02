"""TeamLibrary: parsing, name resolution, pairing, raw packed passthrough."""

from __future__ import annotations

from pathlib import Path

import pytest

from pokeagent.teams import (
    BUILTIN_TEAMS_DIR,
    TeamEntry,
    TeamLibrary,
    parse_team_file,
    resolve_team,
)

SAMPLE = """\
# name: Test Squad
# archetype: test
# description: A test team.
Mon1|Mon1|Choice Band|Intimidate|Tackle,Protect|Adamant|252,252,0,0,0,4|31,31,31,31,31,31||50|255,,,,,Water
"""


@pytest.fixture
def library(tmp_path: Path) -> TeamLibrary:
    (tmp_path / "squad.txt").write_text(SAMPLE, encoding="utf8")
    (tmp_path / "empty.txt").write_text("# name: Empty\n", encoding="utf8")
    (tmp_path / "raw-no-header.txt").write_text(
        "Mon2|Mon2|||Splash|||,,,,,,||50|255\n", encoding="utf8"
    )
    return TeamLibrary(tmp_path)


def test_parse_team_file(tmp_path: Path):
    f = tmp_path / "t.txt"
    f.write_text(SAMPLE, encoding="utf8")
    entry = parse_team_file(f)
    assert entry.name == "Test Squad"
    assert entry.archetype == "test"
    assert entry.description == "A test team."
    assert entry.packed.startswith("Mon1|Mon1|")
    assert entry.source == f


def test_parse_name_falls_back_to_file_stem(tmp_path: Path):
    f = tmp_path / "no-name.txt"
    f.write_text("Mon|Mon|||Splash|||,,,,,,||50|255\n", encoding="utf8")
    assert parse_team_file(f).name == "no-name"


def test_empty_file_skipped(library: TeamLibrary):
    assert "Empty Squad" not in library.names
    # the file with no headers still loads, keyed by file stem
    assert "raw-no-header" in library.names


def test_get_case_insensitive(library: TeamLibrary):
    entry = library.get("test SQUAD")
    assert entry.name == "Test Squad"


def test_get_unknown_raises_with_choices(library: TeamLibrary):
    with pytest.raises(KeyError, match="raw-no-header"):
        library.get("nope")


def test_resolve_team_raw_string_passthrough(library: TeamLibrary):
    raw = "Mon|Mon|||Splash|||,,,,,,||50|255"
    assert resolve_team(raw, library) == raw
    assert resolve_team(None, library) is None


def test_resolve_team_name(library: TeamLibrary):
    assert resolve_team("Test Squad", library).startswith("Mon1|Mon1|")


def test_random_pair_and_other(tmp_path: Path):
    for i in range(3):
        (tmp_path / f"t{i}.txt").write_text(
            f"Mon{i}|Mon{i}|||Splash|||,,,,,,||50|255\n", encoding="utf8"
        )
    lib = TeamLibrary(tmp_path)
    a, b = lib.random_pair()
    assert a.packed != b.packed
    other = lib.random_other(a.packed)
    assert other.packed != a.packed


def test_random_pair_needs_two(tmp_path: Path):
    (tmp_path / "only.txt").write_text(
        "Mon|Mon|||Splash|||,,,,,,||50|255\n", encoding="utf8"
    )
    with pytest.raises(ValueError, match="at least 2"):
        TeamLibrary(tmp_path).random_pair()


def test_builtin_library_loads_and_has_vgc_teams():
    lib = TeamLibrary(BUILTIN_TEAMS_DIR)
    assert len(lib.names) >= 4
    for name in lib.names:
        entry = lib.get(name)
        assert isinstance(entry, TeamEntry)
        # 6 sets (6-pick-4 needs a full team of 6)
        assert entry.packed.count("]") == 5
        assert entry.archetype


def test_user_dir_shadows_builtin_by_file_name(tmp_path: Path):
    shadow = tmp_path / "shadowbolt.txt"
    shadow.write_text(SAMPLE, encoding="utf8")
    lib = TeamLibrary(BUILTIN_TEAMS_DIR, tmp_path)
    # the user file replaces the built-in shadowbolt.txt, so the built-in
    # "Shadow Bolt" team is gone and only the user's entry remains
    assert "shadow bolt" not in lib.names
    assert lib.get("test squad").source == shadow
    # other built-ins are untouched
    assert "rain rush" in lib.names
