"""Team library: named, validated packed teams for team-preview formats.

Built-in teams live in ``teams/*.txt`` (one packed team per file, with ``#``
comment headers carrying name/archetype/description metadata). Users can drop
additional files into an import directory (``config.teams_dir``) with the same
layout — the library merges both, with user teams shadowing built-ins by file
name. A raw packed string (anything containing ``|``) is also accepted
wherever a team name is, so teams never have to touch this library.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from poke_env.teambuilder import Teambuilder

BUILTIN_TEAMS_DIR = Path(__file__).resolve().parents[1] / "teams"


class RawTeambuilder(Teambuilder):
    """Yields the packed team verbatim.

    poke-env's ConstantTeambuilder (used when passing a raw string to a
    Player) round-trips every set through TeambuilderPokemon, whose packed
    property lower-cases ability slot codes — 'H' becomes 'h', which the
    Showdown server rejects as an invalid ability. Our team files are
    already validated, so they go through unchanged.
    """

    def __init__(self, packed: str):
        self._packed = packed

    def yield_team(self) -> str:
        return self._packed


@dataclass(frozen=True)
class TeamEntry:
    """One parsed team file: metadata plus the Showdown packed string."""

    name: str
    packed: str
    archetype: str | None = None
    description: str | None = None
    source: Path | None = None  # file the team came from (None for raw strings)


class TeamLibrary:
    """Reads and indexes packed team files from one or more directories."""

    def __init__(self, *dirs: Path):
        self._entries: dict[str, TeamEntry] = {}  # keyed by lower-cased name
        self._name_by_file: dict[str, str] = {}  # file stem -> entry key
        for d in dirs:
            if d is None:
                continue
            self._load_dir(Path(d))
        if not self._entries:
            raise ValueError("team library is empty: no team files found")

    @classmethod
    def from_config(cls, config) -> "TeamLibrary":
        """Built-in teams plus the user import directory, if configured."""
        dirs: list[Path] = [BUILTIN_TEAMS_DIR]
        if getattr(config, "teams_dir", None):
            user_dir = Path(config.teams_dir).expanduser()
            if user_dir.is_dir():
                dirs.append(user_dir)
        return cls(*dirs)

    def _load_dir(self, dirpath: Path) -> None:
        for path in sorted(dirpath.glob("*.txt")):
            entry = parse_team_file(path)
            if entry is None:
                continue
            # a file with the same stem in a later dir shadows the earlier
            # one (user import dir wins over built-ins)
            prev_key = self._name_by_file.get(path.stem)
            key = entry.name.lower()
            if prev_key is not None and prev_key != key:
                self._entries.pop(prev_key, None)
            self._entries[key] = entry
            self._name_by_file[path.stem] = key

    # --- accessors ---------------------------------------------------------

    @property
    def names(self) -> list[str]:
        return sorted(self._entries)

    def get(self, name: str) -> TeamEntry:
        try:
            return self._entries[name.lower()]
        except KeyError:
            raise KeyError(
                f"unknown team '{name}' (available: {', '.join(self.names)})"
            ) from None

    def random_pair(self) -> tuple[TeamEntry, TeamEntry]:
        """Two distinct teams, for AI-vs-AI battles."""
        if len(self._entries) < 2:
            raise ValueError("random_pair needs at least 2 teams in the library")
        a, b = random.sample(list(self._entries.values()), 2)
        return a, b

    def random_other(self, packed: str) -> TeamEntry:
        """A team different from ``packed`` (also accepts raw packed strings)."""
        candidates = [
            e for e in self._entries.values() if e.packed.strip() != packed.strip()
        ]
        if not candidates:
            raise ValueError("no different team available in the library")
        return random.choice(candidates)


def parse_team_file(path: Path) -> TeamEntry | None:
    """Parse one ``teams/*.txt`` file into a TeamEntry.

    Layout: ``# key: value`` header lines (name required, fallback = file
    stem), then a single packed team on the remaining non-comment lines.
    """
    name = path.stem
    archetype = None
    description = None
    packed_lines: list[str] = []
    for line in path.read_text(encoding="utf8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            key, sep, value = stripped.lstrip("#").strip().partition(":")
            if not sep:
                continue
            key, value = key.strip().lower(), value.strip()
            if key == "name":
                name = value
            elif key == "archetype":
                archetype = value
            elif key == "description":
                description = value
        else:
            packed_lines.append(stripped)
    if not packed_lines:
        return None
    return TeamEntry(
        name=name,
        packed="".join(packed_lines),
        archetype=archetype,
        description=description,
        source=path,
    )


def resolve_team(spec: str | None, library: TeamLibrary) -> str | None:
    """Turn a config/CLI team spec into a packed string.

    ``None`` -> None (random battles, no team needed). A string containing
    ``|`` is treated as a raw packed team; anything else is a library name.
    """
    if spec is None:
        return None
    if "|" in spec:
        return spec
    return library.get(spec).packed
