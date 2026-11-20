"""Layered cross-battle memory.

- atoms: immutable per-battle facts (``atoms.py``);
- scenes: consolidated species/team/archetype/our profiles (``scenes.py``);
- facts: the deterministic grep projection of scenes (``facts.py``);
- recall: budgeted index + per-turn deltas (``recall.py``).

No embeddings and no index server: keys come from the battle, lines are
greppable, and the same inputs always produce the same output.
"""

from .facts import parse_fact_line, project_scene, read_fact_file
from .grep import Query, excerpt
from .keys import opponent_key, our_team_key, roster_signature, sanitize_key
from .schema import FactLine, format_fact_line, fact_id
from .scope import (
    MemoryScope,
    previous_season_roots,
    regulation_from_id,
    scope_for_config,
    scope_from,
    scope_root,
)
from .tactics import TacticSpec, detect_tactics, load_tactics

__all__ = [
    "FactLine",
    "MemoryScope",
    "Query",
    "TacticSpec",
    "detect_tactics",
    "excerpt",
    "fact_id",
    "format_fact_line",
    "load_tactics",
    "opponent_key",
    "our_team_key",
    "parse_fact_line",
    "previous_season_roots",
    "project_scene",
    "read_fact_file",
    "regulation_from_id",
    "roster_signature",
    "sanitize_key",
    "scope_for_config",
    "scope_from",
    "scope_root",
]