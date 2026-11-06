"""Cross-battle memory: grep-style storage, retrieval, and writing.

No embeddings, no index server. Memory is layered JSONL under
``<runs_dir>/_memory/`` — one file per key per layer — and retrieval is
plain text search with provenance (battle, date, turn), exactly like
grepping a codebase. See ``docs/harness_state.md`` §9.2.
"""

from .grep import Query, excerpt
from .keys import opponent_key, our_team_key, roster_signature, sanitize_key
from .store import LEVELS, MemoryLine, MemoryStore
from .tactics import TacticSpec, detect_tactics, load_tactics
from .writer import MemoryWriter

__all__ = [
    "LEVELS",
    "MemoryLine",
    "MemoryStore",
    "MemoryWriter",
    "Query",
    "TacticSpec",
    "detect_tactics",
    "excerpt",
    "load_tactics",
    "opponent_key",
    "our_team_key",
    "roster_signature",
    "sanitize_key",
]