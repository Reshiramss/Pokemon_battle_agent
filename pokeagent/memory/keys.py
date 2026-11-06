"""Stable, filename-safe keys for the layered memory store.

Species / tactic ids are already Showdown-like; team rows use the library
name when the agent was configured with one, otherwise a roster signature
(sorted species ids, hashed). Both sides derive the same key from the same
roster, so the hook can look up a matchup while the battle is still running.
"""

from __future__ import annotations

import hashlib
import re

from ..knowledge.static import normalize_name


def sanitize_key(raw: str) -> str:
    key = re.sub(r"[^a-z0-9_.-]+", "-", (raw or "").lower()).strip("-")
    return key or "unknown"


def roster_signature(species) -> str:
    ids = sorted({normalize_name(s) for s in species if s})
    digest = hashlib.sha1(",".join(ids).encode("utf-8")).hexdigest()
    return digest[:8]


def our_team_key(config, species) -> str:
    name = getattr(config, "team", None)
    if name and "|" not in name:  # a library name, not a raw packed string
        return "team-" + sanitize_key(name)
    return "our-" + roster_signature(species)


def opponent_key(species) -> str:
    return "opp-" + roster_signature(species)