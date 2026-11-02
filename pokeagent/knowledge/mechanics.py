"""Curated game-mechanics knowledge base (``data/mechanics.json``).

The JSON is hand-maintained doubles/VGC knowledge: turn order, spread
moves, protect, redirection, weather, terrain, status, stat stages,
terastallization, doubles-only tools. This module loads it once and
offers fuzzy lookup by topic id, title or alias — the tool layer lets
the model ask for "speed" and land on ``turn_order``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mechanics.json"


def _norm(text: str) -> str:
    """lowercase, non-alphanumerics collapsed — 'Trick Room!' ~ 'trickroom'."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


@dataclass(frozen=True)
class MechanicTopic:
    id: str
    title: str
    summary: str
    details: tuple[str, ...]
    aliases: tuple[str, ...]

    def render(self) -> str:
        lines = [f"{self.title}: {self.summary}"]
        lines.extend(self.details)
        return "\n".join(lines)


class MechanicsKB:
    """Read-only view over mechanics.json with fuzzy topic matching."""

    def __init__(self, data: dict):
        self.topics: dict[str, MechanicTopic] = {}
        for topic_id, raw in data.get("topics", {}).items():
            self.topics[topic_id] = MechanicTopic(
                id=topic_id,
                title=raw["title"],
                summary=raw["summary"],
                details=tuple(raw.get("details", [])),
                aliases=tuple(raw.get("aliases", [])),
            )
        self.context_lines: dict[str, dict[str, str]] = data.get(
            "context_lines", {}
        )

    def context_line(self, kind: str, key: str) -> str | None:
        """A one-line reminder for an active weather/field (enum .name as key)."""
        return self.context_lines.get(kind, {}).get(key)

    def core_doubles_lines(self) -> list[str]:
        """The three lines every doubles prompt carries (topic summaries)."""
        return [
            f"{self.topics[t].title}: {self.topics[t].summary}"
            for t in ("turn_order", "spread_moves", "protect")
            if t in self.topics
        ]

    def match(self, query: str) -> MechanicTopic | None:
        """Fuzzy-match a free-text topic name; None when nothing fits."""
        q = _norm(query)
        if not q:
            return None
        for topic in self.topics.values():
            if q == _norm(topic.id) or q == _norm(topic.title):
                return topic
        for topic in self.topics.values():
            if any(q == _norm(a) for a in topic.aliases):
                return topic
        # substring fallthrough: "trick" lands on turn_order via its alias,
        # "boost" on stat_stages via its alias — ids/titles/aliases combined
        for topic in self.topics.values():
            if any(
                q in _norm(c) or _norm(c) in q
                for c in (topic.id, topic.title, *topic.aliases)
            ):
                return topic
        return None


@lru_cache(maxsize=1)
def get_mechanics_kb() -> MechanicsKB:
    """Singleton loader; a missing/corrupt file yields an empty KB (hook sleeps)."""
    try:
        with DATA_PATH.open(encoding="utf-8") as f:
            return MechanicsKB(json.load(f))
    except (OSError, ValueError):
        return MechanicsKB({})
