"""Grep-style text queries shared by in-battle and cross-battle search.

Semantics deliberately mirror searching a codebase: case-insensitive;
whitespace-separated terms are ANDed (all must appear); ``regex=True``
switches the whole pattern to one regular expression, and an invalid
pattern raises ``ValueError`` so the caller can surface it as a tool error
instead of silently matching nothing.
"""

from __future__ import annotations

import re


class Query:
    """A compiled text query; ``None`` query means 'match everything'."""

    def __init__(self, pattern: str | None, *, regex: bool = False):
        self.pattern = (pattern or "").strip()
        self.regex = bool(regex) and bool(self.pattern)
        self._compiled: re.Pattern[str] | None = None
        self._needles: list[str] = []
        if not self.pattern:
            return
        if self.regex:
            try:
                self._compiled = re.compile(self.pattern, re.IGNORECASE)
            except re.error as e:
                raise ValueError(f"invalid regex {self.pattern!r}: {e}") from e
        else:
            self._needles = [t.lower() for t in self.pattern.split()]

    @property
    def active(self) -> bool:
        return bool(self.pattern)

    def matches(self, text: str | None) -> bool:
        if not self.pattern:
            return True
        text = text or ""
        if self._compiled is not None:
            return self._compiled.search(text) is not None
        low = text.lower()
        return all(needle in low for needle in self._needles)

    def find(self, text: str) -> int:
        """Index of the first match (for excerpt centering), -1 when none."""
        if not self.pattern:
            return 0
        text = text or ""
        if self._compiled is not None:
            match = self._compiled.search(text)
            return match.start() if match else -1
        low = text.lower()
        positions = [low.find(needle) for needle in self._needles]
        return -1 if any(p < 0 for p in positions) else min(positions)


def excerpt(text: str, query: Query, *, width: int = 200) -> str:
    """A hit-centred window of at most ``width`` characters."""
    flat = " ".join((text or "").split())
    if len(flat) <= width:
        return flat
    idx = query.find(flat)
    if idx < 0:
        idx = 0
    start = max(0, idx - width // 5)
    end = min(len(flat), start + width)
    return ("…" if start else "") + flat[start:end] + ("…" if end < len(flat) else "")