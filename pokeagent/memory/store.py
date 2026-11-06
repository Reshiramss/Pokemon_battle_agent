"""Layered JSONL store with grep-style search and bounded files.

Layout (all local, gitignored with ``runs/``):

    <root>/species/<species>.jsonl
    <root>/teams/<team-key>.jsonl
    <root>/tactics/<tactic>.jsonl
    <root>/battles/<date>_<battle-tag>.jsonl
    <root>/index.json            # counts; rebuildable at any time

Every line carries provenance (``battle``, ``date``, ``turn``, ``source``),
so a hit can always be traced back to the battle it came from. Files are
capped: past the cap the oldest rows are dropped and a single
``[compacted older entries]`` marker is kept at the top.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .grep import Query
from .keys import sanitize_key

LEVELS = ("species", "teams", "tactics", "battles")
COMPACTED = "compacted"
CAP_MARKER = "[compacted older entries]"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _flat(text: str, limit: int = 400) -> str:
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


@dataclass
class MemoryLine:
    """One retrievable memory row; the JSONL line schema."""

    battle: str = ""
    date: str = ""
    turn: int | None = None
    level: str = ""
    key: str = ""
    text: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw) -> "MemoryLine | None":
        if not isinstance(raw, dict):
            return None
        text = raw.get("text")
        level = raw.get("level")
        key = raw.get("key")
        if not isinstance(text, str) or not text.strip():
            return None
        if level not in LEVELS or not isinstance(key, str) or not key:
            return None
        turn = raw.get("turn")
        if not isinstance(turn, int) or isinstance(turn, bool):
            turn = None
        return cls(
            battle=str(raw.get("battle") or ""),
            date=str(raw.get("date") or ""),
            turn=turn,
            level=level,
            key=key,
            text=text.strip(),
            source=str(raw.get("source") or ""),
        )

    def render(self) -> str:
        turn = f" T{self.turn}" if self.turn is not None else ""
        battle = f" ({self.battle})" if self.battle else ""
        return f"{self.date}{turn} {self.level}/{self.key}{battle}: {self.text}"


class MemoryStore:
    """Read/write access to one memory root."""

    def __init__(self, root: Path | str, *, lines_cap: int = 500):
        self.root = Path(root)
        self.lines_cap = max(int(lines_cap), 2)

    # --- paths -----------------------------------------------------------

    def _level_dir(self, level: str) -> Path:
        return self.root / level

    def _key_path(self, level: str, key: str) -> Path:
        return self._level_dir(level) / f"{sanitize_key(key)}.jsonl"

    def _paths(self, level: str, key: str | None) -> list[Path]:
        if key:
            path = self._key_path(level, key)
            return [path] if path.is_file() else []
        directory = self._level_dir(level)
        return sorted(directory.glob("*.jsonl")) if directory.is_dir() else []

    # --- reading ---------------------------------------------------------

    def _read_path(self, path: Path) -> list[MemoryLine]:
        out: list[MemoryLine] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = MemoryLine.from_dict(json.loads(line))
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if parsed is not None:
                        out.append(parsed)
        except OSError:
            return []
        return out

    def read(self, level: str, key: str | None = None) -> list[MemoryLine]:
        if level not in LEVELS:
            raise ValueError(f"unknown memory level {level!r}")
        lines: list[MemoryLine] = []
        for path in self._paths(level, key):
            lines.extend(self._read_path(path))
        return lines

    # --- writing ---------------------------------------------------------

    def append(
        self,
        level: str,
        key: str,
        text: str,
        *,
        battle: str = "",
        turn: int | None = None,
        source: str = "deterministic",
        date: str | None = None,
        dedupe: str = "battle",
    ) -> bool:
        """Append one row; returns False when it was empty or a duplicate.

        ``dedupe``: ``"text"`` collapses identical insights across battles,
        ``"battle"`` collapses only re-writes from the same battle (so
        repeated matchup results still accumulate), ``"none"`` appends.
        """
        if level not in LEVELS:
            raise ValueError(f"unknown memory level {level!r}")
        text = _flat(text)
        if not text:
            return False
        line = MemoryLine(
            battle=battle,
            date=date or _today(),
            turn=turn,
            level=level,
            key=sanitize_key(key),
            text=text,
            source=source,
        )
        path = self._key_path(level, key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = self._read_path(path)
            fingerprint = text.lower()
            for old in existing:
                if dedupe == "text" and old.text.lower() == fingerprint:
                    return False
                if (
                    dedupe == "battle"
                    and old.battle
                    and old.battle == battle
                    and old.text.lower() == fingerprint
                ):
                    return False
            if len(existing) + 1 > self.lines_cap:
                keep = existing[-(self.lines_cap - 1) :]
                marker = MemoryLine(
                    battle=battle,
                    date=line.date,
                    turn=None,
                    level=level,
                    key=line.key,
                    text=CAP_MARKER,
                    source=COMPACTED,
                )
                self._write(path, [marker, *keep, line])
            else:
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(line.to_dict(), ensure_ascii=False) + "\n")
        except OSError:
            return False
        return True

    def _write(self, path: Path, lines: list[MemoryLine]) -> None:
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line.to_dict(), ensure_ascii=False) + "\n")
        os.replace(tmp, path)

    # --- search ----------------------------------------------------------

    def search(
        self,
        query: str | None = "",
        *,
        level: str = "all",
        key: str | None = None,
        limit: int = 8,
        regex: bool = False,
    ) -> tuple[list[MemoryLine], int]:
        """Grepped rows (newest first, at most ``limit``) and total hits."""
        q = Query(query, regex=regex)
        levels = LEVELS if level in ("", "all", None) else (level,)
        if any(lv not in LEVELS for lv in levels):
            raise ValueError(f"unknown memory level {level!r}")
        limit = max(1, int(limit))
        hits: list[tuple[float, int, MemoryLine]] = []
        for lv in levels:
            for path in self._paths(lv, key):
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                for idx, line in enumerate(self._read_path(path)):
                    if line.source == COMPACTED:
                        continue
                    # key + text, like grepping a file's path and content
                    if q.matches(f"{line.key} {line.text}"):
                        hits.append((mtime, idx, line))
        hits.sort(key=lambda item: (item[2].date, item[0], item[1]), reverse=True)
        return [line for _mtime, _idx, line in hits[:limit]], len(hits)

    def recent_for_keys(
        self,
        keys,
        *,
        levels=("species", "tactics", "teams"),
        per_key: int = 1,
    ) -> list[tuple[MemoryLine, int]]:
        """For each key: its newest rows and the total count (hook input)."""
        out: list[tuple[MemoryLine, int]] = []
        seen: set[str] = set()
        for key in keys:
            norm = sanitize_key(key)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            lines: list[MemoryLine] = []
            for lv in levels:
                lines.extend(self.read(lv, norm))
            lines = [l for l in lines if l.source != COMPACTED]
            if not lines:
                continue
            lines.sort(key=lambda l: l.date)
            for line in lines[-per_key:]:
                out.append((line, len(lines)))
        return out

    # --- stats / index ---------------------------------------------------

    def _scan_level(self, level: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for path in self._paths(level, None):
            rows = [l for l in self._read_path(path) if l.source != COMPACTED]
            if rows:
                counts[path.stem] = len(rows)
        return counts

    def stats(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for level in LEVELS:
            keys = self._scan_level(level)
            out[level] = {"total": sum(keys.values()), "keys": keys}
        return out

    def rebuild_index(self) -> None:
        """Rewrite ``index.json`` from the files — always safe to skip."""
        try:
            index = {
                "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "levels": self.stats(),
            }
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / "index.json.tmp"
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.root / "index.json")
        except OSError:
            pass