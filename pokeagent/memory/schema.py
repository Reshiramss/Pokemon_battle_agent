"""Line-oriented memory schema shared by atoms, scenes and grep tools.

One memory row is one line::

    id=7c3a5f21 k=incineroar t=set m=obs a=rain p=70 n=4 d=2026-11-18 \
    s=obs se=regmc b=... | moves: fakeout,partingshot; item: safetygoggles

The row is deliberately boring and lossless: every field a recall needs to
filter, rank or explain a line lives before the ``|`` separator, and the
human-readable statement after it is capped by the writers. This is what
makes grep the retrieval mechanism instead of a semantic embedding index.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..knowledge.static import normalize_name
from .keys import sanitize_key

KINDS: tuple[str, ...] = ("set", "event", "trait", "lesson", "matchup")
MATURITIES: tuple[str, ...] = ("prior", "obs", "hyp", "pat", "rule", "ref")
SOURCES: tuple[str, ...] = (
    "obs",
    "model",
    "prior",
    "library",
    "transferred",
    "legacy",
)
SCENE_KINDS: tuple[str, ...] = ("species", "teams", "archetypes", "our")

_SEPARATOR = " | "
_FIELD_ORDER = ("id", "k", "t", "m", "a", "r", "p", "n", "d", "s", "se", "b", "x")
_FIELD_RE = re.compile(r"^([a-z]+)=(.*)$")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def now_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _one_line(text: str, limit: int = 200) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def normalize_subject(subject: str | None) -> str:
    """Species keys are Showdown ids; team/our/archetype keys keep '-' and '_'."""
    raw = str(subject or "").strip()
    if not raw:
        return ""
    if raw.startswith(("opp-", "team-", "our-")) or "_" in raw:
        return sanitize_key(raw)
    return normalize_name(raw)


def fact_id(kind: str, subject: str, signature: str) -> str:
    """Stable id from a structural signature (never from a changing count).

    For a set cluster the signature is the sorted moves + item + ability, so
    support counts can grow without changing the id. Lessons/traits use the
    normalised text as their signature.
    """
    key = "|".join(
        (
            (kind or "").strip().lower(),
            normalize_subject(subject),
            " ".join((signature or "").strip().lower().split()),
        )
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


@dataclass
class FactLine:
    """One retrievable memory row (the projection of an atom or scene fact)."""

    id: str = ""
    subject: str = ""
    kind: str = "trait"
    maturity: str = "obs"
    archetypes: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    priority: int = 0
    evidence: int = 0
    date: str = ""
    source: str = "obs"
    season: str = ""
    battle: str = ""
    conflict: bool = False
    content: str = ""
    # Runtime provenance, never serialised: which scope directory produced it.
    origin: str = ""

    def valid(self) -> bool:
        return bool(self.subject and self.content and self.kind in KINDS)

    def stable_signature(self) -> str:
        return self.content


def format_fact_line(fact: FactLine) -> str:
    parts = [
        f"id={fact.id or fact_id(fact.kind, fact.subject, fact.stable_signature())}",
        f"k={fact.subject}",
        f"t={fact.kind}",
        f"m={fact.maturity or 'obs'}",
    ]
    if fact.archetypes:
        parts.append("a=" + ",".join(sorted({a for a in fact.archetypes if a})))
    if fact.roles:
        parts.append("r=" + ",".join(sorted({r for r in fact.roles if r})))
    parts.append(f"p={max(0, min(100, int(fact.priority or 0)))}")
    parts.append(f"n={max(0, int(fact.evidence or 0))}")
    if fact.date:
        parts.append(f"d={fact.date}")
    parts.append(f"s={fact.source or 'obs'}")
    if fact.season:
        parts.append(f"se={fact.season}")
    if fact.battle:
        parts.append(f"b={fact.battle}")
    if fact.conflict:
        parts.append("x=conflict")
    return " ".join(parts) + _SEPARATOR + _one_line(fact.content)


def parse_fact_line(line: str) -> FactLine | None:
    """Parse one row; malformed or incomplete rows return ``None``."""
    text = (line or "").strip()
    if not text or _SEPARATOR not in text or text.startswith("{"):
        return None
    prefix, content = text.split(_SEPARATOR, 1)
    content = _one_line(content)
    fields: dict[str, str] = {}
    for token in prefix.split():
        match = _FIELD_RE.match(token)
        if match:
            fields[match.group(1)] = match.group(2)
    subject = fields.get("k", "")
    kind = fields.get("t", "")
    if not subject or kind not in KINDS or not content:
        return None
    maturity = fields.get("m", "obs")
    if maturity not in MATURITIES:
        maturity = "obs"
    source = fields.get("s", "obs")
    if source not in SOURCES:
        source = "obs"
    try:
        priority = int(fields.get("p", "0") or 0)
    except ValueError:
        priority = 0
    try:
        evidence = int(fields.get("n", "0") or 0)
    except ValueError:
        evidence = 0
    archetypes = [a for a in fields.get("a", "").split(",") if a]
    roles = [r for r in fields.get("r", "").split(",") if r]
    return FactLine(
        id=fields.get("id") or fact_id(kind, subject, content),
        subject=normalize_subject(subject),
        kind=kind,
        maturity=maturity,
        archetypes=archetypes,
        roles=roles,
        priority=max(0, min(100, priority)),
        evidence=max(0, evidence),
        date=fields.get("d", ""),
        source=source,
        season=fields.get("se", ""),
        battle=fields.get("b", ""),
        conflict=fields.get("x") == "conflict",
        content=content,
    )