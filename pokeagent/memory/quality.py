"""Deterministic battle-quality score for memory writes.

A loss caused by three critical hits should not become a lesson about the
opponent's strategy. The score below is deliberately simple and observable:

``Q = base(opponent) * (1 - 0.5*fallback_rate) * (1 - 0.3*luck_factor) * complete``

The hard gates are separate from Q and are what the consolidation stage reads:
a battle where our own decisions were mostly random fallbacks can never
produce conclusions, regardless of how the opponent played.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

BASE = {
    "human": 1.0,
    "agent": 0.7,
    "heuristic": 0.5,
    "random": 0.2,
    "unknown": 0.6,
}

LUCK_TERMS = (
    "critical hit",
    "missed",
    "flinch",
    "frozen",
    "paraly",
    "confused",
    "hit itself",
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def fallback_rate(records: Iterable[Any]) -> tuple[int, int]:
    records = list(records or [])
    if not records:
        return 0, 0
    falls = sum(1 for record in records if getattr(record, "fell_back", False))
    return falls, len(records)


def luck_factor(records: Iterable[Any], *, turns: int = 0) -> float:
    """Share of turns touched by observable luck, capped at 1.0."""
    records = list(records or [])
    denom = max(int(turns or 0), len(records), 1)
    hits = 0
    for record in records:
        for event in getattr(record, "events", None) or []:
            low = str(event).lower()
            if any(term in low for term in LUCK_TERMS):
                hits += 1
    return _clamp(hits / denom, 0.0, 1.0)


def quality_score(
    battle: Any,
    cm: Any,
    opponent_class: str = "unknown",
    *,
    quality_min_lesson: float = 0.6,
    quality_min_behavior: float = 0.5,
    fallback_hard_gate: float = 0.25,
) -> dict[str, Any]:
    """Return a JSON-safe quality record stored on the battle meta atom."""
    records = list(getattr(cm, "records", None) or [])
    falls, decisions = fallback_rate(records)
    rate = falls / decisions if decisions else 0.0
    turns = int(getattr(battle, "turn", 0) or 0)
    luck = luck_factor(records, turns=turns)
    complete = 1.0 if getattr(battle, "finished", True) else 0.0
    opponent = opponent_class if opponent_class in BASE else "unknown"
    q = (
        BASE[opponent]
        * (1.0 - 0.5 * min(rate, 0.6))
        * (1.0 - 0.3 * luck)
        * complete
    )
    q = _clamp(q)
    behavior_ok = (
        complete > 0
        and opponent != "random"
        and rate <= fallback_hard_gate
        and q >= quality_min_behavior
    )
    lesson_ok = (
        behavior_ok
        and opponent != "random"
        and q >= quality_min_lesson
        and decisions > 0
    )
    return {
        "opponent_class": opponent,
        "q": round(q, 3),
        "fallback_count": falls,
        "fallback_rate": round(rate, 3),
        "luck_factor": round(luck, 3),
        "complete": bool(complete),
        "behavior_ok": bool(behavior_ok),
        "lesson_ok": bool(lesson_ok),
    }


def normalize_opponent(value: str | None) -> str:
    """Map CLI/config spellings to the score vocabulary."""
    text = re.sub(r"[^a-z]+", "", (value or "").lower())
    for key in ("human", "agent", "heuristic", "random"):
        if key in text:
            return key
    return "unknown"