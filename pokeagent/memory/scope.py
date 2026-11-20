"""Rule/season partitioning for cross-battle memory.

A single format id is not a good storage key: ``hidden``/``bo3`` only change
presentation, while Reg M-B and Reg M-C share almost all mechanics but differ
in the legal pool. Memory is therefore partitioned by a structured scope:

- ``game_type``    singles / doubles        (hard boundary)
- ``ruleset``      gen9 / champions / ...   (hard boundary, from profile.mod)
- ``regulation``   regmc / regmb / regi / default  (soft boundary)
- ``format_family`` vgc / bss / ou / random / other (hard boundary)

On disk each scope owns a directory::

    <memory_path>/<game_type>/<ruleset>/<regulation>/

Old regulations are kept as sibling directories, so recall can read them with
a decay instead of forcing every rule change back to a cold start.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import AgentConfig
    from ..formats import FormatProfile

_SUFFIX = re.compile(r"(?:hidden|bo3)+$")
_REGULATION = re.compile(r"(reg[a-z]{1,2})$")
_TIERS = ("ou", "uu", "ru", "nu", "pu", "zu")


def _norm(format_id: str | None) -> str:
    raw = (format_id or "").split("@@@", 1)[0].strip().lower()
    return re.sub(r"[^a-z0-9]+", "", raw)


def regulation_from_id(format_id: str | None) -> str:
    """``gen9championsvgc2026regmchidden`` -> ``regmc``; unknown -> ``default``.

    Showdown puts presentation suffixes after the regulation
    (``regmchidden`` / ``regmcbo3``); strip them first so both variants
    land in the same season bucket.
    """
    norm = _norm(format_id)
    stripped = _SUFFIX.sub("", norm)
    while stripped != norm:  # e.g. "regmchiddenbo3"
        norm, stripped = stripped, _SUFFIX.sub("", stripped)
    match = _REGULATION.search(stripped)
    return match.group(1) if match else "default"


def format_family(format_id: str | None, game_type: str = "singles") -> str:
    norm = _norm(format_id)
    if "vgc" in norm:
        return "vgc"
    if "bss" in norm:
        return "bss"
    if "randombattle" in norm or "randomdoubles" in norm:
        return "random"
    if any(norm.endswith(tier) for tier in _TIERS):
        return "tier"
    return "other"


@dataclass(frozen=True)
class MemoryScope:
    """Immutable storage key for one ruleset + season combination."""

    game_type: str = "singles"
    ruleset: str = "gen9"
    regulation: str = "default"
    format_family: str = "other"

    @property
    def parts(self) -> tuple[str, str, str]:
        return (self.game_type, self.ruleset, self.regulation)

    def __str__(self) -> str:  # pragma: no cover - logging helper
        return "/".join(self.parts)


def scope_from(
    profile: "FormatProfile",
    format_id: str | None = None,
    default_regulation: str = "auto",
) -> MemoryScope:
    """Build the scope for a resolved format profile."""
    fid = format_id or getattr(profile, "format_id", "") or ""
    if default_regulation and default_regulation not in ("auto", "none"):
        regulation = default_regulation
    else:
        regulation = regulation_from_id(fid)

    game_type = getattr(profile, "game_type", None)
    if not game_type:
        norm = _norm(fid)
        game_type = "doubles" if ("doubles" in norm or "vgc" in norm) else "singles"

    ruleset = getattr(profile, "mod", None) or "gen9"
    return MemoryScope(
        game_type=str(game_type),
        ruleset=str(ruleset),
        regulation=regulation,
        format_family=format_family(fid, str(game_type)),
    )


def scope_for_config(config: "AgentConfig") -> MemoryScope:
    """Resolve the scope from a config without importing formats at module load."""
    from ..formats import profile_for

    profile = profile_for(config.battle_format)
    return scope_from(
        profile,
        config.battle_format,
        getattr(config, "default_regulation", "auto"),
    )


def scope_root(config: "AgentConfig", scope: MemoryScope) -> Path:
    return Path(config.memory_path) / scope.game_type / scope.ruleset / scope.regulation


def _scope_meta_path(root: Path) -> Path:
    return root / "scope.json"


def ensure_scope_root(config: "AgentConfig", scope: MemoryScope, format_id: str = "") -> Path:
    """Create the scope directory and its metadata file; safe to repeat."""
    root = scope_root(config, scope)
    root.mkdir(parents=True, exist_ok=True)
    path = _scope_meta_path(root)
    if not path.is_file():
        meta = {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "format": format_id or "",
            "game_type": scope.game_type,
            "ruleset": scope.ruleset,
            "regulation": scope.regulation,
            "format_family": scope.format_family,
        }
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass
    return root


def _created(root: Path) -> str:
    try:
        data = json.loads(_scope_meta_path(root).read_text(encoding="utf-8"))
        value = data.get("created")
        if isinstance(value, str) and value:
            return value
    except (OSError, ValueError, TypeError):
        pass
    try:
        return datetime.fromtimestamp(root.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return ""


def previous_season_roots(
    config: "AgentConfig",
    scope: MemoryScope,
    *,
    limit: int = 2,
) -> list[Path]:
    """Sibling season directories, newest first (current season excluded)."""
    base = Path(config.memory_path) / scope.game_type / scope.ruleset
    if not base.is_dir():
        return []
    roots = [
        path
        for path in base.iterdir()
        if path.is_dir() and path.name != scope.regulation
    ]
    roots.sort(key=_created, reverse=True)
    return roots[: max(0, int(limit))]