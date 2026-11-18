"""BattleKB: the per-battle, per-Pokemon data tree bound to static data.

Built once at team preview (own side: full details; opponent: six species
stubs with types / base stats / every possible ability, everything else
unknown) and refreshed every decision point from the live poke-env objects,
which accumulate reveals (moves, confirmed ability, item, tera type) as
the protocol streams them in. Bound values are ids and numbers — prose
descriptions stay in the tools, which resolve them on demand.

``to_json`` snapshots the tree to ``run_dir/kb_tree.json`` every refresh, so
postmortems can replay exactly what the agent believed at each decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from .showdown_data import get_showdown_data
from .static import normalize_name

if TYPE_CHECKING:
    from poke_env.battle import AbstractBattle


@dataclass
class MonNode:
    """One Pokemon: static identity plus whatever has been revealed."""

    species: str  # showdown species key, lowercase
    side: str  # "us" | "them"
    level: int | None = None
    types: list[str] = field(default_factory=list)
    base_stats: dict[str, int] = field(default_factory=dict)
    possible_abilities: dict[str, str] = field(default_factory=dict)  # slot -> name
    known_ability: str | None = None
    revealed_moves: list[str] = field(default_factory=list)
    item: str | None = None
    tera_type: str | None = None  # revealed tera type
    terastallized: bool = False
    fainted: bool = False
    mega_evolved: bool = False  # form changed via Mega Evolution
    selected: bool | None = None  # picked in team preview (None = singles)
    # current battle state, bound from the live object on every refresh.
    # ``hp_fraction`` is None for opponent preview stubs (nothing revealed);
    # ``item_raw`` keeps Showdown's "unknown_item'' sentinel so the rendered
    # board text stays byte-compatible with records.snapshot.
    hp_fraction: float | None = None
    status: str | None = None  # Status enum name ("BRN", "FNT", ...)
    boosts: dict[str, int] = field(default_factory=dict)
    item_raw: str | None = None
    # own side only (opponent stats are never visible)
    stats: dict[str, int] | None = None
    moves: list[str] | None = None

    def render(self) -> str:
        """One compact knowledge-blurb line for prompts."""
        bits = [
            f"{self.species}",
            f"L{self.level}" if self.level else None,
            "/".join(self.types) if self.types else None,
            (
                "abilities: "
                + ", ".join(
                    f"{name}{' (hidden)' if slot == 'H' else ''}"
                    for slot, name in self.possible_abilities.items()
                )
                if self.possible_abilities and not self.known_ability
                else None
            ),
            f"ability: {self.known_ability}" if self.known_ability else None,
            "moves: " + ", ".join(self.revealed_moves) if self.revealed_moves else None,
            f"item: {self.item}" if self.item else None,
            f"tera: {self.tera_type}" if self.tera_type else None,
            "TERASTALLIZED" if self.terastallized else None,
            "MEGA" if self.mega_evolved else None,
            "fainted" if self.fainted else None,
        ]
        return " | ".join(b for b in bits if b)


def _tname(value) -> str:
    """Enum -> short name ('FIRE'), not the repr-ish str()."""
    return getattr(value, "name", str(value))


def _static_entry(species: str, kb=None, data=None) -> dict[str, Any]:
    """Types / base stats / possible abilities for a species, any source.

    Prefers the profile's Showdown bundle (keeps form keys exact and reflects
    format mods such as Champions); falls back to the GenData-based StaticKB.
    Empty dict if neither knows the species.
    """
    sd = data if data is not None else getattr(kb, "data", None)
    if sd is None:
        sd = get_showdown_data()
    if sd is not None:
        entry = sd.species(species)
        if entry:
            return entry
    if kb is not None:
        entry = kb.species_data(species)
        if entry:
            return entry
    return {}


def _possible_abilities(entry: dict[str, Any]) -> dict[str, str]:
    abilities = entry.get("abilities") or {}
    return {
        str(slot): name
        for slot, name in abilities.items()
        if name and name != "noability"
    }


class BattleKB:
    """The agent's live knowledge tree for one battle, keyed by species."""

    def __init__(self):
        self.us: dict[str, MonNode] = {}
        self.them: dict[str, MonNode] = {}
        # live species -> team-identity key (Mega/Primal forms map to the base
        # species so a forme change never creates a second team node)
        self._key_cache: dict[str, str] = {}
        self.data = None
        self.kb = None

    def _key_for(self, species: str, data=None) -> str:
        """Team-identity key for a live species id (Mega -> base species)."""
        key = normalize_name(species)
        cached = self._key_cache.get(key)
        if cached is not None:
            return cached
        sd = data if data is not None else self.data
        entry = _static_entry(species, self.kb, data=sd) or {}
        base = entry.get("baseSpecies") or entry.get("changesFrom")
        resolved = normalize_name(base) if base else key
        self._key_cache[key] = resolved
        return resolved

    # --- construction ------------------------------------------------------

    def build(self, battle: "AbstractBattle", kb=None) -> None:
        """Preview-time build: own side full, opponent stubbed to species.

        During team preview an opponent's HP is not information we have —
        the stub keeps ``hp_fraction=None`` instead of Showdown's internal
        zero, so nothing downstream mistakes it for a fainted mon.
        """
        self.us.clear()
        self.them.clear()
        self._key_cache.clear()
        self.kb = kb
        self.data = getattr(kb, "data", None)
        preview = self._preview_selected(battle)
        preview_foes = bool(getattr(battle, "teampreview", False))
        for mon in battle.team.values():
            key = self._key_for(mon.species, self.data)
            self.us[key] = self._own_node(
                mon, preview=preview, data=self.data, species_key=key
            )
        for mon in battle.opponent_team.values():
            key = self._key_for(mon.species, self.data)
            self.them[key] = self._foe_node(
                mon, kb, hidden_hp=preview_foes, data=self.data,
                species_key=key,
            )

    def refresh(self, battle: "AbstractBattle", kb=None) -> None:
        """Bind the latest revealed state back into the tree.

        New reveals can also introduce species we had not seen (an
        opponent's mon entering without a preview build, late substitutions
        in formats without preview) — those get nodes on the fly.
        """
        preview = self._preview_selected(battle)
        self.kb = kb
        self.data = getattr(kb, "data", None)
        for mon in battle.team.values():
            self._refresh_own(mon, preview=preview, data=self.data)
        for mon in battle.opponent_team.values():
            self._refresh_foe(mon, kb, data=self.data)

    # --- node factories / updaters ---------------------------------------

    @staticmethod
    def _preview_selected(battle: "AbstractBattle") -> bool:
        """True once a team preview has marked which mons were brought."""
        return any(
            getattr(p, "_selected_in_teampreview", False)
            for p in battle.team.values()
        )

    @staticmethod
    def _bind_common(node: MonNode, mon, *, hidden_hp: bool = False) -> None:
        """Bind the per-turn visible state shared by both sides.

        Everything here is an *observation* from the live object; it never
        invents opponent data. ``hidden_hp`` covers the team-preview stubs,
        where Showdown reports 0 HP that does not mean "fainted".
        """
        node.level = mon.level
        # poke-env has no ``is_mega`` flag; a changed base-stat spread is the
        # reliable signal that Mega Evolution replaced the forme's base stats.
        try:
            live_base = {
                str(k): int(v) for k, v in dict(mon.base_stats).items()
            }
        except Exception:
            live_base = {}
        if live_base:
            if node.base_stats and live_base != node.base_stats:
                node.mega_evolved = True
            node.base_stats = live_base
        try:
            node.types = [_tname(t) for t in mon.types]
        except Exception:
            pass
        node.fainted = bool(mon.fainted)
        if hidden_hp:
            node.hp_fraction = None
        else:
            try:
                node.hp_fraction = float(mon.current_hp_fraction)
            except Exception:
                node.hp_fraction = None
        status = getattr(mon, "status", None)
        node.status = _tname(status) if status else None
        try:
            node.boosts = {
                str(_tname(k)): int(v)
                for k, v in dict(mon.boosts).items()
                if v
            }
        except Exception:
            node.boosts = {}
        node.item_raw = getattr(mon, "item", None)
        ability = getattr(mon, "ability", None)
        if ability:
            node.known_ability = str(ability)
        if getattr(mon, "is_terastallized", False):
            node.terastallized = True
            if getattr(mon, "tera_type", None):
                node.tera_type = _tname(mon.tera_type)
        elif getattr(mon, "tera_type", None):
            node.tera_type = _tname(mon.tera_type)

    def _own_node(
        self, mon, *, preview: bool = False, data=None, species_key=None
    ) -> MonNode:
        entry = _static_entry(mon.species, data=data)
        node = MonNode(
            species=species_key or self._key_for(mon.species, data),
            side="us",
            base_stats=dict(
                entry.get("baseStats")
                or getattr(mon, "base_stats", {})
                or {}
            ),
            possible_abilities=_possible_abilities(entry),
            selected=(
                bool(getattr(mon, "_selected_in_teampreview", False))
                if preview
                else None
            ),
        )
        self._bind_common(node, mon)
        # stats only exist once the preview request carries the actual spread;
        # a lazy build can catch mons whose details have not arrived yet
        try:
            stats = {
                k: v for k, v in dict(mon.stats).items() if isinstance(v, int)
            }
            node.stats = stats or None
        except Exception:
            pass
        if mon.moves:
            node.moves = sorted(mon.moves)
        item = getattr(mon, "item", None)
        node.item = item if item and item != "unknown_item" else None
        return node

    def _refresh_own(self, mon, *, preview: bool = False, data=None) -> None:
        key = self._key_for(mon.species, data)
        node = self.us.get(key)
        if node is None:
            self.us[key] = node = self._own_node(
                mon, preview=preview, data=data, species_key=key
            )
            return
        self._bind_common(node, mon)
        if preview:
            node.selected = bool(getattr(mon, "_selected_in_teampreview", False))
        if mon.moves:
            node.moves = sorted(mon.moves)
        try:
            stats = {
                k: v for k, v in dict(mon.stats).items() if isinstance(v, int)
            }
            if stats:
                node.stats = stats
        except Exception:
            pass
        item = getattr(mon, "item", None)
        if item and item != "unknown_item":
            node.item = item

    def _foe_node(
        self, mon, kb=None, *, hidden_hp: bool = False, data=None,
        species_key=None,
    ) -> MonNode:
        entry = _static_entry(mon.species, kb, data=data)
        node = MonNode(
            species=species_key or self._key_for(mon.species, data),
            side="them",
            base_stats=dict(
                entry.get("baseStats")
                or getattr(mon, "base_stats", {})
                or {}
            ),
            possible_abilities=_possible_abilities(entry),
            revealed_moves=sorted(mon.moves) if mon.moves else [],
        )
        self._bind_common(node, mon, hidden_hp=hidden_hp)
        item = getattr(mon, "item", None)
        node.item = item if item and item != "unknown_item" else None
        return node

    def _refresh_foe(self, mon, kb=None, *, data=None) -> None:
        key = self._key_for(mon.species, data)
        node = self.them.get(key)
        if node is None:
            self.them[key] = node = self._foe_node(
                mon, kb, data=data, species_key=key
            )
            return
        self._bind_common(node, mon)
        if mon.moves:
            node.revealed_moves = sorted(mon.moves)
        item = getattr(mon, "item", None)
        if item and item != "unknown_item":
            node.item = item

    # --- access ------------------------------------------------------------

    def key_for(self, species: str, data=None) -> str:
        """Public team-identity key (see :meth:`_key_for`)."""
        return self._key_for(species, data)

    def query(self, species: str) -> MonNode | None:
        return self.us.get(species) or self.them.get(species)

    # --- persistence ---------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "us": {k: asdict(v) for k, v in self.us.items()},
            "them": {k: asdict(v) for k, v in self.them.items()},
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "BattleKB":
        tree = cls()
        for side, attr in (("us", "us"), ("them", "them")):
            nodes = getattr(tree, attr)
            for species, fields in (data.get(side) or {}).items():
                nodes[species] = MonNode(**fields)
        return tree
