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
    selected: bool | None = None  # picked in team preview (None = singles)
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
            "fainted" if self.fainted else None,
        ]
        return " | ".join(b for b in bits if b)


def _tname(value) -> str:
    """Enum -> short name ('FIRE'), not the repr-ish str()."""
    return getattr(value, "name", str(value))


def _static_entry(species: str, kb=None) -> dict[str, Any]:
    """Types / base stats / possible abilities for a species, any source.

    Prefers the Showdown bundle (keeps form keys exact); falls back to the
    GenData-based StaticKB. Empty dict if neither knows the species.
    """
    sd = get_showdown_data()
    if sd is not None:
        data = sd.species(species)
        if data:
            return data
    if kb is not None:
        data = kb.species_data(species)
        if data:
            return data
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

    # --- construction ------------------------------------------------------

    def build(self, battle: "AbstractBattle", kb=None) -> None:
        """Preview-time build: own side full, opponent stubbed to species."""
        self.us.clear()
        self.them.clear()
        for mon in battle.team.values():
            self.us[mon.species] = self._own_node(mon)
        for mon in battle.opponent_team.values():
            self.them[mon.species] = self._foe_node(mon, kb)

    def refresh(self, battle: "AbstractBattle", kb=None) -> None:
        """Bind the latest revealed state back into the tree.

        New reveals can also introduce species we had not seen (an
        opponent's mon entering without a preview build, late substitutions
        in formats without preview) — those get nodes on the fly.
        """
        for mon in battle.team.values():
            self._refresh_own(mon)
        for mon in battle.opponent_team.values():
            self._refresh_foe(mon, kb)

    # --- node factories / updaters ---------------------------------------

    def _own_node(self, mon) -> MonNode:
        entry = _static_entry(mon.species)
        node = MonNode(
            species=mon.species,
            side="us",
            level=mon.level,
            types=[_tname(t) for t in mon.types],
            base_stats=dict(entry.get("baseStats") or {}),
            possible_abilities=_possible_abilities(entry),
            known_ability=str(mon.ability) if mon.ability else None,
            fainted=mon.fainted,
            selected=getattr(mon, "_selected_in_teampreview", None),
        )
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
        if getattr(mon, "is_terastallized", False):
            node.terastallized = True
            if mon.tera_type:
                node.tera_type = _tname(mon.tera_type)
        return node

    def _refresh_own(self, mon) -> None:
        node = self.us.get(mon.species)
        if node is None:
            self.us[mon.species] = node = self._own_node(mon)
            return
        node.fainted = mon.fainted
        if mon.moves:
            node.moves = sorted(mon.moves)
        if mon.ability:
            node.known_ability = str(mon.ability)
        if getattr(mon, "is_terastallized", False):
            node.terastallized = True
            if mon.tera_type:
                node.tera_type = _tname(mon.tera_type)

    def _foe_node(self, mon, kb=None) -> MonNode:
        entry = _static_entry(mon.species, kb)
        return MonNode(
            species=mon.species,
            side="them",
            level=mon.level,
            types=[_tname(t) for t in mon.types],
            base_stats=dict(entry.get("baseStats") or {}),
            possible_abilities=_possible_abilities(entry),
            known_ability=str(mon.ability) if mon.ability else None,
            revealed_moves=sorted(mon.moves) if mon.moves else [],
            fainted=mon.fainted,
        )

    def _refresh_foe(self, mon, kb=None) -> None:
        node = self.them.get(mon.species)
        if node is None:
            self.them[mon.species] = self._foe_node(mon, kb)
            return
        node.fainted = mon.fainted
        if mon.moves:
            node.revealed_moves = sorted(mon.moves)
        ability = mon.ability
        if ability:
            node.known_ability = str(ability)
        item = getattr(mon, "item", None)
        if item and item != "unknown_item":
            node.item = item
        if getattr(mon, "is_terastallized", False):
            node.terastallized = True
            if mon.tera_type:
                node.tera_type = _tname(mon.tera_type)
        elif getattr(mon, "tera_type", None):
            node.tera_type = _tname(mon.tera_type)

    # --- access ------------------------------------------------------------

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
