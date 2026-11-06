"""Explicit harness state for one battle.

The harness keeps four kinds of state, each with a different truth contract:

- **observed** — hard facts bound from the live poke-env objects at every
  decision point (``BattleKB`` nodes). Monotone: reveal information but
  never invent it; missing data stays ``None``.
- **inferred** — revisable beliefs the agent wrote down (``Claim``).
- **committed** — the plan the agent chose to follow (``PlanBoard`` and
  ``Directive``).
- **history** — immutable ``TurnRecord`` objects, owned by
  ``context.manager.ContextManager`` and never rewritten here.

``BattleState`` glues observed + committed + the reasoning ledger into one
JSON-serializable object persisted as ``run_dir/state.json``. The prompt's
``[NOW]`` section is rendered from it (``state.render``), so what the model
saw is exactly what was saved — the postmortem can replay the agent's
beliefs, not just the protocol log.

Import rule: this module must not import ``pokeagent.context`` (the context
package imports ``manager`` which imports this package). Only poke-env and
``knowledge.tree`` are allowed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from dataclasses import field as _dc_field  # alias: BattleState has a "field" attribute
from typing import TYPE_CHECKING, Any

from poke_env.battle import DoubleBattle

from ..knowledge.tree import BattleKB

if TYPE_CHECKING:
    from poke_env.battle import AbstractBattle

SCHEMA_VERSION = 2


def _name(value: Any) -> str:
    """Enum -> short name ('FIRE'), str stays as-is."""
    return getattr(value, "name", str(value))


def _preview_pick_count(battle: "AbstractBattle") -> int:
    """Picked-team size of a team-preview format (mirrors records)."""
    request = getattr(battle, "_last_request", None) or {}
    for key in ("maxChosenTeamSize", "maxTeamSize"):
        value = request.get(key)
        if value:
            return int(value)
    return battle.max_team_size or len(battle.team) or 6


# --- observed side ----------------------------------------------------------


@dataclass
class SideState:
    """Which Pokemon occupy which slots, and in which team order."""

    active: list[str | None] = field(default_factory=list)  # species per slot
    order: list[str] = field(default_factory=list)  # team order (species)

    @property
    def active_species(self) -> list[str]:
        return [s for s in self.active if s]


@dataclass
class FieldState:
    """Weather / terrain / side conditions, as short enum names."""

    turn: int = 0
    weather: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    side_conditions: list[str] = field(default_factory=list)
    opponent_side_conditions: list[str] = field(default_factory=list)


@dataclass
class ResourceLedger:
    """Cheap, deterministic bookkeeping derived from the observed board.

    HP fractions are copied from the KB nodes (``None`` = not revealed, e.g.
    opponent team-preview stubs). ``tera_used`` accumulates and is never
    cleared: a spent tera cannot come back.
    """

    our_hp: dict[str, float | None] = field(default_factory=dict)
    foe_hp: dict[str, float | None] = field(default_factory=dict)
    fainted: list[str] = field(default_factory=list)  # either side
    tera_used: list[str] = field(default_factory=list)

    @property
    def our_fainted_count(self) -> int:
        return sum(1 for hp in self.our_hp.values() if hp == 0)

    @property
    def foe_fainted_count(self) -> int:
        return sum(1 for hp in self.foe_hp.values() if hp == 0)


# --- inferred / committed ---------------------------------------------------


@dataclass
class Claim:
    """One revisable belief about the battle. ``status`` is the pruning hook."""

    subject: str  # species or "field" / "plan"
    kind: str  # set | speed | item | ability | move | hp | plan
    text: str
    confidence: str = "medium"  # low | medium | high
    source: str = "model"  # model | inference | revealed
    turn: int = 0
    status: str = "active"  # active | confirmed | refuted


@dataclass
class Directive:
    """An imperative the agent committed to following across turns."""

    text: str
    priority: str = "medium"  # high | medium | low
    scope: str = "battle"  # turn | battle
    turn: int = 0
    expires_turn: int | None = None


@dataclass
class PlanBoard:
    """The committed plan, written by the model and revised explicitly."""

    win_conditions: list[str] = field(default_factory=list)
    primary_threats: list[str] = field(default_factory=list)
    line: str | None = None  # current multi-turn line of play
    commitments: list[str] = field(default_factory=list)
    last_outcome: str | None = None
    revision: int = 0


@dataclass
class ReasoningNote:
    """Structured compression of one turn's raw CoT.

    Raw CoT stays in ``llm.jsonl``; this is what later turns actually read.
    ``source`` is "deterministic" (no model call — intent and outcome only)
    or "llm" (cheap-model extraction).
    """

    seq: int = 0
    turn: int = 0
    intent: str | None = None
    claims: list[str] = field(default_factory=list)
    correction: str | None = None
    directives: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    source: str = "deterministic"
    fallback: bool = False  # the decision loop failed and fell back


@dataclass
class StrategyBrief:
    """Rolling L2 summary: how this battle is won, rewritten each few turns.

    Produced by an LLM compressor in a fixed schema (see
    ``context/brief.py``), never by appending. ``covered_seq`` makes the
    swap race-free exactly like history compression: a brief covers notes
    up to a monotonic sequence number and never clobbers newer state.
    """

    covered_seq: int = 0
    turn: int = 0
    stance: str | None = None  # one line: the current winning idea
    win_conditions: list[str] = field(default_factory=list)  # <= 2
    threats: list[str] = field(default_factory=list)  # <= 3
    commitments: list[str] = field(default_factory=list)  # <= 3
    beliefs: list[str] = field(default_factory=list)  # <= 4 (revisable)
    open_questions: list[str] = field(default_factory=list)  # <= 2
    source: str = "llm"  # llm | deterministic

    def empty(self) -> bool:
        return not (
            self.stance
            or self.win_conditions
            or self.threats
            or self.commitments
            or self.beliefs
            or self.open_questions
        )


@dataclass
class ReasoningLedger:
    """Per-battle reasoning memory: notes, lifted directives, rolling brief."""

    notes: list[ReasoningNote] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    directives: list[Directive] = field(default_factory=list)
    brief: StrategyBrief | None = None  # rolling strategy brief (S3.5)
    brief_covered_seq: int = 0

    def add_note(self, note: ReasoningNote) -> None:
        self.notes.append(note)
        for text in note.directives:
            text = (text or "").strip()
            if text:
                self.directives.append(Directive(text=text, turn=note.turn))

    def note_for_seq(self, seq: int) -> ReasoningNote | None:
        for note in reversed(self.notes):
            if note.seq == seq:
                return note
        return None

    def latest_note(self) -> ReasoningNote | None:
        return self.notes[-1] if self.notes else None

    def active_directives(self, *, limit: int = 5) -> list[Directive]:
        """Unexpired directives, newest first (battle-scope only for now)."""
        out: list[Directive] = []
        for directive in reversed(self.directives):
            if directive.scope == "turn":
                continue
            out.append(directive)
            if len(out) >= limit:
                break
        return out


@dataclass
class BattleMeta:
    """Identity and format metadata for the battle."""

    battle_tag: str = ""
    format: str = ""
    player_username: str | None = None
    opponent_username: str | None = None
    doubles: bool = False
    preview_pick: int | None = None
    preview_order: list[int] = field(default_factory=list)  # /team slot numbers
    schema_version: int = SCHEMA_VERSION


# --- the per-battle state ---------------------------------------------------


@dataclass
class BattleState:
    """Observed + inferred + committed state of one battle."""

    meta: BattleMeta = _dc_field(default_factory=BattleMeta)
    kb: BattleKB = _dc_field(default_factory=BattleKB)
    ours: SideState = _dc_field(default_factory=SideState)
    theirs: SideState = _dc_field(default_factory=SideState)
    field: FieldState = _dc_field(default_factory=FieldState)
    resources: ResourceLedger = _dc_field(default_factory=ResourceLedger)
    plan: PlanBoard = _dc_field(default_factory=PlanBoard)
    ledger: ReasoningLedger = _dc_field(default_factory=ReasoningLedger)

    # --- construction ------------------------------------------------------

    @classmethod
    def create(
        cls,
        battle: "AbstractBattle",
        static_kb=None,
        ledger: ReasoningLedger | None = None,
    ) -> "BattleState":
        """Build from a battle at team preview (or first decision)."""
        state = cls(ledger=ledger if ledger is not None else ReasoningLedger())
        state.kb.build(battle, static_kb)
        state._sync(battle)
        return state

    def refresh(self, battle: "AbstractBattle", static_kb=None) -> None:
        """Bind the latest observations; never invents or forgets reveals."""
        self.kb.refresh(battle, static_kb)
        self._sync(battle)

    def _sync(self, battle: "AbstractBattle") -> None:
        doubles = isinstance(battle, DoubleBattle)
        self.meta.battle_tag = battle.battle_tag or ""
        self.meta.format = battle.format or ""
        self.meta.player_username = battle.player_username
        self.meta.opponent_username = battle.opponent_username
        self.meta.doubles = doubles
        # preview metadata only exists while the preview request is live;
        # afterwards the value written by the player must survive refresh
        if getattr(battle, "teampreview", False):
            self.meta.preview_pick = _preview_pick_count(battle)

        self.ours.active = _active_species(battle, doubles=doubles, ours=True)
        self.theirs.active = _active_species(battle, doubles=doubles, ours=False)
        self.ours.order = [p.species for p in battle.team.values()]
        self.theirs.order = [p.species for p in battle.opponent_team.values()]

        self.field.turn = battle.turn
        self.field.weather = [_name(w) for w in battle.weather]
        self.field.fields = [_name(f) for f in battle.fields]
        self.field.side_conditions = [_name(s) for s in battle.side_conditions]
        self.field.opponent_side_conditions = [
            _name(s) for s in battle.opponent_side_conditions
        ]

        self.resources.our_hp = {
            species: node.hp_fraction for species, node in self.kb.us.items()
        }
        self.resources.foe_hp = {
            species: node.hp_fraction for species, node in self.kb.them.items()
        }
        self.resources.fainted = sorted(
            species
            for side in (self.kb.us, self.kb.them)
            for species, node in side.items()
            if node.fainted
        )
        for species, node in list(self.kb.us.items()) + list(self.kb.them.items()):
            if node.terastallized and species not in self.resources.tera_used:
                self.resources.tera_used.append(species)

    # --- helpers -----------------------------------------------------------

    @property
    def selected_known(self) -> bool:
        """True once a team preview marked which mons were brought."""
        return any(node.selected is True for node in self.kb.us.values())

    def node(self, species: str, *, ours: bool | None = None):
        if ours is True:
            return self.kb.us.get(species)
        if ours is False:
            return self.kb.them.get(species)
        return self.kb.query(species)

    # --- persistence --------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.meta.schema_version,
            "meta": asdict(self.meta),
            "kb": self.kb.to_json(),
            "ours": asdict(self.ours),
            "theirs": asdict(self.theirs),
            "field": asdict(self.field),
            "resources": asdict(self.resources),
            "plan": asdict(self.plan),
            "ledger": {
                "notes": [asdict(n) for n in self.ledger.notes],
                "claims": [asdict(c) for c in self.ledger.claims],
                "directives": [asdict(d) for d in self.ledger.directives],
                "brief": (
                    asdict(self.ledger.brief)
                    if self.ledger.brief is not None
                    else None
                ),
                "brief_covered_seq": self.ledger.brief_covered_seq,
            },
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "BattleState":
        ledger_data = data.get("ledger") or {}
        ledger = ReasoningLedger(
            notes=[ReasoningNote(**n) for n in ledger_data.get("notes") or []],
            claims=[Claim(**c) for c in ledger_data.get("claims") or []],
            directives=[
                Directive(**d) for d in ledger_data.get("directives") or []
            ],
            brief=_brief_from_json(ledger_data.get("brief")),
            brief_covered_seq=int(ledger_data.get("brief_covered_seq") or 0),
        )
        return cls(
            meta=BattleMeta(**(data.get("meta") or {})),
            kb=BattleKB.from_json(data.get("kb") or {}),
            ours=SideState(**(data.get("ours") or {})),
            theirs=SideState(**(data.get("theirs") or {})),
            field=FieldState(**(data.get("field") or {})),
            resources=ResourceLedger(**(data.get("resources") or {})),
            plan=PlanBoard(**(data.get("plan") or {})),
            ledger=ledger,
        )


def _brief_from_json(raw: Any) -> StrategyBrief | None:
    """Load a brief; tolerate the schema-1 plain-string form."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return StrategyBrief(stance=raw or None, source="legacy")
    if isinstance(raw, dict):
        fields = StrategyBrief.__dataclass_fields__
        known = {k: v for k, v in raw.items() if k in fields}
        return StrategyBrief(**known)
    return None


def _active_species(
    battle: "AbstractBattle", *, doubles: bool, ours: bool
) -> list[str | None]:
    """Active species per slot; empty during team preview (no roles yet)."""
    try:
        active = battle.active_pokemon if ours else battle.opponent_active_pokemon
    except Exception:  # team preview: roles are not assigned yet
        return [None] * (2 if doubles else 1)
    slots = list(active) if doubles else [active]
    return [p.species if p is not None else None for p in slots]