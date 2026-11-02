"""Shared test helpers: hand-built battles, scripted LLM stub, no server needed."""

from __future__ import annotations

import logging
import re

from poke_env.battle import AbstractBattle, Battle, DoubleBattle, Pokemon

from pokeagent.config import AgentConfig
from pokeagent.knowledge.static import StaticKB
from pokeagent.providers.base import (
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)
from pokeagent.tools.base import ToolContext

LOGGER = logging.getLogger("test_battle")


class ScriptedProvider(LLMProvider):
    """Returns canned LLMResponses in order; records every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, *, system, messages, tools, max_tokens):
        self.calls.append(
            {
                "system": system,
                "messages": list(messages),
                "tools": list(tools),
                "max_tokens": max_tokens,
            }
        )
        if not self.responses:
            return LLMResponse(stop_reason="end_turn")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            return LLMResponse(stop_reason="error", error=str(item))
        return item


class DecisionStubProvider(LLMProvider):
    """Offline decision-maker: parses the [NOW] section and submits move #1.

    Drives a full real battle through the whole agent loop with zero LLM
    cost — used by the end-to-end test to verify plumbing, records, and
    postmortem without an API key. Handles both the singles layout (global
    Moves/Switches lists) and the doubles layout (per-slot sections with
    symbolic move targets).
    """

    def __init__(self):
        self.calls: list[str] = []  # one entry per turn: "turn:move" or "turn:switch"

    async def complete(self, *, system, messages, tools, max_tokens):
        # the assembled prompt for the CURRENT decision point is the first
        # message of this turn's conversation; retried rounds only append
        # nudges and tool errors, which hold no parseable options. Within
        # that prompt the LAST "[NOW" is the current turn (recent history
        # records embed their own stale [NOW] sections).
        latest = None
        for msg in messages:
            text = getattr(msg, "content", None)
            if text and ("[NOW" in text or "[TEAM PREVIEW" in text):
                latest = text
                break
        if latest and "[TEAM PREVIEW" in latest:
            return self._decide_preview(latest, system)
        if latest and "[NOW" in latest:
            latest = latest.rsplit("[NOW", 1)[-1]
        if latest and "Slot 1 (" in latest:
            return self._decide_doubles(latest)
        return self._decide_singles(latest)

    # --- team preview ----------------------------------------------------

    def _decide_preview(self, latest, system):
        """Pick the first N slots (N stated in the prompt since Step 6)."""
        m = re.search(r"Pick exactly (\d+) of", latest)
        pick = int(m.group(1)) if m else 4
        team_block = system.split("Your team:", 1)[-1].split("Opponent's team:", 1)[0]
        team_size = sum(1 for line in team_block.splitlines() if line.startswith("- "))
        order = list(range(1, min(pick, team_size or pick) + 1))
        self.calls.append("preview:" + "".join(map(str, order)))
        return LLMResponse(
            text=f"Reasoning for picking slots {order}.",
            tool_calls=[
                ToolCallRequest(
                    "stub-team",
                    "submit_team",
                    {"order": order, "reason": "stub picks the first N slots"},
                )
            ],
            stop_reason="tool_use",
        )

    # --- singles --------------------------------------------------------

    def _decide_singles(self, latest):
        move_id = None
        switch = None
        if latest and "Moves:" in latest:
            for line in latest.splitlines():
                line = line.strip()
                if line.startswith("- ") and "|" in line and "|" in line[2:]:
                    move_id = line[2:].split("|")[0].strip()
                    break
        if move_id is None and latest and "Switches:" in latest:
            for line in latest.splitlines():
                line = line.strip()
                if line.startswith("- "):
                    switch = line[2:].split()[0]
                    break
        if move_id:
            self.calls.append(f"move:{move_id}")
            arguments = {"actions": [
                {"slot": 1, "kind": "move", "move": move_id}
            ], "reason": "stub chose first available move"}
        elif switch:
            self.calls.append(f"switch:{switch}")
            arguments = {"actions": [
                {"slot": 1, "kind": "switch", "switch_to": switch}
            ], "reason": "stub forced to switch"}
        else:
            return LLMResponse(stop_reason="end_turn", text="no options parsed")
        return self._submit(arguments)

    # --- doubles ----------------------------------------------------------

    # prefer hitting foes; only target our own side if nothing else exists
    TARGET_PREFERENCE = ("opp1", "opp2", "self", "ally")

    def _decide_doubles(self, latest):
        actions = []
        claimed_switches: set[str] = set()
        slot = None
        section = None  # "moves" | "switches" | None
        for line in latest.splitlines():
            stripped = line.strip()
            if stripped.startswith("Slot ") and " (" in stripped:
                slot = int(stripped[len("Slot "):].split("(")[0].strip())
                section = None
                continue
            if slot is None:
                continue
            if stripped == "Moves:":
                section = "moves"
                continue
            if stripped == "Switches:":
                section = "switches"
                continue
            if not stripped.startswith("- "):
                continue
            if "pass" in stripped:
                continue  # auto-pass slots are omitted from the submission
            if any(a["slot"] == slot for a in actions):
                continue  # the stub only ever picks each slot's first option
            if section == "moves" and "|" in stripped[2:]:
                self._finish_move(actions, slot, stripped[2:])
            elif section == "switches":
                name = stripped[2:].split()[0]
                if name in claimed_switches:
                    continue  # a bench mon can only be sent to one slot
                claimed_switches.add(name)
                actions.append(
                    {
                        "slot": slot,
                        "kind": "switch",
                        "switch_to": name,
                    }
                )
        if not actions:
            return LLMResponse(stop_reason="end_turn", text="no options parsed")
        self.calls.append(
            " | ".join(
                f"{a['slot']}:{a['kind']}:{a.get('move') or a.get('switch_to') or ''}"
                + (f"->{a['target']}" if a.get("target") else "")
                for a in actions
            )
        )
        return self._submit(
            {"actions": actions, "reason": "stub chose first option per slot"}
        )

    @classmethod
    def _finish_move(cls, actions, slot, move_line):
        """Record the first move of ``slot`` (with its best non-auto target)."""
        move_id = move_line.split("|")[0].strip()
        action = {"slot": slot, "kind": "move", "move": move_id}
        if "targets: " in move_line:
            targets = [
                t.strip()
                for t in move_line.split("targets: ", 1)[1].split(",")
                if t.strip() and t.strip() != "auto"
            ]
            for preferred in cls.TARGET_PREFERENCE:
                if preferred in targets:
                    action["target"] = preferred
                    break
        actions.append(action)

    def _submit(self, arguments):
        return LLMResponse(
            text=f"Reasoning for {self.calls[-1]}.",
            tool_calls=[
                ToolCallRequest("stub-1", "submit_decision", arguments)
            ],
            stop_reason="tool_use",
        )


def make_battle(*, with_moves: bool = True) -> AbstractBattle:
    """A minimal gen9 battle: active Furret (known stats) vs Torterra (unknown stats)."""
    battle = Battle(
        battle_tag="battle-gen9randombattle-1",
        username="AgentUser",
        logger=LOGGER,
        save_replays=False,
        gen=9,
    )
    battle._format = "gen9randombattle"
    battle._player_role = "p1"

    attacker = Pokemon(9, species="furret", name="Furret")
    attacker._level = 50
    attacker._max_hp = 170
    attacker._current_hp = 170
    attacker._stats = {
        "hp": 170, "atk": 96, "def": 82, "spa": 62, "spd": 72, "spe": 110,
    }
    attacker._add_move("flamethrower")
    if with_moves:
        attacker._add_move("quickattack")
        battle._available_moves = [
            attacker.moves["flamethrower"],
            attacker.moves["quickattack"],
        ]
    attacker._active = True

    defender = Pokemon(9, species="torterra", name="Torterra")
    defender._level = 50
    defender._add_move("stoneedge")
    defender._active = True

    battle._team = {"p1: Furret": attacker}
    battle._opponent_team = {"p2: Torterra": defender}
    return battle


def make_double_battle(
    *,
    force_switch=(False, False),
    fainted=(False, False),
    bench=("pikachu", "charizard"),
    can_tera=(False, False),
    trapped=(False, False),
    moves=(("flamethrower", "quickattack"), ("waterfall", "icefang")),
    wait=False,
) -> DoubleBattle:
    """A minimal doubles battle with two actives, two foes and a bench.

    Knobs cover the decision matrix: force_switch/fainted (forced-switch
    rounds), bench contents (same-name switch tests), can_tera and trapped.
    """
    battle = DoubleBattle(
        battle_tag="battle-gen9vgc2025regihidden-1",
        username="AgentUser",
        logger=LOGGER,
        gen=9,
        save_replays=False,
    )
    battle._format = "gen9vgc2025regihidden"
    battle._player_role = "p1"
    battle._wait = wait
    battle._turn = 5

    team: dict[str, Pokemon] = {}
    for i, species in enumerate(("furret", "gyarados")):
        mon = Pokemon(9, species=species, name=species.capitalize())
        mon._level = 50
        mon._max_hp = 170
        mon._current_hp = 0 if fainted[i] else 170
        mon._stats = {
            "hp": 170, "atk": 96, "def": 82, "spa": 62, "spd": 72, "spe": 90 + i,
        }
        mon._active = True
        team[f"p1: {species.capitalize()}"] = mon
        battle._active_pokemon[f"p1{'ab'[i]}"] = mon

    opp_team: dict[str, Pokemon] = {}
    for i, species in enumerate(("torterra", "incineroar")):
        foe = Pokemon(9, species=species, name=species.capitalize())
        foe._level = 50
        foe._max_hp = 180
        foe._current_hp = 180
        foe._add_move("tackle")
        foe._active = True
        opp_team[f"p2: {species.capitalize()}"] = foe
        battle._opponent_active_pokemon[f"p2{'ab'[i]}"] = foe

    bench_mons: list[Pokemon] = []
    for i, species in enumerate(bench):
        mon = Pokemon(9, species=species, name=f"{species.capitalize()} {i + 1}")
        mon._level = 50
        mon._max_hp = 150
        mon._current_hp = 150
        mon._add_move("tackle")
        team[f"p1: {mon.name}"] = mon
        bench_mons.append(mon)

    battle._team = team
    battle._opponent_team = opp_team

    available_moves: list[list] = [[], []]
    for i in range(2):
        mon = battle._active_pokemon[f"p1{'ab'[i]}"]
        if not fainted[i] and not force_switch[i]:
            for mv in moves[i]:
                mon._add_move(mv)
            available_moves[i] = list(mon.moves.values())
    battle._available_moves = available_moves
    battle._available_switches = [list(bench_mons), list(bench_mons)]
    battle._force_switch = list(force_switch)
    battle._can_tera = list(can_tera)
    battle._trapped = list(trapped)
    return battle


def make_kb() -> StaticKB:
    return StaticKB("gen9randombattle")


def make_ctx(battle, kb, cm=None, config=None) -> ToolContext:
    return ToolContext(
        battle=battle, kb=kb, context=cm, config=config or AgentConfig()
    )
