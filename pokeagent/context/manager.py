"""Per-battle context: prompt assembly, JSONL persistence, intel store.

Disk is the source of truth for the full record (verbatim CoT, tool calls,
events). The assembled prompt carries only a bounded window: recent turns
verbatim, older turns compacted (TruncatingCompressor now; background LLM
summarization wired in step 6).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ..providers.base import UserMessage
from ..tasks.base import IntelStore
from .compressor import TruncatingCompressor, _compact_record
from .records import TurnRecord, available_actions, snapshot

if TYPE_CHECKING:
    from poke_env.battle import AbstractBattle

    from ..config import AgentConfig

SYSTEM_PROMPT = """You are an expert competitive Pokemon battle agent playing on Pokemon Showdown. \
Each turn you receive the battle state and must commit your action(s) by calling the \
`submit_decision` tool.

How to decide:
- You may first call other tools (damage calculations, move/type/species lookups, battle history) \
to inform your decision. Multiple tool calls requested in the same round run in parallel.
- You MUST finish by calling `submit_decision`. Plain text is never interpreted as a move.
- `submit_decision` arguments: actions is a list of action objects. \
Each has slot (1-based active slot), kind, and the fields for that kind:
  - kind="move": move=<exact move id from the available list> and, in doubles, \
target=<one of that move's listed targets: auto/self/ally/opp1/opp2>.
  - kind="switch": switch_to=<exact species from the available switches list>.
  - kind="pass": only where the slot list says pass is a legal choice.
Choose ONLY from the listed options; anything else is rejected and you lose time.
- Singles: submit exactly one action (slot is ignored). \
Doubles: submit one action per active slot that has choices; slots shown as pass auto-pass.
- Pass terastallize=true only when the notes say it is available and it is the better play.
- You have a hard per-turn time budget (~45s). Be decisive: decide within 1-2 tool rounds.

Battle guidance: weigh type matchups, damage ranges vs both parties' HP, your win condition, \
the opponent's revealed moves and likely sets, status conditions, priority, and whether \
switching preserves momentum better than attacking. In doubles, weigh protect timing, \
speed order, spread moves (0.75x power against both foes), and redirection. \
Random-battle teams are uneven — identify your best win condition early."""

NUDGE = (
    "No action committed yet. Call `submit_decision` now with your chosen action, "
    "or make one final round of tool calls if you are still missing critical information."
)


def _sanitize_tag(battle_tag: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in battle_tag)


class ContextManager:
    """One instance per battle. Owns records, intel, and the run directory."""

    def __init__(self, battle_tag: str, config: "AgentConfig", kb=None):
        self.config = config
        self.battle_tag = battle_tag
        self.kb = kb
        self.intel = IntelStore()
        self.records: list[TurnRecord] = []
        self.seen_species: set[str] = set()

        # run directory + persistence
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir: Path = config.runs_dir / f"{ts}_{_sanitize_tag(battle_tag)}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self.run_dir / "events.jsonl"
        self._llm_path = self.run_dir / "llm.jsonl"
        self._write_meta(ts)
        self.log_event(
            "battle_start",
            battle_tag=battle_tag,
            format=config.battle_format,
            model=config.model,
        )

        # bookkeeping
        self._seq = 0
        self._replay_offset = 0
        self.last_now_text: str = ""  # the [NOW] section the model saw last
        # per-battle knowledge tree (knowledge/tree.BattleKB), built by the
        # player at team preview and refreshed every decision point
        self.battle_kb = None
        # compression state: an LLM summary covering records with seq <= covered
        self.summary: str | None = None
        self.summary_covered_seq = 0
        self.compressor = None  # LLM-backed compressor, attached by the player
        self._compressing = False
        self._compressor_fallback = TruncatingCompressor()

    # --- persistence ------------------------------------------------------

    def _write_meta(self, ts: str) -> None:
        """Config snapshot per battle: which model/endpoint/parameters ran here."""
        try:
            config_dict = asdict(self.config)
            if config_dict.get("api_key"):
                config_dict["api_key"] = "<set>"  # never write secrets to disk
            meta = {
                "started": ts,
                "battle_tag": self.battle_tag,
                "format": self.config.battle_format,
                "provider": self.config.provider,
                "model": self.config.model,
                "config": config_dict,
            }
            with (self.run_dir / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, default=str, indent=2)
        except Exception:  # meta is a convenience — never block a battle on it
            pass

    def log_event(self, kind: str, **fields) -> None:
        entry = {
            "type": kind,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **fields,
        }
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def log_llm(self, kind: str, **fields) -> None:
        """Append one raw-conversation entry to llm.jsonl (the debugging dump).

        events.jsonl stays the lean, postmortem-friendly stream; everything
        the model actually saw and said — full prompts, per-round responses,
        tool results — lands here so any turn can be replayed offline.
        """
        entry = {
            "type": kind,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **fields,
        }
        try:
            with self._llm_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:  # pragma: no cover - disk issues must not kill a turn
            pass

    def record_turn(self, record: TurnRecord) -> None:
        self._seq += 1
        record.seq = self._seq
        self.records.append(record)
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "turn", **asdict(record)}, ensure_ascii=False) + "\n")
        self._maybe_schedule_compression()

    # --- background compression ------------------------------------------------

    def _uncovered_history_chars(self) -> int:
        return sum(
            len(r.request or "") + len(r.reason or "") + len(str(r.decision or ""))
            + sum(len(e) for e in (r.events or []))
            for r in self.records
            if r.seq > self.summary_covered_seq
        )

    def _maybe_schedule_compression(self) -> None:
        """Fire a background summarization when history outgrows the trigger.

        Runs in whatever loop scheduled record_turn (choose_move is always
        async). While a compression is in flight the trigger is suppressed
        and the prompt falls back to compact records — the critical path
        never waits on the LLM.
        """
        if self.compressor is None or self._compressing:
            return
        if self._uncovered_history_chars() // 4 < self.config.compression_trigger_tokens:
            return
        end_seq = self._seq
        self._compressing = True
        asyncio.create_task(self._compress(end_seq))

    async def _compress(self, end_seq: int) -> None:
        try:
            records = [r for r in self.records if r.seq <= end_seq]
            summary = await self.compressor.compress(
                records, self.config.summary_token_budget
            )
            if summary:
                # sequence swap: only records <= end_seq are covered; anything
                # appended after the snapshot stays verbatim/compact
                self.summary = summary
                self.summary_covered_seq = end_seq
                self.log_event(
                    "compressed",
                    covered_seq=end_seq,
                    chars=len(summary),
                    summary=summary,
                )
        except Exception as e:
            self.log_event("compress_error", covered_seq=end_seq,
                           error=f"{type(e).__name__}: {e}")
        finally:
            self._compressing = False

    def new_events(self, battle: "AbstractBattle") -> list[str]:
        """Protocol events that arrived since the previous decision point."""
        replay = getattr(battle, "_replay_data", None) or []
        events = [" ".join(str(part) for part in msg) for msg in replay[self._replay_offset :]]
        self._replay_offset = len(replay)
        return events

    def save_battle_kb(self) -> None:
        """Snapshot the knowledge tree to kb_tree.json (postmortem replay).

        Written via a temp file so a crash mid-write never corrupts the
        previous snapshot; failures are swallowed — the tree is an aid,
        not a dependency.
        """
        if self.battle_kb is None:
            return
        try:
            path = self.run_dir / "kb_tree.json"
            tmp = path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self.battle_kb.to_json(), f, ensure_ascii=False, indent=1)
            tmp.replace(path)
        except Exception:
            pass

    # --- assembly ------------------------------------------------------------

    def build_now(self, battle: "AbstractBattle") -> str:
        now = snapshot(battle) + "\n\nAvailable actions:\n" + available_actions(battle)
        self.last_now_text = now
        return now

    def _history_text(self) -> str | None:
        """Bounded view of past decisions: summary for old, verbatim for recent."""
        if not self.records:
            return None
        recent_k = self.config.recent_turns_verbatim
        split = max(0, len(self.records) - recent_k)
        old, recent = self.records[:split], self.records[split:]

        parts: list[str] = []
        if old:
            if self.summary:
                covered = [r for r in old if r.seq <= self.summary_covered_seq]
                uncovered = [r for r in old if r.seq > self.summary_covered_seq]
                parts.append(
                    f"[SUMMARY of earlier decisions (up to #{self.summary_covered_seq})]\n{self.summary}"
                )
                parts.extend(_compact_record(r) for r in uncovered)
            else:
                parts.extend(_compact_record(r) for r in old)
        if recent:
            for r in recent:
                lines = [f"--- Decision #{r.seq} (turn {r.turn}) ---", r.request.rstrip()]
                if r.events:
                    lines.append("Events: " + "; ".join(r.events[:8]))
                lines.append(
                    f"You decided: {r.decision} ({r.reason})"
                    if not r.fell_back
                    else f"FALLBACK to {r.decision} ({r.fallback_reason})"
                )
                parts.append("\n".join(lines))
        return "\n\n".join(parts) if parts else None

    def assemble_prompt(self, battle: "AbstractBattle", hook_fragments: list[str]):
        """Build (system, messages) for this decision point."""
        sections: list[str] = []
        info = (
            f"Format: {battle.format}\nTurn: {battle.turn}"
            f"\nYou are {battle.player_username}; opponent: {battle.opponent_username}"
        )
        sections.append(f"[BATTLE INFO]\n{info}")

        intel = self.intel.latest()
        if intel:
            intel_lines = [f"- {key}: {text}" for key, text in intel.items()]
            sections.append(
                "[INTEL — background analyses from your sub-agents]\n" + "\n".join(intel_lines)
            )

        if hook_fragments:
            sections.append(
                "[KNOWLEDGE — precomputed for this turn]\n" + "\n\n".join(hook_fragments)
            )

        history = self._history_text()
        if history:
            sections.append(f"[HISTORY]\n{history}")

        sections.append(f"[NOW — your current decision]\n{self.build_now(battle)}")

        user = "\n\n".join(sections)
        # soft-cap: shed verbatim history by dropping compact old records first
        cap_chars = self.config.context_token_soft_cap * 4
        return SYSTEM_PROMPT, [UserMessage(user)] if len(user) <= cap_chars else [
            UserMessage(user[:cap_chars] + "\n…[context truncated to fit budget]")
        ]
