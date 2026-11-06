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
from ..state.model import (
    BattleState,
    Claim,
    ReasoningLedger,
    ReasoningNote,
    StrategyBrief,
)
from ..state.prune import prune_state
from ..state.render import render_observed
from ..tasks.base import IntelStore
from .brief import deterministic_brief
from .compressor import TruncatingCompressor, _compact_record
from .records import TurnRecord, available_actions, snapshot

if TYPE_CHECKING:
    from poke_env.battle import AbstractBattle

    from ..config import AgentConfig

SYSTEM_PROMPT = """You are an expert competitive Pokemon battle agent playing on Pokemon Showdown. \
Each turn runs in two phases:

1. INFORMATION phase: gather what you need with the information tools \
(damage calculation, move/type/species/ability lookups, battle history — including \
your own past reasoning). Tool calls requested in the same round run in parallel. \
The phase ends as soon as you answer without tool calls. If you already know the \
best play, call `submit_decision` immediately and skip this phase.
2. DECISION phase: information tools are closed. You MUST call `submit_decision` \
to commit your action(s). Plain text is never interpreted as a move. If your \
submit is rejected, read the error and submit again.

`submit_decision` arguments: actions is a list of action objects. \
Each has slot (1-based active slot), kind, and the fields for that kind:
  - kind="move": move=<exact move id from the available list> and, in doubles, \
target=<one of that move's listed targets: auto/self/ally/opp1/opp2>.
  - kind="switch": switch_to=<exact species from the available switches list>.
  - kind="pass": only where the slot list says pass is a legal choice.
Choose ONLY from the listed options; anything else is rejected and you lose time.
- Singles: submit exactly one action (slot is ignored). \
Doubles: submit one action per active slot that has choices; slots shown as pass auto-pass.
- Pass terastallize=true only when the notes say it is available and it is the better play.
- Optional `reasoning_update`: when your plan changes (a threat is answered, the \
old line failed), send `win_conditions` (<=2), `commitments` (<=3) and/or `correction` \
to rewrite your committed plan. `[STRATEGY]` is your rolling brief — it is a summary, \
not a fact: if fresh observations contradict it, trust the observations and update the plan.
- You have a hard per-turn time budget (~45s). Be decisive: 1-2 rounds of \
information are usually enough — then commit.

Battle guidance: weigh type matchups, damage ranges vs both parties' HP, your win condition, \
the opponent's revealed moves and likely sets, status conditions, priority, and whether \
switching preserves momentum better than attacking. In doubles, weigh protect timing, \
speed order, spread moves (0.75x power against both foes), and redirection. \
Identify your best win condition early."""

NUDGE = (
    "No action committed yet. Call `submit_decision` now with your chosen action, "
    "or make one final round of tool calls if you are still missing critical information."
)

DECISION_PHASE_NUDGE = (
    "[DECISION PHASE] Information gathering is over — information tools are no "
    "longer available. Call `submit_decision` NOW with your chosen action(s) for "
    "this turn."
)

DECISION_PHASE_FORCED = (
    "[DECISION PHASE] You reached the information-gathering limit (rounds or "
    "time budget). Call `submit_decision` NOW with your best judgment."
)


def _sanitize_tag(battle_tag: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in battle_tag)


def _note_lines(note: ReasoningNote) -> list[str]:
    """Compact [REFLECTION] lines from a structured reasoning note."""
    lines: list[str] = []
    if note.intent:
        lines.append(f"Intent then: {note.intent}")
    if note.claims:
        lines.append("Beliefs then: " + "; ".join(note.claims))
    if note.correction:
        lines.append(f"Correction: {note.correction}")
    if note.directives:
        lines.append("Carry forward: " + "; ".join(note.directives))
    if note.open_questions:
        lines.append("Still unknown: " + "; ".join(note.open_questions))
    return lines


class ContextManager:
    """One instance per battle. Owns records, intel, and the run directory."""

    def __init__(
        self,
        battle_tag: str,
        config: "AgentConfig",
        kb=None,
        label: str | None = None,
    ):
        self.config = config
        self.battle_tag = battle_tag
        self.kb = kb
        self.intel = IntelStore()
        self.records: list[TurnRecord] = []
        self.seen_species: set[str] = set()

        # run directory + persistence. The label (the player's username)
        # keeps AI-vs-AI run dirs from colliding: both sides of one battle
        # share the battle_tag and can be created within the same second.
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{ts}_{_sanitize_tag(battle_tag)}"
        if label:
            name = f"{ts}_{_sanitize_tag(label)}_{_sanitize_tag(battle_tag)}"
        self.run_dir: Path = config.runs_dir / name
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
        # explicit per-battle harness state (state/model.BattleState) plus the
        # reasoning ledger. The player builds/refreshes ``state`` at team
        # preview and every decision point; ``battle_kb`` stays the live
        # alias the knowledge hooks read (it is ``state.kb``).
        self.state: BattleState | None = None
        self.ledger = ReasoningLedger()
        # NoteExtractor (context.reasoning), attached by the player when the
        # cheap model exists; None keeps the legacy raw-CoT reflection.
        self.note_extractor = None
        self._extracting = False
        self._reasoning_path = self.run_dir / "reasoning.jsonl"
        # rolling strategy brief (context/brief.LLMBriefCompressor), attached
        # by the player when the cheap model exists. ``_brief_in_flight``
        # suppresses overlapping rewrites; the swap is sequence-guarded.
        self.brief_compressor = None
        self._brief_in_flight = False
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
        data = asdict(record)
        # structured replay rows feed deterministic pruning; the joined
        # ``events`` text is the durable, human-readable form on disk
        data.pop("event_rows", None)
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "turn", **data}, ensure_ascii=False) + "\n")
        self._apply_pruning(record)
        self._maybe_schedule_reasoning_note(record)
        self._maybe_schedule_compression()

    # --- reasoning notes -------------------------------------------------------

    def _maybe_schedule_reasoning_note(self, record: TurnRecord) -> None:
        """Turn the raw CoT into a compact note (sync when possible).

        Deterministic extractors run inline so the note is available to the
        very next reflection; model-backed ones run as a background task so
        the critical path never waits on a second LLM call.
        """
        extractor = self.note_extractor
        if extractor is None:
            return
        note = extractor.extract_now(record, self.state)
        if note is not None:
            self._store_note(note, record.seq)
            return
        if self._extracting or record.cot is None:
            return
        self._extracting = True
        try:
            asyncio.create_task(self._extract_note(record))
        except RuntimeError:  # no running loop (sync callers/tests)
            self._extracting = False

    async def _extract_note(self, record: TurnRecord) -> None:
        try:
            note = await self.note_extractor.extract(record, self.state)
            if note is not None:
                self._store_note(note, record.seq)
        except Exception as e:
            self.log_event(
                "reasoning_note_error",
                seq=record.seq,
                error=f"{type(e).__name__}: {e}",
            )
        finally:
            self._extracting = False

    def _store_note(self, note: ReasoningNote, seq: int) -> None:
        if not note.seq:
            note.seq = seq
        self.ledger.add_note(note)
        self.log_event(
            "reasoning_note",
            seq=note.seq,
            turn=note.turn,
            source=note.source,
            intent=note.intent,
            claims=note.claims,
            correction=note.correction,
            directives=note.directives,
            open_questions=note.open_questions,
        )
        try:
            with self._reasoning_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps({"type": "note", **asdict(note)}, ensure_ascii=False)
                    + "\n"
                )
        except Exception:  # pragma: no cover - disk issues must not kill a turn
            pass
        self._maybe_schedule_brief()

    # --- deterministic claim pruning ------------------------------------------

    def _apply_pruning(self, record: TurnRecord) -> None:
        """Fold turn-order and reveal evidence into the inferred layer.

        Purely deterministic and failure-safe: a bad bound is worse than no
        bound, so the pruner is conservative (see ``state/prune.py``).
        """
        if self.state is None or not record.event_rows:
            return
        try:
            result = prune_state(
                self.state, record.event_rows, static_kb=self.kb
            )
        except Exception as e:
            self.log_event(
                "claims_prune_error",
                seq=record.seq,
                error=f"{type(e).__name__}: {e}",
            )
            return
        if not result.changed:
            return
        self.log_event(
            "claims_pruned",
            seq=record.seq,
            added=result.added,
            confirmed=result.confirmed,
            refuted=result.refuted,
            dropped=result.dropped,
            speed_bounds=result.speed_bounds,
        )
        self.save_state()

    # --- rolling strategy brief (L2) ------------------------------------------

    def _maybe_schedule_brief(self) -> None:
        """Rewrite the brief after enough new notes (background, race-free).

        Without a model the deterministic builder runs inline — it only
        lifts what the agent already committed, so it is cheap.
        """
        ledger = self.ledger
        covered = ledger.brief_covered_seq
        pending = [n for n in ledger.notes if n.seq > covered]
        if not pending:
            return

        if self.brief_compressor is None:
            ledger.brief = deterministic_brief(
                ledger,
                plan=self.state.plan if self.state is not None else None,
                records=self.records,
                covered_seq=pending[-1].seq,
                turn=pending[-1].turn,
            )
            ledger.brief_covered_seq = pending[-1].seq
            self.log_event(
                "strategy_brief",
                source="deterministic",
                covered_seq=ledger.brief_covered_seq,
                brief=asdict(ledger.brief),
            )
            self.save_state()
            return

        if self._brief_in_flight:
            return
        if len(pending) < max(1, self.config.reasoning_brief_every):
            return
        end_seq = pending[-1].seq
        self._brief_in_flight = True
        try:
            asyncio.create_task(self._compress_brief(end_seq, pending))
        except RuntimeError:  # no running loop (sync callers/tests)
            self._brief_in_flight = False
            self._store_fallback_brief(end_seq)

    async def _compress_brief(
        self, end_seq: int, pending: list[ReasoningNote]
    ) -> None:
        """Rewrite covering exactly the pending (uncovered) notes.

        The note snapshot is taken at schedule time; ``end_seq`` fences the
        records for the same reason, so a slower run cannot absorb — or
        clobber — state that arrived after it was scheduled.
        """
        try:
            notes = [n for n in pending if n.seq <= end_seq]
            records = [r for r in self.records if r.seq <= end_seq]
            brief = await self.brief_compressor.compress(
                brief=self.ledger.brief,
                notes=notes,
                records=records[-4:],
                plan=self.state.plan if self.state is not None else None,
                turn=records[-1].turn if records else 0,
            )
            if brief is not None:
                self._store_brief(brief, end_seq)
            elif self.ledger.brief is None:
                self._store_fallback_brief(end_seq)
        except Exception as e:
            self.log_event(
                "strategy_brief_error",
                covered_seq=end_seq,
                error=f"{type(e).__name__}: {e}",
            )
            if self.ledger.brief is None:
                self._store_fallback_brief(end_seq)
        finally:
            self._brief_in_flight = False

    def _store_brief(self, brief: StrategyBrief, end_seq: int) -> None:
        # sequence swap: only notes <= end_seq are covered; later notes stay
        # for the next rewrite and can never be clobbered by this snapshot
        brief.covered_seq = end_seq
        self.ledger.brief = brief
        self.ledger.brief_covered_seq = end_seq
        self.log_event(
            "strategy_brief",
            source=brief.source,
            covered_seq=end_seq,
            brief=asdict(brief),
        )
        self.save_state()

    def _store_fallback_brief(self, end_seq: int) -> None:
        notes = [n for n in self.ledger.notes if n.seq <= end_seq]
        brief = deterministic_brief(
            self.ledger,
            plan=self.state.plan if self.state is not None else None,
            records=[r for r in self.records if r.seq <= end_seq],
            covered_seq=end_seq,
            turn=notes[-1].turn if notes else 0,
        )
        self._store_brief(brief, end_seq)

    def _strategy_text(self) -> str | None:
        """``[STRATEGY]``: the rolling brief, capped to keep prompts tight."""
        brief = self.ledger.brief
        if brief is None or brief.empty():
            return None
        lines = ["[STRATEGY — rolling brief, revise as evidence changes]"]
        if brief.stance:
            lines.append(brief.stance)
        if brief.win_conditions:
            lines.append("Win: " + "; ".join(brief.win_conditions))
        if brief.threats:
            lines.append("Answer: " + "; ".join(brief.threats))
        if brief.commitments:
            lines.append("Commit: " + "; ".join(brief.commitments))
        if brief.beliefs:
            lines.append("Reads: " + "; ".join(brief.beliefs))
        if brief.open_questions:
            lines.append("Open: " + "; ".join(brief.open_questions))
        text = "\n".join(lines)
        return text if len(text) <= 600 else text[:599].rstrip() + "…"

    # --- model-written plan updates -------------------------------------------

    def apply_reasoning_update(self, update, turn: int) -> None:
        """Apply ``submit_decision.reasoning_update`` to the committed layer.

        Replacement semantics (not append): the model revises its own plan.
        Only called after a decision validated, so a rejected submission
        cannot rewrite the plan. Failures are logged, never raised.
        """
        if not isinstance(update, dict) or self.state is None:
            return
        plan = self.state.plan
        changed = False
        try:
            if isinstance(update.get("win_conditions"), list):
                plan.win_conditions = [
                    str(v).strip()[:120]
                    for v in update["win_conditions"]
                    if str(v).strip()
                ][:2]
                changed = True
            if isinstance(update.get("commitments"), list):
                plan.commitments = [
                    str(v).strip()[:120]
                    for v in update["commitments"]
                    if str(v).strip()
                ][:3]
                changed = True
            correction = update.get("correction")
            if isinstance(correction, str) and correction.strip():
                self.ledger.claims.append(
                    Claim(
                        subject="plan",
                        kind="plan",
                        text=correction.strip()[:200],
                        confidence="high",
                        source="model",
                        turn=turn,
                    )
                )
                changed = True
        except Exception as e:
            self.log_event(
                "plan_update_error", turn=turn, error=f"{type(e).__name__}: {e}"
            )
            return
        if not changed:
            return
        plan.revision += 1
        self.log_event(
            "plan_update",
            turn=turn,
            revision=plan.revision,
            win_conditions=plan.win_conditions,
            commitments=plan.commitments,
            correction=update.get("correction"),
        )
        self.save_state()

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

    def new_events(
        self,
        battle: "AbstractBattle",
        *,
        rows_out: list[list[str]] | None = None,
    ) -> list[str]:
        """Protocol events that arrived since the previous decision point.

        ``rows_out`` (optional) receives the structured replay rows as well;
        deterministic claim pruning parses those instead of the joined text,
        which is ambiguous for names containing spaces.
        """
        replay = getattr(battle, "_replay_data", None) or []
        rows = [list(msg) for msg in replay[self._replay_offset :]]
        self._replay_offset = len(replay)
        if rows_out is not None:
            rows_out.extend(rows)
        return [" ".join(str(part) for part in msg) for msg in rows]

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

    def save_state(self) -> None:
        """Snapshot the explicit harness state to state.json (atomic write)."""
        if self.state is None:
            return
        try:
            path = self.run_dir / "state.json"
            tmp = path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self.state.to_json(), f, ensure_ascii=False, indent=1)
            tmp.replace(path)
        except Exception:
            pass

    # --- assembly ------------------------------------------------------------

    def build_now(self, battle: "AbstractBattle") -> str:
        """``[NOW]`` text: the observed board, then this turn's legal actions.

        Rendered from persisted state when available (so the saved state is
        exactly what the model saw); ``snapshot`` is the fallback before the
        player has initialized state.
        """
        board = (
            render_observed(self.state) if self.state is not None else snapshot(battle)
        )
        now = board + "\n\nAvailable actions:\n" + available_actions(battle)
        self.last_now_text = now
        return now

    def _reflection_text(self) -> str | None:
        """[REFLECTION]: the last decision and what came of it.

        Shown from the second decision onward so the model can adjust instead
        of repeating itself. Only the most recent record — the verbatim
        [HISTORY] window already covers anything older.
        """
        if not self.records:
            return None
        r = self.records[-1]
        lines = ["[REFLECTION — your last decision and its outcome]"]
        if r.fell_back:
            lines.append(
                f"Last decision loop FAILED — fell back to {r.decision} "
                f"(reason: {r.fallback_reason}). Do not repeat that failure: "
                "call `submit_decision` correctly this time."
            )
        else:
            lines.append(f"Your last decision: {r.decision} — {r.reason}")
            note = self.ledger.note_for_seq(r.seq)
            if note is not None and (
                note.claims
                or note.correction
                or note.directives
                or note.open_questions
                or (note.intent and note.intent != (r.reason or ""))
            ):
                lines.extend(_note_lines(note))
            elif r.cot:
                excerpt = r.cot[:600]
                lines.append(
                    "Your reasoning then (excerpt): " + excerpt
                    + ("…" if len(r.cot) > 600 else "")
                )
        if r.events:
            lines.append("What happened: " + "; ".join(r.events[:6]))
        lines.append("Learn from that outcome, then decide this turn.")
        return "\n".join(lines)

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

        strategy = self._strategy_text()
        if strategy:
            sections.append(strategy)

        reflection = self._reflection_text()
        if reflection:
            sections.append(reflection)

        sections.append(f"[NOW — your current decision]\n{self.build_now(battle)}")

        user = "\n\n".join(sections)
        # soft-cap: shed verbatim history by dropping compact old records first
        cap_chars = self.config.context_token_soft_cap * 4
        return SYSTEM_PROMPT, [UserMessage(user)] if len(user) <= cap_chars else [
            UserMessage(user[:cap_chars] + "\n…[context truncated to fit budget]")
        ]
