"""Battle-level COT text: capping, sanitizing, and maintainer input.

The COT is a short narrative — "how this game is won" — rewritten after every
turn. Everything here is pure and bounded so a runaway model output can never
bloat the prompt: the maintainer's input is built from explicit per-part
budgets (thinking gets the largest share, because it is the new information)
and the stored text is hard-capped to fit the ``[STRATEGY]`` section.

Why a tail extractor: on the reasoning gateway most completions return
thinking only (the visible answer is empty or truncated at max_tokens). When
that happens the maintainer falls back to the last *finished* paragraph of
the thinking, which is the closest thing to a conclusion the model produced.
"""

from __future__ import annotations

import re

MAX_COT_CHARS = 520  # + the [STRATEGY] header stays under the 600-char cap
MAX_THINKING_HEAD = 1500
MAX_THINKING_TAIL = 3500
MIN_TAIL_CHARS = 40
MIN_THINKING_CHARS = 400  # below this the thinking section is not worth a call

_SENTENCE_END = (".", "!", "?", "。", "！", "？", "…", '"')
_TAIL_MARKERS = ("plan:", "next step:", "conclusion:")
_FENCE = re.compile(r"```")


def cap_thinking(
    text: str | None,
    *,
    head: int = MAX_THINKING_HEAD,
    tail: int = MAX_THINKING_TAIL,
) -> str | None:
    """Head + tail of a long thinking dump, with the middle elided.

    The model states the setup at the start and its conclusion at the end;
    the middle is usually arithmetic. Keeping both ends costs nothing and
    preserves the line of reasoning that matters.
    """
    if not text:
        return None
    text = text.strip()
    if len(text) <= head + tail:
        return text
    skipped = len(text) - head - tail
    return f"{text[:head]}\n…[middle {skipped} chars elided]…\n{text[-tail:]}"


def sanitize_cot(text: str | None, *, limit: int = MAX_COT_CHARS) -> str:
    """One clean paragraph from a model answer (or '' when unusable).

    Strips code fences and markdown heading lines (the agent may imitate the
    JSON-style prompts of its neighbours), collapses whitespace, and cuts at
    the last sentence boundary that fits ``limit`` so the text never ends
    mid-word.
    """
    if not text:
        return ""
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or _FENCE.search(line):
            continue
        if line.startswith("#") or line.lower().startswith(("plan:", "cot:")):
            line = line.split(":", 1)[1].strip() if ":" in line else ""
            if not line:
                continue
        lines.append(line)
    body = " ".join(lines)
    body = " ".join(body.split())
    if not body:
        return ""
    if len(body) > limit:
        cut = body[:limit]
        # prefer the last finished sentence inside the budget
        best = max(cut.rfind(ch) for ch in (".", "!", "?", "。", "！", "？"))
        body = cut[: best + 1] if best >= MIN_TAIL_CHARS else cut.rstrip()
        if len(body) > limit:
            body = body[: limit - 1].rstrip() + "…"
    return body.strip()


def thinking_tail(text: str | None) -> str:
    """The final usable paragraph of a thinking block, or ''.

    Used only when the visible answer is empty: prefer an explicit marker
    (``Plan:`` / ``Next step:`` / ``Conclusion:``, the last one wins), else
    the last paragraph that ends with sentence punctuation and is long
    enough to be a conclusion rather than a stray label.
    """
    if not text:
        return ""
    lowered = text.lower()
    marks = [
        (lowered.rfind(marker), marker)
        for marker in _TAIL_MARKERS
    ]
    marks = [(i, marker) for i, marker in marks if i != -1]
    if marks:
        index, marker = max(marks)
        after = text[index + len(marker) :].strip()
        if len(after) >= MIN_TAIL_CHARS:
            return after
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for paragraph in reversed(paragraphs):
        if len(paragraph) >= MIN_TAIL_CHARS and paragraph.endswith(_SENTENCE_END):
            return paragraph
    return ""


def build_cot_input(
    *,
    turn: int,
    seq: int,
    previous: str | None,
    board: str | None,
    record,
    prev_record=None,
    plan=None,
    limit: int = 8000,
) -> str:
    """The maintainer prompt body, budgeted per part.

    ``record.events`` are the protocol events *since the previous decision
    point* — the outcome of ``prev_record``'s decision, never of ``record``'s,
    which has not resolved yet. The two are labelled apart so the maintainer
    cannot credit a result to the move that did not produce it.

    The thinking gets whatever room the fixed parts leave: the final clamp
    can only cut from the tail, and the tail is the conclusion this rewrite
    depends on.
    """
    board_budget = max(800, int(limit * 0.30))

    parts: list[str] = [f"Turn {turn} (decision #{seq})"]
    if previous:
        parts.append("PREVIOUS COT (revise this, do not append):\n" + previous[:800])

    decision = getattr(record, "decision", None) or "none"
    reason = (getattr(record, "reason", None) or "").strip()
    events = list(getattr(record, "events", []) or [])[-12:]
    last_decision = getattr(prev_record, "decision", None) if prev_record else None
    last_reason = ((getattr(prev_record, "reason", None) or "").strip() if prev_record else "")

    outcome: list[str] = []
    if last_decision:
        line = f"YOUR PREVIOUS DECISION: {last_decision}"
        if last_reason:
            line += f" — {last_reason[:300]}"
        outcome.append(line)
        if events:
            outcome.append(
                "WHAT CAME OF IT (the only outcome known — the agent saw this "
                "before committing): " + "; ".join(e[:120] for e in events)
            )
    elif events:
        outcome.append(
            "EVENTS BEFORE THE DECISION: " + "; ".join(e[:120] for e in events)
        )
    outcome.append(f"JUST COMMITTED (outcome unknown at this point): {decision}")
    if reason:
        outcome.append(f"YOUR STATED REASON: {reason[:400]}")
    parts.append("\n".join(outcome))

    if plan is not None:
        bits = []
        if getattr(plan, "line", None):
            bits.append(f"line: {plan.line}")
        if getattr(plan, "win_conditions", None):
            bits.append("win: " + "; ".join(plan.win_conditions))
        if getattr(plan, "commitments", None):
            bits.append("commit: " + "; ".join(plan.commitments))
        if bits:
            parts.append("COMMITTED PLAN (written by you, authoritative): " + " | ".join(bits))

    if board:
        parts.append("BOARD YOU SAW:\n" + board[:board_budget])

    fixed = "\n\n".join(parts)
    header = "PRIVATE THINKING (verbatim, may be truncated):\n"
    # reserve the separators plus the elision marker: ``cap_thinking`` adds
    # the marker on top of head + tail, and overrunning would cut the tail
    room = min(
        MAX_THINKING_HEAD + MAX_THINKING_TAIL,
        limit - len(fixed) - len(header) - 96,
    )
    if room >= MIN_THINKING_CHARS:
        head = min(MAX_THINKING_HEAD, room // 3)
        thinking = cap_thinking(
            getattr(record, "thinking", None), head=head, tail=room - head
        )
        if thinking:
            fixed = fixed + "\n\n" + header + thinking
    return fixed[:limit]


def seed_cot_text(reason: str | None = None, plan=None) -> str:
    """The T1 COT before any maintenance call has run, or ''.

    Deterministic and honest: it only restates what the agent already
    committed at preview. When there is nothing to restate it returns ''
    (and no COT is stored) rather than inventing a placeholder plan — an
    empty ``[STRATEGY]`` slot is better than a fake one, and the rolling
    brief still renders on its own.
    """
    reason = " ".join((reason or "").split()).strip()
    if reason:
        return sanitize_cot("Opening line: " + reason)
    if plan is not None and getattr(plan, "line", None):
        return sanitize_cot(plan.line)
    if plan is not None and getattr(plan, "win_conditions", None):
        return sanitize_cot("Win by: " + "; ".join(plan.win_conditions))
    return ""