"""Summarize runs/ directories: turn counts, fallback rate, latency, tokens.

Usage:
    python scripts/postmortem.py [runs_dir] [--last N]

Reads each run's events.jsonl (the disk truth) and prints per-battle and
aggregate metrics. No LLM calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct / 100), len(ordered) - 1)
    return ordered[idx]


def read_run(run_dir: Path) -> dict | None:
    events_path = run_dir / "events.jsonl"
    if not events_path.is_file():
        return None
    turns, fallbacks, subagents, elapsed = [], 0, 0, []
    in_tok = out_tok = 0
    compressed = 0
    won = lost = None
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = entry.get("type")
        if kind == "turn":
            turns.append(entry)
            if entry.get("fell_back"):
                fallbacks += 1
            elapsed.append(float(entry.get("elapsed_s") or 0))
            in_tok += int(entry.get("input_tokens") or 0)
            out_tok += int(entry.get("output_tokens") or 0)
            if entry.get("cot"):
                pass  # CoT present; not counted, just recorded verbatim on disk
        elif kind == "subagent_done":
            subagents += 1
        elif kind == "compressed":
            compressed += 1
        elif kind == "battle_finished":
            won = bool(entry.get("won"))
            lost = bool(entry.get("lost"))
    if not turns:
        return None
    return {
        "run": run_dir.name,
        "turns": len(turns),
        "fallbacks": fallbacks,
        "fallback_rate": fallbacks / len(turns),
        "cot_logged": sum(1 for t in turns if t.get("cot")),
        "reasoned": sum(1 for t in turns if t.get("reason")),
        "subagents": subagents,
        "compressions": compressed,
        "elapsed": elapsed,
        "in_tokens": in_tok,
        "out_tokens": out_tok,
        "won": won,
        "lost": lost,
    }


def fmt(result: dict) -> str:
    outcome = {True: "WIN", False: "LOSS", None: "?"}[result["won"]]
    mean = sum(result["elapsed"]) / len(result["elapsed"]) if result["elapsed"] else 0
    p95 = percentile(result["elapsed"], 95)
    return (
        f"{result['run']}: {outcome} | turns {result['turns']} "
        f"| fallbacks {result['fallbacks']} ({result['fallback_rate']:.0%}) "
        f"| CoT logged {result['cot_logged']}/{result['turns']} "
        f"| latency mean {mean:.1f}s p95 {p95:.1f}s "
        f"| tokens in {result['in_tokens']} out {result['out_tokens']} "
        f"| sub-agents {result['subagents']} | compressions {result['compressions']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", nargs="?", default="runs")
    parser.add_argument("--last", type=int, default=None,
                        help="only the N most recent runs")
    args = parser.parse_args()

    root = Path(args.runs_dir)
    if not root.is_dir():
        print(f"no runs directory at {root}", file=sys.stderr)
        raise SystemExit(1)

    results = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        result = read_run(run_dir)
        if result:
            results.append(result)

    if args.last:
        results = results[-args.last:]

    if not results:
        print("no recorded turns found")
        return

    for result in results:
        print(fmt(result))

    total_turns = sum(r["turns"] for r in results)
    total_fb = sum(r["fallbacks"] for r in results)
    all_elapsed = [e for r in results for e in r["elapsed"]]
    print("-" * 72)
    print(
        f"TOTAL: {len(results)} battles, {total_turns} turns, "
        f"fallback rate {total_fb / total_turns:.1%}, "
        f"latency mean {sum(all_elapsed) / len(all_elapsed):.1f}s "
        f"p95 {percentile(all_elapsed, 95):.1f}s, "
        f"sub-agents {sum(r['subagents'] for r in results)}, "
        f"wins {sum(1 for r in results if r['won'])}"
    )


if __name__ == "__main__":
    main()
