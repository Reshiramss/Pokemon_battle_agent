"""Tests for ContextManager compression and prompt assembly (Step 6)."""

from __future__ import annotations

import asyncio
import json
import logging

from helpers import ScriptedProvider, make_battle, make_kb

from pokeagent.config import AgentConfig
from pokeagent.context.compressor import LLMCompressor, TruncatingCompressor
from pokeagent.context.manager import ContextManager
from pokeagent.context.records import TurnRecord
from pokeagent.logging_setup import setup_logging
from pokeagent.providers.base import LLMResponse


def make_cm(tmp_path, **overrides) -> ContextManager:
    config = AgentConfig(runs_dir=tmp_path / "runs", **overrides)
    return ContextManager(battle_tag="battle-1", config=config, kb=None)


def add_turns(cm: ContextManager, n: int, request_chars: int = 200) -> None:
    for i in range(n):
        cm.record_turn(
            TurnRecord(
                seq=0, turn=i + 1,
                request=f"[NOW] turn {i + 1} " + "x" * request_chars,
                events=[f"|damage|p1a: {i}"],
                decision=f"move:m{i}", reason=f"reason {i}",
                cot=f"thinking about {i}",
                input_tokens=100, output_tokens=50,
                elapsed_s=1.5,
            )
        )


def test_compression_triggers_swaps_summary(tmp_path):
    async def scenario():
        # small trigger so 20 turns (~1250 estimated tokens) overflow it
        cm = make_cm(tmp_path, compression_trigger_tokens=1000)
        provider = ScriptedProvider(
            [LLMResponse(text="SUMMARY: opponent keeps switching.", stop_reason="end_turn")]
        )
        cm.compressor = LLMCompressor(provider)
        add_turns(cm, 20)

        # background task was scheduled; let it finish
        for _ in range(50):
            if cm.summary:
                break
            await asyncio.sleep(0.02)
        assert cm.summary == "SUMMARY: opponent keeps switching."
        covered = cm.summary_covered_seq
        assert 1 <= covered < 20  # fired as soon as the estimate crossed the trigger
        assert len(provider.calls) == 1

        # prompt now carries the summary section instead of full history
        battle = make_battle()
        _system, messages = cm.assemble_prompt(battle, [])
        prompt = messages[0].content
        assert f"[SUMMARY of earlier decisions (up to #{covered})]" in prompt
        assert "SUMMARY: opponent keeps switching." in prompt

        events = (cm.run_dir / "events.jsonl").read_text()
        assert '"compressed"' in events
        assert f'"covered_seq": {covered}' in events

    asyncio.run(scenario())


def test_compression_suppressed_while_in_flight(tmp_path):
    async def scenario():
        cm = make_cm(tmp_path, compression_trigger_tokens=100)
        provider = ScriptedProvider([])  # never resolves from the model's view
        # make the compressor slow: wrap with a sleeping provider
        class _SlowProvider(ScriptedProvider):
            async def complete(self, **kw):
                self.calls.append(kw)
                await asyncio.sleep(0.3)
                return LLMResponse(text="late summary", stop_reason="end_turn")

        slow = _SlowProvider([LLMResponse(text="late summary", stop_reason="end_turn")])
        cm.compressor = LLMCompressor(slow)

        add_turns(cm, 10)  # triggers a compression
        assert cm._compressing
        await asyncio.sleep(0.05)  # let the background task start its LLM call
        assert len(slow.calls) == 1
        add_turns(cm, 10)  # suppressed: still one in flight
        assert len(slow.calls) == 1
        assert not cm.summary  # not swapped yet

        await asyncio.sleep(0.5)
        assert cm.summary == "late summary"
        assert cm._compressing is False
        # end_seq was captured when the compression was scheduled —
        # partway through the first 10 turns, before the rest were added
        assert 1 <= cm.summary_covered_seq <= 10
        assert 20 >= len(cm.records)  # nothing was lost; records only append

    asyncio.run(scenario())


def test_compression_error_is_logged_not_raised(tmp_path):
    async def scenario():
        cm = make_cm(tmp_path, compression_trigger_tokens=100)
        provider = ScriptedProvider([RuntimeError("model down")])
        cm.compressor = LLMCompressor(provider)
        add_turns(cm, 10)
        for _ in range(50):
            if not cm._compressing:
                break
            await asyncio.sleep(0.02)
        assert cm.summary is None
        events = (cm.run_dir / "events.jsonl").read_text()
        assert "compress_error" in events

    asyncio.run(scenario())


def test_no_compressor_no_trigger(tmp_path):
    async def scenario():
        cm = make_cm(tmp_path, compression_trigger_tokens=1)
        add_turns(cm, 10)  # would trigger if a compressor were attached
        await asyncio.sleep(0.05)
        assert cm.summary is None
        # prompt still assembles, with compact history
        _system, messages = cm.assemble_prompt(make_battle(), [])
        assert "move:m9" in messages[0].content

    asyncio.run(scenario())


def test_prompt_recent_turns_verbatim(tmp_path):
    async def scenario():
        cm = make_cm(tmp_path, recent_turns_verbatim=3)
        add_turns(cm, 10)
        prompt = cm.assemble_prompt(make_battle(), [])[1][0].content
        # old turns compacted: their verbatim request text is gone,
        # but the compact line keeps the decision + reason
        assert "[NOW] turn 1 " not in prompt
        assert "(reason 0)" in prompt
        # recent 3 verbatim, with the request text they saw
        assert "reason 9" in prompt and "reason 8" in prompt
        assert "[NOW] turn 8 " + "x" * 200 in prompt

    asyncio.run(scenario())


def test_prompt_soft_cap_truncates(tmp_path):
    async def scenario():
        cm = make_cm(tmp_path, context_token_soft_cap=500)
        add_turns(cm, 10, request_chars=300)
        prompt = cm.assemble_prompt(make_battle(), [])[1][0].content
        assert len(prompt) <= 500 * 4 + 100
        assert "context truncated" in prompt

    asyncio.run(scenario())


def test_truncating_compressor_fits_budget():
    async def scenario():
        compressor = TruncatingCompressor()
        records = [
            TurnRecord(seq=i + 1, turn=i + 1, request="r" * 100,
                       decision=f"move:m{i}", reason=f"reason {i}")
            for i in range(50)
        ]
        text = await compressor.compress(records, budget_tokens=100)
        assert len(text) <= 100 * 4
        assert "oldest decisions omitted" in text

    asyncio.run(scenario())


def test_run_artifacts_meta_and_llm_trace(tmp_path):
    """Every run dir persists a config snapshot and an llm.jsonl trace."""
    cm = make_cm(tmp_path)
    meta = json.loads((cm.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["battle_tag"] == "battle-1"
    assert meta["config"]["server_port"] == 8964
    assert meta["model"]

    cm.log_llm("turn_prompt", turn=1, prompt="hello", tools=["damage_calc"])
    cm.log_llm(
        "llm_response", turn=1, round=0, text="thinking",
        tool_calls=[{"id": "1", "name": "submit_decision", "arguments": {}}],
        stop_reason="tool_use",
    )
    entries = [
        json.loads(line)
        for line in (cm.run_dir / "llm.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    kinds = [e["type"] for e in entries]
    assert kinds == ["turn_prompt", "llm_response"]
    assert entries[0]["prompt"] == "hello"
    assert entries[1]["tool_calls"][0]["name"] == "submit_decision"


def test_setup_logging_writes_file_and_is_idempotent(tmp_path):
    root = logging.getLogger()
    before = len(root.handlers)
    config = AgentConfig(runs_dir=tmp_path / "runs")
    path = setup_logging(config)
    logging.getLogger("test_ctx").warning("boom marker")
    for handler in root.handlers:
        handler.flush()
    assert "boom marker" in path.read_text(encoding="utf-8")

    # a second call swaps the file handler instead of stacking another one
    # (same-second calls reuse the timestamped path; that's fine)
    setup_logging(config)
    ours = [h for h in root.handlers if getattr(h, "_pokeagent_log", False)]
    assert len(ours) == 1
    logging.getLogger("test_ctx").warning("second marker")
    for handler in root.handlers:
        handler.flush()
    assert "second marker" in path.read_text(encoding="utf-8")
    # leave the root logger as we found it
    for handler in ours:
        root.removeHandler(handler)
    assert len(root.handlers) == before


def test_knowledge_fragment_and_intel_sections(tmp_path):
    async def scenario():
        cm = make_cm(tmp_path)
        cm.intel.put("threat:torterra", "switch Furret in safely")
        add_turns(cm, 2)
        prompt = cm.assemble_prompt(
            make_battle(), ["KNOWLEDGE: fire beats grass"]
        )[1][0].content
        assert "[INTEL" in prompt and "threat:torterra" in prompt
        assert "[KNOWLEDGE" in prompt and "fire beats grass" in prompt
        assert "[NOW" in prompt

    asyncio.run(scenario())
