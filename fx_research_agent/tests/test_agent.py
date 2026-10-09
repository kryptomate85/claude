"""Tests for the optional LLM analyst (skipped when the openai package is absent)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, List

import pytest

pytest.importorskip("openai")

import agent as agent_mod  # noqa: E402


def _tool_call(i: int, name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(id=f"call_{i}", type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class ScriptedLLM:
    """Stand-in chat client that replays a fixed tool-calling script."""

    def __init__(self, steps: List[SimpleNamespace]) -> None:
        self.steps = steps
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(choices=[SimpleNamespace(message=self.steps.pop(0))])


def test_agent_full_loop(settings, candles, monkeypatch) -> None:
    monkeypatch.setattr(agent_mod, "fetch_data", lambda **k: {"status": "ok", "rows": len(candles)})
    real_suite = agent_mod.run_backtest_suite
    monkeypatch.setattr(agent_mod, "run_backtest_suite", lambda **k: real_suite(data=candles, **k))
    final = "*Report*\n" + "analysis line\n" * 60
    script = [
        SimpleNamespace(content=None, tool_calls=[_tool_call(1, "send_telegram_update", {"message": "started"})]),
        SimpleNamespace(content=None, tool_calls=[_tool_call(2, "fetch_data", {})]),
        SimpleNamespace(content=None, tool_calls=[_tool_call(3, "run_backtest_suite", {})]),
        SimpleNamespace(content=None, tool_calls=[_tool_call(4, "send_telegram_update", {"message": final})]),
    ]
    sent: List[str] = []
    bot = agent_mod.ResearchAgent(settings, client=ScriptedLLM(script), send=lambda m: sent.append(m) or True)
    summary = bot.run()
    assert summary["status"] == "ok" and summary["final_report_sent"]
    assert summary["tools_called"] == ["send_telegram_update", "fetch_data", "run_backtest_suite", "send_telegram_update"]
    assert sent[0] == "started" and sent[-1] == final


def test_agent_llm_failure_uses_fallback(settings, candles, monkeypatch) -> None:
    monkeypatch.setattr(agent_mod, "fetch_data", lambda **k: {"status": "ok"})
    real_suite = agent_mod.run_backtest_suite
    monkeypatch.setattr(agent_mod, "run_backtest_suite", lambda **k: real_suite(data=candles, **k))

    class DeadLLM:
        chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **k: (_ for _ in ()).throw(RuntimeError("down"))))

    sent: List[str] = []
    bot = agent_mod.ResearchAgent(settings, client=DeadLLM(), send=lambda m: sent.append(m) or True)
    monkeypatch.setattr(bot, "_complete", lambda messages: (_ for _ in ()).throw(RuntimeError("LLM unavailable")))
    summary = bot.run()
    assert summary["status"] == "ok" and summary["llm_error"]
    assert "LLM analyst unavailable" in sent[-1] and "Scorecard" in sent[-1]
