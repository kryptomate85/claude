"""Tests for the agent loop, notifier chunking and OANDA parsing (offline)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, List

import pandas as pd

import agent as agent_mod
from tools import data_fetcher
from tools.notifier import TelegramNotifier, split_message


def test_split_message_respects_limit() -> None:
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(300))
    chunks = split_message(text, 1000)
    assert all(len(c) <= 1000 for c in chunks)
    assert "".join(c.replace("\n", "") for c in chunks) == text.replace("\n", "")


def test_notifier_disabled_returns_false() -> None:
    assert TelegramNotifier("", "").send("hi") is False


class FakeResponse:
    def __init__(self, status: int, payload: Any) -> None:
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> Any:
        return self._payload


class FakeSession:
    def __init__(self, responses: List[FakeResponse]) -> None:
        self.responses = responses
        self.headers: dict = {}
        self.calls: List[dict] = []

    def get(self, url: str, params: dict, timeout: float) -> FakeResponse:
        self.calls.append(params)
        return self.responses.pop(0)

    def post(self, url: str, json: dict, timeout: float) -> FakeResponse:
        self.calls.append(json)
        return self.responses.pop(0)


def test_notifier_markdown_fallback() -> None:
    session = FakeSession([FakeResponse(400, {"description": "Bad Request: can't parse entities"}), FakeResponse(200, {"ok": True})])
    assert TelegramNotifier("tok", "1", session=session).send("*broken_md") is True
    assert "parse_mode" not in session.calls[1]


def _candle(ts: str, complete: bool = True) -> dict:
    return {"time": ts, "complete": complete, "volume": 10, "mid": {"o": "1.1", "h": "1.2", "l": "1.0", "c": "1.15"}}


def test_oanda_client_parses_and_drops_incomplete(settings) -> None:
    payload = {"candles": [_candle("2026-06-01T07:00:00.000000000Z"), _candle("2026-06-01T07:05:00.000000000Z", complete=False)]}
    client = data_fetcher.OandaClient(settings, session=FakeSession([FakeResponse(200, payload)]))
    df = client.fetch_candles("EUR_USD", "M5", pd.Timestamp("2026-06-01", tz="UTC").to_pydatetime(), pd.Timestamp("2026-06-02", tz="UTC").to_pydatetime())
    assert len(df) == 1 and df["Close"].iloc[0] == 1.15


def test_fetch_data_falls_back_to_cache(settings, candles, monkeypatch) -> None:
    settings.parquet_path.parent.mkdir(parents=True)
    candles.to_parquet(settings.parquet_path)

    def boom(*_a: Any, **_k: Any) -> None:
        raise data_fetcher.DataFetchError("network down")

    monkeypatch.setattr(data_fetcher.OandaClient, "fetch_candles", boom)
    result = data_fetcher.fetch_data(settings=settings)
    assert result["status"] == "stale_cache" and result["rows"] == len(candles)


def test_fetch_data_error_without_cache(settings, monkeypatch) -> None:
    monkeypatch.setattr(data_fetcher.OandaClient, "fetch_candles", lambda *a, **k: (_ for _ in ()).throw(data_fetcher.DataFetchError("down")))
    assert data_fetcher.fetch_data(settings=settings)["status"] == "error"


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
    assert "rule-based" in sent[-1] and "Scorecard" in sent[-1]
