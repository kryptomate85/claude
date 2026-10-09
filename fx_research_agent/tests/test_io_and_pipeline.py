"""Tests for notifier chunking, OANDA parsing, config and the default pipeline (offline)."""

from __future__ import annotations

import json
from typing import Any, List

import pandas as pd
import pytest

import config
import main
from tools import data_fetcher
from tools.backtest_runner import run_backtest_suite
from tools.notifier import TelegramNotifier, split_message
from tools.report import build_report


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


def test_config_does_not_require_openai(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for key in config.SECRET_KEYS:
        monkeypatch.setenv(key, "x")
    assert config.load_settings().openai_api_key == ""
    with pytest.raises(config.ConfigError, match="OPENAI_API_KEY"):
        config.load_settings(config.SECRET_KEYS + config.LLM_SECRET_KEYS)


def test_build_report(settings, candles) -> None:
    report = run_backtest_suite(settings=settings, notify=False, data=candles)
    text = build_report(report, {"status": "stale_cache"})
    for section in ("*Scorecard*", "*Analysis*", "*Verdict & Tactical Recommendations*", "`london_orb`", "cached candles"):
        assert section in text
    assert build_report(report, None) == build_report(report, None)  # deterministic


def test_default_pipeline_needs_no_llm(settings, candles, monkeypatch) -> None:
    import tools.backtest_runner as runner
    import tools.data_fetcher as fetcher
    import tools.notifier as notifier

    sent: List[str] = []
    monkeypatch.setattr(notifier, "send_telegram_message", lambda m, **k: sent.append(m) or True)
    monkeypatch.setattr(runner, "send_telegram_message", lambda m, **k: sent.append(m) or True)
    monkeypatch.setattr(fetcher, "fetch_data", lambda **k: {"status": "ok"})
    real_suite = runner.run_backtest_suite
    monkeypatch.setattr(runner, "run_backtest_suite", lambda **k: real_suite(settings=settings, data=candles, **k))
    summary = main.run_pipeline()
    assert summary["status"] == "ok"
    assert sent[0].startswith("🚀") and "*Scorecard*" in sent[-1]
