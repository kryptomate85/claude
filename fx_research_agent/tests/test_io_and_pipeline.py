"""Tests for notifier chunking, OANDA/Dukascopy parsing, config and the default pipeline (offline)."""

from __future__ import annotations

import dataclasses
import json
import lzma
from pathlib import Path
from typing import Any, List

import numpy as np
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
    settings = dataclasses.replace(settings, data_source="oanda")
    settings.parquet_path.parent.mkdir(parents=True)
    candles.to_parquet(settings.parquet_path)

    def boom(*_a: Any, **_k: Any) -> None:
        raise data_fetcher.DataFetchError("network down")

    monkeypatch.setattr(data_fetcher.OandaClient, "fetch_candles", boom)
    result = data_fetcher.fetch_data(settings=settings)
    assert result["status"] == "stale_cache" and result["rows"] == len(candles)


def test_fetch_data_error_without_cache(settings, monkeypatch) -> None:
    settings = dataclasses.replace(settings, data_source="oanda")
    monkeypatch.setattr(data_fetcher.OandaClient, "fetch_candles", lambda *a, **k: (_ for _ in ()).throw(data_fetcher.DataFetchError("down")))
    assert data_fetcher.fetch_data(settings=settings)["status"] == "error"


def _bi5(day_seconds: List[int], price: int, volume: float) -> bytes:
    """Build a Dukascopy 1-minute file with flat bars at ``price`` points."""
    rec = np.zeros(len(day_seconds), dtype=data_fetcher.DUKASCOPY_RECORD)
    rec["sec"], rec["v"] = day_seconds, volume
    for col in ("o", "c", "l", "h"):
        rec[col] = price
    rec["h"] += 2
    return lzma.compress(rec.tobytes(), format=lzma.FORMAT_ALONE)


class FakeFeed:
    """Serves BID/ASK files for 2026-06-01 (a Monday) and 404 for every other day."""

    def __init__(self, first_status: int = 200) -> None:
        self.headers: dict = {}
        self.urls: List[str] = []
        self.first_status = first_status

    def get(self, url: str, timeout: float) -> FakeResponse:
        self.urls.append(url)
        if self.first_status != 200:
            status, self.first_status = self.first_status, 200
            return FakeResponse(status, {})
        if "/2026/05/01/" not in url:  # Dukascopy months are zero-based
            return FakeResponse(404, {})
        seconds = [7 * 3600 + 60 * m for m in range(10)]
        resp = FakeResponse(200, {})
        if "BID" in url:
            resp.content = _bi5(seconds, 108000, 1.0)
        else:  # last minute has no ticks on the ask side -> dropped
            resp.content = _bi5(seconds[:-1], 108010, 1.0)
        return resp


@pytest.fixture()
def no_sleep(monkeypatch) -> None:
    monkeypatch.setattr(data_fetcher.time, "sleep", lambda _s: None)


def test_dukascopy_mid_resample_and_cache(settings, no_sleep) -> None:
    feed = FakeFeed()
    client = data_fetcher.DukascopyClient(settings, session=feed)
    start, end = pd.Timestamp("2026-05-30", tz="UTC"), pd.Timestamp("2026-06-03", tz="UTC")
    df = client.fetch_candles("EUR_USD", "M5", start.to_pydatetime(), end.to_pydatetime())
    assert list(df.index.strftime("%H:%M")) == ["07:00", "07:05"]
    assert df["Open"].iloc[0] == pytest.approx(1.08005) and df["High"].iloc[0] == pytest.approx(1.08007)
    assert df["Volume"].tolist() == [10.0, 8.0]  # 07:09 has no ask bar
    assert len(feed.urls) == 4 and not any("/2026/04/30/" in u or "/2026/04/31/" in u for u in feed.urls)  # weekend skipped
    client.fetch_candles("EUR_USD", "M5", start.to_pydatetime(), end.to_pydatetime())
    assert len(feed.urls) == 6  # 2026-06-01 served from cache; 404 days are retried


def test_dukascopy_retries_rate_limit(settings, no_sleep) -> None:
    client = data_fetcher.DukascopyClient(settings, session=FakeFeed(first_status=429))
    df = client.fetch_candles("EUR_USD", "M5", pd.Timestamp("2026-06-01", tz="UTC").to_pydatetime(), pd.Timestamp("2026-06-02", tz="UTC").to_pydatetime())
    assert len(df) == 2


def test_yahoo_parses_and_drops_forming_bar(settings, no_sleep, monkeypatch) -> None:
    now = pd.Timestamp("2026-06-01T08:12:00", tz="UTC")
    monkeypatch.setattr(data_fetcher, "datetime", type("FrozenDT", (data_fetcher.datetime,), {"now": staticmethod(lambda tz=None: now.to_pydatetime())}))
    times = [int(pd.Timestamp(f"2026-06-01T{t}", tz="UTC").timestamp()) for t in ("08:00", "08:05", "08:10")]
    quote = {"open": [1.1, None, 1.2], "high": [1.2, None, 1.3], "low": [1.0, None, 1.1], "close": [1.15, None, 1.25], "volume": [0, 0, 0]}
    payload = {"chart": {"result": [{"timestamp": times, "indicators": {"quote": [quote]}}], "error": None}}
    session = FakeSession([FakeResponse(429, {}), FakeResponse(200, payload)])
    df = data_fetcher.YahooClient(settings, session=session).fetch_candles("EUR_USD", "M5", (now - pd.Timedelta(days=90)).to_pydatetime())
    assert list(df.index.strftime("%H:%M")) == ["08:00"]  # null bar dropped, 08:10 still forming
    assert session.calls[-1]["period1"] == int((now - pd.Timedelta(days=data_fetcher.YAHOO_MAX_DAYS)).timestamp())


def test_default_source_needs_no_keys(monkeypatch) -> None:
    for key in config.SECRET_KEYS + config.LLM_SECRET_KEYS + config.TELEGRAM_SECRET_KEYS + ("DATA_SOURCE",):
        monkeypatch.delenv(key, raising=False)
    assert config.load_settings().data_source == "dukascopy"
    monkeypatch.setenv("DATA_SOURCE", "oanda")
    with pytest.raises(config.ConfigError, match="OANDA_API_KEY"):
        config.load_settings()


def test_config_only_requires_oanda(monkeypatch) -> None:
    for key in config.LLM_SECRET_KEYS + config.TELEGRAM_SECRET_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DATA_SOURCE", "oanda")
    for key in config.SECRET_KEYS:
        monkeypatch.setenv(key, "x")
    loaded = config.load_settings()
    assert loaded.openai_api_key == "" and not loaded.telegram_enabled
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
    summary = main.run_pipeline(settings)
    assert summary["status"] == "ok" and summary["telegram"] is False
    assert sent[0].startswith("🚀") and "*Scorecard*" in sent[-1]
    saved = Path(summary["markdown_path"])
    assert saved.parent == settings.reports_dir and saved.read_text().strip() == sent[-1]


def test_half_configured_telegram_is_rejected(monkeypatch) -> None:
    for key in config.SECRET_KEYS:
        monkeypatch.setenv(key, "x")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with pytest.raises(config.ConfigError, match="TELEGRAM_CHAT_ID"):
        config.load_settings()


def test_disabled_notifier_logs_to_console(caplog) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="tools.notifier"):
        assert TelegramNotifier("", "").send("*Report*\nbody") is False
    assert "[console] Report" in caplog.text
