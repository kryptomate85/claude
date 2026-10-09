"""Shared fixtures: put the project root on sys.path and build synthetic candles."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Settings  # noqa: E402
from tools.data_fetcher import filter_liquid_hours  # noqa: E402


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Settings with dummy secrets and temp output directories."""
    return Settings(
        openai_api_key="sk-test",
        oanda_api_key="oanda-test",
        oanda_account_id="101-000",
        telegram_bot_token="",
        telegram_chat_id="",
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
    )


@pytest.fixture()
def candles(settings: Settings) -> pd.DataFrame:
    """40 calendar days of random-walk EUR/USD M5 candles, liquid hours only."""
    idx = pd.date_range("2026-06-01", periods=40 * 288, freq="5min", tz="UTC")
    rng = np.random.default_rng(7)
    close = 1.08 * np.exp(np.cumsum(rng.normal(0, 0.0004, len(idx))))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0002, len(idx))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0002, len(idx))))
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 100}, index=idx)
    return filter_liquid_hours(df, settings.liquid_window)
