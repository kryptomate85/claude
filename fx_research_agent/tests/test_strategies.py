"""Tests for the zero-overnight constraint, indicators and the backtest suite."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.backtest_runner import run_backtest_suite, run_single_backtest, split_by_days
from tools.data_fetcher import filter_liquid_hours
from tools.strategies import STRATEGIES, IntradaySessionStrategy, opening_range, release_hour_fade


class AlwaysLong(IntradaySessionStrategy):
    """Worst case for the flat rule: wants to be long on every bar, no stops."""

    def on_bar(self, now: int) -> None:
        if not self.position:
            self.buy(size=self.size)


def _exit_minutes(trades: pd.DataFrame) -> pd.Series:
    t = pd.to_datetime(trades["ExitTime"], utc=True)
    return t.dt.hour * 60 + t.dt.minute


def test_liquid_filter_bounds(candles: pd.DataFrame) -> None:
    mod = candles.index.hour * 60 + candles.index.minute
    assert mod.min() == 7 * 60
    assert mod.max() == 20 * 60 + 30
    assert (candles.index.dayofweek < 5).all()


def test_filter_handles_naive_index(settings) -> None:
    idx = pd.date_range("2026-06-01", periods=288, freq="5min")
    df = pd.DataFrame({"Close": np.arange(288.0)}, index=idx)
    out = filter_liquid_hours(df, settings.liquid_window)
    assert len(out) == (20 * 60 + 30 - 7 * 60) // 5 + 1
    assert str(out.index.tz) == "UTC"


def test_hard_close_flattens_every_day(candles: pd.DataFrame, settings) -> None:
    _, trades = run_single_backtest(candles, AlwaysLong, settings)
    assert len(trades) > 0
    entry = pd.to_datetime(trades["EntryTime"], utc=True)
    exit_ = pd.to_datetime(trades["ExitTime"], utc=True)
    assert (entry.dt.date == exit_.dt.date).all(), "position carried overnight"
    assert (_exit_minutes(trades) == 20 * 60 + 25).all()
    assert len(trades) == candles.index.normalize().nunique()


def test_hard_close_with_missing_evening_bars(candles: pd.DataFrame, settings) -> None:
    """Early close (data ends 17:00 on a day) must still be flat that day."""
    day = candles.index.normalize().unique()[3]
    mod = candles.index.hour * 60 + candles.index.minute
    gappy = candles[~((candles.index.normalize() == day) & (mod > 17 * 60))]
    _, trades = run_single_backtest(gappy, AlwaysLong, settings)
    entry = pd.to_datetime(trades["EntryTime"], utc=True)
    exit_ = pd.to_datetime(trades["ExitTime"], utc=True)
    assert (entry.dt.date == exit_.dt.date).all()


def test_all_strategies_respect_constraint(candles: pd.DataFrame, settings) -> None:
    for name, cls in STRATEGIES.items():
        _, trades = run_single_backtest(candles, cls, settings)
        if trades.empty:
            continue
        entry = pd.to_datetime(trades["EntryTime"], utc=True)
        exit_ = pd.to_datetime(trades["ExitTime"], utc=True)
        assert (entry.dt.date == exit_.dt.date).all(), name
        assert (_exit_minutes(trades) <= 20 * 60 + 25).all(), name
        entry_mod = entry.dt.hour * 60 + entry.dt.minute
        assert (entry_mod >= cls.entry_start).all() and (entry_mod <= cls.entry_end).all(), name


def test_release_fade_signal_on_spike_day_only() -> None:
    """A spike day gets one fade signal on the last bar of the 08:00 New York hour; calm days get none."""
    idx = pd.date_range("2026-06-01 07:00", "2026-07-31 20:30", freq="5min", tz="UTC")
    idx = idx[(idx.dayofweek < 5) & (idx.hour * 60 + idx.minute >= 420) & (idx.hour * 60 + idx.minute <= 1230)]
    close = np.full(len(idx), 1.10)
    high, low = close + 0.0001, close - 0.0001
    spike_day = pd.Timestamp("2026-07-15", tz="UTC")
    hour = (idx.normalize() == spike_day) & (idx.hour == 12)  # 08:00-09:00 EDT
    close[hour] += np.linspace(0.0002, 0.002, hour.sum())  # 20-pip rally inside the hour
    high = np.maximum(high, close + 0.0001)
    sig = release_hour_fade(idx, close, high, low, close)
    hits = np.flatnonzero(sig)
    assert len(hits) == 1 and sig[hits[0]] == -1.0  # up-move faded with a short
    assert idx[hits[0]] == pd.Timestamp("2026-07-15 12:55", tz="UTC")


def test_opening_range_has_no_lookahead(candles: pd.DataFrame) -> None:
    bands = opening_range(candles.index, candles["High"].to_numpy(), candles["Low"].to_numpy(), 420, 480)
    mod = candles.index.hour * 60 + candles.index.minute
    assert np.isnan(bands[0][mod < 480]).all()
    day = candles.index.normalize()[0]
    first = candles[(candles.index.normalize() == day)]
    expected = first[(first.index.hour == 7)]["High"].max()
    i = np.where((candles.index.normalize() == day) & (mod == 480))[0][0]
    assert bands[0][i] == expected


def test_split_by_days_is_chronological(candles: pd.DataFrame) -> None:
    ins, oos = split_by_days(candles, 0.3)
    assert ins.index.max() < oos.index.min()
    assert len(ins) + len(oos) == len(candles)


def test_suite_report(candles: pd.DataFrame, settings) -> None:
    pings: list = []
    report = run_backtest_suite(settings=settings, notifier=pings.append, data=candles)
    assert report["status"] == "ok"
    assert report["overnight_violations_total"] == 0
    assert set(report["strategies"]) == set(STRATEGIES)
    assert len(pings) == 2 * len(STRATEGIES)
    for entry in report["strategies"].values():
        m = entry["full_sample"]
        for key in ("net_return_pct", "win_rate_pct", "max_drawdown_pct", "sharpe_ratio", "profit_factor", "total_trades"):
            assert key in m
    assert report["report_path"] and settings.reports_dir.exists()


def test_suite_missing_data_file(settings) -> None:
    report = run_backtest_suite(settings=settings, notify=False)
    assert report["status"] == "error"
    assert "not found" in report["error"]
