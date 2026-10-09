"""Intraday EUR/USD strategies for backtesting.py with zero overnight exposure.

Every strategy derives from :class:`IntradaySessionStrategy`, whose ``next()``
enforces the non-negotiable flat rule *before* any strategy logic runs:

* If the current bar's open time is ``>= 20:25 UTC`` all open positions are
  closed immediately and every pending order is cancelled; no new entries are
  allowed for the rest of the day.
* As a defensive guard for data gaps (holidays / early closes / missing
  bars) the same liquidation is applied on the last available bar of each
  trading day, so no position can ever be carried into the next session.

Backtests are run with ``trade_on_close=True`` so the liquidation market order
is filled at the close of the very bar that triggered it (no fill on the next
day's open).

Strategies:

* ``london_momentum``      – EMA crossover momentum, entries 07:30–15:00 UTC.
* ``ny_mean_reversion``    – RSI extreme fade, entries 12:30–18:30 UTC.
* ``london_orb``           – 07:00–08:00 opening-range breakout, entries 08:00–11:00 UTC.
* ``us_release_fade``      – fade an outsized 08:00–09:00 New York hour (US 8:30 ET data
  releases), entry at 09:00 ET, exit 20:00 UTC.
"""

from __future__ import annotations

from typing import Dict, Optional, Type

import numpy as np
import pandas as pd
from backtesting import Strategy

HARD_CLOSE_MINUTE: int = 20 * 60 + 25  # 20:25 UTC


# ---------------------------------------------------------------------------
# Indicator helpers (pure functions; usable with Strategy.I)
# ---------------------------------------------------------------------------


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average (span ``period``, non-adjusted)."""
    return pd.Series(values).ewm(span=int(period), adjust=False).mean().to_numpy()


def rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's Relative Strength Index."""
    close = pd.Series(values)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    alpha = 1.0 / int(period)
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=int(period)).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=int(period)).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss != 0.0, 100.0)  # no losses -> RSI 100
    out[avg_gain.isna()] = np.nan
    return out.to_numpy()


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's Average True Range."""
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    prev_close = c.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / int(period), adjust=False, min_periods=int(period)).mean().to_numpy()


def minute_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """UTC minutes after midnight for each bar-open timestamp."""
    return np.asarray(index.hour * 60 + index.minute, dtype=float)


def last_bar_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """1.0 where a bar is the final available bar of its UTC calendar date."""
    dates = pd.Series(index.date)
    return (dates != dates.shift(-1)).to_numpy(dtype=float)


def opening_range(
    index: pd.DatetimeIndex,
    high: np.ndarray,
    low: np.ndarray,
    start_minute: int,
    end_minute: int,
) -> np.ndarray:
    """Per-day opening range high/low, published only once the range is complete.

    For each date, the range is the max(high)/min(low) of bars with open time in
    ``[start_minute, end_minute)``. Values are NaN for bars before
    ``end_minute`` so there is no look-ahead.

    Returns:
        A ``(2, n)`` array: row 0 = range high, row 1 = range low.
    """
    df = pd.DataFrame({"High": high, "Low": low}, index=index)
    mod = index.hour * 60 + index.minute
    in_range = (mod >= start_minute) & (mod < end_minute)
    day = index.normalize()
    rng_high = df["High"].where(in_range).groupby(day).transform("max")
    rng_low = df["Low"].where(in_range).groupby(day).transform("min")
    published = mod >= end_minute
    rng_high = rng_high.where(published)
    rng_low = rng_low.where(published)
    return np.vstack([rng_high.to_numpy(), rng_low.to_numpy()])


def release_hour_fade(
    index: pd.DatetimeIndex,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k: float = 2.0,
    window: int = 20,
) -> np.ndarray:
    """Fade direction after an outsized US release hour (08:00–09:00 New York time).

    For each day, the release hour's range (as a fraction of its open) is
    compared with ``k`` × the median of the previous ``window`` days' release
    hours. On a spike, the value on the hour's *last* bar is the opposite sign
    of the hour's move (+1 long, -1 short); every other bar is 0. The value
    uses only that bar and earlier data, so with ``trade_on_close=True`` the
    entry fills at the 09:00 New York price.
    """
    et = index.tz_convert("America/New_York")
    in_hour = np.asarray(et.hour == 8)
    out = np.zeros(len(index))
    if not in_hour.any():
        return out
    df = pd.DataFrame({"o": open_, "h": high, "l": low, "c": close}, index=index)[in_hour]
    day = index[in_hour].normalize()
    hour = df.groupby(day).agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    rng = (hour["h"] - hour["l"]) / hour["o"]
    baseline = rng.rolling(window, min_periods=15).median().shift(1)
    fade = (-np.sign(hour["c"] - hour["o"])).where(rng > k * baseline, 0.0).fillna(0.0)
    positions = np.flatnonzero(in_hour)
    last_in_day = pd.Series(positions, index=day).groupby(level=0).max()
    out[last_in_day.to_numpy()] = fade.reindex(last_in_day.index).to_numpy()
    return out


# ---------------------------------------------------------------------------
# Base strategy with the hard liquidation rule
# ---------------------------------------------------------------------------


class IntradaySessionStrategy(Strategy):
    """Base class enforcing strict intraday-only exposure.

    Subclasses implement :meth:`on_bar` (entries/exits) and may override
    :meth:`setup`. They must **not** override :meth:`next`.
    """

    hard_close_minute: int = HARD_CLOSE_MINUTE
    size: float = 0.95
    entry_start: int = 7 * 60
    entry_end: int = 20 * 60

    def init(self) -> None:
        """Register time helpers, then delegate to :meth:`setup`."""
        self.mod = self.I(minute_of_day, self.data.index, name="minute_of_day", plot=False)
        self.eod = self.I(last_bar_of_day, self.data.index, name="last_bar_of_day", plot=False)
        self.setup()

    def setup(self) -> None:
        """Hook for subclass indicator registration."""

    def next(self) -> None:
        """Apply the hard liquidation rule, then run strategy logic.

        If the current bar's time is >= 20:25 UTC (or it is the last available
        bar of the day), close every open position, cancel pending orders and
        block new entries.
        """
        now = int(self.mod[-1])
        if now >= self.hard_close_minute or self.eod[-1] > 0.5:
            self.liquidate()
            return
        self.on_bar(now)

    def liquidate(self) -> None:
        """Cancel all pending orders and flatten every open trade."""
        for order in list(self.orders):
            order.cancel()
        if self.position:
            self.position.close()

    def in_entry_window(self, now: int) -> bool:
        """True if ``now`` (minute of day) is inside the entry window."""
        return self.entry_start <= now <= self.entry_end and now < self.hard_close_minute

    def on_bar(self, now: int) -> None:  # pragma: no cover - abstract
        """Strategy logic for a bar inside the tradable day."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Strategy A: London Session Momentum
# ---------------------------------------------------------------------------


class LondonMomentum(IntradaySessionStrategy):
    """EMA crossover momentum during the London session (07:30–15:00 UTC).

    Long when the fast EMA crosses above the slow EMA *and* the bar's true
    range exceeds ``expansion`` × ATR (range-expansion confirmation); short on
    the mirror condition. Exits: ATR stop / target, opposite crossover, or the
    hard 20:25 UTC liquidation.
    """

    fast: int = 9
    slow: int = 21
    atr_period: int = 14
    expansion: float = 1.0
    sl_atr: float = 1.5
    tp_atr: float = 3.0
    entry_start: int = 7 * 60 + 30
    entry_end: int = 15 * 60

    def setup(self) -> None:
        """Register EMAs and ATR."""
        self.ema_fast = self.I(ema, self.data.Close, self.fast, name=f"EMA{self.fast}")
        self.ema_slow = self.I(ema, self.data.Close, self.slow, name=f"EMA{self.slow}")
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close, self.atr_period, name="ATR")

    def on_bar(self, now: int) -> None:
        """Crossover entries/exits."""
        if len(self.data) < self.slow + 2:
            return
        f0, f1 = self.ema_fast[-1], self.ema_fast[-2]
        s0, s1 = self.ema_slow[-1], self.ema_slow[-2]
        a = self.atr[-1]
        if np.isnan([f0, f1, s0, s1, a]).any() or a <= 0:
            return
        cross_up = f1 <= s1 and f0 > s0
        cross_down = f1 >= s1 and f0 < s0

        if self.position.is_long and cross_down:
            self.position.close()
        elif self.position.is_short and cross_up:
            self.position.close()

        if self.position or not self.in_entry_window(now):
            return
        bar_range = self.data.High[-1] - self.data.Low[-1]
        if bar_range < self.expansion * a:
            return
        price = self.data.Close[-1]
        if cross_up:
            self.buy(size=self.size, sl=price - self.sl_atr * a, tp=price + self.tp_atr * a)
        elif cross_down:
            self.sell(size=self.size, sl=price + self.sl_atr * a, tp=price - self.tp_atr * a)


# ---------------------------------------------------------------------------
# Strategy B: New York Session Mean Reversion
# ---------------------------------------------------------------------------


class NYMeanReversion(IntradaySessionStrategy):
    """RSI fade during the New York session (12:30–18:30 UTC).

    Long when RSI crosses back above ``oversold`` from below, short when it
    crosses back below ``overbought`` from above. Exit when RSI reverts through
    ``exit_level`` (50), on the ATR stop / target, or at the hard liquidation.
    """

    rsi_period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    exit_level: float = 50.0
    atr_period: int = 14
    sl_atr: float = 2.0
    tp_atr: float = 2.0
    entry_start: int = 12 * 60 + 30
    entry_end: int = 18 * 60 + 30

    def setup(self) -> None:
        """Register RSI and ATR."""
        self.rsi = self.I(rsi, self.data.Close, self.rsi_period, name="RSI")
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close, self.atr_period, name="ATR")

    def on_bar(self, now: int) -> None:
        """RSI threshold entries and mean-reversion exits."""
        if len(self.data) < self.rsi_period + 2:
            return
        r0, r1, a = self.rsi[-1], self.rsi[-2], self.atr[-1]
        if np.isnan([r0, r1, a]).any() or a <= 0:
            return

        if self.position.is_long and r0 >= self.exit_level:
            self.position.close()
        elif self.position.is_short and r0 <= self.exit_level:
            self.position.close()

        if self.position or not self.in_entry_window(now):
            return
        price = self.data.Close[-1]
        if r1 < self.oversold <= r0:
            self.buy(size=self.size, sl=price - self.sl_atr * a, tp=price + self.tp_atr * a)
        elif r1 > self.overbought >= r0:
            self.sell(size=self.size, sl=price + self.sl_atr * a, tp=price - self.tp_atr * a)


# ---------------------------------------------------------------------------
# Strategy C: London Opening Range Breakout
# ---------------------------------------------------------------------------


class LondonOpeningRangeBreakout(IntradaySessionStrategy):
    """Breakout of the 07:00–08:00 UTC opening range (entries 08:00–11:00 UTC).

    At most one trade per day. Long on a close above the range high, short on
    a close below the range low. Stop at the opposite side of the range
    (capped at ``max_range_pips``), target ``rr`` × risk.
    """

    range_start: int = 7 * 60
    range_end: int = 8 * 60
    entry_start: int = 8 * 60
    entry_end: int = 11 * 60
    rr: float = 1.5
    min_range_pips: float = 5.0
    max_range_pips: float = 40.0

    def setup(self) -> None:
        """Register opening-range bands and per-day trade tracking."""
        bands = self.I(
            opening_range,
            self.data.index,
            self.data.High,
            self.data.Low,
            self.range_start,
            self.range_end,
            name=["OR_high", "OR_low"],
            overlay=True,
        )
        self.or_high, self.or_low = bands[0], bands[1]
        self._last_trade_day: Optional[pd.Timestamp] = None

    def on_bar(self, now: int) -> None:
        """Single breakout entry per day."""
        hi, lo = self.or_high[-1], self.or_low[-1]
        if np.isnan(hi) or np.isnan(lo) or self.position or not self.in_entry_window(now):
            return
        today = self.data.index[-1].normalize()
        if self._last_trade_day == today:
            return
        width_pips = (hi - lo) * 10_000
        if not self.min_range_pips <= width_pips <= self.max_range_pips:
            return
        price = self.data.Close[-1]
        if price > hi:
            risk = price - lo
            self.buy(size=self.size, sl=lo, tp=price + self.rr * risk)
            self._last_trade_day = today
        elif price < lo:
            risk = hi - price
            self.sell(size=self.size, sl=hi, tp=price - self.rr * risk)
            self._last_trade_day = today


# ---------------------------------------------------------------------------
# Strategy D: US Data-Release Fade (fundamental event + technical trigger)
# ---------------------------------------------------------------------------


class USReleaseFade(IntradaySessionStrategy):
    """Fade outsized moves in the US 8:30 ET data-release hour.

    Scheduled US releases (CPI, retail sales, GDP, payrolls, claims) come out
    at 08:30 New York time. When the 08:00–09:00 New York hour's range exceeds
    ``k`` × its trailing ``window``-day median, enter against that hour's move
    at 09:00 New York time and hold, without stop or target, until
    ``exit_minute`` (20:00 UTC) or the hard liquidation.

    The rule was specified before testing (k=2, exit 20:00). On 2.8 years of
    Yahoo H1 EUR/USD it netted about +6.6 bp per trade (n≈100, t≈2.2) after
    costs, but it was weaker on other USD pairs, so treat it as a candidate to
    validate on longer M5 data, not a proven edge.
    """

    k: float = 2.0
    window: int = 20
    exit_minute: int = 20 * 60
    entry_start: int = 12 * 60  # 08:55 New York is 12:55 UTC (EDT) or 13:55 UTC (EST)
    entry_end: int = 15 * 60

    def setup(self) -> None:
        """Register the release-hour fade signal."""
        self.signal = self.I(
            release_hour_fade,
            self.data.index,
            self.data.Open,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.k,
            self.window,
            name="release_fade",
            plot=False,
        )

    def on_bar(self, now: int) -> None:
        """Enter on the signal bar; exit at ``exit_minute``."""
        if self.position and now >= self.exit_minute:
            self.position.close()
            return
        if self.position or not self.in_entry_window(now):
            return
        signal = self.signal[-1]
        if signal > 0:
            self.buy(size=self.size)
        elif signal < 0:
            self.sell(size=self.size)


STRATEGIES: Dict[str, Type[IntradaySessionStrategy]] = {
    "london_momentum": LondonMomentum,
    "ny_mean_reversion": NYMeanReversion,
    "london_orb": LondonOpeningRangeBreakout,
    "us_release_fade": USReleaseFade,
}

STRATEGY_DESCRIPTIONS: Dict[str, str] = {
    "london_momentum": "London Session Momentum: EMA(9/21) crossover with range-expansion filter, entries 07:30-15:00 UTC, 1.5/3.0 ATR stop/target.",
    "ny_mean_reversion": "New York Session Mean Reversion: RSI(14) 30/70 re-entry fade, exit at RSI 50, entries 12:30-18:30 UTC, 2.0 ATR stop/target.",
    "london_orb": "London Opening Range Breakout: 07:00-08:00 UTC range, entries 08:00-11:00 UTC, stop at opposite range edge, 1.5R target, 1 trade/day.",
    "us_release_fade": "US Data-Release Fade: when the 08:00-09:00 New York hour (8:30 ET releases) ranges > 2x its 20-day median, fade its move from 09:00 New York to 20:00 UTC, no stop/target.",
}
