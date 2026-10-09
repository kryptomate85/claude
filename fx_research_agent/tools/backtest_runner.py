"""Backtest suite dispatcher.

Runs every registered intraday strategy over the cached liquid-hours candle
file on three samples (full, in-sample, out-of-sample split by trading day),
sends Telegram progress pings as each strategy starts and finishes, audits
for any overnight exposure, and emits a structured JSON report (returned and
written to ``reports/``).
"""

from __future__ import annotations

import json
import logging
import math
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd
from backtesting import Backtest

from config import Settings, get_settings
from tools.data_fetcher import DataFetchError, load_candles
from tools.notifier import send_telegram_message
from tools.strategies import (
    HARD_CLOSE_MINUTE,
    STRATEGIES,
    STRATEGY_DESCRIPTIONS,
    IntradaySessionStrategy,
)

logger = logging.getLogger(__name__)

Notifier = Callable[[str], Any]
MIN_BARS = 500


def _clean(value: Any, digits: int = 4) -> Optional[float]:
    """Convert numpy/pandas scalars to JSON-safe rounded floats (NaN/inf -> None)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, digits)


def run_single_backtest(
    data: pd.DataFrame,
    strategy_cls: Type[IntradaySessionStrategy],
    settings: Settings,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Execute one backtest and return backtesting.py stats and its trades frame.

    Positions are sized as ``settings.position_size`` of available equity and
    costs are modelled with a relative per-side ``commission`` (default
    0.00008 ≈ 0.86 pip at 1.08 each way). ``trade_on_close=True`` ensures the
    hard-liquidation order fills on the triggering bar.
    """
    bt_kwargs: Dict[str, Any] = dict(
        cash=settings.cash,
        commission=settings.commission,
        margin=settings.margin,
        trade_on_close=True,
        exclusive_orders=False,
        hedging=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            bt = Backtest(data, strategy_cls, finalize_trades=True, **bt_kwargs)
        except TypeError:  # backtesting < 0.6 has no finalize_trades
            bt = Backtest(data, strategy_cls, **bt_kwargs)
        stats = bt.run(size=settings.position_size)
    trades = stats["_trades"] if "_trades" in stats else pd.DataFrame()
    return stats, trades


def extract_metrics(stats: pd.Series, trades: pd.DataFrame, data: pd.DataFrame) -> Dict[str, Any]:
    """Build the metrics dict for one backtest run.

    Core metrics: Net Return %, Win Rate %, Max Drawdown %, Sharpe Ratio,
    Profit Factor and Total Trades, plus risk/reward diagnostics and an
    overnight-exposure audit.
    """
    n_trades = int(stats.get("# Trades", 0) or 0)
    trading_days = int(data.index.normalize().nunique())
    metrics: Dict[str, Any] = {
        "net_return_pct": _clean(stats.get("Return [%]"), 3),
        "win_rate_pct": _clean(stats.get("Win Rate [%]"), 2),
        "max_drawdown_pct": _clean(stats.get("Max. Drawdown [%]"), 3),
        "sharpe_ratio": _clean(stats.get("Sharpe Ratio"), 3),
        "profit_factor": _clean(stats.get("Profit Factor"), 3),
        "total_trades": n_trades,
        "sortino_ratio": _clean(stats.get("Sortino Ratio"), 3),
        "calmar_ratio": _clean(stats.get("Calmar Ratio"), 3),
        "expectancy_pct": _clean(stats.get("Expectancy [%]"), 4),
        "sqn": _clean(stats.get("SQN"), 3),
        "exposure_time_pct": _clean(stats.get("Exposure Time [%]"), 2),
        "buy_hold_return_pct": _clean(stats.get("Buy & Hold Return [%]"), 3),
        "max_drawdown_duration": str(stats.get("Max. Drawdown Duration", "")) or None,
        "trading_days": trading_days,
        "trades_per_day": _clean(n_trades / trading_days, 3) if trading_days else None,
        "period_start_utc": data.index[0].isoformat(),
        "period_end_utc": data.index[-1].isoformat(),
    }

    if trades is None or trades.empty:
        metrics.update(
            {
                "avg_win_pct": None,
                "avg_loss_pct": None,
                "payoff_ratio": None,
                "breakeven_win_rate_pct": None,
                "avg_hold_minutes": None,
                "long_trades": 0,
                "short_trades": 0,
                "long_win_rate_pct": None,
                "short_win_rate_pct": None,
                "max_consecutive_losses": 0,
                "hard_close_exits": 0,
                "gross_pnl": 0.0,
                "total_commissions": 0.0,
                "overnight_violations": 0,
            }
        )
        return metrics

    ret = trades["ReturnPct"] * 100.0
    wins, losses = ret[ret > 0], ret[ret <= 0]
    avg_win = wins.mean() if len(wins) else np.nan
    avg_loss = losses.mean() if len(losses) else np.nan
    payoff = abs(avg_win / avg_loss) if len(wins) and len(losses) and avg_loss != 0 else np.nan
    breakeven = 100.0 / (1.0 + payoff) if not np.isnan(payoff) else np.nan

    entry_t = pd.to_datetime(trades["EntryTime"], utc=True)
    exit_t = pd.to_datetime(trades["ExitTime"], utc=True)
    hold_min = (exit_t - entry_t).dt.total_seconds() / 60.0
    exit_mod = exit_t.dt.hour * 60 + exit_t.dt.minute
    overnight = int(((entry_t.dt.date != exit_t.dt.date) | (exit_mod > 20 * 60 + 30)).sum())

    longs = trades[trades["Size"] > 0]
    shorts = trades[trades["Size"] < 0]
    losing = (ret <= 0).astype(int).to_numpy()
    max_streak, streak = 0, 0
    for flag in losing:
        streak = streak + 1 if flag else 0
        max_streak = max(max_streak, streak)

    commissions = float(trades["Commission"].sum()) if "Commission" in trades else None
    metrics.update(
        {
            "avg_win_pct": _clean(avg_win, 4),
            "avg_loss_pct": _clean(avg_loss, 4),
            "payoff_ratio": _clean(payoff, 3),
            "breakeven_win_rate_pct": _clean(breakeven, 2),
            "avg_hold_minutes": _clean(hold_min.mean(), 1),
            "long_trades": int(len(longs)),
            "short_trades": int(len(shorts)),
            "long_win_rate_pct": _clean((longs["PnL"] > 0).mean() * 100, 2) if len(longs) else None,
            "short_win_rate_pct": _clean((shorts["PnL"] > 0).mean() * 100, 2) if len(shorts) else None,
            "max_consecutive_losses": int(max_streak),
            "hard_close_exits": int((exit_mod >= HARD_CLOSE_MINUTE).sum()),
            "gross_pnl": _clean(trades["PnL"].sum() + (commissions or 0.0), 2),
            "total_commissions": _clean(commissions, 2),
            "overnight_violations": overnight,
        }
    )
    return metrics


def split_by_days(data: pd.DataFrame, oos_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split ``data`` into in-sample / out-of-sample by whole days."""
    days = data.index.normalize().unique()
    n_oos = max(1, int(round(len(days) * oos_fraction)))
    if len(days) - n_oos < 1:
        return data, data.iloc[0:0]
    cutoff = days[len(days) - n_oos]
    return data[data.index < cutoff], data[data.index >= cutoff]


def _degradation(is_m: Optional[Dict[str, Any]], oos_m: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare in-sample vs out-of-sample results for an overfitting signal."""
    if not is_m or not oos_m:
        return {"assessable": False}
    out: Dict[str, Any] = {"assessable": True}
    for key in ("sharpe_ratio", "profit_factor", "win_rate_pct"):
        a, b = is_m.get(key), oos_m.get(key)
        out[f"{key}_is"] = a
        out[f"{key}_oos"] = b
        out[f"{key}_delta"] = _clean(b - a, 3) if a is not None and b is not None else None
    is_days = is_m.get("trading_days") or 1
    oos_days = oos_m.get("trading_days") or 1
    is_daily = (is_m.get("net_return_pct") or 0.0) / is_days
    oos_daily = (oos_m.get("net_return_pct") or 0.0) / oos_days
    out["net_return_per_day_is_pct"] = _clean(is_daily, 4)
    out["net_return_per_day_oos_pct"] = _clean(oos_daily, 4)
    out["sign_flip"] = bool(is_daily > 0 >= oos_daily)
    return out


def _fmt(value: Any, suffix: str = "") -> str:
    """Pretty-print a possibly-missing metric for Telegram."""
    return "n/a" if value is None else f"{value}{suffix}"


def run_backtest_suite(
    strategies: Optional[Sequence[str]] = None,
    notify: bool = True,
    settings: Optional[Settings] = None,
    notifier: Optional[Notifier] = None,
    data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Run the strategy suite and return a JSON-serialisable report.

    This is the function exposed to the agent as ``run_backtest_suite``. It
    never raises; failures are reported per strategy or at top level.

    Args:
        strategies: Subset of strategy keys (defaults to all registered).
        notify: Send Telegram start/finish pings for each strategy.
        settings: Optional explicit settings (defaults to global settings).
        notifier: Optional callable used instead of Telegram (e.g. in tests).
        data: Optional pre-loaded candle frame (defaults to the Parquet cache).
    """
    started = time.perf_counter()
    try:
        settings = settings or get_settings()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"configuration error: {exc}"}

    send = notifier or send_telegram_message

    def ping(text: str) -> None:
        if not notify:
            return
        try:
            send(text)
        except Exception:  # noqa: BLE001 - pings are best-effort
            logger.exception("Progress ping failed")

    if data is None:
        try:
            data = load_candles(settings.parquet_path)
        except DataFetchError as exc:
            logger.error("Cannot run backtests: %s", exc)
            return {"status": "error", "error": str(exc)}

    if len(data) < MIN_BARS:
        return {
            "status": "error",
            "error": f"Only {len(data)} bars available; need at least {MIN_BARS}.",
        }

    selected = list(strategies) if strategies else list(STRATEGIES)
    unknown = [s for s in selected if s not in STRATEGIES]
    selected = [s for s in selected if s in STRATEGIES]
    if not selected:
        return {"status": "error", "error": f"No valid strategies requested: {unknown}"}

    in_sample, out_sample = split_by_days(data, settings.oos_fraction)
    report: Dict[str, Any] = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "instrument": settings.instrument,
        "granularity": settings.granularity,
        "data": {
            "bars": int(len(data)),
            "start_utc": data.index[0].isoformat(),
            "end_utc": data.index[-1].isoformat(),
            "trading_days": int(data.index.normalize().nunique()),
            "session_filter_utc": "07:00-20:30",
            "in_sample_end_utc": in_sample.index[-1].isoformat() if len(in_sample) else None,
            "out_of_sample_start_utc": out_sample.index[0].isoformat() if len(out_sample) else None,
        },
        "assumptions": {
            "cash": settings.cash,
            "commission_per_side_relative": settings.commission,
            "approx_round_trip_cost_pips": _clean(
                2 * settings.commission * float(data["Close"].mean()) * 10_000, 2
            ),
            "leverage": settings.leverage,
            "position_size_fraction": settings.position_size,
            "fill_model": "market orders fill at signal-bar close (trade_on_close)",
            "hard_close_utc": "20:25 (flatten + block entries), zero overnight exposure",
            "oos_fraction": settings.oos_fraction,
        },
        "strategies": {},
        "unknown_strategies": unknown,
    }

    for position, key in enumerate(selected, start=1):
        cls = STRATEGIES[key]
        ping(f"⏳ *Backtest {position}/{len(selected)} started:* `{key}`\n_{STRATEGY_DESCRIPTIONS[key]}_")
        logger.info("Running strategy %s", key)
        t0 = time.perf_counter()
        entry: Dict[str, Any] = {"description": STRATEGY_DESCRIPTIONS[key]}
        try:
            stats, trades = run_single_backtest(data, cls, settings)
            entry["full_sample"] = extract_metrics(stats, trades, data)
            samples: Dict[str, Optional[Dict[str, Any]]] = {}
            for label, frame in (("in_sample", in_sample), ("out_of_sample", out_sample)):
                if len(frame) >= MIN_BARS // 2:
                    s, t = run_single_backtest(frame, cls, settings)
                    samples[label] = extract_metrics(s, t, frame)
                else:
                    samples[label] = None
            entry.update(samples)
            entry["overfitting_check"] = _degradation(samples["in_sample"], samples["out_of_sample"])
            entry["status"] = "ok"
            entry["runtime_sec"] = round(time.perf_counter() - t0, 2)
            m = entry["full_sample"]
            ping(
                f"✅ *Finished* `{key}` in {entry['runtime_sec']}s\n"
                f"Return: {_fmt(m['net_return_pct'], '%')} | Win: {_fmt(m['win_rate_pct'], '%')} | "
                f"MaxDD: {_fmt(m['max_drawdown_pct'], '%')}\n"
                f"Sharpe: {_fmt(m['sharpe_ratio'])} | PF: {_fmt(m['profit_factor'])} | "
                f"Trades: {m['total_trades']} | Overnight: {m['overnight_violations']}"
            )
        except Exception as exc:  # noqa: BLE001 - isolate per-strategy failures
            logger.exception("Strategy %s failed", key)
            entry.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            ping(f"❌ *Backtest failed:* `{key}` — {type(exc).__name__}")
        report["strategies"][key] = entry

    ok = [k for k, v in report["strategies"].items() if v.get("status") == "ok"]
    if not ok:
        report["status"] = "error"
        report["error"] = "All strategies failed."
    report["overnight_violations_total"] = sum(
        v["full_sample"]["overnight_violations"] for v in report["strategies"].values() if v.get("status") == "ok"
    )
    report["runtime_sec"] = round(time.perf_counter() - started, 2)

    try:
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = settings.reports_dir / f"backtest_report_{stamp}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(path)
        logger.info("Backtest report written to %s", path)
    except OSError as exc:
        logger.error("Could not write report file: %s", exc)
        report["report_path"] = None
    return report
