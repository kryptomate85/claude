"""Rule-based research report.

Turns the backtest suite's JSON into a Telegram-Markdown report with a
scorecard, a per-strategy quantitative assessment (edge vs. breakeven win
rate, cost drag, out-of-sample stability, drawdown) and a verdict. This is the
default analyst; it needs no LLM and produces identical output for identical
inputs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def save_report_markdown(text: str, reports_dir: Path) -> Optional[Path]:
    """Write a final report to ``reports/research_report_<UTC stamp>.md``.

    Returns the path, or None if the file could not be written.
    """
    try:
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = reports_dir / f"research_report_{stamp}.md"
        path.write_text(text + "\n", encoding="utf-8")
        logger.info("Research report written to %s", path)
        return path
    except OSError as exc:
        logger.error("Could not write research report: %s", exc)
        return None


def _v(value: Any, suffix: str = "") -> str:
    """Format a metric that may be None."""
    return "n/a" if value is None else f"{value}{suffix}"


def assess_strategy(name: str, entry: Dict[str, Any]) -> List[str]:
    """Rule-based quant assessment of one strategy's metrics."""
    m = entry.get("full_sample") or {}
    notes: List[str] = []
    trades = m.get("total_trades", 0)
    if trades < 30:
        notes.append(f"only {trades} trades — low statistical confidence")
    wr, be = m.get("win_rate_pct"), m.get("breakeven_win_rate_pct")
    if wr is not None and be is not None:
        edge = round(wr - be, 2)
        notes.append(f"win rate {wr}% vs breakeven {be}% (payoff {_v(m.get('payoff_ratio'))}) → edge {edge:+} pp")
    gross, costs = m.get("gross_pnl"), m.get("total_commissions")
    if gross is not None and costs:
        if gross > 0 and gross < costs:
            notes.append(f"gross edge {gross} is consumed by costs {costs}")
        elif gross <= 0:
            notes.append(f"negative before costs (gross {gross}) — no raw edge")
        else:
            notes.append(f"edge survives costs (gross {gross} vs costs {costs})")
    oc = entry.get("overfitting_check") or {}
    if oc.get("assessable"):
        if oc.get("sign_flip"):
            notes.append("in-sample profits flip to OOS losses — likely overfit/regime-dependent")
        d = oc.get("sharpe_ratio_delta")
        if d is not None and d < -1.0:
            notes.append(f"Sharpe decays {d} out-of-sample")
    dd, streak = m.get("max_drawdown_pct"), m.get("max_consecutive_losses")
    if dd is not None:
        notes.append(f"max DD {dd}% with {streak} consecutive losses")
    return notes


def build_report(
    report: Dict[str, Any],
    data_result: Optional[Dict[str, Any]],
    llm_error: Optional[str] = None,
) -> str:
    """Build the Telegram-Markdown research report from the backtest JSON."""
    data = report.get("data", {})
    assumptions = report.get("assumptions", {})
    lines = [
        "*EUR/USD Intraday Research Report*",
        f"Data: {data.get('start_utc', '?')[:10]} → {data.get('end_utc', '?')[:10]}, "
        f"{data.get('bars')} bars, {data.get('trading_days')} days, 07:00-20:30 UTC",
        f"Costs: ~{assumptions.get('approx_round_trip_cost_pips')} pips round trip | "
        f"Overnight violations: {report.get('overnight_violations_total')}",
    ]
    if data_result and data_result.get("status") == "stale_cache":
        lines.append("⚠️ Live fetch failed — results use cached candles.")
    if llm_error:
        lines.append("⚠️ LLM analyst unavailable; rule-based assessment below.")

    lines += ["", "*Scorecard*"]
    ranked = []
    for name, entry in report.get("strategies", {}).items():
        if entry.get("status") != "ok":
            lines.append(f"- `{name}`: failed ({entry.get('error')})")
            continue
        m = entry["full_sample"]
        lines.append(
            f"- `{name}`: Ret {_v(m['net_return_pct'], '%')} | Win {_v(m['win_rate_pct'], '%')} | "
            f"DD {_v(m['max_drawdown_pct'], '%')} | Sharpe {_v(m['sharpe_ratio'])} | "
            f"PF {_v(m['profit_factor'])} | N {m['total_trades']}"
        )
        oos = (entry.get("out_of_sample") or {}).get("sharpe_ratio")
        score = (m.get("sharpe_ratio") or -99) + (oos if oos is not None else -99) * 0.5
        ranked.append((score, name, entry))

    lines += ["", "*Analysis*"]
    for _, name, entry in sorted(ranked, reverse=True):
        lines.append(f"`{name}`:")
        lines += [f"- {note}" for note in assess_strategy(name, entry)]

    lines += ["", "*Verdict & Tactical Recommendations*"]
    viable = [
        name
        for _, name, entry in sorted(ranked, reverse=True)
        if (entry["full_sample"].get("profit_factor") or 0) > 1.1
        and entry["full_sample"].get("total_trades", 0) >= 30
        and not (entry.get("overfitting_check") or {}).get("sign_flip")
    ]
    if viable:
        lines.append(f"- Candidates for paper trading: {', '.join(f'`{n}`' for n in viable)}")
    else:
        lines.append("- No strategy clears PF > 1.1, N ≥ 30 and OOS stability; do not deploy.")
    lines.append("- Next: walk-forward the top candidate's parameters and test a volatility (ATR percentile) filter.")
    lines.append("- Re-run with a longer lookback to raise trade counts before drawing conclusions.")
    return "\n".join(lines)
