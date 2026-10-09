"""Autonomous FX research agent driven by OpenAI tool calling.

The model is given three tools — ``fetch_data``, ``run_backtest_suite`` and
``send_telegram_update`` — and a system prompt instructing it to announce the
run, refresh data, run the backtest suite, analyse the metrics as a senior
quant, and publish a Markdown report to Telegram.

Reliability guarantees enforced *outside* the model:

* LLM API failures are retried; if the model remains unavailable the agent
  falls back to a deterministic pipeline with a rule-based analysis so the
  user still receives a report.
* If the model stops without sending the final report, it is nudged once; if
  it still does not comply, its last answer (or the rule-based report) is
  sent on its behalf.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

from config import Settings, get_settings
from tools.backtest_runner import run_backtest_suite
from tools.data_fetcher import fetch_data
from tools.notifier import send_telegram_message
from tools.strategies import STRATEGIES

logger = logging.getLogger(__name__)

FINAL_REPORT_MIN_CHARS = 400

SYSTEM_PROMPT = """You are a senior quantitative FX researcher operating an autonomous research pipeline for EUR/USD on M5 candles restricted to liquid London/New York hours (07:00-20:30 UTC). Every strategy is strictly intraday: positions are force-liquidated at 20:25 UTC and nothing is ever held overnight.

Follow this procedure exactly, using the tools:
1. Call send_telegram_update with a short alert that the EUR/USD research pipeline has started.
2. Call fetch_data to pull fresh intraday candles. If status is "stale_cache", continue but mention the stale data in the report. If status is "error", send a Telegram message explaining the failure and stop.
3. Call run_backtest_suite (all strategies). The suite sends its own per-strategy progress pings.
4. Analyse the returned JSON as a senior quantitative analyst:
   - Win rate vs. risk-reward: compare win_rate_pct with breakeven_win_rate_pct (derived from payoff_ratio). State the edge in percentage points.
   - Cost drag: compare gross_pnl with total_commissions; judge whether the edge survives realistic spread/slippage.
   - Overfitting / robustness: use in_sample vs out_of_sample and overfitting_check (sign flips, Sharpe/PF decay). Flag small samples (fewer than ~30 trades means low statistical confidence).
   - Drawdown sustainability without overnight bias: relate max_drawdown_pct, max_consecutive_losses and return per day; note that zero overnight exposure removes gap risk, so intraday drawdowns are the true risk. Confirm overnight_violations_total is 0.
   - Long/short asymmetry and hard_close_exits (trades cut by the 20:25 rule).
5. Call send_telegram_update ONCE with the final report. Format for Telegram legacy Markdown: *bold* headers, `code` for every identifier containing underscores (e.g. `london_orb`), plain '-' bullets, no tables, no '#' headings, under 3500 characters. Structure:
   *EUR/USD Intraday Research Report* (data window, bars, cost assumptions)
   *Scorecard* - one line per strategy: Return %, Win %, MaxDD %, Sharpe, PF, Trades
   *Analysis* - the points from step 4, quantitative and specific
   *Verdict & Tactical Recommendations* - rank strategies, say which (if any) merit paper trading, and give concrete next experiments (parameter, session or filter changes).
Never invent numbers: quote only values present in the tool results. After sending the final report, reply with the single word DONE."""


TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fetch_data",
            "description": (
                "Fetch recent complete EUR/USD M5 candles from OANDA, keep only 07:00-20:30 UTC "
                "bars, save them to Parquet and return a data summary. Falls back to the cached "
                "file if the API is unavailable (status 'stale_cache')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lookback_days": {
                        "type": "integer",
                        "minimum": 5,
                        "maximum": 365,
                        "description": "Calendar days of history. Omit to use the configured default.",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_backtest_suite",
            "description": (
                "Backtest the intraday strategies on the cached candles (full sample plus "
                "chronological in-sample / out-of-sample split) with 0.8+ pip costs and a hard "
                "20:25 UTC liquidation. Returns a JSON metrics report."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategies": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(STRATEGIES)},
                        "description": "Subset of strategies to run. Omit to run all.",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_telegram_update",
            "description": "Send a Telegram legacy-Markdown message to the research channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Markdown message body."}
                },
                "required": ["message"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass
class RunState:
    """Tracks pipeline milestones so the harness can guarantee completion."""

    alert_sent: bool = False
    data_result: Optional[Dict[str, Any]] = None
    backtest_report: Optional[Dict[str, Any]] = None
    final_report_sent: bool = False
    messages_sent: int = 0
    tool_log: List[str] = field(default_factory=list)


class ResearchAgent:
    """OpenAI tool-calling loop orchestrating the research pipeline."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[OpenAI] = None,
        send: Callable[[str], bool] = send_telegram_message,
    ) -> None:
        """Create the agent.

        Args:
            settings: Runtime settings (defaults to global settings).
            client: Optional pre-built OpenAI(-compatible) client.
            send: Telegram send function (injectable for testing).
        """
        self.settings = settings or get_settings()
        self.client = client or OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            timeout=120.0,
            max_retries=2,
        )
        self.send = send
        self.state = RunState()

    # ------------------------------------------------------------------ tools

    def _tool_fetch_data(self, lookback_days: Optional[int] = None) -> Dict[str, Any]:
        result = fetch_data(lookback_days=lookback_days, settings=self.settings)
        self.state.data_result = result
        return result

    def _tool_run_backtest_suite(self, strategies: Optional[List[str]] = None) -> Dict[str, Any]:
        report = run_backtest_suite(strategies=strategies, settings=self.settings, notifier=self.send)
        if report.get("status") == "ok":
            self.state.backtest_report = report
        return report

    def _tool_send_telegram_update(self, message: str) -> Dict[str, Any]:
        delivered = bool(self.send(message))
        self.state.messages_sent += 1
        if not self.state.alert_sent:
            self.state.alert_sent = True
        elif self.state.backtest_report is not None and len(message) >= FINAL_REPORT_MIN_CHARS:
            self.state.final_report_sent = True
        return {"delivered": delivered, "chars": len(message)}

    def dispatch(self, name: str, raw_args: str) -> Dict[str, Any]:
        """Execute a tool call by name with JSON-encoded arguments; never raises."""
        try:
            args = json.loads(raw_args) if raw_args and raw_args.strip() else {}
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            return {"status": "error", "error": f"Invalid arguments for {name}: {exc}"}

        handlers: Dict[str, Callable[..., Dict[str, Any]]] = {
            "fetch_data": self._tool_fetch_data,
            "run_backtest_suite": self._tool_run_backtest_suite,
            "send_telegram_update": self._tool_send_telegram_update,
        }
        handler = handlers.get(name)
        if handler is None:
            return {"status": "error", "error": f"Unknown tool {name!r}"}
        self.state.tool_log.append(name)
        logger.info("Tool call: %s(%s)", name, ", ".join(f"{k}=" for k in args) or "")
        try:
            return handler(**args)
        except TypeError as exc:
            return {"status": "error", "error": f"Bad arguments for {name}: {exc}"}
        except Exception as exc:  # noqa: BLE001 - surface to model, keep running
            logger.exception("Tool %s crashed", name)
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------- loop

    def _complete(self, messages: List[Dict[str, Any]]) -> Any:
        """Call the chat completions API with retries on transient errors."""
        delay = 2.0
        for attempt in range(1, 5):
            try:
                return self.client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.2,
                )
            except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
                logger.warning("LLM transient error (attempt %d/4): %s", attempt, exc)
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("LLM unavailable after 4 attempts")

    def run(self) -> Dict[str, Any]:
        """Run the agentic loop; always returns a run summary and never raises."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Run today's research cycle for {self.settings.instrument} "
                    f"({self.settings.lookback_days}-day lookback)."
                ),
            },
        ]
        last_text = ""
        nudged = False
        llm_error: Optional[str] = None

        for step in range(1, self.settings.agent_max_steps + 1):
            try:
                response = self._complete(messages)
            except (RuntimeError, APIError) as exc:
                llm_error = f"{type(exc).__name__}: {exc}"
                logger.error("LLM failure at step %d: %s", step, llm_error)
                break

            message = response.choices[0].message
            tool_calls = [tc for tc in (message.tool_calls or []) if getattr(tc, "type", "function") == "function"]
            assistant: Dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            if tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant)
            if message.content:
                last_text = message.content.strip()

            if tool_calls:
                for tc in tool_calls:
                    result = self.dispatch(tc.function.name, tc.function.arguments or "{}")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result, default=str),
                        }
                    )
                data_failed = (self.state.data_result or {}).get("status") == "error"
                if self.state.final_report_sent or (data_failed and self.state.messages_sent >= 2):
                    break
                continue

            # The model answered without tool calls.
            if self.state.final_report_sent:
                break
            if not nudged:
                nudged = True
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The pipeline is not complete. Continue with the remaining steps and "
                            "dispatch the final Markdown report via send_telegram_update."
                        ),
                    }
                )
                continue
            break

        return self._finalize(last_text, llm_error)

    def _finalize(self, last_text: str, llm_error: Optional[str]) -> Dict[str, Any]:
        """Guarantee that data, backtests and a final report were produced."""
        state = self.state
        if llm_error and not state.alert_sent:
            self.send("🚀 *EUR/USD research pipeline started* (deterministic mode — LLM unavailable)")
            state.alert_sent = True

        data_status = (state.data_result or {}).get("status")
        if state.backtest_report is None and data_status != "error":
            if state.data_result is None:
                logger.info("Fallback: fetching data")
                state.data_result = fetch_data(settings=self.settings)
            if state.data_result.get("status") != "error":
                logger.info("Fallback: running backtest suite")
                report = run_backtest_suite(settings=self.settings, notifier=self.send)
                if report.get("status") == "ok":
                    state.backtest_report = report

        if (state.data_result or {}).get("status") == "error" and state.backtest_report is None:
            if not state.final_report_sent:
                self.send(
                    "❌ *EUR/USD research aborted*\nNo market data available: "
                    f"`{state.data_result.get('error', 'unknown error')}`"
                )
            return self._summary("data_error", llm_error)

        if state.backtest_report is None:
            self.send("❌ *EUR/USD research failed*: backtest suite produced no results. Check logs.")
            return self._summary("backtest_error", llm_error)

        if not state.final_report_sent:
            if len(last_text) >= FINAL_REPORT_MIN_CHARS and not llm_error:
                logger.info("Model did not send the report itself; forwarding its final answer.")
                self.send(last_text)
            else:
                logger.info("Sending rule-based fallback report.")
                self.send(build_fallback_report(state.backtest_report, state.data_result, llm_error))
            state.final_report_sent = True

        return self._summary("ok", llm_error)

    def _summary(self, status: str, llm_error: Optional[str]) -> Dict[str, Any]:
        """Compact run summary for logging / exit codes."""
        report = self.state.backtest_report or {}
        return {
            "status": status,
            "llm_error": llm_error,
            "tools_called": self.state.tool_log,
            "telegram_messages": self.state.messages_sent,
            "data_status": (self.state.data_result or {}).get("status"),
            "report_path": report.get("report_path"),
            "final_report_sent": self.state.final_report_sent,
        }


# ---------------------------------------------------------------------------
# Deterministic analysis used when the LLM is unavailable
# ---------------------------------------------------------------------------


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


def build_fallback_report(
    report: Dict[str, Any],
    data_result: Optional[Dict[str, Any]],
    llm_error: Optional[str] = None,
) -> str:
    """Build a Telegram-Markdown report using deterministic rules only."""
    data = report.get("data", {})
    assumptions = report.get("assumptions", {})
    lines = [
        "*EUR/USD Intraday Research Report* (rule-based)",
        f"Data: {data.get('start_utc', '?')[:10]} → {data.get('end_utc', '?')[:10]}, "
        f"{data.get('bars')} bars, {data.get('trading_days')} days, 07:00-20:30 UTC",
        f"Costs: ~{assumptions.get('approx_round_trip_cost_pips')} pips round trip | "
        f"Overnight violations: {report.get('overnight_violations_total')}",
    ]
    if data_result and data_result.get("status") == "stale_cache":
        lines.append("⚠️ Live fetch failed — results use cached candles.")
    if llm_error:
        lines.append("⚠️ LLM analyst unavailable; deterministic assessment below.")

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
