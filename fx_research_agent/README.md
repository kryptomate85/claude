# EUR/USD Intraday Research Agent

Quant research pipeline. It fetches OANDA M5 EUR/USD candles and keeps only the liquid 07:00–20:30 UTC bars. It backtests three intraday strategies with **zero overnight exposure** and writes a rule-based research report. The report is printed to the console and saved to `reports/research_report_<UTC>.md`. **Only OANDA credentials are required.** Telegram delivery and an OpenAI tool-calling analyst (`--llm`) are both optional.

## Layout

| File | Role |
|---|---|
| `config.py` | Loads and validates `.env` secrets and tuning settings |
| `tools/notifier.py` | *Optional* Telegram `sendMessage` (timeouts, retries, 429 handling, chunking, plain-text fallback); logs to the console when Telegram isn't configured |
| `tools/data_fetcher.py` | OANDA `/v3/instruments/EUR_USD/candles` (M5, mid, complete only), paginated; filters to 07:00–20:30 UTC and writes `data/EUR_USD_M5.parquet`; uses the cache if the API is down |
| `tools/strategies.py` | `london_momentum`, `ny_mean_reversion`, `london_orb`; the shared base `next()` enforces the 20:25 UTC flat rule |
| `tools/backtest_runner.py` | Runs the suite on full, in-sample and out-of-sample data, posts progress messages, audits for overnight trades, writes the JSON report to `reports/` |
| `tools/report.py` | Rule-based analyst (the default): scorecard, edge vs. breakeven win rate, cost drag, out-of-sample stability, drawdown, verdict |
| `agent.py` | *Optional* OpenAI tool-calling analyst (`--llm`); falls back to `tools/report.py` if the model is unavailable |
| `main.py` | CLI entry point with console and rotating-file logging (`logs/agent.log`) |

## Setup

```bash
cd fx_research_agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OANDA_API_KEY and OANDA_ACCOUNT_ID
python main.py                  # fetch -> backtest -> rule-based report
python main.py --lookback 60    # custom lookback window

# Optional LLM analyst
pip install openai              # and set OPENAI_API_KEY in .env
python main.py --llm
python -m pytest tests -q
```

## Zero-overnight guarantee

`IntradaySessionStrategy.next()` runs before any strategy logic. On a bar at or after **20:25 UTC** it cancels pending orders, closes the position and returns, so no entry can happen. It does the same on the last available bar of each day, which covers early closes and data gaps. Backtests use `trade_on_close=True`, so the liquidation fills on that same bar. Every run is also checked after the fact (`overnight_violations`), and the tests include a strategy that is always long to prove the rule holds.

## Cost model

The cost model is `commission=0.00008` relative, charged on each side: about 0.86 pip per side at 1.08, so about 1.7 pips per round trip. `config.py` rejects any value below 0.00008. Leverage defaults to 1x (`BACKTEST_LEVERAGE`).
