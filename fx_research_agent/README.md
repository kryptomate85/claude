# EUR/USD Intraday Research Agent

Quant research pipeline. It fetches M5 EUR/USD mid-price candles and keeps only the liquid 07:00–20:30 UTC bars. It backtests three intraday strategies with **zero overnight exposure** and writes a rule-based research report. The report is printed to the console and saved to `reports/research_report_<UTC>.md`. By default candles come from Dukascopy's public datafeed, so **no API key or account is needed**. Yahoo Finance (`DATA_SOURCE=yahoo`, also keyless) and OANDA (`DATA_SOURCE=oanda`) are alternatives; only OANDA needs credentials. Telegram delivery and an OpenAI tool-calling analyst (`--llm`) are both optional.

## Layout

| File | Role |
|---|---|
| `config.py` | Loads and validates `.env` secrets and tuning settings |
| `tools/notifier.py` | *Optional* Telegram `sendMessage` (timeouts, retries, 429 handling, chunking, plain-text fallback); logs to the console when Telegram isn't configured |
| `tools/data_fetcher.py` | Dukascopy daily 1-minute bid/ask files (default; averaged to mid, resampled to M5, raw days cached in `data/dukascopy/`), Yahoo M5 bars (last 60 days) or OANDA `/v3/instruments/EUR_USD/candles` (M5, mid, complete only); filters to 07:00–20:30 UTC and writes `data/EUR_USD_M5.parquet`; uses the cache if the source is down |
| `tools/strategies.py` | `london_momentum`, `ny_mean_reversion`, `london_orb`, `us_release_fade` (candidate; see docstring); the shared base `next()` enforces the 20:25 UTC flat rule |
| `tools/backtest_runner.py` | Runs the suite on full, in-sample and out-of-sample data, posts progress messages, audits for overnight trades, writes the JSON report to `reports/` |
| `tools/report.py` | Rule-based analyst (the default): scorecard, edge vs. breakeven win rate, cost drag, out-of-sample stability, drawdown, verdict |
| `agent.py` | *Optional* OpenAI tool-calling analyst (`--llm`); falls back to `tools/report.py` if the model is unavailable |
| `main.py` | CLI entry point with console and rotating-file logging (`logs/agent.log`) |

## Setup

```bash
cd fx_research_agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py                  # fetch (Dukascopy, no key) -> backtest -> rule-based report
DATA_SOURCE=yahoo python main.py --lookback 59   # keyless alternative, last 60 days only
python main.py --lookback 1825  # 5 years from Dukascopy (first run downloads ~2,600 day files; later runs use the cache)
python main.py --lookback 60    # custom lookback window

# Optional: OANDA as the data source
cp .env.example .env            # set DATA_SOURCE=oanda, OANDA_API_KEY, OANDA_ACCOUNT_ID

# Optional LLM analyst
pip install openai              # and set OPENAI_API_KEY in .env
python main.py --llm
python -m pytest tests -q
```

## Data sources

| `DATA_SOURCE` | Key needed | Notes |
|---|---|---|
| `dukascopy` (default) | No | Bid and ask 1-minute bars averaged to mid, minutes with no ticks dropped, resampled to M5. One request per weekday per side, paced and retried because the feed rate-limits; completed days are cached, so reruns only download new days. The current UTC day is not published until it ends. |
| `yahoo` | No | Native M5 bars from Yahoo's chart API, last 60 days only (a longer `LOOKBACK_DAYS` is truncated with a warning). No volume, and bars are thinner than broker data (smaller ranges, more flat bars), which can understate breakout and momentum signals. Unofficial API that can change without notice. |
| `oanda` | `OANDA_API_KEY`, `OANDA_ACCOUNT_ID` | Complete M5 mid candles up to the latest closed bar. |

The two feeds come from different liquidity providers, so prices can differ by a fraction of a pip; that is small next to the ~1.7-pip round-trip cost the backtests charge.

## Zero-overnight guarantee

`IntradaySessionStrategy.next()` runs before any strategy logic. On a bar at or after **20:25 UTC** it cancels pending orders, closes the position and returns, so no entry can happen. It does the same on the last available bar of each day, which covers early closes and data gaps. Backtests use `trade_on_close=True`, so the liquidation fills on that same bar. Every run is also checked after the fact (`overnight_violations`), and the tests include a strategy that is always long to prove the rule holds.

## Cost model

The cost model is `commission=0.00008` relative, charged on each side: about 0.86 pip per side at 1.08, so about 1.7 pips per round trip. `config.py` rejects any value below 0.00008. Leverage defaults to 1x (`BACKTEST_LEVERAGE`).
