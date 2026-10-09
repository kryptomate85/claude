"""Entry point for the autonomous EUR/USD intraday research pipeline.

Examples::

    python main.py                 # full agentic run (OpenAI + OANDA + Telegram)
    python main.py --no-llm        # deterministic pipeline with rule-based report
    python main.py --lookback 30   # override the lookback window
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import List, Optional

from config import PROJECT_ROOT, SECRET_KEYS, ConfigError, load_settings, set_settings


def setup_logging(verbose: bool = False) -> None:
    """Configure console + rotating file logging (``logs/agent.log``)."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    try:
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(log_dir / "agent.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError as exc:
        root.warning("File logging disabled: %s", exc)

    for noisy in ("urllib3", "httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Autonomous EUR/USD intraday research agent")
    parser.add_argument("--no-llm", action="store_true", help="Skip the LLM and use the rule-based analyst.")
    parser.add_argument("--lookback", type=int, default=None, help="Override LOOKBACK_DAYS.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser.parse_args(argv)


def run_deterministic() -> dict:
    """Run fetch → backtest → rule-based report without any LLM."""
    from agent import build_fallback_report
    from tools.backtest_runner import run_backtest_suite
    from tools.data_fetcher import fetch_data
    from tools.notifier import send_telegram_message

    send_telegram_message("🚀 *EUR/USD research pipeline started* (deterministic mode)")
    data_result = fetch_data()
    if data_result.get("status") == "error":
        send_telegram_message(f"❌ *EUR/USD research aborted*: `{data_result.get('error')}`")
        return {"status": "data_error", "data": data_result}
    report = run_backtest_suite()
    if report.get("status") != "ok":
        send_telegram_message(f"❌ *Backtest suite failed*: `{report.get('error')}`")
        return {"status": "backtest_error", "error": report.get("error")}
    send_telegram_message(build_fallback_report(report, data_result))
    return {"status": "ok", "report_path": report.get("report_path")}


def main(argv: Optional[List[str]] = None) -> int:
    """Execute the pipeline and return a process exit code."""
    args = parse_args(argv)
    setup_logging(args.verbose)
    log = logging.getLogger("main")

    required = tuple(k for k in SECRET_KEYS if not (args.no_llm and k == "OPENAI_API_KEY"))
    try:
        settings = load_settings(required)
        if args.lookback:
            settings = dataclasses.replace(settings, lookback_days=max(1, min(args.lookback, 365)))
    except ConfigError as exc:
        log.error("Configuration error: %s", exc)
        return 2
    set_settings(settings)
    log.info(
        "Starting research pipeline | instrument=%s granularity=%s lookback=%dd model=%s env=%s",
        settings.instrument,
        settings.granularity,
        settings.lookback_days,
        "none" if args.no_llm else settings.openai_model,
        settings.oanda_environment,
    )
    log.debug("Secrets: %s", settings.masked())

    try:
        if args.no_llm:
            summary = run_deterministic()
        else:
            from agent import ResearchAgent

            summary = ResearchAgent(settings).run()
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        return 130
    except Exception:  # noqa: BLE001 - last-resort guard; components already degrade gracefully
        log.exception("Unhandled error in research pipeline")
        return 1

    log.info("Run summary: %s", json.dumps(summary, default=str))
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
