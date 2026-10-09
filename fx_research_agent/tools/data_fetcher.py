"""OANDA v20 candle ingestion.

Pulls recent *complete* M5 mid-price candles for an instrument from
``/v3/instruments/{instrument}/candles``, paginating forward in 5000-candle
pages, filters them strictly to the liquid London + New York daytime window
(07:00–20:30 UTC bar-open times, inclusive) and persists them as Parquet
(e.g. ``data/EUR_USD_M5.parquet``).

If the API is unreachable, the previously cached Parquet file is used as a
fallback so that the pipeline can still proceed (and the result says so).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from config import SessionWindow, Settings, get_settings

logger = logging.getLogger(__name__)

MAX_CANDLES_PER_REQUEST = 5000
MAX_PAGES = 200
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataFetchError(RuntimeError):
    """Raised when candles cannot be retrieved or are unusable."""


class OandaClient:
    """Thin OANDA v20 REST client for instrument candles."""

    def __init__(self, settings: Settings, session: Optional[requests.Session] = None) -> None:
        """Create a client bound to the configured environment and token."""
        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {settings.oanda_api_key}",
                "Accept-Datetime-Format": "RFC3339",
                "Content-Type": "application/json",
            }
        )

    def _get(self, path: str, params: Dict[str, Any], retries: int = 4) -> Dict[str, Any]:
        """GET ``path`` with retries for timeouts, 429 and 5xx responses.

        Raises:
            DataFetchError: On authentication errors, bad requests, or when all
                retries are exhausted.
        """
        url = f"{self.settings.oanda_base_url}{path}"
        last_error = "unknown error"
        for attempt in range(1, retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.settings.http_timeout)
            except requests.Timeout:
                last_error = f"timeout after {self.settings.http_timeout}s"
                logger.warning("OANDA %s (attempt %d/%d)", last_error, attempt, retries)
                time.sleep(min(2**attempt, 16))
                continue
            except requests.RequestException as exc:
                last_error = f"connection error: {exc}"
                logger.warning("OANDA %s (attempt %d/%d)", last_error, attempt, retries)
                time.sleep(min(2**attempt, 16))
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise DataFetchError("OANDA returned non-JSON payload") from exc

            message = _error_message(resp)
            if resp.status_code in (401, 403):
                raise DataFetchError(
                    f"OANDA authentication failed ({resp.status_code}): {message}. "
                    "Check OANDA_API_KEY and OANDA_ENVIRONMENT."
                )
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}: {message}"
                logger.warning("OANDA %s (attempt %d/%d)", last_error, attempt, retries)
                time.sleep(min(2**attempt, 16))
                continue
            raise DataFetchError(f"OANDA request failed ({resp.status_code}): {message}")

        raise DataFetchError(f"OANDA request failed after {retries} attempts: {last_error}")

    def fetch_candles(
        self,
        instrument: str,
        granularity: str,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch all complete mid-price candles between ``start`` and ``end``.

        Args:
            instrument: OANDA instrument name, e.g. ``"EUR_USD"``.
            granularity: Candle granularity, e.g. ``"M5"``.
            start: Inclusive UTC start time.
            end: Exclusive UTC end time (defaults to now).

        Returns:
            DataFrame indexed by UTC bar-open time with columns
            ``Open, High, Low, Close, Volume``.
        """
        end = end or datetime.now(timezone.utc)
        cursor = start
        include_first = True
        rows: List[Dict[str, Any]] = []

        for page in range(1, MAX_PAGES + 1):
            params: Dict[str, Any] = {
                "granularity": granularity,
                "price": "M",
                "from": _rfc3339(cursor),
                "count": MAX_CANDLES_PER_REQUEST,
                "includeFirst": "true" if include_first else "false",
            }
            payload = self._get(f"/v3/instruments/{instrument}/candles", params)
            candles = payload.get("candles", [])
            if not candles:
                break

            for candle in candles:
                if not candle.get("complete", False):
                    continue
                mid = candle.get("mid") or {}
                try:
                    rows.append(
                        {
                            "time": candle["time"],
                            "Open": float(mid["o"]),
                            "High": float(mid["h"]),
                            "Low": float(mid["l"]),
                            "Close": float(mid["c"]),
                            "Volume": int(candle.get("volume", 0)),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    logger.debug("Skipping malformed candle: %s", candle)

            last_time = pd.Timestamp(candles[-1]["time"]).to_pydatetime()
            logger.info(
                "OANDA page %d: %d candles up to %s", page, len(candles), last_time.isoformat()
            )
            if len(candles) < MAX_CANDLES_PER_REQUEST or last_time >= end:
                break
            cursor = last_time
            include_first = False
        else:
            logger.warning("Reached MAX_PAGES=%d; data may be truncated.", MAX_PAGES)

        if not rows:
            raise DataFetchError(f"No complete candles returned for {instrument} {granularity}.")

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
        df = df.drop_duplicates("time").set_index("time").sort_index()
        df = df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]
        df.index.name = "time"
        return df[REQUIRED_COLUMNS]


def filter_liquid_hours(df: pd.DataFrame, window: SessionWindow) -> pd.DataFrame:
    """Keep only bars whose UTC open time lies inside ``window`` on weekdays.

    Args:
        df: Candle frame indexed by tz-aware (or naive UTC) timestamps.
        window: Inclusive liquid window, default 07:00–20:30 UTC.

    Returns:
        A filtered copy; everything outside the window is dropped.
    """
    if df.empty:
        return df.copy()
    index = df.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    minute_of_day = index.hour * 60 + index.minute
    mask = (
        (minute_of_day >= window.start_minute)
        & (minute_of_day <= window.end_minute)
        & (index.dayofweek < 5)
    )
    out = df.loc[mask].copy()
    out.index = index[mask]
    return out


def load_candles(path: Path) -> pd.DataFrame:
    """Load and validate a cached candle Parquet file.

    Raises:
        DataFetchError: If the file is missing, unreadable or malformed.
    """
    if not path.exists():
        raise DataFetchError(f"Candle file not found: {path}. Run fetch_data first.")
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - pyarrow raises many types
        raise DataFetchError(f"Could not read {path}: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataFetchError(f"{path} is missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataFetchError(f"{path} does not have a DatetimeIndex.")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        raise DataFetchError(f"{path} contains no usable candles.")
    return df[REQUIRED_COLUMNS]


def summarize_candles(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a JSON-friendly description of a candle frame."""
    if df.empty:
        return {"rows": 0}
    return {
        "rows": int(len(df)),
        "start_utc": df.index[0].isoformat(),
        "end_utc": df.index[-1].isoformat(),
        "trading_days": int(df.index.normalize().nunique()),
        "first_bar_time_utc": df.index.min().strftime("%H:%M"),
        "last_bar_time_utc": max(df.index.strftime("%H:%M")),
        "close_last": round(float(df["Close"].iloc[-1]), 5),
        "avg_daily_range_pips": round(
            float(
                (df["High"].groupby(df.index.date).max() - df["Low"].groupby(df.index.date).min()).mean()
                * 10_000
            ),
            1,
        ),
    }


def fetch_data(lookback_days: Optional[int] = None, settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Fetch, filter and persist recent candles; fall back to cache on failure.

    This is the function exposed to the agent as the ``fetch_data`` tool. It
    never raises: failures are reported in the returned dict.

    Args:
        lookback_days: Calendar days of history to pull (default from settings).
        settings: Optional explicit settings (defaults to global settings).

    Returns:
        Dict with ``status`` (``"ok"``, ``"stale_cache"`` or ``"error"``),
        ``source``, ``path`` and a candle summary.
    """
    settings = settings or get_settings()
    days = int(lookback_days or settings.lookback_days)
    days = max(1, min(days, 365))
    path = settings.parquet_path
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    try:
        client = OandaClient(settings)
        raw = client.fetch_candles(settings.instrument, settings.granularity, start, end)
        filtered = filter_liquid_hours(raw, settings.liquid_window)
        if filtered.empty:
            raise DataFetchError("No candles left after liquid-hours filter.")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        filtered.to_parquet(tmp, engine="pyarrow")
        tmp.replace(path)  # atomic swap: never leave a half-written cache
        summary = summarize_candles(filtered)
        logger.info(
            "Saved %d liquid-hour candles (%d raw) to %s", len(filtered), len(raw), path
        )
        return {
            "status": "ok",
            "source": "oanda",
            "instrument": settings.instrument,
            "granularity": settings.granularity,
            "lookback_days": days,
            "raw_candles": int(len(raw)),
            "dropped_outside_session": int(len(raw) - len(filtered)),
            "session_filter_utc": "07:00-20:30",
            "path": str(path),
            **summary,
        }
    except Exception as exc:  # noqa: BLE001 - tool must never crash the agent
        logger.error("Live fetch failed: %s", exc)
        try:
            cached = filter_liquid_hours(load_candles(path), settings.liquid_window)
        except DataFetchError as cache_exc:
            return {
                "status": "error",
                "source": None,
                "error": str(exc),
                "cache_error": str(cache_exc),
                "path": str(path),
            }
        return {
            "status": "stale_cache",
            "source": "cache",
            "error": str(exc),
            "warning": "Live fetch failed; using previously cached candles.",
            "path": str(path),
            **summarize_candles(cached),
        }


def _rfc3339(value: datetime) -> str:
    """Format a datetime as an OANDA-compatible RFC3339 UTC string."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f000Z")


def _error_message(resp: requests.Response) -> str:
    """Extract OANDA's ``errorMessage`` field from a failed response."""
    try:
        return str(resp.json().get("errorMessage", resp.text[:300]))
    except ValueError:
        return resp.text[:300]
