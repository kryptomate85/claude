"""Candle ingestion from Dukascopy (default, no key), Yahoo (no key) or OANDA v20.

Pulls recent *complete* M5 candles for an instrument from one of:

* Dukascopy's public datafeed: daily 1-minute bid and ask files, averaged to
  mid and resampled to M5.
* Yahoo Finance's chart API: native M5 bars, last 60 days only.
* OANDA's ``/v3/instruments/{instrument}/candles``: M5 mid candles, paginated
  forward in 5000-candle pages.
 Candles are filtered strictly to the
liquid London + New York daytime window (07:00–20:30 UTC bar-open times,
inclusive) and persisted as Parquet (e.g. ``data/EUR_USD_M5.parquet``).

If the API is unreachable, the previously cached Parquet file is used as a
fallback so that the pipeline can still proceed (and the result says so).
"""

from __future__ import annotations

import logging
import lzma
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from config import MAX_LOOKBACK_DAYS, SessionWindow, Settings, get_settings

logger = logging.getLogger(__name__)

MAX_CANDLES_PER_REQUEST = 5000
MAX_PAGES = 200
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

DUKASCOPY_URL = "https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/{side}_candles_min_1.bi5"
# One record per minute: seconds into the day, open, close, low, high (big-endian
# int32 in price points) and volume (big-endian float32).
DUKASCOPY_RECORD = np.dtype([("sec", ">i4"), ("o", ">i4"), ("c", ">i4"), ("l", ">i4"), ("h", ">i4"), ("v", ">f4")])
DUKASCOPY_PAUSE_SEC = 0.25  # pacing between requests; the feed rate-limits bursts
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_INTERVALS = {"M5": ("5m", 5), "M15": ("15m", 15), "M30": ("30m", 30), "H1": ("60m", 60)}
YAHOO_MAX_DAYS = 59  # Yahoo serves intraday bars for the last 60 days only
RESAMPLE_RULES = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h"}


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


class DukascopyClient:
    """Client for Dukascopy's public datafeed (no account or API key needed).

    Each weekday is one request per price side. Files for completed days never
    change, so they are cached under ``data/dukascopy/`` and reruns only
    download new days.
    """

    def __init__(self, settings: Settings, session: Optional[requests.Session] = None) -> None:
        """Create a client that caches raw day files under the data directory."""
        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "fx-research-agent/1.0"})
        self.cache_dir = settings.data_dir / "dukascopy"

    def _get(self, url: str, retries: int = 6) -> Optional[bytes]:
        """GET ``url`` with retries for timeouts, resets, 429 and 5xx responses.

        Returns:
            The response body, or ``None`` when the feed has no file (404).

        Raises:
            DataFetchError: When all retries are exhausted or the request is refused.
        """
        last_error = "unknown error"
        for attempt in range(1, retries + 1):
            try:
                resp = self.session.get(url, timeout=self.settings.http_timeout)
            except requests.RequestException as exc:
                last_error = f"connection error: {exc}"
            else:
                if resp.status_code == 200:
                    return resp.content
                if resp.status_code == 404:
                    return None
                if resp.status_code != 429 and resp.status_code < 500:
                    raise DataFetchError(f"Dukascopy request failed ({resp.status_code}) for {url}")
                last_error = f"HTTP {resp.status_code}"
            logger.warning("Dukascopy %s (attempt %d/%d)", last_error, attempt, retries)
            time.sleep(min(2**attempt, 30))
        raise DataFetchError(f"Dukascopy request failed after {retries} attempts: {last_error}")

    def _day_file(self, symbol: str, day: pd.Timestamp, side: str) -> Optional[bytes]:
        """Return one day's compressed 1-minute file, from cache when possible."""
        cached = self.cache_dir / symbol / f"{day:%Y-%m-%d}_{side}.bi5"
        if cached.exists():
            return cached.read_bytes()
        # Dukascopy months are zero-based.
        url = DUKASCOPY_URL.format(symbol=symbol, year=day.year, month=day.month - 1, day=day.day, side=side)
        body = self._get(url)
        time.sleep(DUKASCOPY_PAUSE_SEC)
        if body is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(body)
        return body

    def fetch_candles(
        self,
        instrument: str,
        granularity: str,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch mid-price candles for whole weekdays between ``start`` and ``end``.

        Bid and ask 1-minute bars are averaged to mid, minutes with no ticks
        (zero volume) are dropped, and the result is resampled to ``granularity``.
        The current UTC day is excluded because its file is not published yet.

        Args:
            instrument: OANDA-style instrument name, e.g. ``"EUR_USD"``.
            granularity: Candle granularity, e.g. ``"M5"``.
            start: Inclusive UTC start time.
            end: Exclusive UTC end time (defaults to now).

        Returns:
            DataFrame indexed by UTC bar-open time with columns
            ``Open, High, Low, Close, Volume``.
        """
        if granularity not in RESAMPLE_RULES:
            raise DataFetchError(f"Unsupported granularity for Dukascopy: {granularity}")
        symbol = instrument.replace("_", "").upper()
        point = 1e3 if symbol.endswith("JPY") else 1e5
        end = end or datetime.now(timezone.utc)
        today = pd.Timestamp(datetime.now(timezone.utc)).normalize()
        first = pd.Timestamp(start).normalize()
        last = min(pd.Timestamp(end).ceil("D"), today)
        days = [d for d in pd.date_range(first, last, freq="D", inclusive="left") if d.dayofweek < 5]

        frames: List[pd.DataFrame] = []
        for i, day in enumerate(days, 1):
            sides = [_decode_day(self._day_file(symbol, day, side), day, point) for side in ("BID", "ASK")]
            if any(side.empty for side in sides):
                continue
            bid, ask = sides[0].align(sides[1], join="inner")
            mid = (bid[["Open", "High", "Low", "Close"]] + ask[["Open", "High", "Low", "Close"]]) / 2
            mid["Volume"] = bid["Volume"] + ask["Volume"]
            frames.append(mid[mid["Volume"] > 0])
            if i % 20 == 0 or i == len(days):
                logger.info("Dukascopy %s: %d/%d days fetched (through %s)", symbol, i, len(days), day.date())

        if not frames:
            raise DataFetchError(f"No Dukascopy candles returned for {symbol}.")
        minute = pd.concat(frames)
        df = (
            minute.resample(RESAMPLE_RULES[granularity], label="left", closed="left")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna(subset=["Open"])
        )
        df = df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]
        df.index.name = "time"
        return df[REQUIRED_COLUMNS]


def _decode_day(body: Optional[bytes], day: pd.Timestamp, point: float) -> pd.DataFrame:
    """Decode one Dukascopy ``*_candles_min_1.bi5`` file into a 1-minute frame."""
    if not body:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    try:
        raw = np.frombuffer(lzma.decompress(body), dtype=DUKASCOPY_RECORD)
    except (lzma.LZMAError, ValueError) as exc:
        raise DataFetchError(f"Corrupt Dukascopy file for {day.date()}: {exc}") from exc
    index = day + pd.to_timedelta(raw["sec"].astype(np.int64), unit="s")
    return pd.DataFrame(
        {
            "Open": raw["o"] / point,
            "High": raw["h"] / point,
            "Low": raw["l"] / point,
            "Close": raw["c"] / point,
            "Volume": raw["v"].astype(float),
        },
        index=index,
    )


class YahooClient:
    """Client for Yahoo Finance's chart API (no account or API key needed).

    Intraday history is limited to the last 60 days; longer lookbacks are
    truncated with a warning. Yahoo reports no FX volume, so ``Volume`` is 0.
    """

    def __init__(self, settings: Settings, session: Optional[requests.Session] = None) -> None:
        """Create a client; Yahoo refuses requests without a browser-like User-Agent."""
        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def _get(self, url: str, params: Dict[str, Any], retries: int = 4) -> Dict[str, Any]:
        """GET ``url`` with retries for timeouts, truncated bodies, 429 and 5xx."""
        last_error = "unknown error"
        for attempt in range(1, retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.settings.http_timeout)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code != 429 and resp.status_code < 500:
                    raise DataFetchError(f"Yahoo request failed ({resp.status_code}): {resp.text[:300]}")
                last_error = f"HTTP {resp.status_code}"
            except (requests.RequestException, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Yahoo %s (attempt %d/%d)", last_error, attempt, retries)
            time.sleep(min(2**attempt, 16))
        raise DataFetchError(f"Yahoo request failed after {retries} attempts: {last_error}")

    def fetch_candles(
        self,
        instrument: str,
        granularity: str,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch complete candles between ``start`` and ``end`` (at most 60 days back).

        Args:
            instrument: OANDA-style instrument name, e.g. ``"EUR_USD"``.
            granularity: Candle granularity, e.g. ``"M5"``.
            start: Inclusive UTC start time.
            end: Exclusive UTC end time (defaults to now).

        Returns:
            DataFrame indexed by UTC bar-open time with columns
            ``Open, High, Low, Close, Volume``.
        """
        if granularity not in YAHOO_INTERVALS:
            raise DataFetchError(f"Unsupported granularity for Yahoo: {granularity}")
        interval, minutes = YAHOO_INTERVALS[granularity]
        now = datetime.now(timezone.utc)
        end = min(end or now, now)
        earliest = now - timedelta(days=YAHOO_MAX_DAYS)
        if start < earliest:
            logger.warning("Yahoo keeps 60 days of intraday bars; starting at %s", earliest.date())
            start = earliest
        symbol = instrument.replace("_", "").upper() + "=X"
        params = {"interval": interval, "period1": int(start.timestamp()), "period2": int(end.timestamp())}
        chart = self._get(YAHOO_URL.format(symbol=symbol), params).get("chart") or {}
        if chart.get("error") or not chart.get("result"):
            raise DataFetchError(f"Yahoo returned no data for {symbol}: {chart.get('error')}")
        result = chart["result"][0]
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        df = pd.DataFrame(
            {
                "Open": quote.get("open"),
                "High": quote.get("high"),
                "Low": quote.get("low"),
                "Close": quote.get("close"),
                "Volume": 0,
            },
            index=pd.to_datetime(result.get("timestamp", []), unit="s", utc=True),
        ).dropna(subset=["Open", "High", "Low", "Close"])
        # Keep complete bars only: the newest bar is still forming until its close time.
        df = df[df.index + pd.Timedelta(minutes=minutes) <= pd.Timestamp(end)]
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df[df.index >= pd.Timestamp(start)]
        if df.empty:
            raise DataFetchError(f"No complete Yahoo candles for {symbol}.")
        df.index.name = "time"
        return df[REQUIRED_COLUMNS]


def make_client(settings: Settings) -> Any:
    """Return the candle client for ``settings.data_source``."""
    if settings.data_source == "oanda":
        return OandaClient(settings)
    if settings.data_source == "yahoo":
        return YahooClient(settings)
    return DukascopyClient(settings)


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
    days = max(1, min(days, MAX_LOOKBACK_DAYS))
    path = settings.parquet_path
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    try:
        client = make_client(settings)
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
            "source": settings.data_source,
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
