"""Configuration loading and validation for the FX research agent.

All secrets are read from environment variables (optionally populated from a
``.env`` file located next to this module or in the current working directory).
Non-secret tuning knobs also have environment overrides with sane defaults.

Usage::

    from config import get_settings
    settings = get_settings()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Load .env from project root first, then fall back to the CWD (no override of
# variables that are already exported in the shell).
load_dotenv(PROJECT_ROOT / ".env", override=False)
load_dotenv(override=False)

OANDA_HOSTS: Dict[str, str] = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}

# Candle sources. Dukascopy and Yahoo need no account or key.
DATA_SOURCES: Tuple[str, ...] = ("dukascopy", "yahoo", "oanda")

# Upper bound for LOOKBACK_DAYS / --lookback. Dukascopy serves many years of
# history; Yahoo truncates to 60 days on its own.
MAX_LOOKBACK_DAYS: int = 3650

# Secrets needed only when DATA_SOURCE=oanda.
SECRET_KEYS: Tuple[str, ...] = (
    "OANDA_API_KEY",
    "OANDA_ACCOUNT_ID",
)
# Optional: set both to receive progress pings and reports on Telegram.
TELEGRAM_SECRET_KEYS: Tuple[str, ...] = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")
# Extra secret needed only for the optional LLM analyst (``main.py --llm``).
LLM_SECRET_KEYS: Tuple[str, ...] = ("OPENAI_API_KEY",)


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _env_str(name: str, default: str = "") -> str:
    """Return a stripped string environment variable or ``default``."""
    value = os.getenv(name)
    return value.strip() if value is not None and value.strip() else default


def _env_float(name: str, default: float) -> float:
    """Return a float environment variable, raising ``ConfigError`` if malformed."""
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    """Return an int environment variable, raising ``ConfigError`` if malformed."""
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class SessionWindow:
    """A UTC intraday window expressed as minutes after midnight (inclusive)."""

    start_minute: int
    end_minute: int

    @classmethod
    def from_hhmm(cls, start: str, end: str) -> "SessionWindow":
        """Build a window from ``"HH:MM"`` strings."""
        return cls(_hhmm_to_minutes(start), _hhmm_to_minutes(end))

    def contains(self, minute_of_day: int) -> bool:
        """Return True if ``minute_of_day`` lies inside the window (inclusive)."""
        return self.start_minute <= minute_of_day <= self.end_minute


def _hhmm_to_minutes(value: str) -> int:
    """Convert ``"HH:MM"`` into minutes after midnight."""
    try:
        hours, minutes = value.split(":")
        total = int(hours) * 60 + int(minutes)
    except (ValueError, AttributeError) as exc:
        raise ConfigError(f"Invalid HH:MM time {value!r}") from exc
    if not 0 <= total < 24 * 60:
        raise ConfigError(f"Time out of range: {value!r}")
    return total


@dataclass(frozen=True)
class Settings:
    """Immutable, validated runtime settings."""

    # Secrets (OpenAI and Telegram are optional)
    openai_api_key: str
    oanda_api_key: str
    oanda_account_id: str
    telegram_bot_token: str
    telegram_chat_id: str

    # Data
    data_source: str = "dukascopy"
    oanda_environment: str = "practice"
    instrument: str = "EUR_USD"
    granularity: str = "M5"
    lookback_days: int = 90
    data_dir: Path = PROJECT_ROOT / "data"
    reports_dir: Path = PROJECT_ROOT / "reports"
    http_timeout: float = 20.0

    # Liquid session filter (UTC, inclusive bar-open times)
    liquid_window: SessionWindow = field(
        default_factory=lambda: SessionWindow.from_hhmm("07:00", "20:30")
    )
    # Hard flat time: positions are liquidated at/after this bar-open time.
    hard_close_minute: int = 20 * 60 + 25

    # Backtest
    cash: float = 100_000.0
    commission: float = 0.00008
    leverage: float = 1.0
    position_size: float = 0.95
    oos_fraction: float = 0.3

    # LLM
    openai_model: str = "gpt-4o"
    openai_base_url: Optional[str] = None
    agent_max_steps: int = 12

    @property
    def oanda_base_url(self) -> str:
        """REST host for the configured OANDA environment."""
        return OANDA_HOSTS[self.oanda_environment]

    @property
    def parquet_path(self) -> Path:
        """Location of the cached candle file, e.g. ``data/EUR_USD_M5.parquet``."""
        return self.data_dir / f"{self.instrument}_{self.granularity}.parquet"

    @property
    def telegram_enabled(self) -> bool:
        """True when Telegram delivery is configured (otherwise console only)."""
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def margin(self) -> float:
        """backtesting.py ``margin`` parameter derived from leverage."""
        return 1.0 / self.leverage

    def masked(self) -> Dict[str, str]:
        """Return a log-safe view of the secrets (values partially redacted)."""

        def _mask(value: str) -> str:
            return "<missing>" if not value else f"{value[:4]}…({len(value)} chars)"

        return {
            "OPENAI_API_KEY": _mask(self.openai_api_key),
            "OANDA_API_KEY": _mask(self.oanda_api_key),
            "OANDA_ACCOUNT_ID": _mask(self.oanda_account_id),
            "TELEGRAM_BOT_TOKEN": _mask(self.telegram_bot_token),
            "TELEGRAM_CHAT_ID": _mask(self.telegram_chat_id),
        }


def load_settings(required: Tuple[str, ...] = ()) -> Settings:
    """Read, validate and return :class:`Settings`.

    Args:
        required: Names of secret variables that must be present. The OANDA
            secrets are added automatically when ``DATA_SOURCE=oanda``; add
            ``LLM_SECRET_KEYS`` when the LLM analyst is enabled.

    Raises:
        ConfigError: If a required variable is missing or a value is invalid.
    """
    data_source = _env_str("DATA_SOURCE", "dukascopy").lower()
    if data_source not in DATA_SOURCES:
        raise ConfigError(f"DATA_SOURCE must be one of {list(DATA_SOURCES)}, got {data_source!r}")
    if data_source == "oanda":
        required = tuple(dict.fromkeys(SECRET_KEYS + tuple(required)))
    missing = [key for key in required if not _env_str(key)]
    if missing:
        raise ConfigError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill them in."
        )

    telegram_set = [key for key in TELEGRAM_SECRET_KEYS if _env_str(key)]
    if len(telegram_set) == 1:
        missing_tg = next(k for k in TELEGRAM_SECRET_KEYS if k not in telegram_set)
        raise ConfigError(
            f"{telegram_set[0]} is set but {missing_tg} is not. Set both to enable "
            "Telegram, or neither to report to the console only."
        )

    environment = _env_str("OANDA_ENVIRONMENT", "practice").lower()
    if environment not in OANDA_HOSTS:
        raise ConfigError(
            f"OANDA_ENVIRONMENT must be one of {sorted(OANDA_HOSTS)}, got {environment!r}"
        )

    data_dir = Path(_env_str("DATA_DIR", "data"))
    reports_dir = Path(_env_str("REPORTS_DIR", "reports"))
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    if not reports_dir.is_absolute():
        reports_dir = PROJECT_ROOT / reports_dir

    settings = Settings(
        openai_api_key=_env_str("OPENAI_API_KEY"),
        oanda_api_key=_env_str("OANDA_API_KEY"),
        oanda_account_id=_env_str("OANDA_ACCOUNT_ID"),
        telegram_bot_token=_env_str("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_env_str("TELEGRAM_CHAT_ID"),
        data_source=data_source,
        oanda_environment=environment,
        instrument=_env_str("INSTRUMENT", "EUR_USD").upper(),
        granularity=_env_str("GRANULARITY", "M5").upper(),
        lookback_days=_env_int("LOOKBACK_DAYS", 90),
        data_dir=data_dir,
        reports_dir=reports_dir,
        http_timeout=_env_float("HTTP_TIMEOUT", 20.0),
        cash=_env_float("BACKTEST_CASH", 100_000.0),
        commission=_env_float("BACKTEST_COMMISSION", 0.00008),
        leverage=_env_float("BACKTEST_LEVERAGE", 1.0),
        position_size=_env_float("POSITION_SIZE", 0.95),
        oos_fraction=_env_float("OOS_FRACTION", 0.3),
        openai_model=_env_str("OPENAI_MODEL", "gpt-4o"),
        openai_base_url=_env_str("OPENAI_BASE_URL") or None,
        agent_max_steps=_env_int("AGENT_MAX_STEPS", 12),
    )
    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    """Cross-field sanity checks; raises :class:`ConfigError` on failure."""
    if settings.granularity != "M5":
        raise ConfigError("Only M5 granularity is supported by the strategy suite.")
    if not 1 <= settings.lookback_days <= MAX_LOOKBACK_DAYS:
        raise ConfigError(f"LOOKBACK_DAYS must be between 1 and {MAX_LOOKBACK_DAYS}.")
    if settings.commission < 0.00008:
        raise ConfigError(
            "BACKTEST_COMMISSION must be >= 0.00008 (0.8 pip equivalent cost floor)."
        )
    if settings.cash <= 0:
        raise ConfigError("BACKTEST_CASH must be positive.")
    if not 1.0 <= settings.leverage <= 50.0:
        raise ConfigError("BACKTEST_LEVERAGE must be between 1 and 50.")
    if not 0.0 < settings.position_size < 1.0:
        raise ConfigError("POSITION_SIZE must be a fraction in (0, 1).")
    if not 0.0 < settings.oos_fraction < 0.9:
        raise ConfigError("OOS_FRACTION must be in (0, 0.9).")
    if settings.http_timeout <= 0:
        raise ConfigError("HTTP_TIMEOUT must be positive.")
    if settings.agent_max_steps < 3:
        raise ConfigError("AGENT_MAX_STEPS must be at least 3.")


_SETTINGS: Optional[Settings] = None


def get_settings(required: Tuple[str, ...] = ()) -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings(required)
    return _SETTINGS


def set_settings(settings: Settings) -> None:
    """Install ``settings`` as the process-wide instance (used by main/tests)."""
    global _SETTINGS
    _SETTINGS = settings
