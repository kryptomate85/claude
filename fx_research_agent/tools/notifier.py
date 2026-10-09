"""Telegram Bot API notifier.

Sends Markdown-formatted messages via ``sendMessage`` with timeouts, retries on
transient failures / rate limits, automatic chunking for Telegram's 4096-char
limit, and a plain-text fallback when Telegram rejects the Markdown entities.
Never raises on network errors: callers get a boolean result instead.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"
MAX_MESSAGE_LEN = 4096
_CHUNK_LEN = 3900  # headroom for chunk markers


class TelegramNotifier:
    """Minimal, robust Telegram ``sendMessage`` client."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        timeout: float = 15.0,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
    ) -> None:
        """Create a notifier.

        Args:
            bot_token: Telegram bot token from @BotFather.
            chat_id: Destination chat/channel id.
            timeout: Per-request timeout in seconds.
            max_retries: Attempts per chunk for transient errors.
            session: Optional pre-configured ``requests.Session``.
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.session = session or requests.Session()

    @property
    def enabled(self) -> bool:
        """True when both token and chat id are configured."""
        return bool(self.bot_token and self.chat_id)

    def send(self, text: str, parse_mode: Optional[str] = "Markdown") -> bool:
        """Send ``text`` (chunked if necessary).

        Args:
            text: Message body. Telegram legacy ``Markdown`` syntax by default.
            parse_mode: ``"Markdown"``, ``"MarkdownV2"``, ``"HTML"`` or ``None``.

        Returns:
            True if every chunk was delivered, False otherwise.
        """
        if not self.enabled:
            logger.warning("Telegram not configured; message dropped: %.120s", text)
            return False
        if not text or not text.strip():
            logger.warning("Refusing to send empty Telegram message.")
            return False

        chunks = split_message(text, _CHUNK_LEN)
        ok = True
        for idx, chunk in enumerate(chunks, start=1):
            if len(chunks) > 1:
                chunk = f"{chunk}\n\n({idx}/{len(chunks)})"
            ok = self._send_chunk(chunk, parse_mode) and ok
        return ok

    def _send_chunk(self, text: str, parse_mode: Optional[str]) -> bool:
        """POST one chunk with retry, rate-limit and Markdown-fallback handling."""
        url = f"{TELEGRAM_API}/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout)
            except requests.Timeout:
                logger.warning("Telegram timeout (attempt %d/%d)", attempt, self.max_retries)
                self._backoff(attempt)
                continue
            except requests.RequestException as exc:
                logger.warning(
                    "Telegram connection error (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    _redact(str(exc), self.bot_token),
                )
                self._backoff(attempt)
                continue

            if resp.status_code == 200:
                return True

            description = _describe(resp)
            if resp.status_code == 429:
                retry_after = _retry_after(resp)
                logger.warning("Telegram rate-limited; sleeping %ss", retry_after)
                time.sleep(retry_after)
                continue
            if resp.status_code == 400 and "parse" in description.lower() and "parse_mode" in payload:
                logger.warning("Telegram rejected Markdown (%s); resending as plain text.", description)
                payload.pop("parse_mode", None)
                continue
            if 500 <= resp.status_code < 600:
                logger.warning("Telegram server error %s: %s", resp.status_code, description)
                self._backoff(attempt)
                continue

            logger.error("Telegram sendMessage failed (%s): %s", resp.status_code, description)
            return False

        logger.error("Telegram sendMessage gave up after %d attempts.", self.max_retries)
        return False

    @staticmethod
    def _backoff(attempt: int) -> None:
        """Exponential backoff capped at 8 seconds."""
        time.sleep(min(2 ** (attempt - 1), 8))


def split_message(text: str, limit: int = _CHUNK_LEN) -> List[str]:
    """Split ``text`` into chunks no longer than ``limit``, preferring line breaks."""
    if len(text) <= limit:
        return [text]
    chunks: List[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:  # pathological single long line
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) > limit:
            chunks.append(current)
            current = ""
        current += line
    if current.strip():
        chunks.append(current)
    return [c.rstrip("\n") for c in chunks if c.strip()]


def _describe(resp: requests.Response) -> str:
    """Extract Telegram's error description from a response."""
    try:
        return str(resp.json().get("description", resp.text))
    except ValueError:
        return resp.text[:300]


def _retry_after(resp: requests.Response) -> int:
    """Read ``retry_after`` seconds from a 429 response (default 3, max 60)."""
    try:
        value = int(resp.json().get("parameters", {}).get("retry_after", 3))
    except (ValueError, TypeError, AttributeError):
        value = 3
    return max(1, min(value, 60))


def _redact(message: str, secret: str) -> str:
    """Remove the bot token from log messages (it is embedded in the URL)."""
    return message.replace(secret, "<token>") if secret else message


_DEFAULT: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Return a notifier built from the global settings (lazily created)."""
    global _DEFAULT
    if _DEFAULT is None:
        from config import get_settings

        settings = get_settings()
        _DEFAULT = TelegramNotifier(
            settings.telegram_bot_token,
            settings.telegram_chat_id,
            timeout=settings.http_timeout,
        )
    return _DEFAULT


def send_telegram_message(text: str, parse_mode: Optional[str] = "Markdown") -> bool:
    """Convenience wrapper around the default notifier; never raises."""
    try:
        return get_notifier().send(text, parse_mode=parse_mode)
    except Exception:  # noqa: BLE001 - notification must never crash the pipeline
        logger.exception("Unexpected error while sending Telegram message.")
        return False
