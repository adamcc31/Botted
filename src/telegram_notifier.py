"""
telegram_notifier.py — Telegram notifications for engine operations.

Production behavior:
- If `TELEGRAM_BOT_TOKEN` (or legacy `TELEGRAM_TOKEN`) and `TELEGRAM_CHAT_ID`
  are not present, the notifier becomes a no-op (never crashes the bot).
- All methods are async and use small timeouts.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx

from src.config_manager import ConfigManager


class TelegramNotifier:
    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        self._enabled = bool(self._config.get("telegram.enabled", False))

        self._token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
        self._chat_id = os.getenv("TELEGRAM_CHAT_ID")

        # If Telegram env vars exist, allow sending even if config enabled=false,
        # but keep config as the default.
        if self._token and self._chat_id:
            self._enabled = True

        self._rest_timeout_s = float(self._config.get("telegram.timeout_seconds", 10.0))

    async def send_message(
        self,
        title: str,
        message: str,
        *,
        parse_mode: Optional[str] = "HTML",
        disable_notification: bool = False,
    ) -> None:
        if not self._enabled:
            return
        if not self._token or not self._chat_id:
            return

        text = f"🔥 <b>{title}</b>\n{message}"
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        payload = {
            "chat_id": str(self._chat_id),
            "text": text,
            "disable_web_page_preview": True,
            "disable_notification": disable_notification,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        async with httpx.AsyncClient(timeout=self._rest_timeout_s) as client:
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            except Exception:
                # Never fail the trading engine if Telegram errors.
                return

    async def send_document(
        self,
        file_path: str,
        caption: str = "",
        disable_notification: bool = False,
    ) -> None:
        if not self._enabled or not self._token or not self._chat_id:
            return
        
        url = f"https://api.telegram.org/bot{self._token}/sendDocument"
        data = {
            "chat_id": str(self._chat_id),
            "caption": caption,
            "disable_notification": disable_notification,
        }
        
        import os
        if not os.path.exists(file_path):
            return

        # Prepare multipart/form-data for the file
        files = {
            "document": (os.path.basename(file_path), open(file_path, "rb"))
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(url, data=data, files=files)
                resp.raise_for_status()
            except Exception:
                pass
            finally:
                files["document"][1].close()

