"""
notifier.py — Slingger V6 Shared Telegram Notification Module
==============================================================
Standalone module imported by all V6 pipeline scripts.
Loads credentials from .env — NEVER hardcoded.

Usage:
    from notifier import send_telegram
    send_telegram("✅ Model retrained successfully")
"""

import os
import logging

import requests
from dotenv import load_dotenv

# Load .env from the parent directory (slingger/ → project root)
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH, override=False)

logger = logging.getLogger(__name__)


def send_telegram(message: str) -> bool:
    """
    Send a message to the configured Telegram bot.

    Args:
        message: The message text to send. Supports Markdown formatting.

    Returns:
        True if the message was sent successfully, False otherwise.

    Note:
        This function will NEVER raise an exception — a failed notification
        must not crash the trading pipeline. Failures are logged as errors.
    """
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.error(
            "[NOTIFIER] TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set in environment. "
            "Notification skipped."
        )
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("[NOTIFIER] Telegram message sent successfully.")
        return True
    except requests.exceptions.Timeout:
        logger.error("[NOTIFIER] Telegram request timed out after 10s.")
    except requests.exceptions.HTTPError as e:
        logger.error(f"[NOTIFIER] Telegram HTTP error: {e} — Response: {response.text}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"[NOTIFIER] Telegram connection error: {e}")
    except Exception as e:
        logger.error(f"[NOTIFIER] Unexpected Telegram error: {e}")

    return False


def send_telegram_safe(message: str) -> None:
    """
    Fire-and-forget wrapper. Same as send_telegram but returns None.
    Use in contexts where the return value is irrelevant.
    """
    send_telegram(message)


if __name__ == "__main__":
    # Quick smoke test — run directly to verify Telegram connectivity
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] — %(message)s")
    print("Testing Telegram notifier...")
    success = send_telegram("🔔 *Slingger V6 Notifier* — Smoke test OK. Pipeline online.")
    if success:
        print("✅ Telegram notification sent successfully!")
        sys.exit(0)
    else:
        print("❌ Telegram notification failed. Check .env variables.")
        sys.exit(1)
