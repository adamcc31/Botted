"""
config_manager.py — Hot-reloadable configuration management.

Watches config.json for changes and reloads automatically.
Supports dot-notation access: config.get("regime.vol_upper_threshold")
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

try:
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    structlog = None

logger = structlog.get_logger(__name__) if structlog else logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.json"


class ConfigManager:
    """Thread-safe, hot-reloadable configuration manager."""

    _instance: Optional[ConfigManager] = None
    _lock = threading.Lock()

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path else _DEFAULT_CONFIG_PATH
        self._config: dict = {}
        self._last_modified: float = 0.0
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._load()
        self._start_file_watcher()

    @classmethod
    def get_instance(cls, path: Optional[Path] = None) -> ConfigManager:
        """Singleton access — ensures one config manager globally."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton — used in tests."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.stop()
            cls._instance = None

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve config value by dot-notation key.

        Example: config.get("regime.vol_upper_threshold", 0.80)
        """
        parts = key.split(".")
        val = self._config
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part)
                if val is None:
                    return default
            else:
                return default
        return val

    def set(self, key: str, value: Any) -> None:
        """
        Set config value by dot-notation key and persist to file.

        Example: config.set("regime.vol_upper_threshold", 0.75)
        """
        parts = key.split(".")
        d = self._config
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
        self._save()
        logger.info("config_updated", key=key, value=value)

    def get_section(self, section: str) -> dict:
        """Get an entire config section as dict."""
        return self._config.get(section, {})

    def all(self) -> dict:
        """Return full config dict (read-only copy)."""
        return dict(self._config)

    # ── Internal ──────────────────────────────────────────────

    def _load(self) -> None:
        """Load config from JSON file."""
        try:
            if not self._path.exists():
                logger.warning("config_file_not_found", path=str(self._path))
                self._config = {}
                return

            with open(self._path, "r", encoding="utf-8") as f:
                self._config = json.load(f)

            self._last_modified = self._path.stat().st_mtime
            logger.info("config_loaded", path=str(self._path))
        except json.JSONDecodeError as e:
            logger.error("config_parse_error", path=str(self._path), error=str(e))
        except Exception as e:
            logger.error("config_load_error", path=str(self._path), error=str(e))

    def _save(self) -> None:
        """Persist current config to JSON file."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            self._last_modified = self._path.stat().st_mtime
        except Exception as e:
            logger.error("config_save_error", path=str(self._path), error=str(e))

    def _start_file_watcher(self) -> None:
        """Start a background thread polling for config file changes."""
        self._watch_thread = threading.Thread(
            target=self._watch_loop,
            name="ConfigWatcher",
            daemon=True,
        )
        self._watch_thread.start()

    def _watch_loop(self) -> None:
        """Poll config file for modification every 5 seconds."""
        while not self._stop_event.is_set():
            try:
                if self._path.exists():
                    mtime = self._path.stat().st_mtime
                    if mtime > self._last_modified:
                        logger.info("config_file_changed", path=str(self._path))
                        self._load()
            except Exception as e:
                logger.error("config_watch_error", error=str(e))
            self._stop_event.wait(timeout=5.0)

    def stop(self) -> None:
        """Stop the file watcher thread."""
        self._stop_event.set()
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=2.0)

    def __repr__(self) -> str:
        return f"ConfigManager(path={self._path}, keys={len(self._config)})"
