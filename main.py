"""
main.py — Entry point for Polymarket Mispricing Detection Bot.

Usage:
  python main.py --mode dry-run              # Paper trading (default)
  python main.py --mode live --confirm-live   # Live trading (triple-gated)
  python main.py --config show               # Show current config
  python main.py --config set KEY VALUE      # Hot-update config
  python main.py --rollback-model            # Rollback to previous model
"""

from __future__ import annotations

import asyncio
import html
import os
import signal
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

import click
import logging
try:
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    structlog = None
from dotenv import load_dotenv
try:
    from rich.live import Live  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    Live = None

# Load .env before anything else
load_dotenv()

# Configure logging (structlog if available, stdlib otherwise).
if structlog:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
            if os.getenv("ENVIRONMENT", "development") == "development"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(
                __import__("logging"),
                os.getenv("LOG_LEVEL", "INFO").upper(),
                20,
            )
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logger = structlog.get_logger(__name__)
else:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    )
    logger = logging.getLogger(__name__)


from src.binance_feed import BinanceFeed
from src.clob_feed import CLOBFeed
from src.config_manager import ConfigManager

from src.dry_run import DryRunEngine
from src.dual_feed import DualFeed
from src.execution import ExecutionClient
from src.exporter import Exporter
from src.feature_engine import FeatureEngine
from src.market_discovery import MarketDiscovery
from src.model import ModelEnsemble
from src.fair_probability import FairProbabilityEngine
from src.risk_manager import RiskManager
from src.signal_generator import SignalGenerator
from src.spread_filter import SpreadFilter
from src.telegram_notifier import TelegramNotifier
from src.database import DatabaseManager
from sqlalchemy import text

SLUG_PREFIX = os.getenv("POLYMARKET_SLUG_PREFIX", "btc-updown-5m")
IS_ULTRASHORT = "5m" in SLUG_PREFIX


class TradingBot:
    """
    Main orchestrator — wires all modules together.

    Lifecycle:
      1. Initialize all components
      2. Bootstrap historical data
      3. Start WebSocket feeds + market discovery
      4. On each bar close: features → model → signal → risk → trade
      5. On market resolution: settle trades, discover next market
    """

    def __init__(self, mode: str = "dry-run", confirm_live: bool = False) -> None:
        self._requested_mode = mode
        # Effective mode: always start with dry-run simulation when user requests live,
        # then enable live only after the go-live gate passes.
        self._mode = "dry-run" if mode == "live" else mode
        self._confirm_live = confirm_live
        self._running = False
        self._live_enabled = False
        self._go_live_pass_streak = 0
        self._stopping = False
        self._run_started_at = datetime.now(timezone.utc)
        self._stop_reason: str = "UNKNOWN"

        # Initialize components
        self._config = ConfigManager.get_instance()

        self._binance = BinanceFeed(self._config)
        self._dual_feed = DualFeed(self._config, self._binance)
        self._discovery = MarketDiscovery(self._config, self._dual_feed)
        self._clob = CLOBFeed(self._config)
        self._feature_engine = FeatureEngine(self._config)
        self._model = ModelEnsemble(self._config)
        self._signal_gen = SignalGenerator(self._config)
        self._risk_mgr = RiskManager(self._config)
        self._execution = ExecutionClient(self._config)
        self._fair_prob_engine = FairProbabilityEngine(self._config)
        self._spread_filter = SpreadFilter(self._config)
        self._exporter: Exporter | None = None
        self._telegram = TelegramNotifier(self._config)
        self._db = DatabaseManager()

        # Dry run / live engine
        initial_capital = 50.0 if self._requested_mode == "dry-run" else 50.0
        self._dry_run = DryRunEngine(self._config, self._db, initial_capital=initial_capital)
        self._exporter = Exporter(self._dry_run.session_id)

        # Dashboard state
        self._latest_signal = None
        self._latest_metrics = None
        self._telegram_heartbeat_minutes = float(
            self._config.get("telegram.heartbeat_minutes", 15.0)
        )
        self._post_mortem_tracker = {}
        self._active_bets: dict[str, object] = {}  # market_id → active SignalResult
        
        from collections import deque
        self._odds_history: dict[str, deque] = {}   # per market_id
        self._binance_price_history: deque = deque()  # global

    def _get_value_n_seconds_ago(
        self,
        history: 'deque',
        seconds: int,
        tolerance_seconds: int = 30
    ) -> float | None:
        """
        Ambil nilai dari history yang paling mendekati N detik lalu.
        Return None jika tidak ada entri dalam toleransi waktu.
        """
        if not history:
            return None
        from datetime import timedelta
        target = datetime.now(timezone.utc) - timedelta(seconds=seconds)
        closest = min(history, key=lambda x: abs((x[0] - target).total_seconds()))
        if abs((closest[0] - target).total_seconds()) > tolerance_seconds:
            return None
        return closest[1]

    async def _send_telegram(self, title: str, message: str) -> None:
        """Telegram send helper (never raises)."""
        try:
            await self._telegram.send_message(title=title, message=message)
        except Exception:
            return

    @staticmethod
    def _tg_kv(data: dict) -> str:
        lines = []
        for k, v in data.items():
            key = html.escape(str(k))
            val = html.escape(str(v))
            lines.append(f"<b>{key}</b>: {val}")
        return "\n".join(lines)

    async def _telegram_heartbeat_loop(self) -> None:
        """Periodic market/watch heartbeat for operational visibility."""
        interval_s = max(60.0, self._telegram_heartbeat_minutes * 60.0)
        while self._running:
            try:
                market = self._discovery.active_market
                btc_now = self._binance.latest_price
                latest_signal = self._latest_signal

                # Dual feed health for heartbeat
                dual_snapshot = self._dual_feed.get_snapshot()
                oracle_price = self._dual_feed.get_oracle_price()
                spread_stats = self._dual_feed.get_rolling_spread_stats()

                msg = {
                    "session_id": self._dry_run.session_id,
                    "mode": self._mode,
                    "live_enabled": self._live_enabled,
                    "market_id": getattr(market, "market_id", "N/A"),
                    "strike_price": getattr(market, "strike_price", "N/A"),
                    "ttr_minutes": round(getattr(market, "TTR_minutes", 0.0), 3)
                    if market
                    else "N/A",
                    "btc_binance": btc_now if btc_now is not None else "N/A",
                    "btc_chainlink": round(oracle_price, 2) if oracle_price else "N/A",
                    "spread_pct": round(dual_snapshot.spread_pct, 4) if dual_snapshot else "N/A",
                    "spread_mean_60s": round(spread_stats.get("mean_spread_pct", 0), 4),
                    "rtds_msgs": self._dual_feed.messages_received,
                    "signal": getattr(latest_signal, "signal", "N/A"),
                    "edge_yes": round(getattr(latest_signal, "edge_yes", 0.0), 6)
                    if latest_signal
                    else "N/A",
                    "edge_no": round(getattr(latest_signal, "edge_no", 0.0), 6)
                    if latest_signal
                    else "N/A",
                }
                await self._send_telegram(
                    "HEARTBEAT / MARKET WATCH",
                    self._tg_kv(msg),
                )
            except Exception:
                pass
            await asyncio.sleep(interval_s)

    async def _get_signal_summary(self) -> dict:
        """Fetch signal aggregation metrics from SQLite for Telegram reporting."""
        summary = {
            "Total Signals": 0,
            "BUY_UP": 0,
            "BUY_DOWN": 0,
            "ABSTAIN": 0,
            "SKIP": 0,
            "SKIP (Spread)": 0,
            "SKIP (Oracle)": 0,
            "Win Rate": "N/A",
            "Avg Spread Pct": "N/A",
            "Binance Fallbacks": 0,
        }
        try:
            async with self._db.engine.connect() as conn:
                # Total & Breakdowns
                res = await conn.execute(text("""
                    SELECT signal_type, COUNT(*) 
                    FROM signals 
                    WHERE session_id = :sid 
                    GROUP BY signal_type
                """), {"sid": self._dry_run.session_id})
                total = 0
                for row in res.fetchall():
                    stype = row[0]
                    count = row[1]
                    total += count
                    if stype in summary:
                        summary[stype] = count
                summary["Total Signals"] = total
                
                # SKIP Breakdown
                res = await conn.execute(text("""
                    SELECT abstain_reason, COUNT(*) 
                    FROM signals 
                    WHERE session_id = :sid AND signal_type = 'SKIP'
                    GROUP BY abstain_reason
                """), {"sid": self._dry_run.session_id})
                for row in res.fetchall():
                    reason = str(row[0]).lower()
                    count = row[1]
                    if "spread" in reason:
                        summary["SKIP (Spread)"] += count
                    elif "oracle" in reason or "stale" in reason:
                        summary["SKIP (Oracle)"] += count
                
                # Win Rate
                res = await conn.execute(text("""
                    SELECT 
                        SUM(CASE WHEN signal_correct = 'TRUE' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN signal_correct = 'FALSE' THEN 1 ELSE 0 END) as losses
                    FROM signals 
                    WHERE session_id = :sid AND signal_correct IN ('TRUE', 'FALSE')
                """), {"sid": self._dry_run.session_id})
                wr_row = res.fetchone()
                if wr_row:
                    wins = wr_row[0] or 0
                    losses = wr_row[1] or 0
                    if wins + losses > 0:
                        summary["Win Rate"] = f"{(wins / (wins + losses) * 100):.1f}%"
                        
                # Avg Spread Pct
                res = await conn.execute(text("""
                    SELECT AVG(spread_pct) 
                    FROM signals 
                    WHERE session_id = :sid AND spread_pct IS NOT NULL
                """), {"sid": self._dry_run.session_id})
                avg_sp_row = res.fetchone()
                if avg_sp_row and avg_sp_row[0] is not None:
                    summary["Avg Spread Pct"] = f"{avg_sp_row[0]:.4f}%"
                    
                # Binance Fallbacks
                res = await conn.execute(text("""
                    SELECT COUNT(*) 
                    FROM signals 
                    WHERE session_id = :sid AND settlement_price_source = 'BINANCE_FALLBACK'
                """), {"sid": self._dry_run.session_id})
                bf_row = res.fetchone()
                if bf_row:
                    summary["Binance Fallbacks"] = bf_row[0]
                    
        except Exception as e:
            logger.error("signal_summary_query_failed", error=str(e))
            
        return summary

    async def _telegram_periodic_report_loop(self) -> None:
        """Periodic summary report (default: every 2 hours)."""
        report_hours = float(self._config.get("telegram.report_interval_hours", 2.0))
        interval_s = report_hours * 3600.0
        while self._running:
            await asyncio.sleep(interval_s)
            if not self._running:
                break
                
            try:
                metrics = self._dry_run.compute_session_metrics(self._model.version)
                summary = {
                    "trades_executed": metrics.trades_executed,
                    "win_rate (trades)": f"{metrics.win_rate*100:.1f}%" if metrics.win_rate is not None else "N/A",
                    "pnl_usd": f"${metrics.total_pnl_usd:.2f}" if metrics.total_pnl_usd is not None else "N/A",
                    "capital": f"${metrics.capital_end:.2f}" if metrics.capital_end is not None else "N/A",
                    "duration_hours": f"{metrics.duration_hours:.1f}" if metrics.duration_hours else "N/A",
                }
                
                # Append signal aggregation
                sig_summary = await self._get_signal_summary()
                
                # Combine reports
                combined = {**sig_summary, **{"---": "---"}, **summary}
                sum_text = self._tg_kv(combined)
                
                await self._send_telegram(
                    f"Session Report ({report_hours:.0f}h)", sum_text
                )
                logger.info("telegram_periodic_report_sent_text_only")
                    
            except Exception as e:
                logger.error("periodic_report_loop_error", error=str(e), exc_info=True)

    async def _dry_run_time_guard(self) -> None:
        """Stop after max duration unless live gate has already enabled live."""
        max_hours = float(self._config.get("dry_run.max_duration_hours", 48))
        await asyncio.sleep(max_hours * 3600)

        if not self._running:
            return
        if self._live_enabled:
            # Live gate passed; no longer considered dry-run stage.
            return

        self._stop_reason = "DRY_RUN_TIME_LIMIT_EXCEEDED"
        await self._send_telegram(
            "DRY RUN TIME LIMIT",
            f"Dry-run belum mencapai gate live dalam maksimal {max_hours} jam.\n"
            f"session_id={self._dry_run.session_id}",
        )
        await self.stop()

    async def start(self) -> None:
        """Start all subsystems and enter main loop."""
        logger.info(
            "bot_starting",
            mode=self._mode,
            session_id=self._dry_run.session_id,
        )

        self._running = True

        await self._db.init_db()

        logger.info("market_filter_active", slug_prefix=SLUG_PREFIX, is_ultrashort=IS_ULTRASHORT)

        # Load model
        if not self._model.load_latest():
            logger.warning(
                "no_model_loaded_running_in_data_collection_mode",
                info="ML model not available; trading uses settlement-aligned fair probability.",
            )

        # Bootstrap historical data
        bars_loaded = await self._binance.bootstrap_historical(limit=500)
        logger.info("bootstrap_complete", bars=bars_loaded)

        # System health report on first bot active (Railway start).
        # This should be lightweight and never crash the bot.
        try:
            binance_health = self._binance.health.model_dump()
        except Exception:
            binance_health = {}
        try:
            clob_state = self._clob.clob_state
            clob_health = clob_state.model_dump() if clob_state else None
        except Exception:
            clob_health = None

        await self._send_telegram(
            "SYSTEM HEALTH START",
            self._tg_kv(
                {
                    "status": "ACTIVE",
                    "session_id": self._dry_run.session_id,
                    "requested_mode": self._requested_mode,
                    "effective_mode": self._mode,
                    "binance_connected": bool(self._binance.latest_price),
                    "clob_state_present": clob_health is not None,
                    "heartbeat_minutes": self._telegram_heartbeat_minutes,
                }
            ),
        )

        # Live mode gate (arm live client), but effective trading starts after go-live metrics pass.
        if self._requested_mode == "live":
            if not self._execution.confirm_live(cli_flag=self._confirm_live):
                logger.error("live_mode_not_confirmed_falling_back_to_dry_run")
                self._live_enabled = False
                self._mode = "dry-run"
            else:
                logger.info("live_preflight_ready")

        # Register signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(self._graceful_shutdown())
            )

        # Register bar close and price update callbacks
        self._binance.set_on_bar_close(self._on_bar_close)
        self._binance.set_on_price_update(self._on_binance_price_update)

        # Start concurrent tasks
        tasks = [
            asyncio.create_task(self._binance.start(), name="binance_feed"),
            asyncio.create_task(self._dual_feed.start(), name="dual_feed_rtds"),
            asyncio.create_task(self._discovery.start(), name="market_discovery"),
            asyncio.create_task(self._clob.start(), name="clob_ws_loop"),
            asyncio.create_task(self._run_clob_loop(), name="clob_feed"),
            asyncio.create_task(self._run_dashboard(), name="dashboard"),
            asyncio.create_task(
                self._telegram_heartbeat_loop(), name="telegram_heartbeat"
            ),
            asyncio.create_task(
                self._telegram_periodic_report_loop(), name="telegram_periodic_report"
            ),
            asyncio.create_task(
                self._ultrashort_market_loop(), name="ultrashort_loop"
            ),
        ]

        # Dry-run must finish within max duration (default 48h).
        if self._requested_mode in ("dry-run", "live"):
            asyncio.create_task(self._dry_run_time_guard(), name="dry_run_time_guard")

        # Wait for shutdown
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("bot_shutting_down")
        finally:
            await self.stop()

    async def _graceful_shutdown(self) -> None:
        """Graceful shutdown triggered by signal with timeout guard."""
        if self._stopping:
            return
            
        logger.info("graceful_shutdown_triggered")
        self._stop_reason = "SIGTERM_RECEIVED"
        
        try:
            async with asyncio.timeout(25):  # 25 seconds, 5 seconds buffer for Railway
                await self.stop()
                logger.info("graceful_shutdown_complete")
        except asyncio.TimeoutError:
            logger.warning(
                "graceful_shutdown_timeout",
                note="Shutdown sequence exceeded 25s; Railway may kill process soon."
            )
            # Ensure stop() is called even if it partially timed out, 
            # though stop() itself might be what timed out.
            await self.stop()
        except Exception as e:
            logger.error("graceful_shutdown_error", error=str(e))
            await self.stop()

    async def stop(self) -> None:
        """Graceful shutdown."""
        if self._stopping:
            return
        self._stopping = True
        self._running = False
        await self._binance.stop()
        await self._dual_feed.stop()
        await self._discovery.stop()
        await self._clob.stop()
        
        # Export signal data before closing the DB
        signals_csv = None
        if self._exporter:
            try:
                signals_csv = await self._exporter.export_signals(self._db)
            except Exception as e:
                logger.error("export_signals_failed_in_stop", error=str(e))
                
        # Generate summary stats
        sig_summary = await self._get_signal_summary()
        
        await self._db.close()

        # Export session data
        metrics = self._dry_run.compute_session_metrics(self._model.version)
        if self._exporter:
            self._exporter.export_session(
                trades=self._dry_run._resolved_trades,
                metrics=metrics,
                initial_capital=self._dry_run._initial_capital,
            )

        session_title = "DRY RUN FINISHED" if not self._live_enabled else "SESSION FINISHED"
        session_prefix = "Dry run selesai." if not self._live_enabled else "Sesi selesai."
        
        summary = {
            "trades_executed": metrics.trades_executed,
            "win_rate (trades)": f"{metrics.win_rate*100:.1f}%" if metrics.win_rate is not None else "N/A",
            "pnl_usd": f"${metrics.total_pnl_usd:.2f}" if metrics.total_pnl_usd is not None else "N/A",
            "capital": f"${metrics.capital_end:.2f}" if metrics.capital_end is not None else "N/A",
            "duration_hours": f"{metrics.duration_hours:.1f}" if metrics.duration_hours else "N/A",
        }
        combined = {**sig_summary, **{"---": "---"}, **summary}
        sum_text = self._tg_kv(combined)
        
        if signals_csv and str(signals_csv).endswith(".csv"):
            try:
                await self._telegram.send_document(
                    file_path=str(signals_csv),
                    caption=f"{session_title}\n\n{session_prefix}\n{self._stop_reason}\n\n{sum_text}",
                )
            except Exception as e:
                logger.error("telegram_signals_send_failed", error=str(e))
                await self._send_telegram(
                    session_title,
                    f"{session_prefix}\nreason={self._stop_reason}\n\n{sum_text}",
                )
        else:
            await self._send_telegram(
                session_title,
                f"{session_prefix}\nreason={self._stop_reason}\n\n{sum_text}",
            )

        self._config.stop()
        logger.info("bot_stopped", session=self._dry_run.session_id)

    # ── Core Trading Loop ─────────────────────────────────────

    async def _on_bar_close(self, bar: dict) -> None:
        """
        Called on each 15-minute bar close.
        Full pipeline: features → model → signal → risk → trade.
        """
        from src.schemas import SignalResult
        self._dry_run.increment_bars()

        # Check if we have an active market
        if not self._discovery.is_market_active:
            return

        market = self._discovery.active_market
        await self._discovery.refresh_ttr()

        # ── STEP 1: min_ttr_minutes Gate ─────────────────────
        min_ttr = float(self._config.get("signal.min_ttr_minutes", 1.5))
        if market.TTR_minutes < min_ttr:
            logger.debug(
                "signal_skipped_late_ttr",
                TTR_minutes=round(market.TTR_minutes, 2),
                min_ttr=min_ttr,
            )
            return

        # ── Bar-close rotation check ──────────────────────────
        # Aligned here (not on an independent timer) so market switches never
        # interrupt a Z-score computation mid-window.
        rotated = await self._discovery.check_and_rotate()
        if rotated:
            # Discard stale CLOB cache — next poll will fetch fresh data
            self._clob._cached_state = None
            market = self._discovery.active_market
            logger.info(
                "bar_close_rotation_applied",
                new_market_id=market.market_id if market else None,
            )
            return  # Skip this bar's signal; let next bar compute on new market

        # Check data staleness
        if self._binance.is_stale:
            logger.warning("binance_data_stale_skipping_signal")
            return

        clob_state = self._clob.clob_state
        
        entry_odds_source = "CLOB_LIVE"
        
        # ── Synthetic CLOB Fallback for Ultra-Short Markets ──
        # Market discovery identifies dynamic 5m markets. If they have no book depth
        # (common in first 60s), we inject a tight synthetic 50/50 book.
        if not clob_state or not clob_state.is_liquid:
            is_ultrashort = (IS_ULTRASHORT and SLUG_PREFIX in market.slug) or (market.T_resolution - market.T_open).total_seconds() / 60.0 <= 10.0
            if is_ultrashort:
                from src.schemas import CLOBState
                clob_state = CLOBState(
                    market_id=market.market_id,
                    timestamp=datetime.now(timezone.utc),
                    yes_ask=0.505,  # 1% spread around 0.50
                    yes_bid=0.495,
                    no_ask=0.505,
                    no_bid=0.495,
                    yes_depth_usd=100.0,
                    no_depth_usd=100.0,
                    market_vig=0.01,
                    is_liquid=True,
                    is_stale=False
                )
                entry_odds_source = "DEFAULT_FALLBACK"
                logger.debug("using_synthetic_clob_fallback", market_id=market.market_id)

        if not clob_state:
            logger.warning("no_clob_data_skipping_signal")
            return

        # Record CLOB snapshot
        self._exporter.record_clob_snapshot(clob_state, market.TTR_minutes)

        # ── ORACLE SNAPSHOT (single fetch per pipeline tick) ──
        # Requirement: oracle snapshot is taken ONCE at the start of
        # the pipeline and reused for all downstream computations.
        dual_snapshot = self._dual_feed.get_snapshot()
        oracle_price, oracle_source = self._dual_feed.get_oracle_price_with_source()

        # ── SPREAD FILTER GATE ────────────────────────────────
        # Must run BEFORE feature/fair_prob computation.
        # No silent fallback: if oracle unavailable → SKIP.
        spread_result = self._spread_filter.check(self._dual_feed)

        if spread_result.recommendation == "SKIP":
            skip_signal = SignalResult(
                signal="ABSTAIN",
                abstain_reason="ORACLE_UNAVAILABLE" if oracle_source == "UNAVAILABLE"
                    else "SPREAD_FILTER_SKIP",
                P_model=0.5,
                uncertainty_u=1.0,
                edge_yes=0.0,
                edge_no=0.0,
                clob_yes_bid=clob_state.yes_bid,
                clob_yes_ask=clob_state.yes_ask,
                clob_no_bid=clob_state.no_bid,
                clob_no_ask=clob_state.no_ask,
                TTR_minutes=market.TTR_minutes,
                strike_price=market.strike_price,
                current_price=self._binance.latest_price or 0.0,
                strike_distance=0.0,
                market_id=market.market_id,
                timestamp=datetime.now(timezone.utc),
                spread_pct_at_signal=spread_result.spread_pct,
                spread_filter_passed=False,
                spread_filter_reason=spread_result.reason,
                entry_odds_source=entry_odds_source,
                oracle_source=oracle_source,
            )
            skip_signal.binance_price_at_signal = self._binance.latest_price
            logger.warning(
                "spread_filter_blocked_entry",
                recommendation=spread_result.recommendation,
                reason=spread_result.reason,
                spread_pct=round(spread_result.spread_pct, 4),
                oracle_price=oracle_price,
                chainlink_stale=self._dual_feed.is_chainlink_stale,
            )
            self._latest_signal = skip_signal
            await self._dry_run.record_signal(skip_signal, slug=market.slug)
            return

        if spread_result.recommendation == "WAIT":
            wait_signal = SignalResult(
                signal="ABSTAIN",
                abstain_reason="SPREAD_FILTER_WAIT",
                P_model=0.5,
                uncertainty_u=1.0,
                edge_yes=0.0,
                edge_no=0.0,
                clob_yes_bid=clob_state.yes_bid,
                clob_yes_ask=clob_state.yes_ask,
                clob_no_bid=clob_state.no_bid,
                clob_no_ask=clob_state.no_ask,
                TTR_minutes=market.TTR_minutes,
                strike_price=market.strike_price,
                current_price=self._binance.latest_price or 0.0,
                strike_distance=0.0,
                market_id=market.market_id,
                timestamp=datetime.now(timezone.utc),
                spread_pct_at_signal=spread_result.spread_pct,
                spread_filter_passed=False,
                spread_filter_reason=spread_result.reason,
                entry_odds_source=entry_odds_source,
            )
            wait_signal.binance_price_at_signal = self._binance.latest_price
            logger.info(
                "spread_filter_wait",
                spread_pct=round(spread_result.spread_pct, 4),
                reason=spread_result.reason,
            )
            self._latest_signal = wait_signal
            await self._dry_run.record_signal(wait_signal, slug=market.slug)
            return

        # ── Feature Computation ───────────────────────────────
        # oracle_price is guaranteed non-None here (spread filter PROCEED)
        try:
            fv = self._feature_engine.compute(
                self._binance, market, clob_state, oracle_price=oracle_price
            )
        except ValueError as e:
            # oracle_price was None/invalid despite spread filter passing
            logger.warning(
                "feature_compute_oracle_unavailable",
                error=str(e),
                oracle_price=oracle_price,
            )
            return
        if fv is None:
            return

        # ── Dry-run debug log for F16/F17/F20 post-fix monitoring ─
        if self._mode == "dry-run":
            f16 = fv.values[15] if len(fv.values) > 15 else None  # strike_distance_pct
            f17 = fv.values[16] if len(fv.values) > 16 else None  # contest_urgency
            f20 = fv.values[19] if len(fv.values) > 19 else None  # ttr_x_strike
            logger.debug(
                "oracle_feature_debug",
                oracle_price=round(oracle_price, 2),
                binance_price=round(self._binance.latest_price or 0, 2),
                oracle_vs_binance_usd=round(
                    abs(oracle_price - (self._binance.latest_price or oracle_price)), 2
                ),
                F16_strike_distance_pct=round(f16, 4) if f16 is not None else None,
                F17_contest_urgency=round(f17, 4) if f17 is not None else None,
                F20_ttr_x_strike=round(f20, 4) if f20 is not None else None,
                strike_price=market.strike_price,
                spread_pct=round(spread_result.spread_pct, 4),
            )

        # ── Fair Probability Computation ──────────────────────
        try:
            fair = self._fair_prob_engine.compute(
                binance_feed=self._binance,
                active_market=market,
                clob_state=clob_state,
                oracle_price=oracle_price,
            )
        except ValueError as e:
            logger.warning(
                "fair_prob_oracle_unavailable",
                error=str(e),
                oracle_price=oracle_price,
            )
            return
        q_fair = fair.q_fair
        uncertainty_u = fair.uncertainty_u

        # ── Probability Source Selection (explicit policy) ────
        prob_source = str(
            self._config.get("signal.probability_source", "fair")
        ).lower()
        p_model = q_fair
        if prob_source in ("ensemble", "hybrid"):
            model_prob = self._model.predict(np.array(fv.values))
            if prob_source == "ensemble":
                p_model = model_prob
            else:
                # Hybrid: fair value remains anchor, model contributes directional prior.
                fair_w = float(self._config.get("signal.hybrid_fair_weight", 0.7))
                model_w = float(self._config.get("signal.hybrid_model_weight", 0.3))
                total_w = max(1e-8, fair_w + model_w)
                p_model = ((fair_w * q_fair) + (model_w * model_prob)) / total_w
                p_model = max(0.0, min(1.0, p_model))
            logger.info(
                "probability_source_applied",
                source=prob_source,
                q_fair=round(q_fair, 4),
                p_model=round(p_model, 4),
                uncertainty_u=round(uncertainty_u, 4),
            )

        # ── ML Features Collection ────────────────────────────
        ml_features = {}
        if fv:
            ml_features = dict(zip(fv.feature_names, fv.values))
            
        m_id = market.market_id
        odds_hist = self._odds_history.get(m_id)
        odds_60s_ago = self._get_value_n_seconds_ago(odds_hist, 60, tolerance_seconds=30)
        ml_features["odds_yes_60s_ago"] = odds_60s_ago
        if odds_60s_ago is not None and clob_state:
            ml_features["odds_delta_60s"] = clob_state.yes_ask - odds_60s_ago
        else:
            ml_features["odds_delta_60s"] = None

        btc_60s_ago = self._get_value_n_seconds_ago(self._binance_price_history, 60, tolerance_seconds=30)
        current_btc = self._binance.latest_price
        if btc_60s_ago is not None and btc_60s_ago > 0 and current_btc is not None:
            ml_features["btc_return_1m"] = (current_btc - btc_60s_ago) / btc_60s_ago
        else:
            ml_features["btc_return_1m"] = None

        # Determine confidence bucket based on ML probability
        # Determine quantitative confidence bucket for ML features
        confidence_bucket = None
        if p_model is not None:
            if p_model < 0.30: confidence_bucket = '0-30'
            elif p_model < 0.50: confidence_bucket = '30-50'
            elif p_model < 0.70: confidence_bucket = '50-70'
            else: confidence_bucket = '70-100'
        ml_features["confidence_bucket"] = confidence_bucket

        # ── Signal Generation ─────────────────────────────────
        signal = self._signal_gen.evaluate(
            p_model, uncertainty_u, clob_state, market, fv
        )
        
        signal.entry_odds_source = entry_odds_source

        # ── Enrich signal with dual-feed tracking fields ──────
        binance_price_now = self._binance.latest_price
        signal.binance_price_at_signal = binance_price_now
        signal.chainlink_price_at_signal = oracle_price
        signal.spread_pct_at_signal = spread_result.spread_pct
        signal.oracle_vs_binance_at_entry = round(
            abs(oracle_price - (binance_price_now or oracle_price)), 2
        )
        signal.spread_filter_passed = spread_result.passed
        signal.spread_filter_reason = spread_result.reason
        signal.strike_price_source = "GAMMA"  # strike always from Gamma API
        signal.odds_source = "CLOB"  # odds always from CLOB order book
        signal.oracle_source = oracle_source

        if entry_odds_source == "DEFAULT_FALLBACK":
            blocked = SignalResult(
                signal="ABSTAIN",
                abstain_reason="CONTAMINATED_FALLBACK_ODDS",
                clob_yes_ask=signal.clob_yes_ask,
                clob_no_ask=signal.clob_no_ask,
                P_model=signal.P_model,
                TTR_minutes=signal.TTR_minutes,
                strike_price=signal.strike_price,
                current_price=signal.current_price,
                strike_distance=signal.strike_distance,
                market_id=signal.market_id,
                timestamp=signal.timestamp,
                entry_odds_source=entry_odds_source,
            )
            blocked.binance_price_at_signal = self._binance.latest_price
            self._latest_signal = blocked
            await self._dry_run.record_signal(blocked, slug=market.slug, ml_features=ml_features)
            logger.info("signal_skipped_contaminated_fallback", market_id=market.market_id)
            return

        self._latest_signal = signal
        await self._dry_run.record_signal(signal, slug=market.slug, ml_features=ml_features)

        # ── ONE-BET-PER-MARKET RULE ───────────────────────────
        # Only one active position/pending signal per market_id.
        m_id = market.market_id
        if signal.signal != "ABSTAIN" and m_id in self._active_bets:
            existing = self._active_bets[m_id]
            # Keep the signal with higher P_model confidence
            existing_conf = existing.P_model if hasattr(existing, 'P_model') else 0.0
            new_conf = signal.P_model
            if new_conf <= existing_conf:
                logger.info(
                    "one_bet_rule_blocked",
                    market_id=m_id,
                    blocked_signal=signal.signal,
                    blocked_p_model=round(new_conf, 4),
                    active_signal=existing.signal if hasattr(existing, 'signal') else 'UNKNOWN',
                    active_p_model=round(existing_conf, 4),
                    reason="one_bet_per_market",
                )
                # Record blocked signal to SQLite for analysis
                blocked = signal.model_copy(update={
                    "signal": "ABSTAIN",
                    "abstain_reason": "BLOCKED_ONE_BET",
                    "entry_odds_source": entry_odds_source,
                })
                await self._dry_run.record_signal(blocked, slug=market.slug, ml_features=ml_features)
                return
            else:
                # New signal is stronger — replace active bet
                logger.info(
                    "one_bet_rule_replaced",
                    market_id=m_id,
                    new_signal=signal.signal,
                    new_p_model=round(new_conf, 4),
                    replaced_signal=existing.signal if hasattr(existing, 'signal') else 'UNKNOWN',
                    replaced_p_model=round(existing_conf, 4),
                )

        # ── Post-Mortem Aggregator ────────────────────────────
        if signal.signal == "ABSTAIN":
            m_id = market.market_id
            if m_id not in self._post_mortem_tracker:
                self._post_mortem_tracker[m_id] = {
                    "evals": 0,
                    "reasons": Counter(),
                    "max_edge": 0.0
                }
                asyncio.create_task(
                    self._schedule_post_mortem(market),
                    name=f"pm_{m_id[:8]}"
                )
            
            stats = self._post_mortem_tracker[m_id]
            stats["evals"] += 1
            if signal.abstain_reason:
                stats["reasons"][signal.abstain_reason] += 1
            
            current_max_edge = max(signal.edge_yes, signal.edge_no)
            if current_max_edge > stats["max_edge"]:
                stats["max_edge"] = current_max_edge
            return

        # ── STEP 2: Live Edge Verification (Execution Gates) ──────────
        fresh_clob = await self._clob.fetch_clob_snapshot(market)
        if not fresh_clob:
            logger.warning("live_verification_failed_clob_unavailable")
            return

        real_best_ask = fresh_clob.yes_ask if signal.signal == "BUY_UP" else fresh_clob.no_ask
        synthetic_edge = signal.edge_yes if signal.signal == "BUY_UP" else signal.edge_no
        p_outcome = signal.P_model if signal.signal == "BUY_UP" else (1.0 - signal.P_model)
        live_edge = p_outcome - signal.uncertainty_u - real_best_ask
        edge_deviation = abs(synthetic_edge - live_edge)

        max_buy_price = float(self._config.get("risk.max_buy_price", 0.75))
        edge_tolerance = float(self._config.get("risk.live_edge_tolerance", 0.05))
        margin_of_safety = float(self._config.get("signal.margin_of_safety", 0.02))

        # Gate 1: Hard Cap
        if real_best_ask > max_buy_price:
            logger.info("trade_aborted", reason="PRICE_EXCEEDS_MAX_CAP", price=real_best_ask, cap=max_buy_price)
            return

        # Gate 2: Tolerance
        if edge_deviation > edge_tolerance:
            logger.info("trade_aborted", reason="EDGE_DEVIATION_TOO_HIGH", synthetic=round(synthetic_edge, 4), live=round(live_edge, 4), dev=round(edge_deviation, 4))
            return

        # Gate 3: Live Edge Positive
        if live_edge <= margin_of_safety:
            logger.info("trade_aborted", reason="LIVE_EDGE_NEGATIVE", live_edge=round(live_edge, 4), threshold=margin_of_safety)
            return

        # Update signal with live verification data for risk and logging
        signal.live_yes_ask = fresh_clob.yes_ask
        signal.live_no_ask = fresh_clob.no_ask
        signal.synthetic_edge = synthetic_edge
        signal.live_edge = live_edge
        
        # Override CLOB prices in signal to ensure Kelly uses real_best_ask
        if signal.signal == "BUY_UP":
            signal.clob_yes_ask = real_best_ask
        else:
            signal.clob_no_ask = real_best_ask

        # ── Risk Management ───────────────────────────────────
        result = await self._risk_mgr.approve(signal, self._dry_run.capital)

        from src.schemas import ApprovedBet, RejectedBet

        if isinstance(result, RejectedBet):
            logger.info("trade_rejected", reason=result.reason)
            return

        approved = result
        assert isinstance(approved, ApprovedBet)

        # ── Execute Trade (Dry Run) ───────────────────────────
        if self._mode == "dry-run":
            trade = self._dry_run.simulate_trade(signal, approved, market)
            self._discovery.mark_trade_executed()
            self._active_bets[market.market_id] = signal  # Register active bet

            # Telegram: trade opened (paper order).
            asyncio.create_task(
                self._send_telegram(
                    "ORDER EXECUTION (DRY-RUN)",
                    self._tg_kv(
                        {
                            "session_id": self._dry_run.session_id,
                            "trade_id": trade.trade_id,
                            "market_id": trade.market_id,
                            "signal": trade.signal_type,
                            "entry_price": trade.entry_price,
                            "bet_size": trade.bet_size,
                            "strike_price": trade.strike_price,
                            "btc_now": self._binance.latest_price,
                            "ttr_minutes": trade.TTR_at_entry,
                        }
                    ),
                ),
                name=f"tg_open_{trade.trade_id[:8]}",
            )

            # Schedule resolution
            asyncio.create_task(
                self._schedule_resolution(trade, market),
                name=f"resolve_{trade.trade_id[:8]}",
            )
        else:
            # Live mode execution
            fill_result = await self._execution.place_order(approved, market)
            logger.info(
                "live_order_result",
                status=fill_result.status if hasattr(fill_result, "status") else "rejected",
            )

            # If nothing filled, release risk-manager position slot.
            status = getattr(fill_result, "status", "").upper()
            filled_size = getattr(fill_result, "filled_size", None)
            effective_bet_size = None
            if status in ("FILLED", "PARTIALLY_FILLED"):
                if filled_size is not None and float(filled_size) > 0:
                    effective_bet_size = float(filled_size)
                else:
                    # Fallback: treat as fully using the risk-approved size.
                    effective_bet_size = float(approved.bet_size)
            fill_price = getattr(fill_result, "fill_price", None)
            if fill_price is None:
                fill_price = (
                    signal.clob_yes_ask
                    if signal.signal == "BUY_UP"
                    else signal.clob_no_ask
                )

            if effective_bet_size is None:
                await self._risk_mgr.on_trade_resolved(0.0)
            else:
                trade = self._dry_run.simulate_trade(
                    signal,
                    approved,
                    market,
                    entry_price_override=float(fill_price),
                    bet_size_override=float(effective_bet_size),
                )
                self._discovery.mark_trade_executed()

                # Telegram: trade opened (shadow paper record for live).
                asyncio.create_task(
                    self._send_telegram(
                        "ORDER EXECUTION (SHADOW LIVE)",
                        self._tg_kv(
                            {
                                "session_id": self._dry_run.session_id,
                                "trade_id": trade.trade_id,
                                "market_id": trade.market_id,
                                "signal": trade.signal_type,
                                "entry_price": trade.entry_price,
                                "bet_size": trade.bet_size,
                                "strike_price": trade.strike_price,
                                "btc_now": self._binance.latest_price,
                                "ttr_minutes": trade.TTR_at_entry,
                                "fill_status": status,
                                "fill_price": fill_price,
                                "filled_size": filled_size,
                            }
                        ),
                    ),
                    name=f"tg_open_live_{trade.trade_id[:8]}",
                )
                asyncio.create_task(
                    self._schedule_resolution(trade, market),
                    name=f"resolve_live_{trade.trade_id[:8]}",
                )

        # Check abort conditions
        abort = self._dry_run.check_abort_conditions(mode=self._mode)
        if abort:
            logger.critical("session_abort", reason=abort)
            self._stop_reason = abort
            asyncio.create_task(
                self._send_telegram(
                    "SESSION ABORTED",
                    "Dry-run session abort triggered.\n"
                    f"reason={abort}\n"
                    f"session_id={self._dry_run.session_id}",
                ),
                name="tg_abort",
            )
            asyncio.create_task(self.stop(), name="stop_after_abort")

        # Update metrics
        self._latest_metrics = self._dry_run.compute_session_metrics(
            self._model.version
        )

    def _on_binance_price_update(self, price: float) -> None:
        """Handler for real-time Binance price ticks."""
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        self._binance_price_history.append((now, price))
        
        # Temporary debug log per user request
        if len(self._binance_price_history) % 100 == 0:  # Log every 100 ticks to avoid noise
            logger.debug("binance_price_history_append", 
                         buffer_size=len(self._binance_price_history), 
                         price=price, 
                         ts=now)
        
        # Prune > 90s
        cutoff = now - timedelta(seconds=90)
        while self._binance_price_history and self._binance_price_history[0][0] < cutoff:
            self._binance_price_history.popleft()

    async def _schedule_resolution(self, trade, market) -> None:
        """Wait for market resolution and settle trade."""
        now = datetime.now(timezone.utc)
        wait_seconds = (market.T_resolution - now).total_seconds()

        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds + 5)  # +5s buffer for price settlement

        # Get BTC price at resolution
        price = await self._binance.get_1m_settlement_price(
            resolution_time=market.T_resolution,
            price_type=(market.settlement_price_type or "close"),
        )
        if price is None:
            # Final fallback: use last observed price (less correct than candle-aligned resolution).
            price = self._binance.latest_price

        resolved = await self._dry_run.resolve_trade(trade, price)
        await self._risk_mgr.on_trade_resolved(resolved.pnl_usd or 0)

        # Clear one-bet-per-market lock
        self._active_bets.pop(trade.market_id, None)

        # Telegram: trade resolved (PnL final for this paper/live record).
        asyncio.create_task(
            self._send_telegram(
                "ORDER RESULT",
                self._tg_kv(
                    {
                        "session_id": self._dry_run.session_id,
                        "trade_id": resolved.trade_id,
                        "market_id": resolved.market_id,
                        "signal": resolved.signal_type,
                        "outcome": resolved.outcome,
                        "entry_price": resolved.entry_price,
                        "btc_at_resolution": resolved.btc_at_resolution,
                        "pnl_usd": resolved.pnl_usd,
                        "capital_after": resolved.capital_after,
                    }
                ),
            ),
            name=f"tg_resolve_{resolved.trade_id[:8]}",
        )
        await self._maybe_enable_live()

    async def _schedule_post_mortem(self, market) -> None:
        """Independent watcher to log abstention stats when a market resolves."""
        now = datetime.now(timezone.utc)
        wait_seconds = (market.T_resolution - now).total_seconds()
        
        # Wait until the market officially resolves + 5 seconds buffer
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds + 5)
            
        m_id = market.market_id
        strike_price = market.strike_price
        
        # ── SETTLEMENT PRICE (Use Chainlink from dual_feed) ──
        snapshot = self._dual_feed.get_snapshot()
        
        if snapshot and not snapshot.is_stale:
            settlement_price = snapshot.chainlink_price
            price_source = "CHAINLINK"
        else:
            settlement_price = self._binance.latest_price
            price_source = "BINANCE_FALLBACK"
            logger.warning("settlement_price_fallback", 
                           market_id=m_id, 
                           reason="chainlink_stale_or_unavailable")
        
        if settlement_price is not None:
            actual_outcome = "BUY_UP" if settlement_price >= strike_price else "BUY_DOWN"
            
            try:
                async with self._db.engine.begin() as conn:
                    await conn.execute(text("""
                        UPDATE signals 
                        SET actual_outcome = :actual_outcome,
                            signal_correct = CASE 
                                WHEN signal_type = :actual_outcome THEN 'TRUE'
                                WHEN signal_type IN ('SKIP', 'ABSTAIN') THEN 'N/A'
                                ELSE 'FALSE'
                            END,
                            theoretical_pnl = CASE
                                WHEN signal_type IN ('BUY_UP', 'BUY_DOWN') THEN
                                    CASE 
                                        WHEN signal_type = :actual_outcome THEN ROUND((1.0 / NULLIF(entry_odds, 0)) - 1.0, 4)
                                        ELSE -1.0
                                    END
                                ELSE 0.0
                            END,
                            settlement_price = :settlement_price,
                            settlement_price_source = :price_source
                        WHERE market_id = :market_id AND actual_outcome = 'PENDING'
                    """), {
                        "actual_outcome": actual_outcome,
                        "settlement_price": float(settlement_price),
                        "price_source": price_source,
                        "market_id": m_id
                    })
            except Exception as e:
                logger.error("post_mortem_db_update_failed", error=str(e), market_id=m_id)

        if m_id in self._post_mortem_tracker:
            data = self._post_mortem_tracker.pop(m_id) # Safe extract & delete
            
            top_blockers = ", ".join([f"{k}({v}x)" for k, v in data["reasons"].most_common(3)])
            
            logger.info(
                "epoch_post_mortem",
                market_id=m_id,
                total_evaluations=data["evals"],
                max_edge_seen=round(data["max_edge"], 4),
                top_blockers=top_blockers
            )

    async def _maybe_enable_live(self) -> None:
        """Enable actual live trading after dry-run performance gates."""
        if self._requested_mode != "live":
            return
        if self._live_enabled:
            return
        if self._mode != "dry-run":
            return

        min_total_trades = int(
            self._config.get("dry_run.go_live_min_total_trades", 100)
        )
        consec_pass = int(self._config.get("dry_run.go_live_consecutive_pass", 5))
        metrics = self._dry_run.compute_session_metrics(self._model.version)

        if metrics.trades_executed >= min_total_trades and metrics.pass_fail == "PASS":
            self._go_live_pass_streak += 1
        else:
            self._go_live_pass_streak = 0

        if self._go_live_pass_streak >= consec_pass:
            self._mode = "live"
            self._live_enabled = True
            logger.critical(
                "go_live_enabled",
                trades_executed=metrics.trades_executed,
                dry_run_score=metrics.dry_run_score,
                win_rate=metrics.win_rate,
                pass_fail=metrics.pass_fail,
            )

            # Telegram: go-live enabled after gate.
            asyncio.create_task(
                self._send_telegram(
                    "GO LIVE ENABLED",
                    "Go-live enabled after dry-run gate.\n"
                    f"session_id={self._dry_run.session_id}\n"
                    f"trades_executed={metrics.trades_executed}\n"
                    f"win_rate={metrics.win_rate}\n"
                    f"total_pnl_usd={metrics.total_pnl_usd}\n"
                    f"dry_run_score={metrics.dry_run_score}\n"
                    f"pass_fail={metrics.pass_fail}",
                ),
                name="tg_go_live_enabled",
            )

    # ── Ultra-Short Market Evaluation Loop ─────────────────────

    async def _ultrashort_market_loop(self) -> None:
        """30-second evaluation loop for markets ≤ 10 minutes."""
        while self._running:
            await asyncio.sleep(10)

            if not self._discovery.is_market_active:
                continue

            market = self._discovery.active_market
            if market is None:
                continue

            is_ultrashort = (IS_ULTRASHORT and SLUG_PREFIX in market.slug) or (market.T_resolution - market.T_open).total_seconds() <= 600
            if not is_ultrashort:
                continue

            btc_price = self._binance.latest_price
            if btc_price is None:
                logger.warning("ultrashort_loop_skipped", reason="btc_price_is_none")
                continue

            # Check dual feed availability for ultrashort loop
            oracle_price_us = self._dual_feed.get_oracle_price()
            if oracle_price_us is None:
                logger.warning(
                    "ultrashort_loop_skipped",
                    reason="oracle_unavailable",
                    chainlink_stale=self._dual_feed.is_chainlink_stale,
                )
                continue

            logger.info(
                "ultrashort_loop_triggering_evaluation",
                btc_binance=btc_price,
                btc_chainlink=round(oracle_price_us, 2),
                market_id=market.market_id,
            )

            synthetic_bar = {
                "close": btc_price,
                "is_synthetic": True,
            }
            try:
                await self._on_bar_close(synthetic_bar)
            except Exception as e:
                logger.error("ultrashort_loop_error", error=str(e))

    # ── CLOB Polling Loop ─────────────────────────────────────

    async def _run_clob_loop(self) -> None:
        """
        Poll CLOB data when market is active.

        Circuit breaker: if CLOBFeed accumulates max_consecutive_404 errors,
        the market has almost certainly expired. We call force_rediscover() to
        immediately restart the discovery state machine, then reset the breaker
        so it is ready for the next market cycle.
        """
        while self._running:
            if self._discovery.is_market_active:
                market = self._discovery.active_market
                try:
                    state = await self._clob.fetch_clob_snapshot(market)
                    if state:
                        self._clob._cached_state = state
                        self._clob._last_fetch_time = __import__("time").time()
                        
                        from datetime import datetime, timezone, timedelta
                        now = datetime.now(timezone.utc)
                        m_id = market.market_id
                        if m_id not in self._odds_history:
                            self._odds_history[m_id] = __import__("collections").deque()
                        self._odds_history[m_id].append((now, state.yes_ask))
                        
                        cutoff = now - timedelta(seconds=90)
                        while self._odds_history[m_id] and self._odds_history[m_id][0][0] < cutoff:
                            self._odds_history[m_id].popleft()
                except Exception as e:
                    logger.error("clob_loop_error", error=str(e))

                # ── Circuit breaker check ─────────────────────
                if self._clob.circuit_breaker_tripped:
                    logger.warning(
                        "clob_circuit_breaker_triggering_rediscover",
                        market_id=market.market_id if market else None,
                    )
                    self._discovery.force_rediscover()
                    self._clob.reset_circuit_breaker()

            poll_interval = self._config.get("clob.poll_interval_seconds", 5)
            await asyncio.sleep(poll_interval)

    # ── Dashboard ─────────────────────────────────────────────

    async def _run_dashboard(self) -> None:
        """Update Rich dashboard every 5 seconds."""
        if Live is None:
            logger.info("dashboard_disabled_rich_missing")
            return

        try:
            from src.cli import build_dashboard  # rich-dependent

            console = __import__("rich.console", fromlist=["Console"]).Console()
            with Live(
                build_dashboard(),
                refresh_per_second=0.2,
                console=console,
            ) as live:
                while self._running:
                    dashboard = build_dashboard(
                        market=self._discovery.active_market,
                        clob=self._clob.clob_state,
                        signal=self._latest_signal,
                        metrics=self._latest_metrics,
                        ws_health=self._binance.health,
                        current_price=self._binance.latest_price,
                        mode="DRY RUN" if self._mode == "dry-run" else "LIVE",
                        session_id=self._dry_run.session_id,
                    )
                    live.update(dashboard)
                    await asyncio.sleep(5)
        except Exception as e:
            # Dashboard failure should not crash the bot
            logger.warning("dashboard_error", error=str(e))


# ============================================================
# CLI Entry Point
# ============================================================


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["dry-run", "live"]),
    default="dry-run",
    help="Trading mode",
)
@click.option(
    "--confirm-live",
    is_flag=True,
    default=False,
    help="Confirm live trading (required with --mode live)",
)
@click.option(
    "--config",
    "config_cmd",
    type=click.Choice(["show", "set", "get"]),
    default=None,
    help="Config management command",
)
@click.option("--key", default=None, help="Config key (for set/get)")
@click.option("--value", default=None, help="Config value (for set)")
@click.option(
    "--rollback-model",
    is_flag=True,
    default=False,
    help="Rollback to previous model version",
)
def main(
    mode: str,
    confirm_live: bool,
    config_cmd: str | None,
    key: str | None,
    value: str | None,
    rollback_model: bool,
) -> None:
    """Polymarket Bitcoin Up/Down — Probability Mispricing Detection Bot."""

    # Config commands (non-trading)
    if config_cmd:
        cfg = ConfigManager.get_instance()
        if config_cmd == "show":
            import json
            click.echo(json.dumps(cfg.all(), indent=2))
        elif config_cmd == "get" and key:
            click.echo(f"{key} = {cfg.get(key)}")
        elif config_cmd == "set" and key and value:
            # Auto-convert types
            try:
                typed_value = float(value)
            except ValueError:
                typed_value = value
            cfg.set(key, typed_value)
            click.echo(f"Set {key} = {typed_value}")
        cfg.stop()
        return

    # Model rollback
    if rollback_model:
        cfg = ConfigManager.get_instance()
        model = ModelEnsemble(cfg)
        if model.rollback():
            click.echo("✓ Model rolled back successfully")
        else:
            click.echo("✗ Rollback failed — no previous version available")
        cfg.stop()
        return

    # Trading mode
    click.echo(f"\n🚀 Starting Polymarket Bot — Mode: {mode.upper()}\n")

    bot = TradingBot(mode=mode, confirm_live=confirm_live)

    # Graceful shutdown handler
    def handle_shutdown(sig, frame):
        click.echo("\n\n⏹  Shutting down gracefully...")
        asyncio.get_event_loop().call_soon_threadsafe(
            lambda: asyncio.create_task(bot.stop())
        )

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Run
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        click.echo("\nBot stopped.")


if __name__ == "__main__":
    main()