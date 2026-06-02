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
import json
import os
import signal
import sys
import traceback
import tracemalloc
from collections import Counter
from datetime import datetime, timedelta, timezone
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

# TODO(Sprint 6): Remove DryRunEngine dependency — V5 has native database
from src.dry_run import DryRunEngine  # [ISOLATED - ALPHA V1 ARTIFACT]
from src.dual_feed import DualFeed
from src.execution import ExecutionClient
from src.exporter import Exporter
from src.feature_engine import FeatureEngine
from src.market_discovery import MarketDiscovery
# [FIX-03] Alpha V1 XGBoostGate removed — Slingger V5 is sole trading engine
from model_training.dual_inference import SlingshotHunterV5
from src.telegram_notifier import SlingshotAlerts
from src.fair_probability import FairProbabilityEngine
from src.risk_manager import RiskManager
from src.signal_generator import SignalGenerator
from src.spread_filter import SpreadFilter
from src.telegram_notifier import TelegramNotifier
from src.database import DatabaseManager
from src.v5_database import V5DatabaseManager
from src.vatic_feed import VaticFeed
from src.utils import compute_position_size, fmt_money
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
        # [FIX-04] Clean mode assignment — V1 hard lock removed
        self._mode = mode
        self._confirm_live = confirm_live

        self._running = False
        self._live_enabled = False
        self._go_live_pass_streak = 0
        self._stopping = False
        self._run_started_at = datetime.now(timezone.utc)
        self._stop_reason: str = "UNKNOWN"
        self._last_mem_snapshot = None  # [HOTFIX] tracemalloc delta baseline

        # Initialize components
        self._config = ConfigManager.get_instance()

        self._binance = BinanceFeed(self._config)
        self._dual_feed = DualFeed(self._config, self._binance)
        self._discovery = MarketDiscovery(self._config, self._dual_feed)
        self._vatic_feed = VaticFeed(on_strike_price=self._discovery.inject_vatic_strike)
        self._clob = CLOBFeed(self._config)
        self._feature_engine = FeatureEngine(self._config)
        # [FIX-03] XGBoostGate removed — Alpha V1 retired
        self._signal_gen = SignalGenerator(self._config)
        self._db = DatabaseManager()
        self._v5_db = V5DatabaseManager()  # [FIX-14] V5 native database
        self._risk_mgr = RiskManager(self._config, self._db)
        self._execution = ExecutionClient(self._config)
        self._fair_prob_engine = FairProbabilityEngine(self._config)
        self._spread_filter = SpreadFilter(self._config)
        self._exporter: Exporter | None = None
        self._telegram = TelegramNotifier(self._config)

        # Dry run / live engine
        initial_capital = 50.0 if self._requested_mode == "dry-run" else 50.0
        self._dry_run = DryRunEngine(self._config, self._db, initial_capital=initial_capital)
        
        from src.paper_trading import PaperTradingEngine
        self._paper_engine = PaperTradingEngine(self._config, self._clob)
        
        self._exporter = Exporter(self._dry_run.session_id)

        # Slingger Hunter V5
        self._slingger = SlingshotHunterV5()
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._shadow_scalps: dict[str, dict] = {}
        # [FIX-MEM-1] Bounded completed_markets set — capped at 500 entries to prevent
        # unbounded RAM growth. At 12 markets/hr, 500 entries = ~41 hours of lookback,
        # more than sufficient to prevent duplicate processing.
        self._completed_markets: set[str] = set()
        self._completed_markets_max_size: int = 500
        self._enable_dual_execution = os.getenv("ENABLE_DUAL_EXECUTION", "False").lower() == "true"
        
        initial_capital = 50.0
        self._session_stats = {
            'date':            datetime.utcnow().date().isoformat(),
            'trades':          [],
            'total_fees':      0.0,
            'current_capital': initial_capital,
        }
        
        # [FIX-MEM-2] slingger_daily_stats is reset every day via _roll_session_day_if_needed.
        self._slingger_daily_stats = {'hit': 0, 'miss': 0, 'emergency': 0, 'pnls': []}
        MAX_CONCURRENT_SLINGGER_TASKS = int(os.getenv("MAX_SLINGGER_TASKS", "10"))
        self._max_slingger_tasks = MAX_CONCURRENT_SLINGGER_TASKS

        # Dashboard state
        self._latest_signal = None
        self._latest_metrics = None
        self._telegram_heartbeat_minutes = float(
            self._config.get("telegram.heartbeat_minutes", 15.0)
        )
        self._post_mortem_tracker = {}
        # [FIX-MEM-3] _post_mortem_tracker bounded to prevent unbounded growth.
        # Without this cap, markets evaluated but never traded (ABSTAIN) would accumulate
        # indefinitely. At 12 markets/hr x 16hr = 192 entries accruing + counter growth.
        self._MAX_POST_MORTEM_ENTRIES: int = 200
        self._active_bets: dict[str, object] = {}  # market_id → active SignalResult
        
        import uuid as _uuid_mod  # [FIX-15] for trade_id generation
        self._uuid_mod = _uuid_mod
        from collections import deque
        self._odds_history: dict[str, deque] = {}   # per market_id
        self._binance_price_history: deque = deque(maxlen=4500)  # [FIX-08] hard cap: 90s × 50 ticks/s
        
        # [FIX-V5-VISIBILITY] V5 evaluation counter for monitoring
        # Tracks total evaluations so Telegram heartbeat can confirm V5 is running
        self._v5_eval_count: int = 0
        self._v5_enter_count: int = 0

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

    async def _send_alerts_sequential(self, *payloads):
        """Dispatches multiple Telegram alerts in strict sequence to preserve ordering."""
        for channel, msg in payloads:
            await self._send_telegram(channel, msg)

    def _roll_session_day_if_needed(self):
        """
        Auto daily rollover for session stats.
        Sends summary and resets for the new day.
        [FIX-MEM-2] Purges unbounded trades list and pnls list daily to prevent RAM growth.
        V1 gold standard: RiskManager.reset_daily() clears _trade_history every day.
        V5 now mirrors this behavior.
        """
        today = datetime.utcnow().date().isoformat()

        if self._session_stats['date'] != today:
            asyncio.create_task(self._send_daily_summary())

            self._session_stats = {
                'date': today,
                'trades': [],          # [FIX-MEM-2] Purge daily — mirrors V1 RiskManager.reset_daily()
                'total_fees': 0.0,
                'current_capital':
                    self._session_stats['current_capital'],
            }
            # [FIX-MEM-2] Purge daily pnl list — was unbounded in previous code
            self._slingger_daily_stats = {'hit': 0, 'miss': 0, 'emergency': 0, 'pnls': []}
            # [FIX-MEM-2] Reset daily risk counters in V1 RiskManager (alignment)
            self._risk_mgr.reset_daily()
            # [FIX-MEM-1] Clear completed_markets on daily rollover (bounded by day)
            self._completed_markets.clear()
            asyncio.create_task(self._save_v5_state())

    async def _send_daily_summary(self):
        """
        Computes and sends the daily summary via SlingshotAlerts.
        """
        stats = self._session_stats
        trades = stats['trades']
        
        if not trades:
            return

        hit = sum(1 for t in trades if t['result'] == 'HIT')
        miss = sum(1 for t in trades if t['result'] == 'MISS')
        emergency = sum(1 for t in trades if t['result'] in ('EMERGENCY_EXIT', 'HOLD_TO_MATURITY'))
        emergency_exit_now = sum(1 for t in trades if t['result'] == 'EMERGENCY_EXIT')
        emergency_hold = sum(1 for t in trades if t['result'] == 'HOLD_TO_MATURITY')
        
        net_pnls = [t['net_pnl'] for t in trades]
        # We need more detailed stats for the new daily_summary
        # Let's assume we track these in trades
        gross_pnl = sum(t.get('gross_pnl', 0.0) for t in trades)
        total_fees = stats['total_fees']
        net_pnl = sum(net_pnls)
        
        wins = [p for p in net_pnls if p > 0]
        losses = [p for p in net_pnls if p <= 0]
        
        best_trade = max(net_pnls) if net_pnls else 0.0
        worst_trade = min(net_pnls) if net_pnls else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        avg_hold_seconds = np.mean([t['hold_seconds'] for t in trades]) if trades else 0.0
        
        # Sharpe 1D Safety Fix
        returns = net_pnls
        if len(returns) < 2 or np.std(returns) == 0:
            sharpe_1d = 0.0
        else:
            sharpe_1d = (
                np.mean(returns)
                / np.std(returns)
            )

        msg = SlingshotAlerts.daily_summary(
            date_str=stats['date'],
            total=len(trades),
            hit=hit,
            miss=miss,
            emergency=emergency,
            emergency_exit_now=emergency_exit_now,
            emergency_hold=emergency_hold,
            gross_pnl=gross_pnl,
            total_fees=total_fees,
            net_pnl=net_pnl,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_hold_seconds=avg_hold_seconds,
            sharpe_1d=sharpe_1d,
            current_capital=stats['current_capital']
        )
        
        await self._send_telegram("Daily Summary", msg)

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
                    "zone_id": getattr(latest_signal, "zone_id", "N/A")
                    if latest_signal
                    else "N/A",
                    # [FIX-V5-VISIBILITY] V5 evaluation diagnostics in heartbeat
                    # v5_evals > 0 confirms V5 is evaluating; v5_enters/v5_evals = entry rate
                    "v5_evals_total": self._v5_eval_count,
                    "v5_enters_total": self._v5_enter_count,
                    "v5_enter_rate": f"{(self._v5_enter_count / max(self._v5_eval_count, 1) * 100):.1f}%",
                    "v5_active_scalps": len(self._shadow_scalps),
                }
                await self._send_telegram(
                    "HEARTBEAT / MARKET WATCH",
                    SlingshotAlerts.heartbeat(msg),
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
                # [FIX-17] Trigger C: Query V5 session state as Single Source of Truth
                v5_session = await self._v5_db.get_session_state()

                if v5_session:
                    trades_exec = v5_session["trades_executed"]
                    wins = v5_session["trades_win"]
                    total_pnl = v5_session["total_pnl_usd"]
                    capital = v5_session["capital_current"]
                    wr = f"{wins / max(trades_exec, 1) * 100:.1f}%"
                else:
                    trades_exec = 0
                    wins = 0
                    total_pnl = 0.0
                    capital = 50.0
                    wr = "N/A"

                summary = {
                    "trades_executed": trades_exec,
                    "win_rate (trades)": wr,
                    "pnl_usd": f"${total_pnl:.2f}",
                    "capital": f"${capital:.2f}",
                }

                # Append signal aggregation
                sig_summary = await self._get_signal_summary()

                # V5 Heartbeat Health
                v5_health = {
                    "v5_active_scalps": len(self._shadow_scalps),
                    "v5_capital": f"${capital:.2f}",
                    "v5_persistence": "ACTIVE (V5 SQLite)"
                }

                # Combine reports
                combined = {**sig_summary, **{"---": "---"}, **v5_health, **{"---": "---"}, **summary}

                msg_text = SlingshotAlerts.session_report(f"Session Report ({report_hours:.0f}h)", combined)
                await self._send_telegram(
                    f"Session Report ({report_hours:.0f}h)", msg_text
                )
                logger.info("telegram_periodic_report_sent_text_only")

            except Exception as e:
                logger.error("periodic_report_loop_error", error=str(e), exc_info=True)

    async def _dry_run_time_guard(self) -> None:
        """Stop after max duration unless live gate has already enabled live."""
        max_hours = float(self._config.get("dry_run.max_duration_hours", 720))
        await asyncio.sleep(max_hours * 3600)

        if not self._running:
            return
        if self._live_enabled:
            # Live gate passed; no longer considered dry-run stage.
            return

        self._stop_reason = "DRY_RUN_TIME_LIMIT_EXCEEDED"
        await self._send_telegram(
            "DRY RUN TIME LIMIT",
            SlingshotAlerts.dry_run_limit(max_hours, self._dry_run.session_id),
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

        # [FIX-14] Initialize V5 native database
        await self._v5_db.init_db()
        # TODO(Sprint 6): Generate session_id independently from DryRunEngine
        await self._v5_db.init_session(
            session_id=self._dry_run.session_id,
            capital_start=50.0,
        )

        # Guardrail 2: Hydrate V5 State (Capital & Trades)
        await self._hydrate_v5_state()

        logger.info("market_filter_active", slug_prefix=SLUG_PREFIX, is_ultrashort=IS_ULTRASHORT)

        # [FIX-03] Alpha V1 model loading removed — only V5 Slingger active

        # Load Slingger Hunter V5
        try:
            self._slingger.load()
        except Exception as e:
            logger.warning("slingger_v5_not_loaded", info=str(e))

        # Bootstrap historical data
        bars_loaded = await self._binance.bootstrap_historical(limit=500)
        bars_1m_loaded = await self._binance.bootstrap_1m_historical(limit=100)
        logger.info("bootstrap_complete", bars=bars_loaded, bars_1m=bars_1m_loaded)

        # System health report on first bot active (Railway start).
        # This should be lightweight and never crash the bot.
        try:
            binance_health = self._binance.health.model_dump()
        except Exception:
            binance_health = {}
        try:
            clob_health = clob_state.model_dump() if clob_state else None
        except Exception:
            clob_health = None

        await self._send_telegram(
            "SYSTEM HEALTH START",
            SlingshotAlerts.system_health(
                {
                    "architecture": "Predator V3 (Zoned Kelly)",
                    "status": "ACTIVE",
                    "session_id": self._dry_run.session_id,
                    "requested_mode": self._requested_mode,
                    "effective_mode": self._mode,
                    "max_positions": self._config.get("risk.max_positions", 3),
                    "binance_connected": bool(self._binance.latest_price),
                    "clob_state_present": clob_health is not None,
                    "heartbeat_minutes": self._telegram_heartbeat_minutes,
                }
            ),
        )

        # Live mode gate (arm live client), but effective trading starts after go-live metrics pass.
        # HARD LOCK: Tidak akan pernah dijalankan jika tanggal < 4 Mei 2026.
        if self._requested_mode == "live":
            if not self._execution.confirm_live(cli_flag=self._confirm_live):
                logger.error("live_mode_not_confirmed_falling_back_to_dry_run")
                self._live_enabled = False
                self._mode = "dry-run"
            else:
                logger.info("live_preflight_ready")

        # Register signal handlers for graceful shutdown (POSIX only)
        if os.name != 'nt':
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda: asyncio.create_task(self._graceful_shutdown())
                )
        else:
            # On Windows, signal handling is limited in asyncio. 
            # SIGINT (Ctrl+C) is usually handled by the default loop's KeyboardInterrupt.
            pass

        # Register bar close and price update callbacks
        self._binance.set_on_bar_close(self._on_bar_close)
        self._binance.set_on_price_update(self._on_binance_price_update)

        # Start concurrent tasks
        tasks = [
            asyncio.create_task(self._binance.start(), name="binance_feed"),
            asyncio.create_task(self._dual_feed.start(), name="dual_feed_rtds"),
            asyncio.create_task(self._vatic_feed.start(), name="vatic_feed"),
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

        # Slingger daily summary scheduler
        asyncio.create_task(
            self._daily_summary_loop(), name="slingger_daily_summary"
        )

        # [HOTFIX] Memory audit loop — tracemalloc snapshot every 10 min
        asyncio.create_task(
            self._memory_audit_loop(), name="memory_audit_loop"
        )

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

        # Path 5: Shutdown — cleanup all active slingger markets
        for mid in list(self._shadow_scalps.keys()):
            await self.cleanup_market(mid, source='shutdown')

        await self._binance.stop()
        await self._dual_feed.stop()
        await self._discovery.stop()
        await self._clob.stop()
        
        # Flush paper trading engine buffers
        try:
            self._paper_engine._flush_summary()
        except Exception as e:
            logger.error("paper_engine_flush_failed", error=str(e))
        
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
        # TODO(Sprint 6): Replace DryRunEngine metrics with V5 native DB in stop()
        metrics = self._dry_run.compute_session_metrics("Slingger V5")
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
        sum_text = SlingshotAlerts._tg_kv(combined)
        
        if signals_csv and str(signals_csv).endswith(".csv"):
            try:
                await self._telegram.send_document(
                    file_path=str(signals_csv),
                    caption=SlingshotAlerts.session_finished(
                        session_title, session_prefix, self._stop_reason, sum_text
                    ),
                )
            except Exception as e:
                logger.error("telegram_signals_send_failed", error=str(e))
                await self._send_telegram(
                    session_title,
                    SlingshotAlerts.session_finished(
                        session_title, session_prefix, self._stop_reason, sum_text
                    ),
                )
        else:
            await self._send_telegram(
                session_title,
                SlingshotAlerts.session_finished(
                    session_title, session_prefix, self._stop_reason, sum_text
                ),
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
        self._roll_session_day_if_needed()

        # Check if we have an active market
        if not self._discovery.is_market_active:
            return

        market = self._discovery.active_market
        await self._discovery.refresh_ttr()

        # (Moved TTR gate down to line ~950 to prevent blocking Slingger V5)

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
            # Build V5 features from CLOBState for DB recording
            v5_feats_skip = self._build_v5_features_from_clob(market, clob_state)
            self._latest_signal = skip_signal
            await self._dry_run.record_signal(skip_signal, slug=market.slug, v5_features=v5_feats_skip)
            # Shadow prediction: only if oracle available (not ORACLE_UNAVAILABLE)
            if self._mode == "dry-run" and oracle_source != "UNAVAILABLE" and clob_state:
                await self._run_shadow_prediction(market, clob_state, oracle_price, spread_result, v5_feats_skip)
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
            # Build V5 features from CLOBState for DB recording
            v5_feats_wait = self._build_v5_features_from_clob(market, clob_state)
            self._latest_signal = wait_signal
            await self._dry_run.record_signal(wait_signal, slug=market.slug, v5_features=v5_feats_wait)
            # Shadow prediction: oracle is available in WAIT case
            if self._mode == "dry-run" and clob_state:
                await self._run_shadow_prediction(market, clob_state, oracle_price, spread_result, v5_feats_wait)
            return

        # ── Feature Computation ───────────────────────────────
        # oracle_price is guaranteed non-None here (spread filter PROCEED)
        try:
            fv = self._feature_engine.compute(
                self._binance, self._clob, market, clob_state, oracle_price=oracle_price
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

        # ── SLINGGER HUNTER V5 (Section 9) ───────────────────
        # Runs even if TTR is low or spread is marginal.
        try:
            await self._run_slingger_v5(market, clob_state, fv, oracle_price)
        except Exception as e:
            logger.error("slingger_v5_execution_error", error=str(e), traceback=traceback.format_exc())

        # ── ALPHA V1 KILL SWITCH (NO-GO VERDICT) ──
        # Alpha V1 is officially disabled to save CPU cycles and prevent negative EV bleeding.
        # Slingger V5 remains 100% active and healthy.
        logger.debug("alpha_v1_disabled_skipping_directional_flow")
        return

        # [FIX-02] Alpha V1 pipeline removed (430 lines of dead code).
        # Only Slingger V5 pipeline above this point is active.

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

    async def _schedule_resolution(self, trade, market, paper_record=None) -> None:
        """Wait for market resolution and settle trade."""
        from src.zone_matrix import classify_zone
        now = datetime.now(timezone.utc)
        wait_seconds = (market.T_resolution - now).total_seconds()

        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds + 16)  # +16s buffer for Vatic to arrive

        # Get BTC price at resolution using VATIC API / Chainlink (No Binance Fallback)
        epoch_ts = int(market.T_resolution.timestamp())
        cached_sot = self._discovery._epoch_strike_cache.get(epoch_ts)
        
        if cached_sot:
            price = cached_sot[0]
        else:
            oracle_price, oracle_source = self._dual_feed.get_oracle_price_with_source()
            if oracle_price is not None and oracle_source != "UNAVAILABLE":
                price = oracle_price
            else:
                price = float('nan')
                logger.error("resolution_price_unavailable", 
                             trade_id=trade.trade_id, 
                             market_id=trade.market_id)

        if np.isnan(price):
            logger.warning("skipping_trade_resolution_due_to_nan_price", trade_id=trade.trade_id)
            return

        resolved = await self._dry_run.resolve_trade(trade, price)
        
        # Resolve Paper Trade if exists
        if paper_record:
            updated_paper_record = self._paper_engine.resolve_position(
                market_id=trade.market_id,
                won=(resolved.outcome == "WIN"),
                settlement_price=float(price),
                current_capital=resolved.capital_after
            )
            if updated_paper_record:
                # Kirim notifikasi Telegram khusus Paper Trade
                msg = SlingshotAlerts.paper_trade_resolved(
                    {
                        "ID": updated_paper_record.trade_id[:8],
                        "Zone": updated_paper_record.zone_id,
                        "Result": '✅ WIN' if updated_paper_record.actual_outcome == 'WIN' else '❌ LOSS',
                        "PnL": f"${updated_paper_record.actual_pnl_usd:+.2f}",
                        "Edge at entry": f"{updated_paper_record.live_edge:.2%}"
                    }
                )
                asyncio.create_task(
                    self._send_telegram("📊 Paper Trade Resolved", msg),
                    name=f"tg_paper_resolve_{updated_paper_record.trade_id[:8]}"
                )
            
        await self._risk_mgr.on_trade_resolved(resolved.pnl_usd or 0)

        # Real-time CSV append for safety against container crashes
        if self._exporter:
            self._exporter.append_trade(resolved)

        # Clear one-bet-per-market lock
        self._active_bets.pop(trade.market_id, None)

        # Path 4: External resolution — centralized GC
        self._clob.cleanup_market(trade.market_id)
        await self.cleanup_market(trade.market_id, source='schedule_resolution')

        # Telegram: trade resolved (PnL final for this paper/live record).
        asyncio.create_task(
            self._send_telegram(
                "ORDER RESULT",
                SlingshotAlerts.order_result(
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
                )
            ),
            name=f"tg_resolve_{resolved.trade_id[:8]}",
        )
        await self._maybe_enable_live()

    async def _schedule_post_mortem(self, market) -> None:
        """Independent watcher to log abstention stats when a market resolves."""
        now = datetime.now(timezone.utc)
        wait_seconds = (market.T_resolution - now).total_seconds()
        
        # Wait until the market officially resolves + 16 seconds buffer for Vatic
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds + 16)
            
        m_id = market.market_id
        strike_price = market.strike_price
        
        # ── SETTLEMENT PRICE (Use Vatic / Chainlink from dual_feed) ──
        oracle_price = None
        oracle_source = "UNAVAILABLE"
        epoch_ts = int(market.T_resolution.timestamp())
        cached_sot = self._discovery._epoch_strike_cache.get(epoch_ts)
        
        if cached_sot:
            settlement_price = cached_sot[0]
            price_source = cached_sot[1]
        else:
            oracle_price, oracle_source = self._dual_feed.get_oracle_price_with_source()
        
            if oracle_price is not None and oracle_source != "UNAVAILABLE":
                settlement_price = oracle_price
                price_source = oracle_source
            else:
                settlement_price = float('nan')
                price_source = "UNAVAILABLE"
                logger.error("settlement_price_unavailable", 
                               market_id=m_id, 
                               reason="chainlink_and_vatic_unavailable_no_binance_fallback")
        
        if settlement_price is not None and not np.isnan(settlement_price):
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

        # ── Garbage Collection for abstain-only markets ──
        self._clob.cleanup_market(m_id)
        await self.cleanup_market(m_id, source='post_mortem')

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
        consec_pass = int(
            self._config.get("dry_run.go_live_consecutive_pass", 5)
        )
        metrics = self._dry_run.compute_session_metrics("Slingger V5")  # [FIX-18]

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
                    SlingshotAlerts.go_live_enabled(
                        {
                            "session_id": self._dry_run.session_id,
                            "trades_executed": metrics.trades_executed,
                            "win_rate": metrics.win_rate,
                            "total_pnl_usd": metrics.total_pnl_usd,
                            "dry_run_score": metrics.dry_run_score,
                            "pass_fail": metrics.pass_fail,
                        }
                    )
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

            if market.T_resolution is None or market.T_open is None:
                logger.warning("ultrashort_loop_missing_timestamps", market_id=market.market_id)
                continue
                
            lifespan_sec = (market.T_resolution - market.T_open).total_seconds()
            is_ultrashort = (IS_ULTRASHORT and SLUG_PREFIX in market.slug) or lifespan_sec <= 600
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
                logger.error("ultrashort_loop_error", error=str(e), traceback=traceback.format_exc())

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
                            self._odds_history[m_id] = __import__("collections").deque(maxlen=4500)  # [FIX-09]
                        self._odds_history[m_id].append((now, state.yes_ask))
                        
                        cutoff = now - timedelta(seconds=90)
                        while self._odds_history[m_id] and self._odds_history[m_id][0][0] < cutoff:
                            self._odds_history[m_id].popleft()
                            
                        # Real-time CLOB log append
                        if self._exporter:
                            ttr_minutes = (market.T_resolution - now).total_seconds() / 60.0
                            self._exporter.record_clob_snapshot(state, ttr_minutes)
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

    # ── Slingger Hunter V5: Dual-Stage Shadow Monitor (DRY-RUN ONLY) ──
    # [MANDAT-DRYRUN] Slingger V5 DILARANG eksekusi riil (Shadow Entry Only).
    def _init_shadow_scalp(self, market_id: str, side: str, entry: float, exit: float, prob: float,
                           stake_usd: float, shares: float, depth_usd_at_entry: float,
                           btc_vs_strike_pct: float, ttr: int) -> None:
        import time as _time
        trade_id = self._uuid_mod.uuid4().hex[:16]  # [FIX-15] unique trade_id
        self._shadow_scalps[market_id] = {
            'trade_id':            trade_id,
            'token_side':          side,
            'entry_odds':          entry,
            'exit_odds':           exit,
            'swing_prob':          prob,
            'phase':               'WAITING_ENTRY',
            'entry_filled':        False,
            'exit_filled':         False,
            'entry_fill_price':    None,
            'exit_fill_price':     None,
            'entry_fill_time':      None,
            'exit_fill_time':      None,
            'emergency_triggered': False,
            'emergency_decision':  None,
            'created_at':          _time.time(),
            
            # New telemetry fields
            'stake_usd':            stake_usd,
            'shares':               shares,
            'depth_at_entry':       depth_usd_at_entry,
            'btc_vs_strike_pct':    btc_vs_strike_pct,
            'ttr_at_entry':         ttr,
            'ttr_at_exit':          None,
            'exit_price_actual':    None,
            'gross_pnl':            None,
            'fee_paid':             None,
            'net_pnl':              None,
        }

    def _verify_binance_buffer(self) -> bool:
        """Check 30min OHLCV buffer exists"""
        buffer = self._binance.ohlcv_1m_buffer
        if len(buffer) < 30:
            logger.error("insufficient_binance_data", bars=len(buffer))
            return False
        return True

    def _compute_btc_realized_vol_live(self, binance_ohlcv_30m: pd.DataFrame) -> float:
        """
        Compute annualized realized volatility from 30min Binance OHLCV
        Matches training pipeline exactly
        """
        import pandas as pd
        import numpy as np
        # Extract close prices
        closes = binance_ohlcv_30m['close']
        
        # Compute log returns
        log_returns = np.log(closes / closes.shift(1)).dropna()
        
        # Standard deviation
        sigma = log_returns.std()
        
        # Annualize (assuming 5-minute bars in 30min window = 6 bars)
        # Trading year = 365 days * 24 hours * 12 (5min bars/hour) = 105,120 bars
        annualization_factor = np.sqrt(105120)
        
        volatility_annualized = sigma * annualization_factor
        
        # Sanity check: Should be 0.30-0.80 typically
        if volatility_annualized < 0.1 or volatility_annualized > 2.0:
            logger.warning("volatility_outside_expected_range", val=round(volatility_annualized, 4))
        
        return float(volatility_annualized)

    def _audit_feature_ranges(self, btc_vol: float) -> float:
        """Verify live features are within acceptable limits"""
        try:
            # 0.05 <= volatility <= 2.50
            if not (0.05 <= btc_vol <= 2.50):
                raise AssertionError(f"Volatility out of bounds: {btc_vol:.4f}")
            return btc_vol
        except AssertionError as e:
            logger.warning("volatility_safety_audit_failed", error=str(e), fallback=0.45)
            return 0.45

    async def _run_slingger_v5(self, market: ActiveMarket, clob_state: CLOBState, fv: FeatureVector, oracle_price: float) -> None:
        """
        Inference engine for Slingger Hunter V5.
        Detects swing patterns on YES and NO tokens independently.
        """
        if not self._slingger.is_loaded or not clob_state:
            return
            
        # Safety Guard: Ensure all required price fields are populated before feature computation
        if any(v is None for v in [clob_state.yes_ask, clob_state.yes_bid, clob_state.no_ask, clob_state.no_bid]):
            logger.debug("slingger_v5_incomplete_clob_skip", market_id=market.market_id)
            return

        m_id = market.market_id
        if m_id in self._active_tasks or m_id in self._shadow_scalps or m_id in self._completed_markets:
            return  # Already tracking or completed this market

        # [FIX-V5-VISIBILITY] Increment evaluation counter for monitoring.
        # Telegram heartbeat will show v5_evals and v5_enters every interval,
        # confirming V5 IS running even during zero-trade periods.
        self._v5_eval_count += 1

        # [FIX-EV-1] UNDERDOG HARD BLOCK: Production data shows UNDERDOG_<35% has EV=-29.1%
        # (68W/338L = 16.7% win rate at avg odds 0.236, theoretical payout 3.24x).
        # V1 gold standard uses Zone Matrix which already excludes these zones.
        yes_bid = clob_state.yes_bid
        no_bid = clob_state.no_bid
        if (yes_bid is not None and yes_bid < 0.35) or (no_bid is not None and no_bid < 0.35):
            logger.debug("slingger_v5_underdog_block", yes_bid=yes_bid, no_bid=no_bid,
                         reason="EV_negative_at_odds_below_0.35")
            return

        if len(self._active_tasks) >= self._max_slingger_tasks:
            logger.warning("slingger_v5_max_tasks_reached", limit=self._max_slingger_tasks)
            return

        now = datetime.now(timezone.utc)
        
        # 1. Prepare Base Features (from FeatureEngine)
        base_features = dict(zip(fv.feature_names, fv.values))
        
        # 2. Compute 30s Velocity and Volatility proxy (using history from CLOBFeed)
        lookback = 30.0
        yes_token = market.clob_token_ids.get("YES", "")
        no_token = market.clob_token_ids.get("NO", "")
        
        hist_yes_snap = self._clob.get_historical_book_snapshot(yes_token, lookback) if yes_token else None
        
        # [VOLATILITY FIX] Compute annualized volatility from 30min Binance OHLCV
        import pandas as pd
        if self._verify_binance_buffer():
            ohlcv_1m = self._binance.ohlcv_1m_buffer
            recent_1m = ohlcv_1m[-30:]
            df_1m = pd.DataFrame(recent_1m)
            df_5m = df_1m.iloc[::5].copy()
            raw_btc_vol = self._compute_btc_realized_vol_live(df_5m)
            btc_vol = self._audit_feature_ranges(raw_btc_vol)
        else:
            btc_vol = 0.45  # fallback to historical average if buffer not ready
            
        logger.info("btc_realized_vol_live", value=round(btc_vol, 4))

        if hist_yes_snap:
            # Simple bid price velocity
            hist_ask = self._clob._best_ask(hist_yes_snap)
            curr_bid = clob_state.yes_bid
            
            if hist_ask is not None and curr_bid is not None:
                hist_bid = 1.0 - hist_ask
                price_velocity_30s = (curr_bid - hist_bid) / lookback
            else:
                price_velocity_30s = 0.0
        else:
            price_velocity_30s = 0.0

        # 3. Predict for YES side swing
        yes_feat = {
            'yes_price_t0':              clob_state.yes_bid,
            'no_price_t0':               clob_state.no_bid,
            'clob_spread_t0':            clob_state.yes_ask - clob_state.yes_bid,
            'yes_depth_t0':              clob_state.yes_depth_usd,
            'no_depth_t0':               clob_state.no_depth_usd,
            'depth_imbalance_t0':        (clob_state.yes_depth_usd - clob_state.no_depth_usd) / max(clob_state.yes_depth_usd + clob_state.no_depth_usd, 1.0),
            'price_velocity_30s':        price_velocity_30s,
            'depth_trend_30s':           base_features.get("clob_depth_delta", 0.0),
            'btc_realized_vol_prior_30m': btc_vol,
            'ttr_at_signal':             (market.T_resolution - now).total_seconds(),
            'market_hour_utc':           now.hour,
            'day_of_week':               now.weekday(),
        }
        
        logger.debug("slingger_v5_features_assembly", market_id=market.market_id, features=yes_feat)
        res_yes = self._slingger.predict(yes_feat, live_entry_odds=clob_state.yes_bid)  # [FIX-11]
        
        if res_yes['signal'] == 'ENTER':
            logger.info("slingger_v5_yes_signal", market_id=market.market_id, prob=res_yes['swing_probability'], kelly=res_yes['full_kelly'])
        
        # 4. Predict for NO side swing (Swap prices/depths as seen by NO side)
        no_feat = yes_feat.copy()
        no_feat['yes_price_t0'] = clob_state.no_bid
        no_feat['no_price_t0']  = clob_state.yes_bid
        no_feat['yes_depth_t0'] = clob_state.no_depth_usd
        no_feat['no_depth_t0']  = clob_state.yes_depth_usd
        no_feat['depth_imbalance_t0'] = -yes_feat['depth_imbalance_t0']
        
        res_no = self._slingger.predict(no_feat, live_entry_odds=clob_state.no_bid)  # [FIX-11]
        if res_no['signal'] == 'ENTER':
            logger.info("slingger_v5_no_signal", market_id=market.market_id, prob=res_no['swing_probability'], kelly=res_no['full_kelly'])

        # 5. Decide
        winner = None
        if res_yes['signal'] == 'ENTER' and res_no['signal'] == 'ENTER':
            winner = 'YES' if res_yes['swing_probability'] >= res_no['swing_probability'] else 'NO'
        elif res_yes['signal'] == 'ENTER':
            winner = 'YES'
        elif res_no['signal'] == 'ENTER':
            winner = 'NO'

        if winner:
            res = res_yes if winner == 'YES' else res_no
            
            # [FIX-15] Capital from V5 native database — no DryRunEngine dependency
            v5_sess = await self._v5_db.get_session_state()
            primary_capital = v5_sess['capital_current'] if v5_sess else 50.0
            self._session_stats['current_capital'] = primary_capital  # Keep in sync

            # Kalkulasi dana tertahan (locked margin) di shadow positions
            locked_capital = sum(
                s['stake_usd'] for s in self._shadow_scalps.values() 
                if s['phase'] in ('WAITING_ENTRY', 'WAITING_EXIT')
            )
            # Dapatkan sisa saldo aktif (dibatasi nol agar tidak minus)
            available_capital = max(0.0, primary_capital - locked_capital)

            # [FIX-PIPELINE-1] Apply V1 consecutive-loss multiplier decay to V5 sizing.
            # V1 RiskManager uses: multiplier = max(kelly_floor, 1.0 - consec_losses * decay)
            # V5 previously ignored consecutive losses entirely.
            consec_losses = self._risk_mgr.consecutive_losses
            kelly_floor = float(self._config.get("risk.kelly_floor_multiplier", 0.25))
            loss_decay = float(self._config.get("risk.consecutive_loss_multiplier", 0.15))
            kelly_multiplier_v1 = max(kelly_floor, 1.0 - consec_losses * loss_decay)

            # Calculate position size using Kelly with V1 multiplier applied
            sizing_raw = compute_position_size(
                capital=available_capital,
                swing_prob=res['swing_probability'],
                entry_odds=res['entry_odds'],
                exit_odds=res['exit_odds']
            )
            # Apply V1 decay multiplier to stake
            sizing = sizing_raw.copy()
            sizing['stake_usd'] = round(sizing_raw['stake_usd'] * kelly_multiplier_v1, 2)
            sizing['shares'] = round(sizing_raw['shares'] * kelly_multiplier_v1, 2)
            
            if kelly_multiplier_v1 < 1.0:
                logger.info("slingger_v5_kelly_decay_applied",
                            consec_losses=consec_losses,
                            multiplier=round(kelly_multiplier_v1, 4),
                            stake_before=sizing_raw['stake_usd'],
                            stake_after=sizing['stake_usd'])
            
            # 1. Pastikan modal cukup untuk minimum trade
            if available_capital < 1.00:
                logger.warning("Capital depleted (< $1.00). Skipping trade.")
                return

            stake_usd = sizing['stake_usd']
            # 2. Terapkan Floor dan Rounding
            stake_usd = max(1.00, stake_usd)
            stake_usd = round(stake_usd, 2)

            # 3. Pastikan stake tidak melebihi sisa modal setelah pembulatan
            stake_usd = min(stake_usd, available_capital)
            
            sizing['stake_usd'] = stake_usd
            sizing['shares'] = round(stake_usd / res['entry_odds'], 2) if res['entry_odds'] > 0 else 0.0

            if sizing['stake_usd'] <= 0:
                logger.debug("slingger_sizing_zero", market_id=m_id, side=winner, prob=res['swing_probability'])
                return

            logger.info("slingger_v5_pattern_detected", 
                        side=winner, prob=round(res['swing_probability'], 4),
                        tier=res['confidence_tier'], stake=sizing['stake_usd'])

            # [FIX-V5-VISIBILITY] Track successful entries for telemetry
            self._v5_enter_count += 1

            # Additional Telemetry Data
            depth_at_entry = clob_state.yes_depth_usd if winner == 'YES' else clob_state.no_depth_usd
            btc_vs_strike_pct = ((oracle_price - market.strike_price) / market.strike_price) * 100
            ttr = int((market.T_resolution - now).total_seconds())

            self._init_shadow_scalp(
                m_id, winner, res['entry_odds'], res['exit_odds'], res['swing_probability'],
                stake_usd=sizing['stake_usd'],
                shares=sizing['shares'],
                depth_usd_at_entry=depth_at_entry,
                btc_vs_strike_pct=btc_vs_strike_pct,
                ttr=ttr
            )

            # [FIX-15] Record trade in V5 native database
            scalp = self._shadow_scalps[m_id]
            await self._v5_db.record_new_trade(
                trade_id=scalp['trade_id'],
                market_id=m_id,
                token_side=winner,
                entry_odds=res['entry_odds'],
                target_odds=res['exit_odds'],
                entry_spread=clob_state.yes_ask - clob_state.yes_bid if clob_state.yes_ask and clob_state.yes_bid else 0.0,
                stake_usd=sizing['stake_usd'],
                shares=sizing['shares'],
            )

            asyncio.create_task(self._save_v5_state())
            self._active_tasks[m_id] = asyncio.create_task(
                self._shadow_scalp_monitor_loop(m_id),
                name=f"slingger_monitor_{m_id[:8]}"
            )


    def _build_v5_features_from_clob(self, market: "ActiveMarket", clob_state: "CLOBState") -> dict:
        """
        Build the 9-column V5 feature dict from CLOBState and Binance buffer.

        This mirrors the feature assembly in _run_slingger_v5 (lines 1527-1540)
        so that features are recorded in the DB for ALL signal types, including
        spread-blocked SKIP/WAIT abstains.
        """
        now = datetime.now(timezone.utc)
        yes_token = market.clob_token_ids.get("YES", "")
        lookback = 30.0

        # Price velocity (30s)
        hist_yes_snap = self._clob.get_historical_book_snapshot(yes_token, lookback) if yes_token else None
        if hist_yes_snap:
            hist_ask = self._clob._best_ask(hist_yes_snap)
            curr_bid = clob_state.yes_bid
            if hist_ask is not None and curr_bid is not None:
                hist_bid = 1.0 - hist_ask
                price_velocity_30s = (curr_bid - hist_bid) / lookback
            else:
                price_velocity_30s = 0.0
        else:
            price_velocity_30s = 0.0

        # BTC realized vol (30min, from 1m OHLCV buffer)
        if self._verify_binance_buffer():
            import pandas as pd
            ohlcv_1m = self._binance.ohlcv_1m_buffer
            recent_1m = ohlcv_1m[-30:]
            df_1m = pd.DataFrame(recent_1m)
            df_5m = df_1m.iloc[::5].copy()
            raw_vol = self._compute_btc_realized_vol_live(df_5m)
            btc_vol = self._audit_feature_ranges(raw_vol)
        else:
            btc_vol = 0.45  # historical average fallback

        # Depth imbalance
        yes_d = clob_state.yes_depth_usd or 0.0
        no_d = clob_state.no_depth_usd or 0.0
        total_d = yes_d + no_d
        depth_imbalance = (yes_d - no_d) / max(total_d, 1.0)

        return {
            "yes_price_t0": clob_state.yes_bid,
            "no_price_t0": clob_state.no_bid,
            "clob_spread_t0": (clob_state.yes_ask or 0.0) - (clob_state.yes_bid or 0.0),
            "yes_depth_t0": yes_d,
            "no_depth_t0": no_d,
            "depth_imbalance_t0": depth_imbalance,
            "price_velocity_30s": price_velocity_30s,
            "depth_trend_30s": 0.0,   # requires FeatureEngine; set 0 for blocked signals
            "btc_realized_vol_prior_30m": btc_vol,
        }

    async def _run_shadow_prediction(
        self,
        market: "ActiveMarket",
        clob_state: "CLOBState",
        oracle_price: float,
        spread_result: "SpreadFilterResult",
        v5_feats: dict,
    ) -> None:
        """
        Run V5 model inference on a spread-blocked signal and log to shadow CSV.

        Called ONLY in dry-run mode, after SKIP or WAIT spread filter decisions.
        Does NOT affect trading execution in any way.

        Shadow CSV path: data/exports/{session_id}/dry_run_shadow_{session_id}.csv
        """
        if not self._slingger.is_loaded:
            return

        try:
            now = datetime.now(timezone.utc)
            ttr_seconds = max(0.0, (market.T_resolution - now).total_seconds())

            # YES side prediction
            yes_feat = {
                "yes_price_t0": clob_state.yes_bid,
                "no_price_t0": clob_state.no_bid,
                "clob_spread_t0": (clob_state.yes_ask or 0.0) - (clob_state.yes_bid or 0.0),
                "yes_depth_t0": clob_state.yes_depth_usd,
                "no_depth_t0": clob_state.no_depth_usd,
                "depth_imbalance_t0": v5_feats.get("depth_imbalance_t0", 0.0),
                "price_velocity_30s": v5_feats.get("price_velocity_30s", 0.0),
                "depth_trend_30s": v5_feats.get("depth_trend_30s", 0.0),
                "btc_realized_vol_prior_30m": v5_feats.get("btc_realized_vol_prior_30m", 0.45),
                "ttr_at_signal": ttr_seconds,
                "market_hour_utc": now.hour,
                "day_of_week": now.weekday(),
            }
            res_yes = self._slingger.predict(yes_feat, live_entry_odds=clob_state.yes_bid)

            # NO side prediction (swap YES/NO perspectives)
            no_feat = yes_feat.copy()
            no_feat["yes_price_t0"] = clob_state.no_bid
            no_feat["no_price_t0"] = clob_state.yes_bid
            no_feat["yes_depth_t0"] = clob_state.no_depth_usd
            no_feat["no_depth_t0"] = clob_state.yes_depth_usd
            no_feat["depth_imbalance_t0"] = -yes_feat["depth_imbalance_t0"]
            res_no = self._slingger.predict(no_feat, live_entry_odds=clob_state.no_bid)

            # Extract spread blocked reason label
            reason = spread_result.reason or ""
            if "ELEVATED" in reason:
                blocked_reason = "ELEVATED"
            elif "TOO_WIDE" in reason:
                blocked_reason = "TOO_WIDE"
            elif "UNAVAILABLE" in reason or "STALE" in reason:
                blocked_reason = "ORACLE_UNAVAILABLE"
            else:
                blocked_reason = "SKIP"

            shadow_record = {
                "timestamp": now.isoformat(),
                "market_id": market.market_id,
                "slug": getattr(market, "slug", ""),
                "ttr_seconds": round(ttr_seconds, 1),
                "spread_pct": round(spread_result.spread_pct, 6),
                "spread_blocked_reason": blocked_reason,
                "yes_price_t0": v5_feats.get("yes_price_t0", ""),
                "no_price_t0": v5_feats.get("no_price_t0", ""),
                "clob_spread_t0": round(v5_feats.get("clob_spread_t0", 0.0), 6),
                "yes_depth_t0": round(v5_feats.get("yes_depth_t0", 0.0), 2),
                "no_depth_t0": round(v5_feats.get("no_depth_t0", 0.0), 2),
                "depth_imbalance_t0": round(v5_feats.get("depth_imbalance_t0", 0.0), 6),
                "price_velocity_30s": round(v5_feats.get("price_velocity_30s", 0.0), 8),
                "depth_trend_30s": round(v5_feats.get("depth_trend_30s", 0.0), 6),
                "btc_realized_vol_prior_30m": round(v5_feats.get("btc_realized_vol_prior_30m", 0.0), 6),
                "ttr_at_signal": round(ttr_seconds, 1),
                "market_hour_utc": now.hour,
                "day_of_week": now.weekday(),
                "shadow_signal_yes": res_yes["signal"],
                "shadow_prob_yes": round(res_yes["swing_probability"], 6),
                "shadow_kelly_yes": round(res_yes["full_kelly"], 6),
                "shadow_tier_yes": res_yes["confidence_tier"],
                "shadow_signal_no": res_no["signal"],
                "shadow_prob_no": round(res_no["swing_probability"], 6),
                "shadow_kelly_no": round(res_no["full_kelly"], 6),
                "shadow_tier_no": res_no["confidence_tier"],
                "actual_outcome": "PENDING",
            }

            self._exporter.record_shadow(shadow_record)

            logger.debug(
                "shadow_prediction_recorded",
                market_id=market.market_id,
                blocked_reason=blocked_reason,
                yes_signal=res_yes["signal"],
                yes_prob=round(res_yes["swing_probability"], 4),
                no_signal=res_no["signal"],
                no_prob=round(res_no["swing_probability"], 4),
            )

        except Exception as e:
            logger.warning("shadow_prediction_error", error=str(e))

    async def _shadow_scalp_monitor_loop(self, market_id: str) -> None:
        """
        Dual-stage monitor for one market (Slingger Hunter V5).
        FASE 1 WAITING_ENTRY: price <= entry_odds -> virtual fill -> FASE 2
        FASE 2 WAITING_EXIT: price >= exit_odds -> HIT | TTR<60 -> EMERGENCY
        """
        import time as _time
        POLL_INTERVAL = 5
        EMERGENCY_TTR = 60

        state = self._shadow_scalps.get(market_id)
        if not state:
            return

        # Cache market metadata at start to survive rotation
        market = self._discovery.active_market
        if not market or market.market_id != market_id:
            # Fallback for late starts
            market_slug = market_id[:16]
            res_time = datetime.now(timezone.utc) + timedelta(minutes=5)
        else:
            market_slug = market.slug
            res_time = market.T_resolution

        try:
            while state['phase'] != 'CLOSED':
                if not self._running:
                    break

                clob = self._clob.clob_state
                # We can track the market even if it's not the primary active one,
                # as long as self._clob still has its data in cache/history.
                # But for polling simplicity, we assume we only track the primary.
                if clob is None or clob.market_id != market_id:
                    # Try to fetch fresh state for this specific market if possible
                    # (This bot design usually assumes 1 active market at a time)
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

                token_side = state['token_side']
                ttr = int((res_time - datetime.now(timezone.utc)).total_seconds())

                if token_side == 'YES':
                    current_price = (1.0 - clob.no_ask) if clob.no_ask is not None else None
                else:
                    current_price = (1.0 - clob.yes_ask) if clob.yes_ask is not None else None

                if current_price is None:
                    logger.warning("slingger_price_unavailable", market_id=market_id)
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

                # FASE 1: WAITING_ENTRY
                if state['phase'] == 'WAITING_ENTRY':
                    # [FIX-12] Sanity guard: reject fills at absurdly low prices
                    if current_price < 0.35:
                        logger.warning(
                            f"HARD BLOCK: fill price {current_price:.3f} < 0.35 minimum, aborting fill. trade_id={state['trade_id']}"
                        )
                        await asyncio.sleep(POLL_INTERVAL)
                        continue
                    if current_price <= state['entry_odds']:
                        state['entry_filled'] = True
                        state['entry_fill_price'] = current_price
                        state['entry_fill_time'] = _time.time()
                        state['phase'] = 'WAITING_EXIT'
                        msg = SlingshotAlerts.entry(
                            market_slug=market_slug,
                            entry_price=current_price,
                            exit_target=state['exit_odds'],
                            ttr=ttr,
                            confidence=state['swing_prob'],
                            side=token_side,
                            stake_usd=state['stake_usd'],
                            shares=state['shares'],
                            depth_available_usd=state['depth_at_entry'],
                            btc_vs_strike_pct=state['btc_vs_strike_pct']
                        )
                        asyncio.create_task(
                            self._send_telegram("SLINGGER V5", msg),
                            name=f"slingger_entry_{market_id[:8]}"
                        )
                        logger.info("slingger_entry_fill", market_id=market_id, price=current_price)

                # FASE 2: WAITING_EXIT
                elif state['phase'] == 'WAITING_EXIT':
                    latency_s = int(_time.time() - state['entry_fill_time'])
                    entry_p = state['entry_fill_price']

                    if current_price >= state['exit_odds']:
                        # Finalize PnL and telemetry
                        gross_pnl = (current_price - entry_p) * state['shares']
                        fee_paid = current_price * state['shares'] * 0.02
                        net_pnl = gross_pnl - fee_paid - 0.005
                        
                        state.update({
                            'exit_filled':         True,
                            'exit_fill_price':     current_price,
                            'exit_fill_time':      _time.time(),
                            'exit_price_actual':   current_price,
                            'ttr_at_exit':         ttr,
                            'gross_pnl':           gross_pnl,
                            'fee_paid':            fee_paid,
                            'net_pnl':             net_pnl,
                            'result':              'HIT',
                            'phase':               'CLOSED'
                        })

                        # Track session stats
                        self._session_stats['trades'].append({
                            'result': 'HIT',
                            'net_pnl': net_pnl,
                            'gross_pnl': gross_pnl,
                            'hold_seconds': latency_s,
                        })
                        self._session_stats['current_capital'] += net_pnl
                        self._session_stats['total_fees'] += fee_paid

                        # [FIX-16] Persist to V5 native database
                        await self._v5_db.close_trade(
                            trade_id=state.get('trade_id', market_id),
                            status='WIN',
                            exit_odds=current_price,
                            pnl_usd=net_pnl,
                        )

                        if not state.get('is_hydrated', False):
                            await self._risk_mgr.on_trade_resolved(net_pnl)
                        else:
                            logger.debug("Ghost trade resolved. Bypassing RiskManager memory.")

                        self._slingger_daily_stats['hit'] += 1
                        self._slingger_daily_stats['pnls'].append(net_pnl)
                        
                        daily_wins = sum(1 for t in self._session_stats['trades'] if t['result'] == 'HIT')
                        daily_losses = sum(1 for t in self._session_stats['trades'] if t['result'] == 'MISS')
                        daily_pnl = sum(t['net_pnl'] for t in self._session_stats['trades'])

                        msg = SlingshotAlerts.exit_hit(
                            market_slug=market_slug,
                            entry_price=entry_p,
                            exit_price=current_price,
                            exit_target=state['exit_odds'],
                            latency_s=latency_s,
                            ttr_at_exit=ttr,
                            shares=state['shares'],
                            stake_usd=state['stake_usd'],
                            daily_pnl=daily_pnl,
                            daily_wins=daily_wins,
                            daily_losses=daily_losses
                        )
                        asyncio.create_task(
                            self._send_telegram("SLINGGER V5", msg),
                            name=f"slingger_hit_{market_id[:8]}"
                        )
                        logger.info("slingger_hit", market_id=market_id, pnl=net_pnl)
                        break

                    elif ttr < EMERGENCY_TTR and not state['emergency_triggered']:
                        state['emergency_triggered'] = True
                        
                        # Use correct ROI-based EV
                        ev_exit = (current_price - entry_p) * 0.98 - 0.005
                        ev_hold = (current_price * (1.0 - entry_p) * 0.98
                                   - (1 - current_price) * entry_p - 0.005)
                        
                        decision = 'EXIT_NOW' if ev_exit >= ev_hold else 'HOLD_TO_MATURITY'
                        state['emergency_decision'] = decision
                        
                        self._slingger_daily_stats['emergency'] += 1
                        
                        emergency_msg = SlingshotAlerts.emergency(
                            market_slug=market_slug,
                            ttr=ttr,
                            decision=decision,
                            ev_exit=ev_exit,
                            ev_hold=ev_hold,
                            current_price=current_price,
                            exit_target=state['exit_odds'],
                            stake_usd=state['stake_usd']
                        )
                        # We don't fire the emergency alert as a task yet; 
                        # if it's EXIT_NOW, we'll combine it with the final outcome alert.
                        
                        if decision == 'HOLD_TO_MATURITY':
                            state['result'] = 'HOLD_TO_MATURITY'
                            state['phase'] = 'CLOSED'
                            
                            # Track session stats (Hold to maturity doesn't have final PnL yet, but we'll count it)
                            self._session_stats['trades'].append({
                                'result': 'HOLD_TO_MATURITY',
                                'net_pnl': 0.0, # Placeholder
                                'hold_seconds': latency_s,
                            })
                            
                            # PnL from HOLD_TO_MATURITY requires on-chain
                            # resolution data — set 0.0 until Sprint 6 resolver.
                            await self._v5_db.close_trade(
                                trade_id=state['trade_id'],
                                status='HOLD_TO_MATURITY',
                                exit_odds=current_price,
                                pnl_usd=0.0,
                                exit_ts=int(_time.time() * 1000)
                            )

                            if not state.get('is_hydrated', False):
                                await self._risk_mgr.on_trade_resolved(0.0)
                            else:
                                logger.debug("Ghost trade resolved. Bypassing RiskManager memory.")
                            
                            asyncio.create_task(
                                self._send_telegram("SLINGGER V5", emergency_msg),
                                name=f"slingger_emerg_{market_id[:8]}"
                            )
                            break
                        else:
                            # EXIT_NOW: close virtual position
                            state['phase'] = 'CLOSED'
                            # Fix: ensure real loss is captured if delta is unrealistic (e.g. price collapse)
                            delta_pnl = (current_price - entry_p) * state['shares'] * 0.98 - 0.005
                            net_pnl = delta_pnl if delta_pnl < -state['stake_usd'] * 0.5 else -state['stake_usd']
                            
                            # Terjemahkan ke HIT/MISS untuk integritas W/L Tracker
                            is_win = net_pnl > 0
                            final_result = 'HIT' if is_win else 'MISS'
                            state['result'] = final_result
                            
                            self._session_stats['trades'].append({
                                'result': final_result,
                                'net_pnl': net_pnl,
                                'hold_seconds': latency_s,
                            })
                            self._session_stats['current_capital'] += net_pnl

                            # [FIX-16] Persist EMERGENCY_EXIT to V5 native database
                            await self._v5_db.close_trade(
                                trade_id=state.get('trade_id', market_id),
                                status='EMERGENCY_EXIT',
                                exit_odds=current_price,
                                pnl_usd=net_pnl,
                            )

                            if not state.get('is_hydrated', False):
                                await self._risk_mgr.on_trade_resolved(net_pnl)
                            else:
                                logger.debug("Ghost trade resolved. Bypassing RiskManager memory.")
                            
                            # Update W/L Rekor Slingger
                            if is_win:
                                self._slingger_daily_stats['hit'] += 1
                                self._slingger_daily_stats['pnls'].append(net_pnl)
                            else:
                                self._slingger_daily_stats['miss'] += 1
                            
                            daily_wins = sum(1 for t in self._session_stats['trades'] if t['result'] == 'HIT')
                            daily_losses = sum(1 for t in self._session_stats['trades'] if t['result'] == 'MISS')
                            daily_pnl = sum(t['net_pnl'] for t in self._session_stats['trades'])

                            # Picu Notifikasi Penutupan Final (Closure Alert)
                            if is_win:
                                outcome_msg = SlingshotAlerts.exit_hit(
                                    market_slug=market_slug,
                                    entry_price=entry_p,
                                    exit_price=current_price,
                                    exit_target=state['exit_odds'],
                                    latency_s=latency_s,
                                    ttr_at_exit=ttr,
                                    shares=state['shares'],
                                    stake_usd=state['stake_usd'],
                                    daily_pnl=daily_pnl,
                                    daily_wins=daily_wins,
                                    daily_losses=daily_losses
                                )
                            else:
                                outcome_msg = SlingshotAlerts.miss(
                                    market_slug=market_slug,
                                    entry_price=entry_p,
                                    exit_target=state['exit_odds'],
                                    final_price=current_price,
                                    stake_usd=state['stake_usd'],
                                    reason="EMERGENCY_EXIT_LOSS",
                                    daily_pnl=daily_pnl,
                                    daily_wins=daily_wins,
                                    daily_losses=daily_losses
                                )
                                
                            # Dispatch EMERGENCY and outcome alerts sequentially to guarantee delivery order
                            asyncio.create_task(
                                self._send_alerts_sequential(
                                    ("SLINGGER V5", emergency_msg),
                                    ("SLINGGER V5", outcome_msg)
                                ),
                                name=f"slingger_emerg_seq_{market_id[:8]}"
                            )
                            logger.info("slingger_emergency_closed", market_id=market_id, pnl=net_pnl)
                            break

                    elif ttr <= 0:
                        # Guardrail 3: Reality Sync - Force HOLD_TO_MATURITY if expired
                        logger.warning("slingger_stale_reality_sync_expired", market_id=market_id)
                        state.update({
                            'emergency_triggered': True,
                            'emergency_decision': 'HOLD_TO_MATURITY',
                            'result': 'HOLD_TO_MATURITY',
                            'phase': 'CLOSED'
                        })
                        self._session_stats['trades'].append({
                            'result': 'HOLD_TO_MATURITY',
                            'net_pnl': 0.0,
                            'hold_seconds': latency_s,
                        })

                        # PnL from HOLD_TO_MATURITY requires on-chain
                        # resolution data — set 0.0 until Sprint 6 resolver.
                        await self._v5_db.close_trade(
                            trade_id=state['trade_id'],
                            status='HOLD_TO_MATURITY',
                            exit_odds=current_price,
                            pnl_usd=0.0,
                            exit_ts=int(_time.time() * 1000)
                        )

                        if not state.get('is_hydrated', False):
                            await self._risk_mgr.on_trade_resolved(0.0)
                        else:
                            logger.debug("Ghost trade resolved. Bypassing RiskManager memory.")

                        msg = SlingshotAlerts.emergency(
                            market_slug=market_slug,
                            ttr=ttr,
                            decision='HOLD_TO_MATURITY',
                            ev_exit=0.0, ev_hold=0.0,
                            current_price=current_price,
                            exit_target=state['exit_odds'],
                            stake_usd=state['stake_usd']
                        )
                        asyncio.create_task(
                            self._send_telegram("SLINGGER V5", msg),
                            name=f"slingger_stale_sync_{market_id[:8]}"
                        )
                        break
                
                # Save state after each phase update or closure
                await self._save_v5_state()
                await asyncio.sleep(POLL_INTERVAL)

        except asyncio.CancelledError:
            logger.info("slingger_monitor_cancelled", market_id=market_id)
        finally:
            # Central GC anchor
            await self.cleanup_market(market_id, source='slingger_monitor')

    # ── Centralized Garbage Collection (Section 10) ───────────

    async def cleanup_market(self, market_id: str, source: str = 'unknown') -> None:
        """Central GC anchor. Called from all 5 exit paths."""
        # Ensure underlying feed history is cleared
        self._clob.cleanup_market(market_id)
        
        cleaned = []
        for attr in ('_shadow_scalps', '_odds_history', '_post_mortem_tracker'):
            store = getattr(self, attr, {})
            if market_id in store:
                if attr == '_shadow_scalps':
                    result = store[market_id].get('result', 'UNKNOWN')
                    cleaned.append(f'{attr}(result={result})')
                else:
                    cleaned.append(attr)
                del store[market_id]
        
        # [FIX-MEM-1] Add to completed set with bounded size enforcement.
        # Prevents indefinite growth while still blocking duplicate market processing.
        self._completed_markets.add(market_id)
        if len(self._completed_markets) > self._completed_markets_max_size:
            # Remove the oldest-approximate entry (sets are unordered; pop is O(1))
            try:
                self._completed_markets.pop()
            except KeyError:
                pass

        # [FIX-TASK-DEADLOCK] Avoid self-cancellation when cleanup_market is called
        # from within the monitor task's finally block. Previously this caused a 2-second
        # asyncio.wait_for timeout on every market close.
        current_task = asyncio.current_task()
        if market_id in self._active_tasks:
            task = self._active_tasks.pop(market_id)
            if not task.done() and task is not current_task:
                # Only cancel if caller is NOT the task itself
                task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                cleaned.append('_active_tasks')
            elif task is current_task:
                # Self-call: task will end naturally; just remove from registry
                cleaned.append('_active_tasks(self-cleanup)')

        if cleaned:
            logger.debug("cleanup_market", market_id=market_id, source=source,
                         cleared=', '.join(cleaned))

        self._cleanup_count = getattr(self, '_cleanup_count', 0) + 1
        if self._cleanup_count % 100 == 0:
            logger.info("memory_report", cleanup_n=self._cleanup_count,
                        shadow_scalps=len(self._shadow_scalps),
                        active_tasks=len(self._active_tasks),
                        completed_markets_size=len(self._completed_markets))
        
        # Ensure state is saved after cleanup
        asyncio.create_task(self._save_v5_state())

    # ── Memory Audit Loop (tracemalloc) ────────────────────────

    async def _memory_audit_loop(self) -> None:
        """[HOTFIX] Periodic tracemalloc DELTA snapshot for memory leak investigation.
        Runs every 10 minutes, compares to previous snapshot to show what GREW.
        """
        INTERVAL = 600  # 10 minutes
        while self._running:
            await asyncio.sleep(INTERVAL)
            if not self._running:
                break
            try:
                current = tracemalloc.take_snapshot()
                # Filter out importlib and tracemalloc internals
                current = current.filter_traces([
                    tracemalloc.Filter(False, '<frozen importlib._bootstrap>'),
                    tracemalloc.Filter(False, '<frozen importlib._bootstrap_external>'),
                    tracemalloc.Filter(False, tracemalloc.__file__),
                ])

                ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

                if self._last_mem_snapshot is not None:
                    # Delta: what grew since last snapshot (traceback mode
                    # splits same file by call chain, e.g. decoder.py via
                    # binance_feed vs decoder.py via clob_feed)
                    top_stats = current.compare_to(self._last_mem_snapshot, 'traceback')
                    growing = [s for s in top_stats if s.size_diff > 0][:10]

                    print(f"[MEM_AUDIT] ===== DELTA {ts} =====")
                    for rank, stat in enumerate(growing, 1):
                        size_diff_kb = stat.size_diff / 1024
                        frame = stat.traceback[0]
                        fname = f"{frame.filename}:{frame.lineno}"
                        print(f"[MEM_AUDIT] DELTA #{rank}: +{size_diff_kb:.1f} KB | +{stat.count_diff} objects | {fname}")
                        # Inline traceback for entries growing > 50 KB
                        if size_diff_kb > 50.0:
                            for i, tb_frame in enumerate(stat.traceback[:3], 1):
                                print(f"[MEM_AUDIT]   Frame {i}: File \"{tb_frame.filename}\", line {tb_frame.lineno}")

                    print(f"[MEM_AUDIT] ===== END DELTA =====")
                else:
                    # First run — baseline only, no delta available
                    stats = current.statistics('traceback')
                    print(f"[MEM_AUDIT] ===== BASELINE {ts} (next cycle will show delta) =====")
                    for rank, stat in enumerate(stats[:5], 1):
                        frame = stat.traceback[0]
                        fname = f"{frame.filename}:{frame.lineno}"
                        size_kb = stat.size / 1024
                        print(f"[MEM_AUDIT] {rank:>4} | {fname:<28} | {size_kb:>9.1f} KB | {stat.count:>5}")
                    print(f"[MEM_AUDIT] ===== END BASELINE =====")

                self._last_mem_snapshot = current
                sys.stdout.flush()
            except Exception as e:
                print(f"[MEM_AUDIT] ERROR taking snapshot: {e}")
                sys.stdout.flush()


    # ── Daily Summary Scheduler (Section 11) ──────────────────

    async def _daily_summary_loop(self) -> None:
        """Scheduled daily summary (legacy compatibility or secondary trigger)."""
        while self._running:
            now = datetime.now(timezone.utc)
            next_midnight = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0)
            await asyncio.sleep((next_midnight - now).total_seconds())
            if not self._running:
                break
            # Trigger via the new roll helper
            self._roll_session_day_if_needed()

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

    # ── Resilience & Recovery (Section 12) ────────────────────

    async def _save_v5_state(self) -> None:
        """Atomic save of session stats and shadow scalps to SQLite.
        [FIX-STATE-SAVE] Only serializes active (non-CLOSED) scalps to reduce
        JSON size and SQLite write frequency. Previously saved all scalps including
        CLOSED ones, causing JSON payload to grow unboundedly in long sessions.
        [FIX-PIPELINE-1] session_stats capital is now synced from DryRunEngine
        before saving to ensure the persisted value is canonical.
        """
        try:
            from src.database import V5StateRecord
            from sqlalchemy.dialects.sqlite import insert
            
            # [FIX-15] V5 DB is canonical source — no DryRunEngine sync needed
            # Capital is managed exclusively through V5DatabaseManager.close_trade()
            
            # [FIX-STATE-SAVE] Only persist active scalps (WAITING_ENTRY | WAITING_EXIT)
            # CLOSED scalps are already cleaned up by cleanup_market() and don't need persistence
            active_scalps = {
                k: v for k, v in self._shadow_scalps.items()
                if v.get('phase') in ('WAITING_ENTRY', 'WAITING_EXIT')
            }
            
            # Slim session_stats for serialization: exclude full trades list (too large)
            # Only persist capital and totals — trades are reconstructed from DB on restart
            slim_stats = {
                'date':            self._session_stats['date'],
                'total_fees':      self._session_stats['total_fees'],
                'current_capital': self._session_stats['current_capital'],
                # Only keep last 50 trades for daily summary, not full history
                'trades':          self._session_stats['trades'][-50:],
            }
            
            async with self._db.session_factory() as session:
                async with session.begin():
                    # Save Session Stats (slim version)
                    stats_json = json.dumps(slim_stats, default=str)
                    stmt_stats = insert(V5StateRecord).values(
                        key='session_stats', data_json=stats_json
                    ).on_conflict_do_update(
                        index_elements=['key'],
                        set_={'data_json': stats_json, 'updated_at': datetime.now(timezone.utc)}
                    )
                    await session.execute(stmt_stats)
                    
                    # Save only active shadow scalps
                    scalps_json = json.dumps(active_scalps, default=str)
                    stmt_scalps = insert(V5StateRecord).values(
                        key='shadow_scalps', data_json=scalps_json
                    ).on_conflict_do_update(
                        index_elements=['key'],
                        set_={'data_json': scalps_json, 'updated_at': datetime.now(timezone.utc)}
                    )
                    await session.execute(stmt_scalps)
            
            logger.debug("v5_state_saved", 
                         capital=self._session_stats['current_capital'],
                         active_scalps=len(active_scalps),
                         total_scalps=len(self._shadow_scalps))
        except Exception as e:
            logger.error("v5_state_save_failed", error=str(e))

    async def _hydrate_v5_state(self) -> None:
        """Bootstrap hydration — Restore session stats and active trades."""
        try:
            from src.database import V5StateRecord
            from sqlalchemy import select
            
            async with self._db.session_factory() as session:
                # Load Session Stats
                res_stats = await session.execute(select(V5StateRecord).where(V5StateRecord.key == 'session_stats'))
                stats_rec = res_stats.scalar_one_or_none()
                if stats_rec:
                    persisted_stats = json.loads(stats_rec.data_json)
                    # Guardrail 2: Restore Equity & Date Rollover Check
                    self._session_stats['current_capital'] = persisted_stats.get('current_capital', self._session_stats['current_capital'])
                    self._session_stats['total_fees'] = persisted_stats.get('total_fees', 0.0)
                    self._session_stats['trades'] = persisted_stats.get('trades', [])
                    self._session_stats['date'] = persisted_stats.get('date', self._session_stats['date'])
                    logger.info("v5_equity_restored", capital=self._session_stats['current_capital'])

                # Load Shadow Scalps
                res_scalps = await session.execute(select(V5StateRecord).where(V5StateRecord.key == 'shadow_scalps'))
                scalps_rec = res_scalps.scalar_one_or_none()
                if scalps_rec:
                    persisted_scalps = json.loads(scalps_rec.data_json)
                    for m_id, state in persisted_scalps.items():
                        if state.get('phase') in ('WAITING_ENTRY', 'WAITING_EXIT'):
                            # Guardrail 3: Reality Sync will happen in the loop
                            state['is_hydrated'] = True
                            self._shadow_scalps[m_id] = state
                            self._active_tasks[m_id] = asyncio.create_task(
                                self._shadow_scalp_monitor_loop(m_id),
                                name=f"slingger_monitor_hydrated_{m_id[:8]}"
                            )
            
            logger.info("v5_hydration_complete", active_trades=len(self._shadow_scalps))
        except Exception as e:
            logger.error("v5_hydration_failed", error=str(e))


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
    click.echo(f"\nStarting Polymarket Bot - Mode: {mode.upper()}\n")

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
    # [HOTFIX] Start tracemalloc before event loop for memory leak investigation
    tracemalloc.start(10)
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        click.echo("\nBot stopped.")


if __name__ == "__main__":
    main()
