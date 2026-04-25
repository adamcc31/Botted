"""
database.py — SQLAlchemy 2.x async ORM with auto SQLite/PostgreSQL switching.

If DATABASE_URL env var is set → PostgreSQL (asyncpg).
Otherwise → SQLite (aiosqlite) at ./data/trading.db.

Tables: markets, signals, trades, performance, model_versions, system_health.
"""

from __future__ import annotations

import os
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import structlog
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logger = structlog.get_logger(__name__)

# ============================================================
# Engine Factory
# ============================================================

_DATA_DIR = Path(__file__).parent.parent / "data"


def get_database_url() -> str:
    """Determine database URL from environment."""
    url = os.getenv("DATABASE_URL")
    if url:
        # Convert postgres:// to postgresql+asyncpg://
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        logger.info("database_using_postgresql", url=url[:30] + "...")
        return url
    else:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        db_path = _DATA_DIR / "trading.db"
        sqlite_url = f"sqlite+aiosqlite:///{db_path}"
        logger.info("database_using_sqlite", path=str(db_path))
        return sqlite_url


def _engine_kwargs(url: str) -> dict:
    """Engine configuration based on backend."""
    if "sqlite" in url:
        return {"echo": False}
    return {
        "echo": False,
        "pool_size": 5,
        "max_overflow": 10,
        "pool_pre_ping": True,
    }


# ============================================================
# Base Model
# ============================================================


class Base(DeclarativeBase):
    pass


# ============================================================
# Table Definitions
# ============================================================


class MarketRecord(Base):
    """Markets table — Polymarket market lifecycle."""

    __tablename__ = "markets"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    strike_price: Mapped[float] = mapped_column(Float, nullable=False)
    t_open: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    t_resolution: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    clob_token_yes: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    clob_token_no: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(
        String, default="DISCOVERED"
    )  # DISCOVERED | ACTIVE | RESOLVED
    btc_at_resolution: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    resolution_source: Mapped[Optional[str]] = mapped_column(
        String, nullable=True
    )  # Pyth | Coinbase | CoinGecko
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_markets_t_resolution", "t_resolution"),
        Index("idx_markets_status", "status"),
    )


class SignalRecord(Base):
    """Signals table — every signal evaluation (including ABSTAIN)."""

    __tablename__ = "signals"

    signal_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    market_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    timestamp_utc: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    signal_type: Mapped[str] = mapped_column(
        String, nullable=False
    )  # BUY_UP | BUY_DOWN | ABSTAIN
    abstain_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    p_model: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    edge_yes: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    edge_no: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clob_yes_ask: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clob_no_ask: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ttr_minutes: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    strike_distance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vol_percentile: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    obi_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tfm_norm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    market_vig: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # ── ML Features ──
    rv_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    depth_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    strike_distance_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    contest_urgency: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_yes_60s_ago: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_delta_60s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    btc_return_1m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence_bucket: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    entry_odds_source: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    
    # ── New Tracking Columns ──
    slug: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    strike_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    binance_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    chainlink_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spread_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spread_filter_passed: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    spread_filter_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    entry_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    theoretical_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    signal_correct: Mapped[Optional[str]] = mapped_column(String, default="PENDING")
    actual_outcome: Mapped[Optional[str]] = mapped_column(String, default="PENDING")
    settlement_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    settlement_price_source: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    model_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    mode: Mapped[str] = mapped_column(String, nullable=False)  # DRY | LIVE

    __table_args__ = (
        Index("idx_signals_timestamp", "timestamp_utc", "market_id"),
        Index("idx_signals_session", "session_id"),
        Index("idx_signals_signal_type", "signal_type"),
    )


class TradeRecord(Base):
    """Trades table — executed trades (dry or live)."""

    __tablename__ = "trades"

    trade_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    signal_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    market_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    timestamp_signal: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    timestamp_entry: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    timestamp_resolution: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )
    signal_type: Mapped[str] = mapped_column(String, nullable=False)
    p_model: Mapped[float] = mapped_column(Float, nullable=False)
    clob_yes_ask: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clob_no_ask: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    edge_yes: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    edge_no: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    bet_size_usd: Mapped[float] = mapped_column(Float, nullable=False)
    kelly_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    kelly_multiplier: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    strike_price: Mapped[float] = mapped_column(Float, nullable=False)
    btc_at_signal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    strike_distance_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ttr_minutes: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    btc_at_resolution: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    outcome: Mapped[Optional[str]] = mapped_column(
        String, nullable=True
    )  # WIN | LOSS | PENDING
    pnl_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl_pct_capital: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    capital_before: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    capital_after: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vol_percentile: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    obi_at_signal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tfm_norm_at_signal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    market_vig: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    mode: Mapped[str] = mapped_column(String, nullable=False)  # DRY | LIVE

    __table_args__ = (
        Index("idx_trades_session", "session_id", "timestamp_signal"),
        Index("idx_trades_market", "market_id"),
        Index("idx_trades_outcome", "outcome"),
        Index("idx_trades_mode_session", "mode", "session_id"),
    )


class PerformanceRecord(Base):
    """Performance table — session-level aggregated metrics."""

    __tablename__ = "performance"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    date_utc: Mapped[date] = mapped_column(Date, nullable=False)
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    duration_hours: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mode: Mapped[str] = mapped_column(String, nullable=False)
    total_bars_processed: Mapped[int] = mapped_column(Integer, default=0)
    total_signals_evaluated: Mapped[int] = mapped_column(Integer, default=0)
    signals_abstained: Mapped[int] = mapped_column(Integer, default=0)
    abstain_regime: Mapped[int] = mapped_column(Integer, default=0)
    abstain_liquidity: Mapped[int] = mapped_column(Integer, default=0)
    abstain_ttr: Mapped[int] = mapped_column(Integer, default=0)
    abstain_no_edge: Mapped[int] = mapped_column(Integer, default=0)
    trades_executed: Mapped[int] = mapped_column(Integer, default=0)
    win_count: Mapped[int] = mapped_column(Integer, default=0)
    loss_count: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    profit_factor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_pnl_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_pnl_pct_capital: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    max_drawdown: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sharpe_rolling: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sortino_rolling: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    brier_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mean_edge_yes_traded: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    mean_edge_no_traded: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    mean_ttr_at_signal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mean_strike_distance: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    capital_start: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    capital_end: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dry_run_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pass_fail: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    ws_drop_rate_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clob_stale_events: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("idx_perf_date", "date_utc"),
        Index("idx_perf_passfail", "pass_fail"),
    )


class ModelVersionRecord(Base):
    """Model versions table — ML model lifecycle tracking."""

    __tablename__ = "model_versions"

    version_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    brier_score_oos: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    auc_oos: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    win_rate_oos: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    training_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    retrain_trigger: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    file_path_lgbm: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    file_path_logreg: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    file_path_scaler: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)


class SystemHealthRecord(Base):
    """System health table — periodic telemetry snapshots."""

    __tablename__ = "system_health"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    timestamp_utc: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ws_drop_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ws_latency_p99_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clob_fetch_latency_ms: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    clob_is_stale: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    queue_depth: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    market_id_active: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("idx_health_session", "session_id", "timestamp_utc"),
    )


# ============================================================
# Database Manager
# ============================================================


class DatabaseManager:
    """Async database manager with session factory."""

    def __init__(self) -> None:
        self._url = get_database_url()
        self._engine = create_async_engine(self._url, **_engine_kwargs(self._url))
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self) -> None:
        """Create all tables if they don't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("database_initialized", url=self._url[:40])

    async def get_session(self) -> AsyncSession:
        """Get a new async session."""
        return self._session_factory()

    async def close(self) -> None:
        """Dispose engine connections."""
        await self._engine.dispose()
        logger.info("database_closed")

    @property
    def engine(self):
        return self._engine

    @property
    def session_factory(self):
        return self._session_factory
