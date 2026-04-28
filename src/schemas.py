"""
schemas.py — All Pydantic v2 models serving as inter-module API contracts.

Every boundary between modules communicates via typed Pydantic models.
No raw dicts are passed across module boundaries.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================
# Dual Feed (Binance + Chainlink via RTDS)
# ============================================================

class DualFeedSnapshot(BaseModel):
    """Point-in-time snapshot of Binance vs Chainlink oracle prices."""

    timestamp: datetime
    binance_price: float = Field(..., description="BTC price from direct Binance WebSocket feed")
    chainlink_price: float = Field(..., description="BTC price from RTDS crypto_prices_chainlink")
    spread: float = Field(..., description="binance_price - chainlink_price (signed)")
    spread_pct: float = Field(
        ...,
        ge=0.0,
        description="abs(spread / chainlink_price) * 100",
    )
    spread_direction: Literal[
        "BINANCE_ABOVE", "CHAINLINK_ABOVE", "CONVERGED"
    ] = Field(..., description="Which feed is higher")
    is_stale: bool = Field(
        default=False,
        description="True if snapshot data is stale. Always False when freshly created by DualFeed.get_snapshot().",
    )
    oracle_source: Optional[Literal["CHAINLINK_LIVE", "CHAINLINK_CACHED", "UNAVAILABLE"]] = Field(
        default="CHAINLINK_LIVE", description="Source of oracle price in this snapshot"
    )


class SpreadFilterResult(BaseModel):
    """Result of the Binance–Chainlink spread filter gate."""

    passed: bool = Field(..., description="True if spread is within acceptable range")
    reason: str = Field(..., description="Human-readable reason for the decision")
    spread_pct: float = Field(..., ge=0.0)
    recommendation: Literal["PROCEED", "SKIP", "WAIT"] = Field(
        ..., description="PROCEED=normal, WAIT=elevated, SKIP=too wide or feed unavailable"
    )


# ============================================================
# Market Discovery
# ============================================================

class ActiveMarket(BaseModel):
    """Represents a currently active Polymarket market."""

    market_id: str = Field(..., description="Polymarket condition_id")
    slug: str = Field(..., description="Polymarket URL slug")
    question: str = Field(..., description="Market question text for verification")
    strike_price: float = Field(
        ...,
        description=(
            "Static strike price from market metadata (parsed from question "
            "text or Gamma API). NOT Binance price at T_open."
        ),
    )
    T_open: datetime = Field(..., description="Market open time (UTC)")
    T_resolution: datetime = Field(..., description="Resolution time (UTC)")
    TTR_minutes: float = Field(..., description="Time-to-resolution in minutes at discovery")
    settlement_exchange: Literal[
        "BINANCE", "PYTH", "COINBASE", "UMA", "UNKNOWN"
    ] = Field(
        default="UNKNOWN",
        description=(
            "Settlement oracle/exchange source for the binary outcome (e.g. 'BINANCE'). "
            "Trading alpha must only be taken when probability is computed against the "
            "same settlement source."
        ),
    )
    settlement_instrument: Optional[str] = Field(
        default=None,
        description=(
            "Settlement instrument identifier (e.g. 'BTCUSDT'), if known. "
            "Used for fetching the correct settlement price."
        ),
    )
    settlement_granularity: Literal["1m", "unknown"] = Field(
        default="unknown",
        description=(
            "Settlement candle granularity for exchange-based settlement (e.g. '1m')."
        ),
    )
    settlement_price_type: Literal["close", "vwap", "unknown"] = Field(
        default="unknown",
        description=(
            "Settlement price type derived from the exchange candle (e.g. 'close', 'vwap')."
        ),
    )
    clob_token_ids: Dict[str, str] = Field(
        ..., description='{"YES": "0x...", "NO": "0x..."}'
    )
    resolution_source: Optional[str] = Field(
        default=None,
        description=(
            "Oracle source for resolution (e.g. 'Pyth', 'Coinbase', 'CoinGecko'). "
            "Parsed from market rules. Used to assess basis risk."
        ),
    )
    strike_price_source: Optional[str] = Field(
        default=None,
        description=(
            "Source of the strike price (e.g. 'VATIC_WS_LIVE', 'VATIC_WS_SNAPSHOT', "
            "'CHAINLINK_FIRST_TICK', 'CHAINLINK_HISTORICAL_HTTP', 'POLYMARKET_PAYLOAD'). "
            "Set by VaticFeed callback or Chainlink fallback resolution."
        ),
    )

    @field_validator("strike_price")
    @classmethod
    def strike_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"strike_price must be positive, got {v}")
        return v


# ============================================================
# CLOB Feed
# ============================================================

class CLOBState(BaseModel):
    """Snapshot of Polymarket CLOB orderbook state."""

    market_id: str
    timestamp: datetime
    yes_ask: float = Field(..., ge=0.0, le=1.0, description="Best ask YES [0,1]")
    yes_bid: float = Field(..., ge=0.0, le=1.0, description="Best bid YES [0,1]")
    no_ask: float = Field(..., ge=0.0, le=1.0, description="Best ask NO [0,1]")
    no_bid: float = Field(..., ge=0.0, le=1.0, description="Best bid NO [0,1]")
    yes_depth_usd: float = Field(..., ge=0.0, description="Total USDC depth YES within 3% of ask")
    no_depth_usd: float = Field(..., ge=0.0, description="Total USDC depth NO within 3% of ask")
    market_vig: float = Field(..., description="yes_ask + no_ask - 1.0")
    is_liquid: bool = Field(..., description="Meets minimum depth & vig requirements")
    is_stale: bool = Field(default=False, description="Data older than STALE_TIMEOUT_S")


# ============================================================
# Feature Engine
# ============================================================

class FeatureVector(BaseModel):
    """24-feature vector output from feature engine."""

    values: List[float] = Field(..., min_length=24, max_length=24)
    feature_names: List[str] = Field(..., min_length=24, max_length=24)
    metadata: FeatureMetadata

    class Config:
        arbitrary_types_allowed = True


class FeatureMetadata(BaseModel):
    """Metadata attached to every feature vector."""

    timestamp: datetime
    bar_close_time: datetime
    market_id: str
    strike_price: float
    current_btc_price: float
    TTR_minutes: float
    TTR_phase: Literal["EARLY", "ENTRY_WINDOW", "LATE"]
    compute_lag_ms: float = Field(default=0.0, description="Feature computation latency")


# ============================================================
# Signal Generator
# ============================================================

class SignalResult(BaseModel):
    """Output of signal evaluation pipeline."""

    signal: Literal["BUY_UP", "BUY_DOWN", "ABSTAIN"]
    abstain_reason: Optional[
        Literal[
            "TTR_PHASE",
            "REGIME_BLOCK",
            "LIQUIDITY_BLOCK",
            "NO_EDGE",
            "NO_TRADE_ZONE",
            "BASIS_RISK_BLOCK",
            "BASIS_RISK_INFLATED",
            "SPREAD_FILTER_SKIP",
            "SPREAD_FILTER_WAIT",
            "ORACLE_UNAVAILABLE",
            "CONTAMINATED_FALLBACK_ODDS",
            "BLOCKED_ONE_BET",
        ]
    ] = None
    P_model: float = Field(..., ge=0.0, le=1.0)
    uncertainty_u: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Probability-point uncertainty buffer used for conservative edge and sizing",
    )
    edge_yes: Optional[float] = None
    edge_no: Optional[float] = None
    clob_yes_bid: Optional[float] = None
    clob_yes_ask: Optional[float] = None
    clob_no_bid: Optional[float] = None
    clob_no_ask: Optional[float] = None
    TTR_minutes: float
    strike_price: float
    current_price: float
    strike_distance: float
    market_id: str
    timestamp: datetime
    features: Dict[str, float] = Field(
        default_factory=dict, description="Full feature snapshot for logging"
    )
    live_yes_ask: Optional[float] = None
    live_no_ask: Optional[float] = None
    synthetic_edge: Optional[float] = None
    live_edge: Optional[float] = None

    # ── Dual Feed / Spread Filter fields ─────────────────────
    binance_price_at_signal: Optional[float] = Field(
        default=None, description="Binance BTC price at signal time (from RTDS)"
    )
    chainlink_price_at_signal: Optional[float] = Field(
        default=None, description="Chainlink oracle BTC price at signal time (from RTDS)"
    )
    spread_pct_at_signal: Optional[float] = Field(
        default=None, description="Binance–Chainlink spread % at signal time"
    )
    oracle_vs_binance_at_entry: Optional[float] = Field(
        default=None, description="Absolute USD difference: |chainlink - binance| at entry"
    )
    spread_filter_passed: Optional[bool] = Field(
        default=None, description="Whether spread filter passed for this signal"
    )
    spread_filter_reason: Optional[str] = Field(
        default=None, description="Reason from spread filter decision"
    )
    strike_price_source: Optional[Literal["CLOB", "GAMMA", "BINANCE"]] = Field(
        default=None, description="Source of strike price — must be CLOB or GAMMA after fix"
    )
    odds_source: Optional[Literal["CLOB", "ESTIMATED"]] = Field(
        default=None, description="Source of odds — must be CLOB after fix"
    )
    entry_odds_source: Optional[str] = Field(
        default=None, description="Source of entry odds: CLOB_LIVE or DEFAULT_FALLBACK"
    )
    oracle_source: Optional[Literal["CHAINLINK_LIVE", "CHAINLINK_CACHED", "UNAVAILABLE"]] = Field(
        default=None, description="Source of oracle price: LIVE, CACHED (degraded mode), or UNAVAILABLE"
    )


# ============================================================
# Risk Manager
# ============================================================

class ApprovedBet(BaseModel):
    """Trade approved by risk manager."""

    signal: SignalResult
    bet_size: float = Field(..., gt=0.0, description="USDC amount to bet")
    kelly_fraction: float = Field(..., ge=0.0)
    kelly_multiplier: float = Field(..., ge=0.0, le=1.0)


class RejectedBet(BaseModel):
    """Trade rejected by risk manager."""

    signal: SignalResult
    reason: Literal[
        "DAILY_LOSS_LIMIT_HIT",
        "SESSION_LOSS_LIMIT_HIT",
        "MAX_POSITIONS_REACHED",
        "CAPITAL_BELOW_FLOOR",
        "BET_BELOW_MINIMUM",
    ]


# ============================================================
# Trade Outcome
# ============================================================

class TradeOutcome(BaseModel):
    """Resolution outcome of a completed trade."""

    trade_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    outcome: Literal["WIN", "LOSS"]
    btc_at_resolution: float
    pnl_usd: float
    pnl_pct_capital: float = 0.0


class PaperTrade(BaseModel):
    """Paper trade record for dry run engine."""

    trade_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    market_id: str
    signal_type: Literal["BUY_UP", "BUY_DOWN"]
    entry_price: float
    bet_size: float
    strike_price: float
    T_resolution: datetime
    TTR_at_entry: float
    P_model: float
    edge_yes: float
    edge_no: float
    kelly_fraction: float
    kelly_multiplier: float
    capital_before: float
    synthetic_edge: Optional[float] = None
    live_edge: Optional[float] = None
    btc_price_at_trigger: Optional[float] = None
    btc_distance_to_strike: Optional[float] = None
    trigger_timestamp: Optional[datetime] = None

    # ── Dual Feed tracking ────────────────────────────────────
    binance_price_at_signal: Optional[float] = None
    chainlink_price_at_signal: Optional[float] = None
    spread_pct_at_signal: Optional[float] = None
    oracle_vs_binance_at_entry: Optional[float] = None
    spread_filter_passed: Optional[bool] = None
    spread_filter_reason: Optional[str] = None
    strike_price_source: Optional[str] = None
    odds_source: Optional[str] = None
    oracle_source: Optional[str] = None

    # Filled post-resolution
    btc_at_resolution: Optional[float] = None
    outcome: Optional[Literal["WIN", "LOSS", "PENDING"]] = "PENDING"
    pnl_usd: Optional[float] = None
    pnl_pct_capital: Optional[float] = None
    capital_after: Optional[float] = None
    timestamp_signal: datetime = Field(default_factory=datetime.utcnow)
    timestamp_resolution: Optional[datetime] = None


# ============================================================
# Session Metrics & Performance
# ============================================================

class SessionMetrics(BaseModel):
    """Aggregated metrics for a trading session."""

    session_id: str
    date_utc: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_hours: float = 0.0
    mode: Literal["DRY", "LIVE"]
    total_bars_processed: int = 0
    total_signals_evaluated: int = 0
    signals_abstained: int = 0
    abstain_regime: int = 0
    abstain_liquidity: int = 0
    abstain_ttr: int = 0
    abstain_no_edge: int = 0
    trades_executed: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl_usd: float = 0.0
    total_pnl_pct_capital: float = 0.0
    max_drawdown: float = 0.0
    sharpe_rolling: float = 0.0
    sortino_rolling: float = 0.0
    brier_score: float = 0.0
    mean_edge_yes_traded: float = 0.0
    mean_edge_no_traded: float = 0.0
    mean_ttr_at_signal: float = 0.0
    mean_strike_distance: float = 0.0
    capital_start: float = 0.0
    capital_end: float = 0.0
    dry_run_score: float = 0.0
    pass_fail: Optional[Literal["PASS", "FAIL"]] = None
    model_version: str = ""
    ws_drop_rate_pct: float = 0.0
    clob_stale_events: int = 0
    consecutive_losses: int = 0


# ============================================================
# Observability
# ============================================================

class WSHealthMetrics(BaseModel):
    """WebSocket connection health metrics."""

    messages_received: int = 0
    messages_expected: int = 0
    messages_dropped: int = 0
    drop_rate: float = 0.0
    reconnect_count: int = 0
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0
    queue_depth_max: int = 0
    last_message_timestamp: Optional[datetime] = None


class StalenessReport(BaseModel):
    """Data staleness status across all feeds."""

    binance_stale: bool = False
    clob_stale: bool = False
    market_stale: bool = False

    @property
    def any_stale(self) -> bool:
        return self.binance_stale or self.clob_stale or self.market_stale


# ============================================================
# Execution
# ============================================================

class FillResult(BaseModel):
    """Result of order fill monitoring."""

    status: Literal[
        "FILLED",
        "PARTIALLY_FILLED",
        "FAILED",
        "TIMEOUT_CANCELLED",
    ]
    fill_price: Optional[float] = None
    filled_size: Optional[float] = None
    remaining_size: Optional[float] = None
    reason: Optional[str] = None
    order_id: Optional[str] = None


class OrderRejected(BaseModel):
    """Pre-order validation rejection."""

    reason: str