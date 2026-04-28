"""
execution.py — Live CLOB order placement (LIVE_MODE gated).

Protected by triple-gate:
  1. LIVE_MODE environment variable == "true"
  2. CLI flag --confirm-live
  3. Interactive confirmation prompt

In dry-run mode, this module is NEVER called for order placement.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import structlog

from src.config_manager import ConfigManager
from src.schemas import (
    ActiveMarket,
    ApprovedBet,
    FillResult,
    OrderRejected,
    TradeOutcome,
)

logger = structlog.get_logger(__name__)


class ExecutionClient:
    """
    Live order execution via py-clob-client SDK.

    LIVE_MODE must be explicitly enabled via env + CLI + interactive prompt.
    Default: all operations return rejection / dry-run response.
    """

    ORDER_POLL_INTERVAL_S = 5
    ORDER_TIMEOUT_S = 60

    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        self._live_mode = os.getenv("LIVE_MODE", "false").lower() == "true"
        self._confirmed = False
        self._clob_client = None

    # ── Live Mode Gating ──────────────────────────────────────

    @property
    def is_live(self) -> bool:
        return self._live_mode and self._confirmed

    def confirm_live(self, cli_flag: bool = False) -> bool:
        """
        Triple-gate confirmation for live trading.

        Requirements:
          1. LIVE_MODE env == "true"
          2. cli_flag == True (--confirm-live)
          3. Interactive confirmation (in non-Railway environments)
        """
        if not self._live_mode:
            logger.info("live_mode_disabled_env")
            return False

        if not cli_flag:
            logger.info("live_mode_disabled_no_cli_flag")
            return False

        # Interactive prompt (only for terminal)
        if os.isatty(0):
            print("\n" + "=" * 60)
            print("⚠️  LIVE TRADING MODE ACTIVATION")
            print("=" * 60)
            print("Anda akan mengeksekusi order NYATA dengan uang sungguhan.")
            print("Semua kerugian adalah PERMANEN dan TIDAK DAPAT DIBATALKAN.")
            print("=" * 60)
            response = input("Ketik 'CONFIRM-LIVE-TRADING' untuk melanjutkan: ").strip()
            if response != "CONFIRM-LIVE-TRADING":
                logger.info("live_mode_rejected_by_user")
                return False

        self._confirmed = True
        self._initialize_client()
        logger.critical("LIVE_MODE_ACTIVATED")
        return True

    def _initialize_client(self) -> None:
        """
        Initialize py-clob-client SDK.

        AUTHENTICATION ARCHITECTURE (V2 Migration):
          - CLOB trading credentials are DERIVED from wallet private key.
          - There is NO separate API key/secret for CLOB trading.
          - py-clob-client-v2 derive_api_key() generates CLOB creds
            deterministically from the wallet's private key signature.
          - V2: POLY_BUILDER_* env vars are FULLY DEPRECATED (removed).
            Builder attribution now uses builderCode (bytes32) per order.
          - V2: Collateral is pUSD (not USDC.e). Wrap via Collateral Onramp.
        """
        try:
            # Lazy import — only needed for live mode
            from py_clob_client.client import ClobClient

            private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")

            if not private_key:
                logger.error(
                    "missing_polymarket_private_key",
                    hint="Set POLYMARKET_PRIVATE_KEY in .env (Polygon wallet private key)",
                )
                self._confirmed = False
                return

            # Step 1: Create temporary client with just the private key
            temp_client = ClobClient(
                host="https://clob.polymarket.com",
                chain=137,  # Polygon mainnet
                key=private_key,
            )

            # Step 2: Derive CLOB API credentials from wallet signature
            # This is deterministic — same private key always produces
            # the same API credentials. No need to store them separately.
            api_creds = temp_client.derive_api_key()
            logger.info("clob_api_creds_derived")

            # Step 3: Initialize the full trading client
            self._clob_client = ClobClient(
                host="https://clob.polymarket.com",
                chain=137,
                key=private_key,
                creds=api_creds,
            )

            # V2_MIGRATION: Startup check for API key invalidation
            try:
                self._clob_client.get_order_book(token_id="test")
            except Exception as e:
                if "401" in str(e) or "Unauthorized" in str(e):
                    logger.critical(
                        "v2_auth_failed_regenerate_api_keys",
                        error=str(e),
                        action="Login to Polymarket UI after 12:00 UTC April 28, regenerate L2 API keys"
                    )
                    raise SystemExit(1)

            logger.info("clob_client_initialized", signature_type="EOA")

        except ImportError:
            logger.error(
                "py_clob_client_not_installed",
                fix="pip install py-clob-client-v2",
            )
            self._confirmed = False
        except Exception as e:
            logger.error("clob_client_init_error", error=str(e))
            self._confirmed = False

    # ── Order Placement ───────────────────────────────────────

    async def place_order(
        self,
        approved_bet: ApprovedBet,
        active_market: ActiveMarket,
    ) -> FillResult | OrderRejected:
        """
        Place a limit order on Polymarket CLOB.

        Only executes in LIVE mode. Returns rejection otherwise.
        """
        if not self.is_live:
            return OrderRejected(reason="LIVE_MODE_NOT_ACTIVE")

        if not self._clob_client:
            return OrderRejected(reason="CLOB_CLIENT_NOT_INITIALIZED")

        # Pre-order validation
        if active_market.TTR_minutes < 3.0:
            return OrderRejected(reason="TTR_TOO_LOW_FOR_FILL")

        signal = approved_bet.signal

        # Determine token
        token_key = "YES" if signal.signal == "BUY_UP" else "NO"
        token_id = active_market.clob_token_ids.get(token_key, "")

        if not token_id:
            return OrderRejected(reason="MISSING_TOKEN_ID")

        # ── Maker vs taker-like pricing ────────────────────────
        if signal.signal == "BUY_UP":
            clob_ask = signal.clob_yes_ask
            clob_bid = signal.clob_yes_bid
            edge = signal.edge_yes
        else:
            clob_ask = signal.clob_no_ask
            clob_bid = signal.clob_no_bid
            edge = signal.edge_no

        taker_edge_threshold = float(self._config.get("execution.taker_edge_threshold", 0.08))
        taker_price_buffer = float(self._config.get("execution.taker_price_buffer", 0.002))
        maker_price_step = float(self._config.get("execution.maker_price_step", 0.0005))
        maker_price_epsilon = float(self._config.get("execution.maker_price_epsilon", 0.0001))

        spread = clob_ask - clob_bid
        use_taker_like = edge >= taker_edge_threshold and spread > 0

        if use_taker_like:
            # Cross more aggressively only when edge is strong.
            order_price = round(clob_ask + taker_price_buffer, 4)
            exec_mode = "TAKER_LIKE"
        else:
            # Post passively: keep limit price below ask when possible.
            candidate = clob_bid + maker_price_step
            candidate = min(candidate, clob_ask - maker_price_epsilon)
            if candidate <= 0:
                candidate = clob_bid
            order_price = round(candidate, 4)
            exec_mode = "MAKER_POST"

        try:
            logger.info(
                "placing_live_order",
                token_id=token_id[:16],
                price=order_price,
                size=approved_bet.bet_size,
                side="BUY",
                signal=signal.signal,
                exec_mode=exec_mode,
                clob_bid=clob_bid,
                clob_ask=clob_ask,
                edge=round(edge, 6),
            )

            # TODO: V2_MIGRATION — re-enable after dry-run confirmed working
            # V2 API: feeRateBps, nonce, taker are REMOVED.
            # Optional: builderCode for builder attribution.
            # Collateral: pUSD (not USDC.e). Ensure wrap() before first trade.
            # order = self._clob_client.create_and_post_order(
            #     token_id=token_id,
            #     price=order_price,
            #     size=approved_bet.bet_size,
            #     side="BUY",
            # )
            # order_id = order.get("id") or order.get("orderID", "")
            logger.info("V2_MIGRATION: write/execute function disabled for dry-run validation")
            order = {"id": "mock_id_v2_migration"}
            order_id = "mock_id_v2_migration"

            if order_id:
                return await self._monitor_fill(order_id)
            else:
                logger.error("order_no_id_returned", response=str(order)[:200])
                return FillResult(status="FAILED", reason="NO_ORDER_ID")

        except Exception as e:
            logger.error("order_placement_error", error=str(e))
            return FillResult(status="FAILED", reason=str(e))

    async def _monitor_fill(self, order_id: str) -> FillResult:
        """Monitor order fill status with timeout."""
        import time

        start = time.time()
        last_avg_price: Optional[float] = None
        last_filled_size: Optional[float] = None
        last_remaining_size: Optional[float] = None

        while (time.time() - start) < self.ORDER_TIMEOUT_S:
            try:
                order_status = self._clob_client.get_order(order_id)
                status = order_status.get("status", "").upper()

                if status == "FILLED":
                    fill_price = float(order_status.get("avg_price", order_status.get("averagePrice", 0)) or 0)
                    filled_size = self._coerce_float(
                        order_status.get("executed_size")
                        or order_status.get("filled_size")
                        or order_status.get("executedSize")
                        or order_status.get("filledSize")
                        or 0
                    )
                    logger.info("order_filled", order_id=order_id, fill_price=fill_price)
                    return FillResult(
                        status="FILLED",
                        fill_price=fill_price,
                        filled_size=filled_size,
                        remaining_size=0.0,
                        order_id=order_id,
                    )
                if status in ("PARTIALLY_FILLED", "PARTIALLYFILLED"):
                    last_avg_price = float(
                        order_status.get("avg_price", order_status.get("averagePrice", 0)) or 0
                    )
                    last_filled_size = self._coerce_float(
                        order_status.get("executed_size")
                        or order_status.get("filled_size")
                        or order_status.get("executedSize")
                        or order_status.get("filledSize")
                        or 0
                    )
                    last_remaining_size = self._coerce_float(
                        order_status.get("remaining_size")
                        or order_status.get("remainingSize")
                        or 0
                    )
                    # Keep monitoring until filled or timeout/cancel.
                elif status in ("CANCELLED", "REJECTED"):
                    # If we have a partial fill history, report partial instead of total failure.
                    if last_filled_size is not None and last_filled_size > 0:
                        return FillResult(
                            status="PARTIALLY_FILLED",
                            fill_price=last_avg_price,
                            filled_size=last_filled_size,
                            remaining_size=last_remaining_size,
                            reason=status,
                            order_id=order_id,
                        )
                    return FillResult(
                        status="FAILED", reason=status, order_id=order_id
                    )

            except Exception as e:
                logger.warning("fill_monitor_error", error=str(e))

            await asyncio.sleep(self.ORDER_POLL_INTERVAL_S)

        # Timeout — cancel order
        try:
            self._clob_client.cancel_order(order_id)
            logger.warning("order_timeout_cancelled", order_id=order_id)
        except Exception as e:
            logger.error("cancel_error", order_id=order_id, error=str(e))

        if last_filled_size is not None and last_filled_size > 0:
            return FillResult(
                status="PARTIALLY_FILLED",
                fill_price=last_avg_price,
                filled_size=last_filled_size,
                remaining_size=last_remaining_size,
                reason="TIMEOUT_CANCELLED",
                order_id=order_id,
            )
        return FillResult(status="TIMEOUT_CANCELLED", order_id=order_id)

    @staticmethod
    def _coerce_float(v: object) -> Optional[float]:
        try:
            if v is None:
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    # ── Post-Resolution ───────────────────────────────────────

    async def redeem_positions(self, market_id: str) -> bool:
        """Redeem winning positions after market resolution."""
        if not self.is_live or not self._clob_client:
            return False

        try:
            self._clob_client.redeem_positions(condition_id=market_id)
            logger.info("positions_redeemed", market_id=market_id)
            return True
        except Exception as e:
            logger.error("redeem_error", market_id=market_id, error=str(e))
            return False
