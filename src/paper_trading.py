"""
paper_trading.py — High-fidelity Paper Trading Engine for Predator V4.

Simulates the full live trade lifecycle (orderbook fetch, fill logic, resolution)
without sending real transactions to Polygon. Records high-quality data
identical to live trades for future model training.
"""

import os
import uuid
import asyncio
import csv
import pandas as pd
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, fields
from typing import Dict, Optional, List
import logging

try:
    import structlog  # type: ignore
except ModuleNotFoundError:
    structlog = None

logger = structlog.get_logger(__name__) if structlog else logging.getLogger(__name__)

@dataclass
class PaperTradeRecord:
    # Entry data
    trade_id: str
    timestamp_signal: str
    market_id: str
    zone_id: str
    signal_direction: str
    
    # Orderbook snapshot at execution
    timestamp_execution: str
    yes_bid_at_exec: float
    yes_ask_at_exec: float
    no_bid_at_exec: float
    no_ask_at_exec: float
    spread_bps_at_exec: float
    
    # Simulated fill
    simulated_fill_price: float
    bet_size_usd: float
    
    # Edge calculation
    p_model: float
    synthetic_edge: float
    live_edge: float
    uncertainty_u: float
    
    # Metadata
    ttr_minutes: float
    distance_usd: float
    entry_odds: float
    data_source: str = "paper_v4"
    kelly_fraction_used: float = 0.0
    kelly_cap_used: float = 0.0

    # Resolution (filled later)
    timestamp_resolution: Optional[str] = None
    resolution_price: Optional[float] = None
    actual_outcome: str = "PENDING"
    actual_pnl_usd: float = 0.0

class PaperTradingEngine:
    def __init__(self, config, clob_client):
        self.config = config
        self.clob = clob_client
        self.open_positions: Dict[str, PaperTradeRecord] = {}
        self.output_dir = "app/data/exports/paper_trades"
        self.slippage_log_path = os.path.join(self.output_dir, "slippage_log.csv")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # O(1) Buffers for high-performance I/O
        self._trade_buffer: List[PaperTradeRecord] = []
        self._summary_cache: Dict[str, dict] = {}
        self._load_summary_cache()

    def _load_summary_cache(self):
        """Load existing summary into memory to avoid R/W overhead in loop."""
        summary_path = os.path.join(self.output_dir, "paper_trades_summary.csv")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, mode='r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        date_str = row['date']
                        self._summary_cache[date_str] = {
                            "date": date_str,
                            "total_trades": int(row['total_trades']),
                            "wins": int(row['wins']),
                            "losses": int(row['losses']),
                            "total_pnl": float(row['total_pnl']),
                            "win_rate": float(row['win_rate'])
                        }
            except Exception as e:
                logger.error("paper_trade_load_summary_failed", error=str(e))

    def _get_daily_file(self) -> str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(self.output_dir, f"paper_trades_{date_str}.csv")

    def _validate_record(self, record: PaperTradeRecord) -> bool:
        try:
            assert 0 < record.simulated_fill_price < 1.0
            assert record.bet_size_usd > 0
            assert record.live_edge > 0.01  # relaxed from 0.02 to catch marginals
            assert record.zone_id.startswith("V4-")
            assert record.data_source == "paper_v4"
            return True
        except Exception as e:
            import traceback as tb
            logger.error("paper_trade_validation_failed",
                error_type=type(e).__name__,
                error_repr=repr(e),
                traceback_last_line=tb.format_exc().strip().split('\n')[-1],
                full_traceback=tb.format_exc(),
                trade_id=record.trade_id
            )
            return False

    async def execute_paper_trade(
        self, signal, zone_result, bet_size, market
    ) -> Optional[PaperTradeRecord]:
        """Simulate entry by fetching fresh orderbook and verifying live edge."""
        
        # 1. Fetch fresh orderbook
        try:
            fresh_clob = await self.clob.fetch_clob_snapshot(market)
        except Exception as e:
            logger.error("paper_trade_fetch_clob_failed", error=str(e))
            return None

        if not fresh_clob:
            return None

        # 2. Determine fill price
        fill_price = fresh_clob.yes_ask if signal.signal == "BUY_UP" else fresh_clob.no_ask
        
        # 3. Check live edge again
        current_live_edge = signal.P_model - signal.uncertainty_u - fill_price
        
        margin_of_safety = float(self.config.get("signal.margin_of_safety", 0.02))
        
        if current_live_edge <= margin_of_safety:
            self._log_slippage(signal, fresh_clob, current_live_edge)
            return None

        # 4. Create record
        record = PaperTradeRecord(
            trade_id=str(uuid.uuid4()),
            timestamp_signal=signal.timestamp.isoformat(),
            market_id=signal.market_id,
            zone_id=zone_result.zone_id,
            signal_direction=signal.signal,
            timestamp_execution=datetime.now(timezone.utc).isoformat(),
            yes_bid_at_exec=fresh_clob.yes_bid,
            yes_ask_at_exec=fresh_clob.yes_ask,
            no_bid_at_exec=fresh_clob.no_bid,
            no_ask_at_exec=fresh_clob.no_ask,
            spread_bps_at_exec=fresh_clob.market_vig * 10000,
            simulated_fill_price=fill_price,
            bet_size_usd=bet_size,
            p_model=signal.P_model,
            live_edge=current_live_edge,
            synthetic_edge=signal.synthetic_edge,
            uncertainty_u=signal.uncertainty_u,
            ttr_minutes=signal.TTR_minutes,
            distance_usd=signal.strike_distance,
            entry_odds=fill_price,
            kelly_fraction_used=zone_result.kelly_fraction,
            kelly_cap_used=zone_result.kelly_cap
        )

        if not self._validate_record(record):
            return None

        # 5. Track and Save
        self.open_positions[signal.market_id] = record
        self._append_to_csv(self._get_daily_file(), record)
        
        logger.info("paper_trade_executed", 
                    trade_id=record.trade_id, 
                    market_id=record.market_id, 
                    fill=round(fill_price, 4), 
                    edge=round(current_live_edge, 4))
        
        return record

    async def check_resolution(self, market_id: str, settlement_price: float) -> None:
        """Update record with actual outcome when market resolves."""
        if market_id not in self.open_positions:
            return
            
        record = self.open_positions[market_id]
        
        # Determine won/lost
        # Note: logic must match Predator resolution
        # Usually BUY_UP wins if settlement_price >= strike_price
        # In PaperTradeRecord we don't have strike_price, let's assume market resolution logic handled externally
        # and we are passed the result. 
        # Actually, let's just pass the 'actual_outcome' string (BUY_UP or BUY_DOWN)
        
        # Wait, the caller should pass the winner
        pass # Placeholder for logic below

    def resolve_position(self, market_id: str, won: bool, settlement_price: float, current_capital: float) -> Optional[PaperTradeRecord]:
        if market_id not in self.open_positions:
            return None
            
        record = self.open_positions[market_id]
        # won parameter is now passed from the source of truth (DryRunEngine)
        
        fee_pct = float(self.config.get("risk.fee_pct", 0.02))
        gas_flat = float(self.config.get("risk.gas_flat_usd", 0.01))
        
        if won:
            payout = record.bet_size_usd / record.simulated_fill_price
            record.actual_pnl_usd = (payout - record.bet_size_usd - gas_flat - (record.bet_size_usd * fee_pct))
            record.actual_outcome = "WIN"
        else:
            record.actual_pnl_usd = -record.bet_size_usd
            record.actual_outcome = "LOSS"
            
        record.timestamp_resolution = datetime.now(timezone.utc).isoformat()
        record.resolution_price = settlement_price
        
        self._update_csv(self._get_daily_file(), record)
        self._update_summary(record)
        del self.open_positions[market_id]
        
        logger.info("paper_trade_resolved", 
                    trade_id=record.trade_id, 
                    outcome=record.actual_outcome, 
                    pnl=round(record.actual_pnl_usd, 2),
                    capital=round(current_capital, 2))
                    
        return record

    def _update_summary(self, record: PaperTradeRecord):
        """Updates summary in memory and flushes periodically (Opsi A - Batching)."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        if date_str not in self._summary_cache:
            self._summary_cache[date_str] = {
                "date": date_str,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0
            }
        
        stats = self._summary_cache[date_str]
        stats['total_trades'] += 1
        if record.actual_outcome == "WIN":
            stats['wins'] += 1
        else:
            stats['losses'] += 1
        
        stats['total_pnl'] += record.actual_pnl_usd
        total = stats['wins'] + stats['losses']
        stats['win_rate'] = stats['wins'] / total if total > 0 else 0.0
        
        # Immediate save for summary (since it's small, Opsi B style but using memory cache to avoid pd.concat)
        self._flush_summary()

    def _flush_summary(self):
        """Write memory summary cache to CSV using Pandas (Opsi A implementation)."""
        summary_path = os.path.join(self.output_dir, "paper_trades_summary.csv")
        if not self._summary_cache:
            return
            
        df = pd.DataFrame(list(self._summary_cache.values()))
        # Sort by date to keep it clean
        df = df.sort_values('date')
        df.to_csv(summary_path, index=False)

    def _append_to_csv(self, file_path: str, record: PaperTradeRecord):
        """Direct CSV Append (Opsi B) - No Pandas, O(1) performance."""
        data = asdict(record)
        header = not os.path.exists(file_path)
        
        with open(file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if header:
                writer.writeheader()
            writer.writerow(data)
            
        # Also keep in buffer for Task 1
        self._trade_buffer.append(record)

    def _update_csv(self, file_path: str, record: PaperTradeRecord):
        """
        Update the resolution fields in the CSV. 
        Still uses Pandas for existing file updates as CSV is not row-addressable,
        but we eliminate pd.concat and futurewarnings.
        """
        if not os.path.exists(file_path):
            return
        
        # Optimization: only read if needed
        df = pd.read_csv(file_path)
        if record.trade_id in df['trade_id'].values:
            idx = df[df['trade_id'] == record.trade_id].index[0]
            for field in fields(record):
                val = getattr(record, field.name)
                # Avoid setting NaNs or incompatible types if possible
                df.at[idx, field.name] = val
            df.to_csv(file_path, index=False)

    def _log_slippage(self, signal, fresh_clob, edge):
        """Direct CSV Append (Opsi B) for slippage logs."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_id": signal.market_id,
            "signal_direction": signal.signal,
            "p_model": signal.P_model,
            "yes_ask": fresh_clob.yes_ask,
            "no_ask": fresh_clob.no_ask,
            "edge": edge,
            "reason": "SLIPPAGE_NEGATIVE_EDGE"
        }
        
        header = not os.path.exists(self.slippage_log_path)
        with open(self.slippage_log_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            if header:
                writer.writeheader()
            writer.writerow(log_entry)
            
        logger.info("paper_trade_slippage", market_id=signal.market_id, edge=round(edge, 4))
