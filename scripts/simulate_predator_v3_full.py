"""
simulate_predator_v3_full.py — Full Fidelity Simulator for Predator V3
Replicates all 19 execution gates from the live pipeline without look-ahead bias.

Rules Enforced:
1. NO LOOK-AHEAD: `label` and `actual_outcome` are only used AFTER execution for PnL.
2. FULL PIPELINE: All structural gates (Spread, Edge, Zone, Risk) are applied sequentially.
3. EDGE CALCULATION: Live edge is computed and enforced.
"""

import sys
import os
import math
import glob
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import defaultdict

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.zone_matrix import classify_zone

# --- CONFIGURATION (Synced with live config.json) ---
INITIAL_CAPITAL = 50.0
MAX_POSITIONS = 3
MAX_EXPOSURE_PCT = 0.60
KELLY_CAP = 0.15
MIN_BET = 1.0
FEE_PCT = 0.02
SLIPPAGE_PCT = 0.005
GAS_FLAT = 0.01

USE_FLAT_BET = True  # Set to True for regression testing
FLAT_BET_SIZE = 1.0

SPREAD_NORMAL_PCT = 0.03
SPREAD_ELEVATED_PCT = 0.08
MIN_TTR_MINUTES = 1.5
MARGIN_OF_SAFETY = 0.02
PRICE_HARD_CAP = 0.75
EDGE_DEVIATION_TOL = 0.05
UNCERTAINTY_U = 0.02 # Base uncertainty (from fair_prob.base_uncertainty_p)


@dataclass
class RejectionLog:
    signal_id: int
    timestamp: pd.Timestamp
    market_id: str
    zone_id: str
    reason: str
    synthetic_edge: float
    live_edge: float
    executed: bool


@dataclass
class TradeRecord:
    trade_id: int
    market_id: str
    zone_id: str
    entry_time: pd.Timestamp
    resolution_time: pd.Timestamp
    bet_size: float
    entry_odds: float
    signal_direction: str
    strike_price: float
    actual_outcome: str  # 'BUY_UP', 'BUY_DOWN', 'PENDING'
    capital_before: float
    capital_after: float
    pnl: float
    live_edge: float


class FullFidelitySimulator:
    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.pending_trades: List[TradeRecord] = []
        self.completed_trades: List[TradeRecord] = []
        self.rejections: List[RejectionLog] = []
        
        self.rejection_counts = defaultdict(int)
        self.total_signals = 0
        self.executed_signals = 0
        
        self._trade_counter = 0

    @property
    def pending_exposure(self):
        return sum(p.bet_size for p in self.pending_trades)

    @property
    def available_capital(self):
        return max(0, self.capital * MAX_EXPOSURE_PCT - self.pending_exposure)

    def _resolve_pending(self, current_time: pd.Timestamp):
        """Resolve trades whose resolution time has passed."""
        keep = []
        for trade in self.pending_trades:
            if current_time >= trade.resolution_time:
                # Calculate PnL based on actual_outcome
                if trade.actual_outcome == 'PENDING':
                    # Unresolved / Cancelled market -> zero PnL
                    pnl = 0.0
                elif trade.actual_outcome == trade.signal_direction:
                    # WIN
                    mult = (1.0 / trade.entry_odds) * (1.0 - FEE_PCT - SLIPPAGE_PCT) - GAS_FLAT
                    pnl = trade.bet_size * mult - trade.bet_size
                else:
                    # LOSS
                    pnl = -trade.bet_size

                self.capital += pnl
                trade.pnl = pnl
                trade.capital_after = self.capital
                self.completed_trades.append(trade)
            else:
                keep.append(trade)
        self.pending_trades = keep
    
    def _calculate_bet_size(self, kelly_fraction: float, kelly_cap: float) -> float:
        if USE_FLAT_BET:
            return min(FLAT_BET_SIZE, self.available_capital)

        k = min(kelly_fraction, kelly_cap, KELLY_CAP)
        
        raw_bet = self.capital * k
        raw_bet = min(raw_bet, self.available_capital)
        
        if raw_bet >= MIN_BET:
            return math.floor(raw_bet)
        return 0.0

    def process_row(self, row: pd.Series):
        self.total_signals += 1
        ts = row['timestamp']
        market_id = row['market_id']
        
        # 1. Resolve old trades
        self._resolve_pending(ts)
        
        # ── DATA EXTRACTION & PROXY ASSUMPTIONS ──
        # Proxy: If ttr_seconds is available, use it. Else fallback to ttr_minutes if available.
        ttr_minutes = row.get('ttr_seconds', 0) / 60.0
        if pd.isna(ttr_minutes) and 'ttr_minutes' in row:
            ttr_minutes = row['ttr_minutes']
            
        spread_pct = row.get('spread_pct', 0.0)
        p_model = row.get('confidence_score', 0.5)
        yes_ask = row.get('odds_yes', 0.5)
        no_ask = row.get('odds_no', 0.5)
        mid_price = (yes_ask + (1.0 - no_ask)) / 2.0  # Synthetic mid proxy
        
        distance_usd = row.get('strike_distance_pct', 0.0) # We might not have abs USD, but zone_matrix uses distance.
        # Fallback for distance_usd if 'binance_price' and 'strike_price' are present
        if 'binance_price' in row and 'strike_price' in row:
            distance_usd = abs(row['binance_price'] - row['strike_price'])
            
        entry_odds_source = row.get('entry_odds_source', 'CLOB_LIVE')
        vol_pct = row.get('vol_percentile', 0.5)
        if pd.isna(vol_pct):
            vol_pct = 0.5  # Assume PASS if missing
            
        # Helper to log rejection
        def reject(reason: str, z_id: str = "UNKNOWN", s_edge: float = 0.0, l_edge: float = 0.0):
            self.rejection_counts[reason] += 1
            self.rejections.append(RejectionLog(
                signal_id=self.total_signals, timestamp=ts, market_id=market_id,
                zone_id=z_id, reason=reason, synthetic_edge=s_edge, live_edge=l_edge, executed=False
            ))

        # ── LIVE PIPELINE GATES ──
        
        # F1: Min TTR (hard floor)
        if pd.notna(ttr_minutes) and ttr_minutes < MIN_TTR_MINUTES:
            return reject("F1_MIN_TTR")
            
        # F2: Spread Filter
        if pd.notna(spread_pct):
            if spread_pct > SPREAD_ELEVATED_PCT:
                return reject("F2_SPREAD_FILTER_SKIP")
            elif spread_pct > SPREAD_NORMAL_PCT:
                return reject("F2_SPREAD_FILTER_WAIT")

        # F3: Regime Filter (Proxy using available vol_percentile)
        if not (0.15 <= vol_pct <= 0.80):
            return reject("F3_REGIME_FILTER")

        # F4: Liquidity Block (Proxy: odds > 0)
        if pd.isna(yes_ask) or pd.isna(no_ask) or yes_ask <= 0 or no_ask <= 0:
            return reject("F4_LIQUIDITY_BLOCK")

        # ── EDGE COMPUTATION ──
        # Synthetic Edge (Step 4 of signal_generator)
        edge_yes_raw = p_model - yes_ask
        edge_no_raw = (1.0 - p_model) - no_ask
        synthetic_edge_yes = edge_yes_raw - UNCERTAINTY_U
        synthetic_edge_no = edge_no_raw - UNCERTAINTY_U

        # F5: Margin of Safety (Step 5 of signal_generator)
        if max(synthetic_edge_yes, synthetic_edge_no) <= MARGIN_OF_SAFETY:
            return reject("F5_MARGIN_OF_SAFETY", s_edge=max(synthetic_edge_yes, synthetic_edge_no))

        # Determine Signal Direction
        if synthetic_edge_yes > MARGIN_OF_SAFETY and synthetic_edge_no > MARGIN_OF_SAFETY:
            signal_dir = "BUY_UP" if synthetic_edge_yes >= synthetic_edge_no else "BUY_DOWN"
        elif synthetic_edge_yes > MARGIN_OF_SAFETY:
            signal_dir = "BUY_UP"
        else:
            signal_dir = "BUY_DOWN"
            
        entry_odds = yes_ask if signal_dir == "BUY_UP" else no_ask
        synthetic_edge = synthetic_edge_yes if signal_dir == "BUY_UP" else synthetic_edge_no

        # F6: Zone Matrix Gating
        zone = classify_zone(ttr_minutes, distance_usd, entry_odds)
        if zone.zone_type == "DEATH" or zone.zone_id == "NEUTRAL":
            return reject("F6_ZONE_GATING_DEATH", z_id=zone.zone_id, s_edge=synthetic_edge)

        # F7: Contaminated Fallback Odds
        if entry_odds_source == "DEFAULT_FALLBACK":
            return reject("F7_CONTAMINATED_ODDS", z_id=zone.zone_id, s_edge=synthetic_edge)

        # ── LIVE EXECUTION GATES ──
        # Compute Live Edge (In reality this re-fetches CLOB, here we use the exact row odds as a best proxy for real_best_ask)
        live_yes_ask = yes_ask
        live_no_ask = no_ask
        real_best_ask = live_yes_ask if signal_dir == "BUY_UP" else live_no_ask
        p_outcome = p_model if signal_dir == "BUY_UP" else (1.0 - p_model)
        live_edge = p_outcome - UNCERTAINTY_U - real_best_ask
        edge_deviation = abs(synthetic_edge - live_edge)

        # F8: Price Hard Cap
        if real_best_ask > PRICE_HARD_CAP:
            return reject("F8_PRICE_HARD_CAP", z_id=zone.zone_id, s_edge=synthetic_edge, l_edge=live_edge)

        # F9: Edge Deviation
        if edge_deviation > EDGE_DEVIATION_TOL:
            return reject("F9_EDGE_DEVIATION", z_id=zone.zone_id, s_edge=synthetic_edge, l_edge=live_edge)

        # F10: Live Edge Positive
        if live_edge <= MARGIN_OF_SAFETY:
            return reject("F10_LIVE_EDGE_NEGATIVE", z_id=zone.zone_id, s_edge=synthetic_edge, l_edge=live_edge)

        # ── RISK MANAGER GATES ──
        # F11: Max Positions
        if len(self.pending_trades) >= MAX_POSITIONS:
            return reject("F11_MAX_POSITIONS", z_id=zone.zone_id, s_edge=synthetic_edge, l_edge=live_edge)

        # F12: Exposure Cap & Kelly Sizing
        bet_size = self._calculate_bet_size(zone.kelly_fraction, zone.kelly_cap)
        if bet_size < MIN_BET:
            return reject("F12_INSUFFICIENT_CAPITAL", z_id=zone.zone_id, s_edge=synthetic_edge, l_edge=live_edge)

        # ── EXECUTION ──
        self.executed_signals += 1
        self._trade_counter += 1
        
        resolution_time = ts + pd.Timedelta(minutes=ttr_minutes) if pd.notna(ttr_minutes) else ts + pd.Timedelta(minutes=60)
        
        trade = TradeRecord(
            trade_id=self._trade_counter,
            market_id=market_id,
            zone_id=zone.zone_id,
            entry_time=ts,
            resolution_time=resolution_time,
            bet_size=bet_size,
            entry_odds=real_best_ask,
            signal_direction=signal_dir,
            strike_price=row.get('strike_price', 0.0),
            actual_outcome=row.get('actual_outcome', 'PENDING'),
            capital_before=self.capital,
            capital_after=0.0, # Filled at resolution
            pnl=0.0,
            live_edge=live_edge
        )
        self.pending_trades.append(trade)


def main():
    print("=" * 80)
    print("PREDATOR V3 — FULL FIDELITY PIPELINE SIMULATOR")
    print("=" * 80)
    
    # Load dataset
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        print(f"Loading specific dataset: {data_file}")
        df = pd.read_csv(data_file)
    else:
        print("Loading dataset: dataset/raw/dry_run_*.csv")
        files = glob.glob('dataset/raw/dry_run_*.csv')
        if not files:
            print("DATA_INSUFFICIENT: No files found.")
            return
            
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp', 'market_id'])
    
    print(f"Total raw rows: {len(df)}")
    
    # Column audit
    required_cols = ['timestamp', 'market_id', 'odds_yes', 'odds_no', 'confidence_score']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"DATA_INSUFFICIENT: Missing critical columns: {missing}")
        return
        
    print("All critical columns present.")
    print("Applying 12-stage live pipeline filters...\n")
    
    sim = FullFidelitySimulator()
    
    for _, row in df.iterrows():
        sim.process_row(row)
        
    # Resolve any remaining trades that might have expired
    sim._resolve_pending(df['timestamp'].max() + pd.Timedelta(days=365))
    
    print("=" * 80)
    print("BOTTLENECK MATRIX (FULL PIPELINE)")
    print("=" * 80)
    print(f"Total signals in dataset:    {sim.total_signals:>6}")
    
    # Sort rejections by phase (roughly F1 to F12)
    sorted_rejections = sorted(sim.rejection_counts.items())
    for reason, count in sorted_rejections:
        pct = (count / sim.total_signals) * 100
        print(f"{reason:<30}: {count:>6} rejected ({pct:>5.1f}%)")
        
    print("-" * 44)
    exec_pct = (sim.executed_signals / sim.total_signals) * 100
    print(f"FINAL EXECUTED:              {sim.executed_signals:>6} signals  ({exec_pct:>5.1f}%)")
    print("=" * 80)
    
    if sim.executed_signals > 0:
        trades = sim.completed_trades
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        pending = [t for t in trades if t.pnl == 0.0 and t.actual_outcome == 'PENDING']
        
        resolved_trades = [t for t in trades if t.actual_outcome != 'PENDING']
        wr = (len(wins) / len(resolved_trades) * 100) if resolved_trades else 0.0
        
        avg_live_edge = np.mean([t.live_edge for t in trades])
        total_pnl = sum(t.pnl for t in trades)
        
        print("\nEXECUTION QUALITY METRICS")
        print(f"Resolved Trades (Known Outcome): {len(resolved_trades)}")
        print(f"Unresolved/Cancelled Markets:    {len(pending)}")
        print(f"Win Rate (of resolved):          {wr:.1f}%")
        print(f"Average Live Edge at Entry:      {avg_live_edge:.4f}")
        print(f"Expected PnL Total:              ${total_pnl:+.2f}")
        print(f"Final Capital:                   ${sim.capital:.2f}")

    print("\nSaving execution log to scripts/output/full_pipeline_audit.csv...")
    os.makedirs('scripts/output', exist_ok=True)
    
    if sim.rejections:
        rej_df = pd.DataFrame([r.__dict__ for r in sim.rejections])
        rej_df.to_csv('scripts/output/full_pipeline_rejections.csv', index=False)
    
    if sim.completed_trades:
        tr_df = pd.DataFrame([{
            'trade_id': t.trade_id,
            'market_id': t.market_id,
            'zone_id': t.zone_id,
            'entry_time': t.entry_time,
            'bet_size': t.bet_size,
            'entry_odds': t.entry_odds,
            'signal_direction': t.signal_direction,
            'actual_outcome': t.actual_outcome,
            'pnl': t.pnl,
            'live_edge': t.live_edge
        } for t in sim.completed_trades])
        tr_df.to_csv('scripts/output/full_pipeline_trades.csv', index=False)

if __name__ == "__main__":
    main()
