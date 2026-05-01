import sys
import os
import math
import glob
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.zone_matrix import classify_zone

# --- CONFIGURATION (Synced with live config.json) ---
MAX_POSITIONS = 3
MAX_EXPOSURE_PCT = 0.60
KELLY_CAP = 0.15
MIN_BET = 1.0
FEE_PCT = 0.02
SLIPPAGE_PCT = 0.005
GAS_FLAT = 0.01
SPREAD_NORMAL_PCT = 0.03
SPREAD_ELEVATED_PCT = 0.08
MIN_TTR_MINUTES = 1.5
MARGIN_OF_SAFETY = 0.02
PRICE_HARD_CAP = 0.75
EDGE_DEVIATION_TOL = 0.05
UNCERTAINTY_U = 0.02 

PRUNED_ZONES = {"A1", "A2", "NEUTRAL", "D1", "D2", "D3", "D4", "D5", "D6", "A1_PRUNED", "A2_PRUNED"}
QUARTER_KELLY_ZONES = {"A3"}

@dataclass
class TradeRecord:
    entry_time: pd.Timestamp
    resolution_time: pd.Timestamp
    bet_size: float
    entry_odds: float
    signal_direction: str
    actual_outcome: str
    pnl: float = 0.0

class ForensicsSimulator:
    def __init__(self, initial_capital=50.0, sizing_mode='kelly', capital_floor=0.0):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.sizing_mode = sizing_mode
        self.capital_floor = capital_floor
        self.pending_trades: List[TradeRecord] = []
        self.completed_trades: List[TradeRecord] = []
        
    @property
    def pending_exposure(self):
        return sum(p.bet_size for p in self.pending_trades)

    @property
    def available_capital(self):
        return max(0, self.capital * MAX_EXPOSURE_PCT - self.pending_exposure)

    def _resolve_pending(self, current_time: pd.Timestamp):
        keep = []
        for trade in self.pending_trades:
            if current_time >= trade.resolution_time:
                if trade.actual_outcome == 'PENDING':
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
                self.completed_trades.append(trade)
            else:
                keep.append(trade)
        self.pending_trades = keep

    def _calculate_bet_size(self, kelly_fraction: float, zone_id: str) -> float:
        if self.capital < self.capital_floor:
            return 0.0
            
        if self.sizing_mode == 'flat':
            bet = 1.0
        else: # kelly
            k = kelly_fraction
            if zone_id in QUARTER_KELLY_ZONES: k /= 4.0
            k = min(k, KELLY_CAP)
            bet = math.floor(self.capital * k)
            
        bet = min(bet, self.available_capital)
        return float(bet) if bet >= MIN_BET else 0.0

    def run(self, df):
        for row in df.itertuples():
            ts = row.timestamp
            self._resolve_pending(ts)
            
            # Extract data
            ttr_minutes = getattr(row, 'ttr_seconds', 0) / 60.0
            if pd.isna(ttr_minutes) and hasattr(row, 'ttr_minutes'):
                ttr_minutes = row.ttr_minutes
            
            spread_pct = getattr(row, 'spread_pct', 0.0)
            p_model = getattr(row, 'confidence_score', 0.5)
            yes_ask = getattr(row, 'odds_yes', 0.5)
            no_ask = getattr(row, 'odds_no', 0.5)
            
            # --- SAME FILTERS AS FULL SIMULATOR ---
            if pd.notna(ttr_minutes) and ttr_minutes < MIN_TTR_MINUTES: continue
            if pd.notna(spread_pct) and spread_pct > SPREAD_ELEVATED_PCT: continue
            if not (0.15 <= getattr(row, 'vol_percentile', 0.5) <= 0.80): continue
            
            edge_yes = (p_model - yes_ask) - UNCERTAINTY_U
            edge_no = ((1.0 - p_model) - no_ask) - UNCERTAINTY_U
            
            if max(edge_yes, edge_no) <= MARGIN_OF_SAFETY: continue
            
            signal_dir = "BUY_UP" if edge_yes >= edge_no else "BUY_DOWN"
            entry_odds = yes_ask if signal_dir == "BUY_UP" else no_ask
            
            # Zone Gating
            dist_usd = getattr(row, 'strike_distance_pct', 0.0)
            if hasattr(row, 'binance_price') and hasattr(row, 'strike_price'):
                dist_usd = abs(row.binance_price - row.strike_price)
                
            zone = classify_zone(ttr_minutes, dist_usd, entry_odds)
            if zone.zone_type == "DEATH" or zone.zone_id in PRUNED_ZONES: continue
            
            # Price & Deviation
            if entry_odds > PRICE_HARD_CAP: continue
            
            # Risk/Sizing
            if len(self.pending_trades) >= MAX_POSITIONS: continue
            
            bet_size = self._calculate_bet_size(zone.kelly_fraction, zone.zone_id)
            if bet_size < MIN_BET: continue
            
            # Execute
            res_time = ts + pd.Timedelta(minutes=ttr_minutes)
            self.pending_trades.append(TradeRecord(
                entry_time=ts, resolution_time=res_time, bet_size=bet_size,
                entry_odds=entry_odds, signal_direction=signal_dir,
                actual_outcome=getattr(row, 'actual_outcome', 'PENDING')
            ))
        
        # Final resolution
        self._resolve_pending(df['timestamp'].max() + pd.Timedelta(days=365))

def main():
    files = glob.glob('dataset/raw/dry_run_*.csv')
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp', 'market_id'])
    
    scenarios = [
        ('Original Kelly', 'kelly', 0.0),
        ('Flat $1 Bet', 'flat', 0.0),
        ('Kelly + $10 Floor', 'kelly', 10.0)
    ]
    
    print("| Scenario | Final Capital | Win Rate | Total PnL |")
    print("|----------|---------------|----------|-----------|")
    
    for name, mode, floor in scenarios:
        sim = ForensicsSimulator(sizing_mode=mode, capital_floor=floor)
        sim.run(df)
        
        trades = sim.completed_trades
        res_trades = [t for t in trades if t.actual_outcome != 'PENDING']
        wins = [t for t in res_trades if t.pnl > 0]
        wr = (len(wins) / len(res_trades) * 100) if res_trades else 0.0
        pnl = sim.capital - sim.initial_capital
        
        print(f"| {name} | ${sim.capital:.2f} | {wr:.1f}% | ${pnl:+.2f} |")

if __name__ == "__main__":
    main()
