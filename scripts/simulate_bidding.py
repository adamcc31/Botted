"""
simulate_bidding.py — Alpha V1 Financial Backtester
====================================================
The Mad Quant's proof-of-concept: does the model's precision
translate to actual profits under realistic compounding?

ABSOLUTE LAWS:
  1. NO look-ahead bias — test on the last 15% of data only
  2. Model was trained on the first 85% — never sees test rows
  3. Kelly sizing with 10% capital cap per trade
  4. Two portfolios: $5 (micro) and $50 (standard)
"""

import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

# Suppress verbose logging from inference pipeline
logging.basicConfig(level=logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from model_training.inference import XGBoostGate

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════
DATA_PATH = BASE_DIR / "dataset" / "raw" / "alpha_v1_master.csv"
MODEL_DIR = BASE_DIR / "models" / "alpha_v1"
TRAIN_RATIO = 0.85          # 85% train, 15% test (by market_id)
MAX_BET_PCT = 0.10           # 10% capital cap per trade
OUTPUT_DIR = BASE_DIR / "scripts"


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════
@dataclass
class Trade:
    idx: int
    timestamp: str
    market_id: str
    entry_odds: float
    p_win: float
    ev: float
    kelly_fraction: float
    bet_size: float
    label: int
    pnl: float
    capital_after: float


@dataclass
class Portfolio:
    name: str
    initial_capital: float
    capital: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    signals_evaluated: int = 0
    signals_rejected: int = 0

    def __post_init__(self):
        self.capital = self.initial_capital
        self.equity_curve = [self.initial_capital]


# ═══════════════════════════════════════════════════════════════════
# Step 1: Load & Split Data (Market-Level, No Leakage)
# ═══════════════════════════════════════════════════════════════════
def load_and_split(path: Path, train_ratio: float):
    """Load CSV, drop unlabeled rows, split by market_id chronologically."""
    df = pd.read_csv(path, low_memory=False)
    
    # Drop rows without labels — these are ABSTAIN signals with no outcome
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    
    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Split by market_id (same logic as trainer.py — prevents tick leakage)
    market_end_times = df.groupby("market_id")["timestamp"].max().sort_values()
    market_ids = market_end_times.index.tolist()
    
    split_idx = int(len(market_ids) * train_ratio)
    train_markets = set(market_ids[:split_idx])
    test_markets = set(market_ids[split_idx:])
    
    df_test = df[df["market_id"].isin(test_markets)].reset_index(drop=True)
    
    print(f"  Total labeled rows:  {len(df)}")
    print(f"  Unique markets:      {len(market_ids)}")
    print(f"  Train markets:       {len(train_markets)} ({split_idx}/{len(market_ids)})")
    print(f"  Test markets:        {len(test_markets)}")
    print(f"  Test rows:           {len(df_test)}")
    print(f"  Test label dist:     WIN={int(df_test['label'].sum())} | LOSS={int((1-df_test['label']).sum())}")
    print(f"  Test date range:     {df_test['timestamp'].iloc[0][:19]} → {df_test['timestamp'].iloc[-1][:19]}")
    
    return df_test


# ═══════════════════════════════════════════════════════════════════
# Step 2: Build raw_features dict from a CSV row
# ═══════════════════════════════════════════════════════════════════
def row_to_raw_features(row: pd.Series) -> dict:
    """Map a CSV row to the raw_features dict expected by XGBoostGate."""
    return {
        "obi_value":           row.get("obi_value"),
        "tfm_value":           row.get("tfm_value"),
        "depth_ratio":         row.get("depth_ratio"),
        "obi_tfm_product":     row.get("obi_tfm_product"),
        "obi_tfm_alignment":   row.get("obi_tfm_alignment"),
        "rv_value":            row.get("rv_value"),
        "vol_percentile":      row.get("vol_percentile"),
        "strike_distance_pct": row.get("strike_distance_pct"),
        "contest_urgency":     row.get("contest_urgency"),
        "ttr_seconds":         row.get("ttr_seconds"),
        "odds_yes":            row.get("odds_yes"),
        "odds_no":             row.get("odds_no"),
        "entry_odds":          row.get("entry_odds"),
        "odds_yes_60s_ago":    row.get("odds_yes_60s_ago"),
        "odds_delta_60s":      row.get("odds_delta_60s"),
        "spread_pct":          row.get("spread_pct"),
        "btc_return_1m":       row.get("btc_return_1m"),
        "confidence_score":    row.get("confidence_score"),
        "timestamp":           row.get("timestamp"),
    }


# ═══════════════════════════════════════════════════════════════════
# Step 3: Run Simulation
# ═══════════════════════════════════════════════════════════════════
def simulate(gate: XGBoostGate, df_test: pd.DataFrame, portfolio: Portfolio):
    """Run the backtester on a single portfolio."""
    for idx, row in df_test.iterrows():
        portfolio.signals_evaluated += 1
        
        entry_odds = row["entry_odds"]
        label = int(row["label"])
        
        # Skip if entry_odds is invalid
        if pd.isna(entry_odds) or entry_odds <= 0 or entry_odds >= 1:
            portfolio.signals_rejected += 1
            continue
        
        raw_features = row_to_raw_features(row)
        result = gate.evaluate_signal(raw_features, entry_odds=entry_odds)
        
        if result["decision"] != "PASS":
            portfolio.signals_rejected += 1
            continue
        
        # ── Kelly sizing with 10% cap ──
        kelly = result["kelly_fraction"]
        bet_size = portfolio.capital * kelly
        bet_size = min(bet_size, portfolio.capital * MAX_BET_PCT)  # Cap at 10%
        bet_size = max(bet_size, 0.0)
        
        if bet_size <= 0 or bet_size > portfolio.capital:
            portfolio.signals_rejected += 1
            continue
        
        # ── Resolve trade ──
        if label == 1:  # WIN
            payout = bet_size * ((1.0 / entry_odds) - 1.0)
            pnl = payout
        else:           # LOSS
            pnl = -bet_size
        
        portfolio.capital += pnl
        portfolio.equity_curve.append(portfolio.capital)
        
        trade = Trade(
            idx=idx,
            timestamp=row["timestamp"],
            market_id=row["market_id"],
            entry_odds=entry_odds,
            p_win=result["p_win"],
            ev=result["ev"],
            kelly_fraction=kelly,
            bet_size=bet_size,
            label=label,
            pnl=pnl,
            capital_after=portfolio.capital,
        )
        portfolio.trades.append(trade)
        
        # Bankruptcy check
        if portfolio.capital <= 0.01:
            print(f"  ☠️  {portfolio.name} BANKRUPT at trade #{len(portfolio.trades)}")
            break


# ═══════════════════════════════════════════════════════════════════
# Step 4: Metrics & Reporting
# ═══════════════════════════════════════════════════════════════════
def compute_max_drawdown(equity_curve: List[float]) -> float:
    """Max drawdown as a percentage."""
    if len(equity_curve) < 2:
        return 0.0
    arr = np.array(equity_curve)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak
    return float(np.min(dd)) * 100  # as percentage


def print_report(portfolio: Portfolio):
    """Print comprehensive portfolio report."""
    trades = portfolio.trades
    n_trades = len(trades)
    wins = sum(1 for t in trades if t.label == 1)
    losses = n_trades - wins
    win_rate = wins / n_trades if n_trades > 0 else 0
    
    net_pnl = portfolio.capital - portfolio.initial_capital
    net_pnl_pct = (net_pnl / portfolio.initial_capital) * 100
    
    max_dd = compute_max_drawdown(portfolio.equity_curve)
    
    avg_bet = np.mean([t.bet_size for t in trades]) if trades else 0
    avg_ev = np.mean([t.ev for t in trades]) if trades else 0
    avg_kelly = np.mean([t.kelly_fraction for t in trades]) if trades else 0
    avg_entry_odds = np.mean([t.entry_odds for t in trades]) if trades else 0
    
    # Profit factor
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = gross_profit / (gross_loss + 1e-8)
    
    # Peak capital
    peak_cap = max(portfolio.equity_curve) if portfolio.equity_curve else portfolio.initial_capital
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  {portfolio.name:^56}  ║
╠══════════════════════════════════════════════════════════════╣
║  Starting Capital:    ${portfolio.initial_capital:>10.2f}                         ║
║  Ending Capital:      ${portfolio.capital:>10.2f}                         ║
║  Peak Capital:        ${peak_cap:>10.2f}                         ║
║  Net PnL:             ${net_pnl:>+10.2f}  ({net_pnl_pct:>+.1f}%)                  ║
║  Max Drawdown:        {max_dd:>10.1f}%                           ║
╠══════════════════════════════════════════════════════════════╣
║  Signals Evaluated:   {portfolio.signals_evaluated:>10}                           ║
║  Signals Rejected:    {portfolio.signals_rejected:>10}                           ║
║  Trades Executed:     {n_trades:>10}                           ║
║  Wins / Losses:       {wins:>5} / {losses:<5}                            ║
║  Win Rate:            {win_rate:>10.1%}                           ║
║  Profit Factor:       {profit_factor:>10.2f}                           ║
╠══════════════════════════════════════════════════════════════╣
║  Avg Bet Size:        ${avg_bet:>10.4f}                         ║
║  Avg Entry Odds:      {avg_entry_odds:>10.4f}                           ║
║  Avg EV per Trade:    {avg_ev:>10.4f}                           ║
║  Avg Kelly Fraction:  {avg_kelly:>10.4f}                           ║
╚══════════════════════════════════════════════════════════════╝""")


def plot_equity_curves(portfolios: List[Portfolio], output_path: Path):
    """Plot equity curves with Matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Alpha V1 Gladiator — Backtest Equity Curves (15% OOS Test Set)",
                     fontsize=14, fontweight="bold")
        
        for ax, pf in zip(axes, portfolios):
            curve = pf.equity_curve
            x = range(len(curve))
            
            # Color segments by profit/loss
            colors = []
            for i in range(1, len(curve)):
                colors.append("#00C853" if curve[i] >= curve[i-1] else "#FF1744")
            
            # Main equity line
            ax.plot(x, curve, color="#2196F3", linewidth=1.5, alpha=0.9)
            ax.fill_between(x, pf.initial_capital, curve, 
                          where=[c >= pf.initial_capital for c in curve],
                          alpha=0.15, color="#00C853")
            ax.fill_between(x, pf.initial_capital, curve,
                          where=[c < pf.initial_capital for c in curve],
                          alpha=0.15, color="#FF1744")
            
            # Start line
            ax.axhline(y=pf.initial_capital, color="gray", linestyle="--", 
                       alpha=0.5, linewidth=0.8)
            
            net_pnl = pf.capital - pf.initial_capital
            net_pct = (net_pnl / pf.initial_capital) * 100
            win_rate = sum(1 for t in pf.trades if t.label == 1) / max(len(pf.trades), 1) * 100
            
            ax.set_title(f"{pf.name}\n"
                        f"${pf.initial_capital:.0f} → ${pf.capital:.2f} "
                        f"({net_pct:+.1f}%) | {len(pf.trades)} trades | "
                        f"WR {win_rate:.0f}%",
                        fontsize=10)
            ax.set_xlabel("Trade #")
            ax.set_ylabel("Capital ($)")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, len(curve) - 1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n📈 Equity curve saved to: {output_path}")
        
    except ImportError:
        print("\n⚠️  Matplotlib not available — skipping chart.")


def print_trade_log(portfolio: Portfolio, n: int = 10):
    """Print last N trades for audit."""
    trades = portfolio.trades
    if not trades:
        return
    
    print(f"\n  Last {min(n, len(trades))} trades ({portfolio.name}):")
    print(f"  {'#':>3} {'Odds':>6} {'P(WIN)':>7} {'EV':>7} {'Kelly':>6} {'Bet$':>8} {'Result':>6} {'PnL':>9} {'Capital':>10}")
    print(f"  {'─'*3} {'─'*6} {'─'*7} {'─'*7} {'─'*6} {'─'*8} {'─'*6} {'─'*9} {'─'*10}")
    
    for t in trades[-n:]:
        result = "WIN" if t.label == 1 else "LOSS"
        print(f"  {trades.index(t)+1:>3} {t.entry_odds:>6.3f} {t.p_win:>7.4f} {t.ev:>+7.4f} "
              f"{t.kelly_fraction:>6.4f} ${t.bet_size:>7.4f} {result:>6} ${t.pnl:>+8.4f} ${t.capital_after:>9.4f}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 62)
    print("  ALPHA V1 GLADIATOR — FINANCIAL SIMULATION ENGINE")
    print("  The Mad Quant Backtest (No Look-Ahead Bias)")
    print("=" * 62)
    
    # ── Load model ──
    print("\n[1/4] Loading model...")
    gate = XGBoostGate()
    gate.load_model(MODEL_DIR)
    print(f"  Model: {gate.model_version} | Features: {len(gate._feature_names)}")
    
    # ── Load & split data ──
    print("\n[2/4] Loading & splitting data (85/15 by market_id)...")
    df_test = load_and_split(DATA_PATH, TRAIN_RATIO)
    
    # ── Run simulations ──
    print("\n[3/4] Running simulations...")
    
    portfolio_a = Portfolio(name="Portfolio A — Micro ($5)", initial_capital=5.0)
    portfolio_b = Portfolio(name="Portfolio B — Standard ($50)", initial_capital=50.0)
    
    print(f"\n  Simulating Portfolio A ($5)...")
    simulate(gate, df_test, portfolio_a)
    
    print(f"  Simulating Portfolio B ($50)...")
    simulate(gate, df_test, portfolio_b)
    
    # ── Report ──
    print("\n[4/4] Results")
    print("=" * 62)
    
    print_report(portfolio_a)
    print_trade_log(portfolio_a, n=15)
    
    print_report(portfolio_b)
    print_trade_log(portfolio_b, n=15)
    
    # ── Plot ──
    output_path = OUTPUT_DIR / "backtest_curve.png"
    plot_equity_curves([portfolio_a, portfolio_b], output_path)
    
    print("\n" + "=" * 62)
    print("  SIMULATION COMPLETE")
    print("=" * 62)


if __name__ == "__main__":
    main()
