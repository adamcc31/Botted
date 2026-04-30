"""
simulate_bidding_v3.py — Full Dataset Regime-Aware Underdog Analysis
====================================================================
Uses ALL 1,412 labeled rows for statistical power.
Splits by weekday/weekend and session, analyzes EV per odds bucket.
"""
import sys, logging, math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from model_training.inference import XGBoostGate

DATA_PATH_WD = BASE_DIR / "dataset" / "raw" / "alpha_v1_master.csv"
DATA_PATH_WE = BASE_DIR / "dataset" / "weekend_market" / "dataset_ml_ready.csv"
MODEL_DIR = BASE_DIR / "models" / "alpha_v1"
OUTPUT_DIR = BASE_DIR / "scripts"
MIN_BUCKET_N = 30
POLYMARKET_FEE = 0.02; SLIPPAGE = 0.005; GAS_USD = 0.01

# ═════════════════════════════════════════════════════════════
# Load & Enrich
# ═════════════════════════════════════════════════════════════
def load_data(path):
    df = pd.read_csv(path, low_memory=False).dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["hour_utc"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    bins = [0, 7, 13, 20, 24]
    labels = ["Asia", "London", "NY", "Late"]
    df["session_bucket"] = pd.cut(df["hour_utc"], bins=bins, labels=labels, right=False, include_lowest=True)
    df["odds_bucket"] = (df["entry_odds"] // 0.05) * 0.05
    df["odds_bucket"] = df["odds_bucket"].round(2)
    return df

def row_to_features(row):
    return {k: row.get(k) for k in [
        "obi_value","tfm_value","depth_ratio","obi_tfm_product","obi_tfm_alignment",
        "rv_value","vol_percentile","strike_distance_pct","contest_urgency","ttr_seconds",
        "odds_yes","odds_no","entry_odds","odds_yes_60s_ago","odds_delta_60s",
        "spread_pct","btc_return_1m","confidence_score","timestamp"]}

# ═════════════════════════════════════════════════════════════
# EV Analysis per Odds Bucket
# ═════════════════════════════════════════════════════════════
def compute_bucket_stats(df_sub, label=""):
    groups = df_sub.groupby("odds_bucket")
    rows = []
    for bucket, grp in sorted(groups):
        n = len(grp)
        avg_odds = grp["entry_odds"].mean()
        breakeven = avg_odds + 0.03  # include fee buffer
        wr = grp["label"].mean()
        # EV = wr * ((1/odds)-1) - (1-wr) - fees
        payout_mult = (1.0 / avg_odds) - 1.0
        gross_ev = wr * payout_mult - (1.0 - wr)
        net_ev = gross_ev - (POLYMARKET_FEE + SLIPPAGE)
        status = "✅ +EV" if net_ev > 0 and n >= MIN_BUCKET_N else ("⚠️ LOW_N" if n < MIN_BUCKET_N else "❌ -EV")
        rows.append({
            "odds_range": f"{bucket:.2f}-{bucket+0.05:.2f}",
            "n": n, "win_rate": wr, "avg_odds": avg_odds,
            "breakeven_wr": breakeven, "gross_ev": gross_ev,
            "net_ev": net_ev, "status": status
        })
    return pd.DataFrame(rows)

def print_table(df_stats, title):
    print(f"\n{'─'*72}")
    print(f"  {title}")
    print(f"{'─'*72}")
    print(f"  {'Odds Range':<12} {'N':>5} {'WinRate':>8} {'AvgOdds':>8} {'GrossEV':>8} {'NetEV':>8} {'Status':<10}")
    print(f"  {'─'*12} {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
    for _, r in df_stats.iterrows():
        wr_str = f"{r['win_rate']:.1%}" if r['n'] >= MIN_BUCKET_N else "  n/a "
        print(f"  {r['odds_range']:<12} {r['n']:>5} {wr_str:>8} {r['avg_odds']:>8.3f} "
              f"{r['gross_ev']:>+8.4f} {r['net_ev']:>+8.4f} {r['status']:<10}")

# ═════════════════════════════════════════════════════════════
# Simulation Engine
# ═════════════════════════════════════════════════════════════
@dataclass
class SimResult:
    name: str; initial_capital: float = 50.0; capital: float = 50.0
    trades: int = 0; wins: int = 0; equity: List[float] = field(default_factory=lambda: [50.0])
    total_fees: float = 0.0
    def wr(self): return self.wins / self.trades if self.trades else 0
    def pnl(self): return self.capital - self.initial_capital
    def pnl_pct(self): return self.pnl() / self.initial_capital * 100
    def max_dd(self):
        if len(self.equity) < 2: return 0
        a = np.array(self.equity); p = np.maximum.accumulate(a)
        return float(np.min((a - p) / p)) * 100

def simulate_ev_only(df, ev_positive_buckets, initial=50.0):
    """Strategy: bet on odds buckets with net +EV, flat 2% sizing."""
    sim = SimResult(name="EV-Only (No ML)", initial_capital=initial, capital=initial)
    sim.equity = [initial]
    for _, row in df.iterrows():
        bucket = row["odds_bucket"]
        odds = row["entry_odds"]
        label = int(row["label"])
        if bucket not in ev_positive_buckets: continue
        if pd.isna(odds) or odds <= 0 or odds >= 1: continue
        bet = sim.capital * 0.02  # flat 2%
        if bet <= 0: break
        if label == 1:
            gross = bet * ((1.0/odds) - 1.0)
            fee = gross * (POLYMARKET_FEE + SLIPPAGE) + GAS_USD
            pnl = gross - fee
        else:
            fee = GAS_USD
            pnl = -bet - fee
        sim.capital += pnl; sim.total_fees += fee
        sim.trades += 1; sim.wins += (1 if label == 1 else 0)
        sim.equity.append(sim.capital)
        if sim.capital <= 0.01: break
    return sim

def simulate_ml_gate(gate, df, ev_positive_buckets, initial=50.0):
    """Strategy: bet on +EV buckets AND model PASS, quarter-Kelly capped 3%."""
    sim = SimResult(name="EV + ML Gate", initial_capital=initial, capital=initial)
    sim.equity = [initial]
    for _, row in df.iterrows():
        bucket = row["odds_bucket"]
        odds = row["entry_odds"]
        label = int(row["label"])
        if bucket not in ev_positive_buckets: continue
        if pd.isna(odds) or odds <= 0 or odds >= 1: continue
        raw = row_to_features(row)
        res = gate.evaluate_signal(raw, entry_odds=odds)
        if res["decision"] != "PASS": continue
        p_win = np.clip(res["p_win"], 0.30, 0.72)
        b = (1.0/odds) - 1.0
        kelly = max((p_win * b - (1.0 - p_win)) / b, 0) * 0.25
        kelly = min(kelly, 0.03)
        bet = sim.capital * kelly
        if bet <= 0: continue
        if label == 1:
            gross = bet * ((1.0/odds) - 1.0)
            fee = gross * (POLYMARKET_FEE + SLIPPAGE) + GAS_USD
            pnl = gross - fee
        else:
            fee = GAS_USD
            pnl = -bet - fee
        sim.capital += pnl; sim.total_fees += fee
        sim.trades += 1; sim.wins += (1 if label == 1 else 0)
        sim.equity.append(sim.capital)
        if sim.capital <= 0.01: break
    return sim

# ═════════════════════════════════════════════════════════════
# Sample Size Calculation
# ═════════════════════════════════════════════════════════════
def min_sample_size(wr_target, odds, confidence=0.95):
    """Min trades for WR CI to exclude breakeven."""
    breakeven_wr = odds + 0.03
    effect = wr_target - breakeven_wr
    if effect <= 0: return float("inf")
    z = 1.96 if confidence == 0.95 else 2.576
    p = wr_target
    n = (z**2 * p * (1 - p)) / (effect**2)
    return int(math.ceil(n))

def estimate_days(n_needed, signals_per_day=174/2):
    """Rough estimate: how many days to collect n labeled signals."""
    return math.ceil(n_needed / signals_per_day)

# ═════════════════════════════════════════════════════════════
# Plot
# ═════════════════════════════════════════════════════════════
def plot(sims, path):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.suptitle("Alpha V1 — EV-Only vs ML-Gated Underdog Strategy", fontsize=14, fontweight="bold")
        colors = ["#F44336", "#4CAF50"]
        for s, c in zip(sims, colors):
            ax.plot(range(len(s.equity)), s.equity, color=c, linewidth=1.8, label=s.name)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel("Trade #"); ax.set_ylabel("Capital ($)")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n📈 Chart saved: {path}")
    except ImportError:
        print("\n⚠️ Matplotlib not available")

# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("  ALPHA V1 — FULL DATASET REGIME-AWARE UNDERDOG ANALYSIS v3")
    print("=" * 72)

    print("\n[1/6] Loading model & data...")
    gate = XGBoostGate(); gate.load_model(MODEL_DIR)
    df_wd = load_data(DATA_PATH_WD)
    df_we = load_data(DATA_PATH_WE)
    df = pd.concat([df_wd, df_we], ignore_index=True)
    print(f"  Labeled rows: {len(df)} | Markets: {df['market_id'].nunique()}")
    print(f"  Weekend: {df['is_weekend'].sum()} | Weekday: {(1-df['is_weekend']).sum()}")
    print(f"  Sessions: {df['session_bucket'].value_counts().to_dict()}")

    # ── Tables 1 & 2: EV per odds bucket ──
    print("\n[2/6] Computing EV per odds bucket...")
    df_wd = df[df["is_weekend"] == 0]
    df_we = df[df["is_weekend"] == 1]
    
    stats_wd = compute_bucket_stats(df_wd)
    stats_we = compute_bucket_stats(df_we)
    
    print_table(stats_wd, "TABLE 1 — WEEKDAY EV per Odds Bucket")
    print_table(stats_we, "TABLE 2 — WEEKEND EV per Odds Bucket")

    # ── Find +EV buckets ──
    print("\n[3/6] Identifying +EV zones...")
    ev_pos_wd = set(stats_wd[(stats_wd["net_ev"] > 0) & (stats_wd["n"] >= MIN_BUCKET_N)]["odds_range"].apply(lambda x: float(x.split("-")[0]))) if len(stats_wd) > 0 else set()
    ev_pos_we = set(stats_we[(stats_we["net_ev"] > 0) & (stats_we["n"] >= MIN_BUCKET_N)]["odds_range"].apply(lambda x: float(x.split("-")[0]))) if len(stats_we) > 0 else set()
    ev_pos_both = ev_pos_wd & ev_pos_we
    
    print(f"  +EV Weekday buckets (n>={MIN_BUCKET_N}): {sorted(ev_pos_wd) if ev_pos_wd else 'NONE'}")
    print(f"  +EV Weekend buckets (n>={MIN_BUCKET_N}): {sorted(ev_pos_we) if ev_pos_we else 'NONE'}")
    print(f"  +EV in BOTH regimes:                     {sorted(ev_pos_both) if ev_pos_both else 'NONE'}")

    # Use all +EV buckets (union) for simulation
    ev_pos_all = ev_pos_wd | ev_pos_we
    if not ev_pos_all:
        print("\n  ⚠️ No +EV buckets found with sufficient sample size!")
        print("  Using all buckets with gross_ev > 0 regardless of n...")
        ev_pos_all_wd = set(stats_wd[stats_wd["gross_ev"] > 0]["odds_range"].apply(lambda x: float(x.split("-")[0]))) if len(stats_wd) > 0 else set()
        ev_pos_all_we = set(stats_we[stats_we["gross_ev"] > 0]["odds_range"].apply(lambda x: float(x.split("-")[0]))) if len(stats_we) > 0 else set()
        ev_pos_all = ev_pos_all_wd | ev_pos_all_we
        print(f"  Gross +EV buckets (any n): {sorted(ev_pos_all)}")

    # ── Tables 3 & 4: Simulation ──
    print(f"\n[4/6] Running simulations on +EV zones: {sorted(ev_pos_all)}...")
    
    sim_ev = simulate_ev_only(df, ev_pos_all)
    sim_ml = simulate_ml_gate(gate, df, ev_pos_all)

    print(f"""
{'─'*72}
  TABLE 3 & 4 — STRATEGY COMPARISON ($50 start, full 1412 rows)
{'─'*72}
  ┌──────────────────┬──────────────┬──────────────┐
  │ Metric           │ EV-Only      │ EV + ML Gate │
  ├──────────────────┼──────────────┼──────────────┤
  │ Final Capital    │ ${sim_ev.capital:>10.2f}  │ ${sim_ml.capital:>10.2f}  │
  │ Net PnL          │ ${sim_ev.pnl():>+10.2f}  │ ${sim_ml.pnl():>+10.2f}  │
  │ Net PnL %        │ {sim_ev.pnl_pct():>+10.1f}%  │ {sim_ml.pnl_pct():>+10.1f}%  │
  │ Max Drawdown     │ {sim_ev.max_dd():>10.1f}%  │ {sim_ml.max_dd():>10.1f}%  │
  │ Trades Executed  │ {sim_ev.trades:>12}  │ {sim_ml.trades:>12}  │
  │ Wins / Losses    │ {sim_ev.wins:>5}/{sim_ev.trades-sim_ev.wins:<6} │ {sim_ml.wins:>5}/{sim_ml.trades-sim_ml.wins:<6} │
  │ Win Rate         │ {sim_ev.wr():>10.1%}  │ {sim_ml.wr():>10.1%}  │
  │ Total Fees       │ ${sim_ev.total_fees:>10.2f}  │ ${sim_ml.total_fees:>10.2f}  │
  └──────────────────┴──────────────┴──────────────┘""")

    # ── Session breakdown ──
    print(f"\n{'─'*72}")
    print(f"  SESSION BREAKDOWN (Win Rate by Trading Session)")
    print(f"{'─'*72}")
    for session in ["Asia", "London", "NY", "Late"]:
        sub = df[df["session_bucket"] == session]
        if len(sub) > 0:
            wr = sub["label"].mean()
            print(f"  {session:<8} n={len(sub):>4}  WR={wr:.1%}  avg_odds={sub['entry_odds'].mean():.3f}")

    # ── Sample size calculation ──
    print(f"\n[5/6] Minimum sample size calculation...")
    print(f"{'─'*72}")
    print(f"  TARGET: WR=20% at odds=0.15 (deep underdog)")
    print(f"  Breakeven WR at 0.15 odds + 3% fee buffer = 18.0%")
    
    n_needed = min_sample_size(0.20, 0.15)
    days = estimate_days(n_needed)
    print(f"  Effect size: 20% - 18% = 2%")
    print(f"  Min trades needed (95% CI): {n_needed}")
    print(f"  Est. dry-run days needed:   {days} days (~{n_needed/87:.0f} sessions)")
    
    print(f"\n  For other common odds levels:")
    print(f"  {'Odds':>6} {'Target WR':>10} {'Breakeven':>10} {'Effect':>8} {'Min N':>8} {'Days':>6}")
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*6}")
    for odds, target_wr in [(0.10, 0.15), (0.15, 0.20), (0.20, 0.26),
                             (0.25, 0.32), (0.30, 0.38), (0.50, 0.56)]:
        be = odds + 0.03
        eff = target_wr - be
        n = min_sample_size(target_wr, odds)
        d = estimate_days(n) if n < 100000 else 99999
        print(f"  {odds:>6.2f} {target_wr:>10.0%} {be:>10.0%} {eff:>+8.0%} {n:>8} {d:>6}")

    # ── Plot ──
    plot([sim_ev, sim_ml], OUTPUT_DIR / "backtest_curve_v3.png")

    # ── Final Answers ──
    print(f"\n{'='*72}")
    print(f"  ANSWERS TO KEY QUESTIONS")
    print(f"{'='*72}")
    
    print(f"""
  Q1: Apakah ada odds bucket dengan EV positif konsisten
      di KEDUA weekday DAN weekend?
  A1: {"YA — buckets: " + str(sorted(ev_pos_both)) if ev_pos_both else "TIDAK — tidak ada bucket yang +EV di kedua regime sekaligus."}

  Q2: Di odds bucket mana ML gate menambah value?
  A2: ML gate {'MENAMBAH' if sim_ml.pnl() > sim_ev.pnl() else 'TIDAK menambah'} value vs EV-only.
      EV-only: {sim_ev.pnl():+.2f} ({sim_ev.trades} trades, WR {sim_ev.wr():.1%})
      EV+ML:   {sim_ml.pnl():+.2f} ({sim_ml.trades} trades, WR {sim_ml.wr():.1%})
      {'ML gate memperbaiki PnL dengan filtering sinyal lemah.' if sim_ml.pnl() > sim_ev.pnl() else 'ML gate belum menambah value — model perlu recalibration.'}

  Q3: Berapa minimum hari dry-run untuk sample valid?
  A3: Untuk deep underdog (odds=0.15, target WR=20%):
      Min {n_needed} trades → ~{days} hari dry-run
      Untuk moderate underdog (odds=0.25, target WR=32%):
      Min {min_sample_size(0.32, 0.25)} trades → ~{estimate_days(min_sample_size(0.32, 0.25))} hari
""")
    print("=" * 72)
    print("  ANALYSIS COMPLETE")
    print("=" * 72)

if __name__ == "__main__":
    main()
