"""
simulate_bidding_v2.py — Reality-Accurate Backtest Engine
=========================================================
Three curves: Original | +Clamp+Fee | +All Fixes
"""
import sys, logging, csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from model_training.inference import XGBoostGate

DATA_PATH = BASE_DIR / "dataset" / "raw" / "alpha_v1_master.csv"
MODEL_DIR = BASE_DIR / "models" / "alpha_v1"
TRAIN_RATIO = 0.85
OUTPUT_DIR = BASE_DIR / "scripts"

# Reality costs
POLYMARKET_FEE = 0.02
SLIPPAGE = 0.005
GAS_USD = 0.01

# Clamp bounds
P_WIN_MIN, P_WIN_MAX = 0.30, 0.72

# Circuit breaker
CB_WINDOW = 20
CB_PAUSE_TRADES = 50
CB_BUFFER = 0.03

@dataclass
class Trade:
    idx: int; timestamp: str; market_id: str; entry_odds: float
    p_win_raw: float; p_win_clamped: float; ev: float
    kelly_raw: float; kelly_final: float; bet_size: float
    label: int; gross_pnl: float; fee_paid: float; net_pnl: float
    capital_after: float; trailing_wr_20: Optional[float]
    circuit_breaker_active: bool

@dataclass
class Portfolio:
    name: str; initial_capital: float; capital: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    signals_evaluated: int = 0; signals_rejected: int = 0
    rejected_after_fee: int = 0; cb_triggers: int = 0
    cb_paused_until: int = 0; total_fees: float = 0.0
    def __post_init__(self):
        self.capital = self.initial_capital
        self.equity_curve = [self.initial_capital]

def load_and_split(path, ratio):
    df = pd.read_csv(path, low_memory=False).dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df.sort_values("timestamp").reset_index(drop=True)
    market_end = df.groupby("market_id")["timestamp"].max().sort_values()
    mids = market_end.index.tolist()
    split = int(len(mids) * ratio)
    test_markets = set(mids[split:])
    df_test = df[df["market_id"].isin(test_markets)].reset_index(drop=True)
    print(f"  Test: {len(df_test)} rows, {len(test_markets)} markets, "
          f"WIN={int(df_test['label'].sum())} LOSS={int((1-df_test['label']).sum())}")
    return df_test

def row_to_features(row):
    return {k: row.get(k) for k in [
        "obi_value","tfm_value","depth_ratio","obi_tfm_product","obi_tfm_alignment",
        "rv_value","vol_percentile","strike_distance_pct","contest_urgency","ttr_seconds",
        "odds_yes","odds_no","entry_odds","odds_yes_60s_ago","odds_delta_60s",
        "spread_pct","btc_return_1m","confidence_score","timestamp"]}

def calc_ev(p_win, odds):
    if odds <= 0: return float("-inf")
    return p_win * ((1.0/odds) - 1.0) - (1.0 - p_win)

def calc_kelly(p_win, odds):
    if odds <= 0: return 0.0
    b = (1.0/odds) - 1.0
    if b <= 0: return 0.0
    k = (p_win * b - (1.0 - p_win)) / b
    return max(k, 0.0)

def simulate(gate, df_test, pf, use_clamp=False, use_fee=False,
             use_cb=False, use_kelly_cap=False):
    outcomes = []
    odds_hist = []
    for idx, row in df_test.iterrows():
        pf.signals_evaluated += 1
        entry_odds = row["entry_odds"]
        label = int(row["label"])
        if pd.isna(entry_odds) or entry_odds <= 0 or entry_odds >= 1:
            pf.signals_rejected += 1; continue

        raw_features = row_to_features(row)
        result = gate.evaluate_signal(raw_features, entry_odds=entry_odds)
        if result["decision"] != "PASS":
            pf.signals_rejected += 1; continue

        p_win_raw = result["p_win"]
        p_win = np.clip(p_win_raw, P_WIN_MIN, P_WIN_MAX) if use_clamp else p_win_raw

        ev = calc_ev(p_win, entry_odds)
        if ev < 0.04:
            pf.signals_rejected += 1; continue

        kelly_raw = calc_kelly(p_win, entry_odds)
        if use_kelly_cap:
            kelly_final = min(kelly_raw * 0.25, 0.03)
        else:
            kelly_final = min(kelly_raw * 0.25, 0.10)
        bet_size = pf.capital * kelly_final
        bet_size = max(bet_size, 0.0)
        if bet_size <= 0 or bet_size > pf.capital:
            pf.signals_rejected += 1; continue

        # Fee check
        if label == 1:
            gross_payout = bet_size * ((1.0/entry_odds) - 1.0)
        else:
            gross_payout = -bet_size
        if use_fee:
            if gross_payout > 0:
                fee = gross_payout * (POLYMARKET_FEE + SLIPPAGE) + GAS_USD
                net_pnl = gross_payout - fee
            else:
                fee = GAS_USD
                net_pnl = gross_payout - fee
            if label == 1:
                # Check if profitable after fee
                expected_net = bet_size * ((1.0/entry_odds) - 1.0) * (1 - POLYMARKET_FEE - SLIPPAGE) - GAS_USD
                if expected_net < bet_size * 0.01:
                    pf.rejected_after_fee += 1; pf.signals_rejected += 1; continue
        else:
            fee = 0.0
            net_pnl = gross_payout

        # Circuit breaker check
        cb_active = False
        if use_cb and len(pf.trades) >= pf.cb_paused_until:
            pass  # Not paused
        elif use_cb:
            cb_active = True
            pf.signals_rejected += 1; continue

        if use_cb and len(outcomes) >= CB_WINDOW:
            trailing_wr = sum(outcomes[-CB_WINDOW:]) / CB_WINDOW
            avg_odds = np.mean(odds_hist[-CB_WINDOW:])
            breakeven_wr = avg_odds + CB_BUFFER
            if trailing_wr < breakeven_wr:
                pf.cb_triggers += 1
                pf.cb_paused_until = len(pf.trades) + CB_PAUSE_TRADES
                print(f"    ⚡ CB triggered at trade #{len(pf.trades)+1}: "
                      f"trailing_wr={trailing_wr:.3f} < breakeven={breakeven_wr:.3f}, "
                      f"paused for {CB_PAUSE_TRADES} trades")
                cb_active = True
                pf.signals_rejected += 1; continue

        pf.capital += net_pnl
        pf.total_fees += fee
        pf.equity_curve.append(pf.capital)
        outcomes.append(label)
        odds_hist.append(entry_odds)
        trailing_wr = sum(outcomes[-CB_WINDOW:]) / min(len(outcomes), CB_WINDOW)

        pf.trades.append(Trade(
            idx=idx, timestamp=row["timestamp"], market_id=row["market_id"],
            entry_odds=entry_odds, p_win_raw=p_win_raw, p_win_clamped=p_win,
            ev=ev, kelly_raw=kelly_raw, kelly_final=kelly_final,
            bet_size=bet_size, label=label, gross_pnl=gross_payout,
            fee_paid=fee, net_pnl=net_pnl, capital_after=pf.capital,
            trailing_wr_20=trailing_wr, circuit_breaker_active=cb_active))

        if pf.capital <= 0.01:
            print(f"    ☠️ {pf.name} BANKRUPT at trade #{len(pf.trades)}"); break

def max_drawdown(curve):
    if len(curve) < 2: return 0.0
    a = np.array(curve); peak = np.maximum.accumulate(a)
    return float(np.min((a - peak) / peak)) * 100

def save_trade_csv(pf, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trade_id","entry_odds","p_win_raw","p_win_clamped",
                     "kelly_raw","kelly_final","bet_size","outcome",
                     "gross_payout","net_payout_after_fee","capital_after",
                     "trailing_wr_20","circuit_breaker_active"])
        for i, t in enumerate(pf.trades):
            w.writerow([i+1, f"{t.entry_odds:.4f}", f"{t.p_win_raw:.4f}",
                        f"{t.p_win_clamped:.4f}", f"{t.kelly_raw:.4f}",
                        f"{t.kelly_final:.4f}", f"{t.bet_size:.4f}",
                        "WIN" if t.label==1 else "LOSS",
                        f"{t.gross_pnl:.4f}", f"{t.net_pnl:.4f}",
                        f"{t.capital_after:.4f}",
                        f"{t.trailing_wr_20:.4f}" if t.trailing_wr_20 else "",
                        t.circuit_breaker_active])

def plot_curves(portfolios, path):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        fig.suptitle("Alpha V1 — Reality-Accurate Backtest Comparison", fontsize=14, fontweight="bold")
        colors = ["#F44336", "#FF9800", "#4CAF50"]
        for pf, color in zip(portfolios, colors):
            ax.plot(range(len(pf.equity_curve)), pf.equity_curve, color=color,
                    linewidth=1.8, label=pf.name, alpha=0.9)
        ax.axhline(y=50.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_xlabel("Trade #"); ax.set_ylabel("Capital ($)")
        ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n📈 Chart saved: {path}")
    except ImportError:
        print("\n⚠️ Matplotlib not available")

def main():
    print("=" * 66)
    print("  ALPHA V1 — REALITY-ACCURATE BACKTEST ENGINE v2")
    print("=" * 66)

    print("\n[1/5] Loading model...")
    gate = XGBoostGate(); gate.load_model(MODEL_DIR)
    print(f"  Model: {gate.model_version}")

    print("\n[2/5] Loading data (85/15 market-level split)...")
    df_test = load_and_split(DATA_PATH, TRAIN_RATIO)

    print("\n[3/5] Running 3 simulations on $50 portfolio...")

    # Curve A: Original (no fixes)
    print("\n  --- Curve A: Original (baseline) ---")
    pf_a = Portfolio(name="A: Original", initial_capital=50.0)
    simulate(gate, df_test, pf_a, use_clamp=False, use_fee=False, use_cb=False, use_kelly_cap=False)

    # Curve B: Clamp + Fee only
    print("\n  --- Curve B: +Clamp +Fee ---")
    pf_b = Portfolio(name="B: +Clamp+Fee", initial_capital=50.0)
    simulate(gate, df_test, pf_b, use_clamp=True, use_fee=True, use_cb=False, use_kelly_cap=False)

    # Curve C: All fixes
    print("\n  --- Curve C: +All Fixes ---")
    pf_c = Portfolio(name="C: All Fixes", initial_capital=50.0)
    simulate(gate, df_test, pf_c, use_clamp=True, use_fee=True, use_cb=True, use_kelly_cap=True)

    # ── Validation assertions ──
    print("\n[4/5] Validation assertions...")
    errors = []
    if pf_b.trades:
        max_p = max(t.p_win_clamped for t in pf_b.trades)
        if max_p > 0.72 + 1e-6: errors.append(f"Clamp fail B: max p_win={max_p:.4f}")
    if pf_c.trades:
        max_p = max(t.p_win_clamped for t in pf_c.trades)
        if max_p > 0.72 + 1e-6: errors.append(f"Clamp fail C: max p_win={max_p:.4f}")
        max_k = max(t.kelly_final for t in pf_c.trades)
        if max_k > 0.03 + 1e-6: errors.append(f"Kelly cap fail C: max kelly={max_k:.4f}")
    if pf_b.trades and any(t.fee_paid <= 0 for t in pf_b.trades if t.label == 1):
        errors.append("Fee not charged on winning trades (B)")
    if pf_c.trades and any(t.fee_paid <= 0 for t in pf_c.trades if t.label == 1):
        errors.append("Fee not charged on winning trades (C)")

    if errors:
        print("  ❌ VALIDATION FAILED:")
        for e in errors: print(f"     {e}")
        return
    else:
        print("  ✅ All assertions passed")
        if pf_b.trades:
            print(f"     max p_win (B): {max(t.p_win_clamped for t in pf_b.trades):.4f} <= 0.72 ✓")
        if pf_c.trades:
            print(f"     max p_win (C): {max(t.p_win_clamped for t in pf_c.trades):.4f} <= 0.72 ✓")
            print(f"     max kelly (C): {max(t.kelly_final for t in pf_c.trades):.4f} <= 0.03 ✓")

    # ── Report ──
    print("\n[5/5] Results")
    print("=" * 66)

    def stats(pf):
        t = pf.trades; n = len(t)
        w = sum(1 for x in t if x.label==1)
        wr = w/n if n else 0
        pnl = pf.capital - pf.initial_capital
        pnl_pct = pnl / pf.initial_capital * 100
        dd = max_drawdown(pf.equity_curve)
        gp = sum(x.net_pnl for x in t if x.net_pnl > 0)
        gl = abs(sum(x.net_pnl for x in t if x.net_pnl < 0))
        pfact = gp / (gl + 1e-8)
        avg_bet = np.mean([x.bet_size for x in t]) if t else 0
        avg_ev = np.mean([x.ev for x in t]) if t else 0
        return {
            "final": pf.capital, "pnl": pnl, "pnl_pct": pnl_pct,
            "dd": dd, "n": n, "w": w, "l": n-w, "wr": wr,
            "pf": pfact, "fees": pf.total_fees, "cb": pf.cb_triggers,
            "rej_fee": pf.rejected_after_fee, "avg_bet": avg_bet, "avg_ev": avg_ev,
        }

    sa, sb, sc = stats(pf_a), stats(pf_b), stats(pf_c)

    print(f"""
┌─────────────────────┬────────────┬────────────┬────────────┐
│ Metric              │  Original  │ +Clamp+Fee │ +All Fixes │
├─────────────────────┼────────────┼────────────┼────────────┤
│ Final Capital       │ ${sa['final']:>8.2f}  │ ${sb['final']:>8.2f}  │ ${sc['final']:>8.2f}  │
│ Net PnL             │ ${sa['pnl']:>+8.2f}  │ ${sb['pnl']:>+8.2f}  │ ${sc['pnl']:>+8.2f}  │
│ Net PnL %           │ {sa['pnl_pct']:>+8.1f}%  │ {sb['pnl_pct']:>+8.1f}%  │ {sc['pnl_pct']:>+8.1f}%  │
│ Max Drawdown        │ {sa['dd']:>8.1f}%  │ {sb['dd']:>8.1f}%  │ {sc['dd']:>8.1f}%  │
│ Trades Executed     │ {sa['n']:>10}  │ {sb['n']:>10}  │ {sc['n']:>10}  │
│ Wins / Losses       │ {sa['w']:>4}/{sa['l']:<5}  │ {sb['w']:>4}/{sb['l']:<5}  │ {sc['w']:>4}/{sc['l']:<5}  │
│ Win Rate            │ {sa['wr']:>8.1%}  │ {sb['wr']:>8.1%}  │ {sc['wr']:>8.1%}  │
│ Profit Factor       │ {sa['pf']:>10.2f}  │ {sb['pf']:>10.2f}  │ {sc['pf']:>10.2f}  │
│ Circuit Breaker Hit │        N/A  │        N/A  │ {sc['cb']:>10}  │
│ Rejected After Fee  │        N/A  │ {sb['rej_fee']:>10}  │ {sc['rej_fee']:>10}  │
│ Total Fees Paid     │       $0.00  │ ${sb['fees']:>8.2f}  │ ${sc['fees']:>8.2f}  │
│ Avg Bet Size        │ ${sa['avg_bet']:>8.4f}  │ ${sb['avg_bet']:>8.4f}  │ ${sc['avg_bet']:>8.4f}  │
│ Avg EV per Trade    │ {sa['avg_ev']:>+10.4f}  │ {sb['avg_ev']:>+10.4f}  │ {sc['avg_ev']:>+10.4f}  │
└─────────────────────┴────────────┴────────────┴────────────┘""")

    # Trade logs
    for pf in [pf_a, pf_b, pf_c]:
        csv_path = OUTPUT_DIR / f"backtest_trades_{pf.name.split(':')[0].strip().lower()}.csv"
        save_trade_csv(pf, csv_path)
        print(f"  📄 Trade log: {csv_path.name}")

    # Plot
    plot_curves([pf_a, pf_b, pf_c], OUTPUT_DIR / "backtest_curve_v2.png")

    # Circuit breaker details
    if pf_c.cb_triggers > 0:
        print(f"\n  Circuit Breaker Details (Curve C):")
        print(f"    Triggers: {pf_c.cb_triggers}")
        print(f"    Paused until trade index: {pf_c.cb_paused_until}")

    print(f"\n{'='*66}")
    print(f"  SIMULATION COMPLETE")
    print(f"{'='*66}")

if __name__ == "__main__":
    main()
