"""
simulate_predator_v3.py — Final validation: A1/A2 pruned, $50 capital, full bet sizing matrix.
"""

import sys, os, math
from dataclasses import dataclass
from typing import List
from collections import defaultdict

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from src.zone_matrix import classify_zone, ZoneResult, ALPHA_ZONES

# ── PRUNED ZONES: A1 and A2 moved to NEUTRAL ────────────────────
PRUNED_ZONES = {"A1", "A2"}
QUARTER_KELLY_ZONES = {"A3"}

@dataclass
class PendingPosition:
    trade_id: int; zone_id: str; entry_time: pd.Timestamp
    resolution_time: pd.Timestamp; bet_size: float; entry_odds: float
    signal_direction: str; strike_price: float; label: float

@dataclass
class TradeResult:
    trade_id: int; zone_id: str; entry_odds: float; bet_size: float
    pnl: float; outcome: str; capital_before: float; capital_after: float
    entry_time: pd.Timestamp; ttr_minutes: float; distance_usd: float

@dataclass
class RejectionRecord:
    reason: str; zone_id: str


class PredatorSimulatorV3:
    def __init__(self, initial_capital=50.0, max_positions=3, max_exposure_pct=0.60,
                 kelly_cap=0.15, min_bet=1.0, fee_pct=0.02, slippage_pct=0.005,
                 gas_flat=0.01, loss_decay=0.10, kelly_floor=0.50):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_positions = max_positions
        self.max_exposure_pct = max_exposure_pct
        self.kelly_cap = kelly_cap
        self.min_bet = min_bet
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.gas_flat = gas_flat
        self.loss_decay = loss_decay
        self.kelly_floor = kelly_floor

        self.pending: List[PendingPosition] = []
        self.trades: List[TradeResult] = []
        self.rejections: List[RejectionRecord] = []
        self.equity_curve: List[float] = [initial_capital]
        self._trade_counter = 0
        self._consecutive_losses = 0
        self._peak = initial_capital

    @property
    def pending_exposure(self):
        return sum(p.bet_size for p in self.pending)

    @property
    def available(self):
        return max(0, self.capital * self.max_exposure_pct - self.pending_exposure)

    def _resolve(self, now):
        keep = []
        for p in self.pending:
            if now >= p.resolution_time:
                won = p.label == 1.0
                cap_before = self.capital
                if won:
                    mult = (1.0 / p.entry_odds) * (1.0 - self.fee_pct - self.slippage_pct) - self.gas_flat
                    pnl = p.bet_size * mult - p.bet_size
                    self._consecutive_losses = 0
                else:
                    pnl = -p.bet_size
                    self._consecutive_losses += 1
                self.capital += pnl
                self._peak = max(self._peak, self.capital)
                self.trades.append(TradeResult(
                    p.trade_id, p.zone_id, p.entry_odds, p.bet_size, pnl,
                    "WIN" if won else "LOSS", cap_before, self.capital,
                    p.entry_time, 0, 0))
                self.equity_curve.append(self.capital)
            else:
                keep.append(p)
        self.pending = keep

    def _bet_size(self, zone):
        k = zone.kelly_fraction
        if zone.zone_id in QUARTER_KELLY_ZONES:
            k /= 4.0
        k = min(k, self.kelly_cap)
        loss_mult = max(self.kelly_floor, 1.0 - self._consecutive_losses * self.loss_decay)
        k *= loss_mult
        raw = self.capital * k
        raw = min(raw, self.available)
        bet = max(self.min_bet, math.floor(raw)) if raw >= self.min_bet else 0.0
        return float(bet), k

    def process(self, ts, ttr, dist, odds, direction, strike, label):
        self._trade_counter += 1
        self._resolve(ts)
        zone = classify_zone(ttr, dist, odds)

        if zone.zone_type == "DEATH":
            self.rejections.append(RejectionRecord("DEATH_ZONE", zone.zone_id)); return
        if zone.zone_type == "NEUTRAL" or zone.zone_id in PRUNED_ZONES:
            self.rejections.append(RejectionRecord("NEUTRAL/PRUNED", zone.zone_id)); return
        if len(self.pending) >= self.max_positions:
            self.rejections.append(RejectionRecord("MAX_POSITIONS", zone.zone_id)); return
        if self.available < self.min_bet:
            self.rejections.append(RejectionRecord("EXPOSURE_CAP", zone.zone_id)); return

        bet, _ = self._bet_size(zone)
        if bet < self.min_bet or bet > self.capital:
            self.rejections.append(RejectionRecord("BELOW_MIN_BET", zone.zone_id)); return

        res_time = ts + pd.Timedelta(minutes=ttr)
        self.pending.append(PendingPosition(
            self._trade_counter, zone.zone_id, ts, res_time, bet, odds, direction, strike, label))

    def finalize(self):
        if self.pending:
            self._resolve(pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=365))


def main():
    print("=" * 100)
    print("PREDATOR V3 — PRUNED A1/A2 | CAPPED KELLY 15% | $50 CAPITAL")
    print("=" * 100)

    df = pd.read_csv("dataset/raw/alpha_v1_master.csv", low_memory=False)
    df = df[df['label'].notnull()].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['ttr_minutes'] = df['ttr_seconds'] / 60.0
    df['distance_to_strike'] = abs(df['binance_price'] - df['strike_price'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"Dataset: {len(df)} labeled rows | {df['timestamp'].min().date()} -> {df['timestamp'].max().date()}")
    print(f"Pruned zones: A1, A2 (moved to NEUTRAL)")
    print(f"Config: Kelly Cap=15%, A3=Quarter-Kelly, Max Pos=3, Exposure=60%")
    print()

    sim = PredatorSimulatorV3(initial_capital=50.0)
    for _, row in df.iterrows():
        sim.process(row['timestamp'], row['ttr_minutes'], row['distance_to_strike'],
                    row['entry_odds'], row['signal_direction'], row['strike_price'], row['label'])
    sim.finalize()

    trades = sim.trades
    wins = [t for t in trades if t.outcome == "WIN"]
    losses = [t for t in trades if t.outcome == "LOSS"]
    total_pnl = sum(t.pnl for t in trades)
    wr = len(wins) / len(trades) * 100 if trades else 0
    gp = sum(t.pnl for t in wins) if wins else 0
    gl = abs(sum(t.pnl for t in losses)) if losses else 1e-8
    pf = gp / (gl + 1e-8)

    ec = np.array(sim.equity_curve)
    peak = np.maximum.accumulate(ec)
    dd = (ec - peak) / (peak + 1e-8)
    max_dd = float(np.min(dd))

    # ── PNL SUMMARY ──────────────────────────────────────────────
    print("=" * 100)
    print("PNL SUMMARY")
    print("=" * 100)
    print(f"  Initial Capital:    ${sim.initial_capital:>10.2f}")
    print(f"  Final Capital:      ${sim.capital:>10.2f}")
    print(f"  Total PnL:          ${total_pnl:>+10.2f} ({total_pnl/sim.initial_capital*100:>+.1f}%)")
    print(f"  Peak Capital:       ${sim._peak:>10.2f}")
    print(f"  Trades Executed:    {len(trades):>10}")
    print(f"  Win / Loss:         {len(wins):>4} / {len(losses):<4} ({wr:.1f}%)")
    print(f"  Profit Factor:      {pf:>10.3f}")
    print(f"  Max Drawdown:       {max_dd:>10.1%}")
    avg_w = np.mean([t.bet_size for t in wins]) if wins else 0
    avg_l = np.mean([t.bet_size for t in losses]) if losses else 0
    print(f"  Avg Bet (Win):      ${avg_w:>10.2f}")
    print(f"  Avg Bet (Loss):     ${avg_l:>10.2f}")
    print(f"  Bet Asymmetry:      {avg_l/(avg_w+1e-8):>10.2f}x")

    # ── ZONE PERFORMANCE ─────────────────────────────────────────
    print()
    print("=" * 100)
    print("ZONE PERFORMANCE")
    print("=" * 100)
    print(f"  {'Zone':<6} {'Trades':>6} {'W':>4} {'L':>4} {'WR':>7} {'PnL':>10} {'TotBet':>10} {'ROI':>8} {'AvgBet':>8}")
    print(f"  {'-'*6} {'-'*6} {'-'*4} {'-'*4} {'-'*7} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    zs = defaultdict(lambda: {"w":0,"l":0,"pnl":0.0,"bets":0.0,"sizes":[]})
    for t in trades:
        z = zs[t.zone_id]
        z["pnl"] += t.pnl; z["bets"] += t.bet_size; z["sizes"].append(t.bet_size)
        if t.outcome == "WIN": z["w"] += 1
        else: z["l"] += 1
    for zid in sorted(zs.keys()):
        z = zs[zid]; n = z["w"]+z["l"]
        wr_z = z["w"]/n*100; roi = z["pnl"]/(z["bets"]+1e-8)*100; avg = z["bets"]/n
        print(f"  {zid:<6} {n:>6} {z['w']:>4} {z['l']:>4} {wr_z:>6.1f}% ${z['pnl']:>+9.2f} ${z['bets']:>9.2f} {roi:>+6.1f}% ${avg:>7.2f}")

    # ── BET SIZING MATRIX BY BALANCE LEVEL ───────────────────────
    print()
    print("=" * 100)
    print("BET SIZING MATRIX — How bet sizes scale with your balance")
    print("=" * 100)

    # Compute theoretical bet sizes at various balance levels
    from src.zone_matrix import ALPHA_ZONES as AZ
    active_zones = [z for z in AZ if z["zone_id"] not in PRUNED_ZONES]

    balance_levels = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000]

    header = f"  {'Balance':>8}"
    for z in active_zones:
        header += f" | {z['zone_id']:>5}"
    header += " | MaxExp"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for bal in balance_levels:
        row = f"  ${bal:>7}"
        for z in active_zones:
            k = z["kelly"]
            if z["zone_id"] in QUARTER_KELLY_ZONES:
                k /= 4.0
            k = min(k, 0.15)
            bet = max(1, math.floor(bal * k))
            bet = min(bet, math.floor(bal * 0.60))  # exposure cap
            row += f" | ${bet:>4}"
        max_exp = math.floor(bal * 0.60)
        row += f" | ${max_exp:>5}"
        print(row)

    print()
    print("  Notes:")
    print("  - A3 uses Quarter-Kelly (kelly/4)")
    print("  - All zones capped at 15% per trade")
    print("  - MaxExp = 60% of balance (max concurrent exposure)")
    print("  - Consecutive losses reduce sizing by 10%/loss (floor 50%)")

    # ── TRADE-BY-TRADE GROWTH LOG (every 10 trades) ──────────────
    print()
    print("=" * 100)
    print("GROWTH LOG — Trade-by-trade progression (every 10 trades)")
    print("=" * 100)
    print(f"  {'#':>4} {'Zone':>5} {'Bet':>6} {'Odds':>6} {'Result':>6} {'PnL':>9} {'Capital':>10} {'DD%':>7}")
    print(f"  {'-'*4} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*9} {'-'*10} {'-'*7}")

    running_peak = sim.initial_capital
    for i, t in enumerate(trades):
        running_peak = max(running_peak, t.capital_after)
        cur_dd = (t.capital_after - running_peak) / (running_peak + 1e-8) * 100

        # Print every 10th trade, first 5, last 5
        show = (i < 5) or (i >= len(trades)-5) or (i % 10 == 0)
        if show:
            print(f"  {i+1:>4} {t.zone_id:>5} ${t.bet_size:>5.0f} {t.entry_odds:>5.3f} {t.outcome:>6} ${t.pnl:>+8.2f} ${t.capital_after:>9.2f} {cur_dd:>+6.1f}%")
        elif i == 5:
            print(f"  {'...':>4}")

    # ── REJECTION SUMMARY ────────────────────────────────────────
    print()
    print("=" * 100)
    print("REJECTION SUMMARY")
    print("=" * 100)
    rej_reasons = defaultdict(int)
    for r in sim.rejections:
        rej_reasons[r.reason] += 1
    total_signals = len(trades) + len(sim.rejections)
    print(f"  Total Signals: {total_signals} | Accepted: {len(trades)} ({len(trades)/total_signals*100:.1f}%) | Rejected: {len(sim.rejections)}")
    for reason, count in sorted(rej_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:<25} {count:>6} ({count/total_signals*100:>5.1f}%)")

    # ── DRAWDOWN PERIODS ─────────────────────────────────────────
    print()
    print("=" * 100)
    print("DRAWDOWN PERIODS (> 15%)")
    print("=" * 100)

    in_dd = False; dd_start = 0; dd_count = 0
    for i in range(len(ec)):
        if dd[i] < -0.15:
            if not in_dd:
                dd_start = i; in_dd = True
        else:
            if in_dd:
                dd_count += 1
                dd_depth = float(np.min(dd[dd_start:i+1]))
                print(f"  Period {dd_count}: Trade {dd_start}-{i} ({i-dd_start} trades) | Depth: {dd_depth:.1%} | Recovery: ${ec[i]:.2f}")
                in_dd = False
    if in_dd:
        dd_count += 1
        dd_depth = float(np.min(dd[dd_start:]))
        print(f"  Period {dd_count}: Trade {dd_start}-{len(ec)-1} ({len(ec)-1-dd_start} trades) | Depth: {dd_depth:.1%} | NO recovery")

    # ── SAVE ─────────────────────────────────────────────────────
    out_dir = "scripts/output"
    os.makedirs(out_dir, exist_ok=True)

    trades_df = pd.DataFrame([{
        "trade_id": t.trade_id, "zone_id": t.zone_id,
        "entry_odds": t.entry_odds, "bet_size": t.bet_size,
        "pnl": round(t.pnl, 4), "outcome": t.outcome,
        "capital_before": round(t.capital_before, 2),
        "capital_after": round(t.capital_after, 2),
        "entry_time": t.entry_time,
    } for t in trades])
    trades_path = os.path.join(out_dir, "predator_v3_trades.csv")
    trades_df.to_csv(trades_path, index=False)

    eq_df = pd.DataFrame({"trade_index": range(len(sim.equity_curve)), "equity": sim.equity_curve})
    eq_path = os.path.join(out_dir, "predator_v3_equity.csv")
    eq_df.to_csv(eq_path, index=False)

    print(f"\n  Trades: {trades_path}")
    print(f"  Equity: {eq_path}")

    # ── FINAL VERDICT ────────────────────────────────────────────
    ret_pct = total_pnl / sim.initial_capital * 100
    print()
    print("=" * 100)
    if ret_pct > 80:
        print(f"VERDICT: RETURN {ret_pct:+.1f}% > 80% THRESHOLD — APPROVED FOR PRODUCTION DEPLOYMENT")
    else:
        print(f"VERDICT: RETURN {ret_pct:+.1f}% — {'PASS' if ret_pct > 0 else 'FAIL'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
