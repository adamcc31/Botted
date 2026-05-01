"""
simulate_predator_v2.py — Capped Kelly Stress Test across multiple capital levels.

Fixes from v1:
  - Kelly fraction CAPPED at 15% per trade (was 25% uncapped Full Kelly)
  - Zones A2 & A3 downgraded to TIER-3 (Quarter-Kelly)
  - Runs $10, $20, $50, $100, $1000 initial capital scenarios
  - Full matrix report with zone-level breakdown per capital level
"""

import sys
import os
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from src.zone_matrix import classify_zone, ZoneResult, ALPHA_ZONES


# ── Data Structures ──────────────────────────────────────────────

@dataclass
class PendingPosition:
    trade_id: int
    zone_id: str
    entry_time: pd.Timestamp
    resolution_time: pd.Timestamp
    bet_size: float
    entry_odds: float
    signal_direction: str
    strike_price: float
    label: float


@dataclass
class TradeResult:
    trade_id: int
    zone_id: str
    entry_odds: float
    bet_size: float
    pnl: float
    outcome: str
    capital_before: float
    capital_after: float
    entry_time: pd.Timestamp


@dataclass
class RejectionRecord:
    reason: str
    zone_id: str


# ── Predator Simulator V2 ───────────────────────────────────────

# Zones that get Quarter-Kelly instead of Full (downgraded from stress test)
QUARTER_KELLY_ZONES = {"A2", "A3"}

class PredatorSimulatorV2:
    def __init__(
        self,
        initial_capital: float = 100.0,
        max_positions: int = 3,
        max_exposure_pct: float = 0.60,
        kelly_cap: float = 0.15,       # <-- CAPPED at 15% (was 25%)
        min_bet: float = 1.0,
        fee_pct: float = 0.02,
        slippage_pct: float = 0.005,
        gas_flat: float = 0.01,
        consecutive_loss_decay: float = 0.10,
        kelly_floor_mult: float = 0.50,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_positions = max_positions
        self.max_exposure_pct = max_exposure_pct
        self.kelly_cap = kelly_cap
        self.min_bet = min_bet
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.gas_flat = gas_flat
        self.consecutive_loss_decay = consecutive_loss_decay
        self.kelly_floor_mult = kelly_floor_mult

        self.pending: List[PendingPosition] = []
        self.trades: List[TradeResult] = []
        self.rejections: List[RejectionRecord] = []
        self.equity_curve: List[float] = [initial_capital]
        self._trade_counter = 0
        self._consecutive_losses = 0
        self._peak_capital = initial_capital

    @property
    def pending_exposure(self) -> float:
        return sum(p.bet_size for p in self.pending)

    @property
    def available_for_new_bets(self) -> float:
        max_total = self.capital * self.max_exposure_pct
        return max(0, max_total - self.pending_exposure)

    def _resolve_expired(self, current_time: pd.Timestamp) -> None:
        still_pending = []
        for pos in self.pending:
            if current_time >= pos.resolution_time:
                won = (pos.label == 1.0)
                capital_before = self.capital

                if won:
                    payout_mult = (1.0 / pos.entry_odds) * (1.0 - self.fee_pct - self.slippage_pct) - self.gas_flat
                    pnl = pos.bet_size * payout_mult - pos.bet_size
                    self._consecutive_losses = 0
                else:
                    pnl = -pos.bet_size
                    self._consecutive_losses += 1

                self.capital += pnl
                if self.capital > self._peak_capital:
                    self._peak_capital = self.capital

                self.trades.append(TradeResult(
                    trade_id=pos.trade_id, zone_id=pos.zone_id,
                    entry_odds=pos.entry_odds, bet_size=pos.bet_size,
                    pnl=pnl, outcome="WIN" if won else "LOSS",
                    capital_before=capital_before, capital_after=self.capital,
                    entry_time=pos.entry_time,
                ))
                self.equity_curve.append(self.capital)
            else:
                still_pending.append(pos)
        self.pending = still_pending

    def _compute_bet_size(self, zone: ZoneResult) -> float:
        if zone.kelly_fraction <= 0:
            return 0.0

        # Determine effective Kelly
        kelly = zone.kelly_fraction

        # Quarter-Kelly for downgraded zones
        if zone.zone_id in QUARTER_KELLY_ZONES:
            kelly = kelly / 4.0

        # Cap at 15%
        kelly = min(kelly, self.kelly_cap)

        # Consecutive loss decay
        loss_mult = max(
            self.kelly_floor_mult,
            1.0 - self._consecutive_losses * self.consecutive_loss_decay,
        )
        kelly *= loss_mult

        raw_bet = self.capital * kelly
        raw_bet = min(raw_bet, self.available_for_new_bets)

        # Integer rounding
        bet = max(self.min_bet, math.floor(raw_bet)) if raw_bet >= self.min_bet else 0.0
        return float(bet)

    def process_signal(self, timestamp, ttr_minutes, distance_usd,
                       entry_odds, signal_direction, strike_price, label):
        self._trade_counter += 1
        self._resolve_expired(timestamp)

        zone = classify_zone(ttr_minutes, distance_usd, entry_odds)

        if zone.zone_type == "DEATH":
            self.rejections.append(RejectionRecord("DEATH_ZONE", zone.zone_id))
            return
        if zone.zone_type == "NEUTRAL":
            self.rejections.append(RejectionRecord("NEUTRAL_ZONE", zone.zone_id))
            return
        if len(self.pending) >= self.max_positions:
            self.rejections.append(RejectionRecord("MAX_POSITIONS", zone.zone_id))
            return
        if self.available_for_new_bets < self.min_bet:
            self.rejections.append(RejectionRecord("EXPOSURE_CAP", zone.zone_id))
            return

        bet_size = self._compute_bet_size(zone)
        if bet_size < self.min_bet or bet_size > self.capital:
            self.rejections.append(RejectionRecord("BELOW_MIN_BET", zone.zone_id))
            return

        resolution_time = timestamp + pd.Timedelta(minutes=ttr_minutes)
        self.pending.append(PendingPosition(
            trade_id=self._trade_counter, zone_id=zone.zone_id,
            entry_time=timestamp, resolution_time=resolution_time,
            bet_size=bet_size, entry_odds=entry_odds,
            signal_direction=signal_direction, strike_price=strike_price,
            label=label,
        ))

    def finalize(self):
        if self.pending:
            future = pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=365)
            self._resolve_expired(future)

    def get_report(self) -> dict:
        trades = self.trades
        if not trades:
            return {"trades": 0}

        wins = [t for t in trades if t.outcome == "WIN"]
        losses = [t for t in trades if t.outcome == "LOSS"]
        total_pnl = sum(t.pnl for t in trades)
        wr = len(wins) / len(trades) * 100

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1e-8
        pf = gross_profit / (gross_loss + 1e-8)

        # Drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / (peak + 1e-8)
        max_dd = float(np.min(dd))

        # Sharpe (per-trade)
        rets = [t.pnl / (t.bet_size + 1e-8) for t in trades]
        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-8)) * math.sqrt(len(trades)) if len(trades) > 1 else 0

        # Zone breakdown
        zone_stats = {}
        for t in trades:
            if t.zone_id not in zone_stats:
                zone_stats[t.zone_id] = {"w": 0, "l": 0, "pnl": 0.0, "bets": 0.0}
            zone_stats[t.zone_id]["bets"] += t.bet_size
            zone_stats[t.zone_id]["pnl"] += t.pnl
            if t.outcome == "WIN":
                zone_stats[t.zone_id]["w"] += 1
            else:
                zone_stats[t.zone_id]["l"] += 1

        # Rejection stats
        rej_reasons = defaultdict(int)
        for r in self.rejections:
            rej_reasons[r.reason] += 1

        # Consecutive loss max
        max_consec = 0
        cur_consec = 0
        for t in trades:
            if t.outcome == "LOSS":
                cur_consec += 1
                max_consec = max(max_consec, cur_consec)
            else:
                cur_consec = 0

        return {
            "initial": self.initial_capital,
            "final": self.capital,
            "pnl": total_pnl,
            "pnl_pct": total_pnl / self.initial_capital * 100,
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "wr": wr,
            "pf": pf,
            "max_dd": max_dd,
            "sharpe": sharpe,
            "peak": self._peak_capital,
            "max_consec_loss": max_consec,
            "zone_stats": zone_stats,
            "rej_reasons": dict(rej_reasons),
            "race_conditions": rej_reasons.get("MAX_POSITIONS", 0) + rej_reasons.get("EXPOSURE_CAP", 0),
            "avg_bet_win": np.mean([t.bet_size for t in wins]) if wins else 0,
            "avg_bet_loss": np.mean([t.bet_size for t in losses]) if losses else 0,
            "equity_curve": self.equity_curve,
        }


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 100)
    print("PREDATOR V2 — CAPPED KELLY MULTI-CAPITAL STRESS TEST")
    print("=" * 100)

    csv_path = "dataset/raw/alpha_v1_master.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df['label'].notnull()].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['ttr_minutes'] = df['ttr_seconds'] / 60.0
    df['distance_to_strike'] = abs(df['binance_price'] - df['strike_price'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"Dataset: {csv_path} | Labeled rows: {len(df)}")
    print(f"Time span: {df['timestamp'].min().date()} -> {df['timestamp'].max().date()}")
    print(f"Config: Kelly Cap=15%, Max Pos=3, Exposure Cap=60%, Loss Decay=10%/loss, Floor=50%")
    print(f"Downgraded zones: A2, A3 -> Quarter-Kelly")
    print()

    capitals = [10, 20, 50, 100, 1000]
    reports = {}

    for cap in capitals:
        sim = PredatorSimulatorV2(initial_capital=float(cap))
        for _, row in df.iterrows():
            sim.process_signal(
                timestamp=row['timestamp'],
                ttr_minutes=row['ttr_minutes'],
                distance_usd=row['distance_to_strike'],
                entry_odds=row['entry_odds'],
                signal_direction=row['signal_direction'],
                strike_price=row['strike_price'],
                label=row['label'],
            )
        sim.finalize()
        reports[cap] = sim.get_report()

    # ── MASTER COMPARISON TABLE ──────────────────────────────────

    print("=" * 100)
    print("MASTER COMPARISON: CAPPED KELLY ACROSS CAPITAL LEVELS")
    print("=" * 100)
    print(f"{'Capital':>8} | {'Final':>10} | {'PnL':>10} | {'PnL%':>8} | {'Trades':>6} | {'WR':>6} | {'PF':>6} | {'MaxDD':>8} | {'Peak':>10} | {'MaxCL':>5} | {'RC':>4}")
    print("-" * 100)
    for cap in capitals:
        r = reports[cap]
        print(f"${cap:>7} | ${r['final']:>9.2f} | ${r['pnl']:>+9.2f} | {r['pnl_pct']:>+6.1f}% | {r['trades']:>6} | {r['wr']:>5.1f}% | {r['pf']:>5.2f} | {r['max_dd']:>7.1%} | ${r['peak']:>9.2f} | {r['max_consec_loss']:>5} | {r['race_conditions']:>4}")

    print()
    print("Legend: PF=Profit Factor, MaxDD=Max Drawdown, Peak=Highest Capital, MaxCL=Max Consecutive Losses, RC=Race Conditions")

    # ── BET SIZE ASYMMETRY CHECK ─────────────────────────────────

    print()
    print("=" * 100)
    print("BET SIZE ASYMMETRY CHECK (Avg Win Bet vs Avg Loss Bet)")
    print("=" * 100)
    print(f"{'Capital':>8} | {'Avg Win Bet':>12} | {'Avg Loss Bet':>13} | {'Ratio (L/W)':>12} | {'Verdict':>10}")
    print("-" * 100)
    for cap in capitals:
        r = reports[cap]
        ratio = r['avg_bet_loss'] / (r['avg_bet_win'] + 1e-8)
        verdict = "OK" if ratio < 1.5 else "WARN" if ratio < 2.0 else "DANGER"
        print(f"${cap:>7} | ${r['avg_bet_win']:>11.2f} | ${r['avg_bet_loss']:>12.2f} | {ratio:>11.2f}x | {verdict:>10}")

    # ── ZONE PERFORMANCE PER CAPITAL ─────────────────────────────

    all_zones_seen = set()
    for cap in capitals:
        all_zones_seen.update(reports[cap]['zone_stats'].keys())
    all_zones_sorted = sorted(all_zones_seen)

    print()
    print("=" * 100)
    print("ZONE PERFORMANCE MATRIX")
    print("=" * 100)

    for zone_id in all_zones_sorted:
        print(f"\n--- Zone {zone_id} ---")
        print(f"  {'Capital':>8} | {'Trades':>6} | {'W':>4} | {'L':>4} | {'WR':>6} | {'PnL':>10} | {'TotalBet':>10} | {'ROI':>8}")
        print(f"  {'-'*8} | {'-'*6} | {'-'*4} | {'-'*4} | {'-'*6} | {'-'*10} | {'-'*10} | {'-'*8}")
        for cap in capitals:
            zs = reports[cap]['zone_stats'].get(zone_id)
            if zs:
                total = zs['w'] + zs['l']
                wr = zs['w'] / total * 100 if total > 0 else 0
                roi = zs['pnl'] / (zs['bets'] + 1e-8) * 100
                print(f"  ${cap:>7} | {total:>6} | {zs['w']:>4} | {zs['l']:>4} | {wr:>5.1f}% | ${zs['pnl']:>+9.2f} | ${zs['bets']:>9.2f} | {roi:>+6.1f}%")
            else:
                print(f"  ${cap:>7} |      0 |    - |    - |     - |         - |         - |       -")

    # ── REJECTION ANALYSIS ───────────────────────────────────────

    print()
    print("=" * 100)
    print("REJECTION ANALYSIS PER CAPITAL")
    print("=" * 100)
    all_reasons = set()
    for cap in capitals:
        all_reasons.update(reports[cap]['rej_reasons'].keys())
    all_reasons_sorted = sorted(all_reasons)

    header = f"{'Reason':<25}"
    for cap in capitals:
        header += f" | ${cap:>6}"
    print(header)
    print("-" * len(header))
    for reason in all_reasons_sorted:
        row = f"{reason:<25}"
        for cap in capitals:
            count = reports[cap]['rej_reasons'].get(reason, 0)
            row += f" | {count:>7}"
        print(row)

    # ── EQUITY CURVE SNAPSHOTS ───────────────────────────────────

    print()
    print("=" * 100)
    print("EQUITY CURVE SNAPSHOTS (Every ~30 trades)")
    print("=" * 100)

    header = f"{'Trade#':>7}"
    for cap in capitals:
        header += f" | ${cap:>8}"
    print(header)
    print("-" * len(header))

    max_trades = max(len(reports[cap]['equity_curve']) for cap in capitals)
    step = max(1, max_trades // 15)
    for i in range(0, max_trades, step):
        row = f"{i:>7}"
        for cap in capitals:
            ec = reports[cap]['equity_curve']
            if i < len(ec):
                row += f" | ${ec[i]:>8.2f}"
            else:
                row += f" | {'---':>9}"
        print(row)
    # Last row
    row = f"{'FINAL':>7}"
    for cap in capitals:
        row += f" | ${reports[cap]['final']:>8.2f}"
    print(row)

    # ── DRAWDOWN DEEP DIVE ───────────────────────────────────────

    print()
    print("=" * 100)
    print("DRAWDOWN ANALYSIS")
    print("=" * 100)
    print(f"{'Capital':>8} | {'MaxDD':>8} | {'Peak->Trough':>20} | {'Recovery':>10} | {'DD Periods(>10%)':>16}")
    print("-" * 80)
    for cap in capitals:
        r = reports[cap]
        ec = np.array(r['equity_curve'])
        peak = np.maximum.accumulate(ec)
        dd = (ec - peak) / (peak + 1e-8)
        max_dd_idx = np.argmin(dd)
        peak_val = peak[max_dd_idx]
        trough_val = ec[max_dd_idx]

        # Did it recover?
        recovered = "YES" if ec[-1] >= peak_val * 0.95 else "NO"

        # DD periods > 10%
        in_dd = dd < -0.10
        dd_periods = 0
        prev = False
        for d in in_dd:
            if d and not prev:
                dd_periods += 1
            prev = d

        print(f"${cap:>7} | {r['max_dd']:>7.1%} | ${peak_val:>8.2f} -> ${trough_val:>7.2f} | {recovered:>10} | {dd_periods:>16}")

    # ── SAVE ALL RESULTS ─────────────────────────────────────────

    out_dir = "scripts/output"
    os.makedirs(out_dir, exist_ok=True)

    # Summary CSV
    summary_rows = []
    for cap in capitals:
        r = reports[cap]
        summary_rows.append({
            "initial_capital": cap,
            "final_capital": round(r['final'], 2),
            "total_pnl": round(r['pnl'], 2),
            "pnl_pct": round(r['pnl_pct'], 2),
            "trades": r['trades'],
            "wins": r['wins'],
            "losses": r['losses'],
            "win_rate": round(r['wr'], 2),
            "profit_factor": round(r['pf'], 3),
            "max_drawdown": round(r['max_dd'], 4),
            "sharpe": round(r['sharpe'], 3),
            "peak_capital": round(r['peak'], 2),
            "max_consecutive_losses": r['max_consec_loss'],
            "race_conditions": r['race_conditions'],
            "avg_bet_win": round(r['avg_bet_win'], 2),
            "avg_bet_loss": round(r['avg_bet_loss'], 2),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "predator_v2_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # Per-capital trade logs
    for cap in capitals:
        r = reports[cap]
        trades_data = [{
            "trade_id": t.trade_id, "zone_id": t.zone_id,
            "entry_odds": t.entry_odds, "bet_size": t.bet_size,
            "pnl": round(t.pnl, 4), "outcome": t.outcome,
            "capital_before": round(t.capital_before, 2),
            "capital_after": round(t.capital_after, 2),
        } for t in [tr for tr in [] or reports[cap].get('_trades_raw', [])]]
        # We save equity curves instead (more useful)
        ec_df = pd.DataFrame({"trade_index": range(len(r['equity_curve'])), "equity": r['equity_curve']})
        ec_path = os.path.join(out_dir, f"predator_v2_equity_{cap}.csv")
        ec_df.to_csv(ec_path, index=False)

    print(f"\n--- FILES SAVED ---")
    print(f"  Summary: {summary_path}")
    for cap in capitals:
        print(f"  Equity ${cap}: {os.path.join(out_dir, f'predator_v2_equity_{cap}.csv')}")

    print(f"\n{'=' * 100}")
    profitable = [cap for cap in capitals if reports[cap]['pnl'] > 0]
    if profitable:
        best = max(profitable, key=lambda c: reports[c]['pnl_pct'])
        print(f"BEST PERFORMER: ${best} capital -> ${reports[best]['final']:.2f} ({reports[best]['pnl_pct']:+.1f}%)")
    else:
        print("NO PROFITABLE SCENARIO FOUND — further optimization needed")

    least_dd = min(capitals, key=lambda c: abs(reports[c]['max_dd']))
    print(f"SAFEST PROFILE: ${least_dd} capital -> MaxDD {reports[least_dd]['max_dd']:.1%}")
    print("=" * 100)


if __name__ == "__main__":
    main()
