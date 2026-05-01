"""
simulate_predator.py — Quarantine Engine for Predator Architecture.

Replays alpha_v1_master.csv through the full Predator execution pipeline:
  - ZoneClassifier gating (Alpha/Death/Neutral)
  - Full Kelly sizing with 25% per-trade cap
  - Concurrent position simulation with time-based capital locking
  - Max 3 positions, 60% max total exposure
  - Exposure management stress testing

This script does NOT touch any production code.
It validates the Predator logic in isolation before any live deployment.
"""

import sys
import os
import math
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
import numpy as np

# Add project root to path for zone_matrix import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from src.zone_matrix import classify_zone, ZoneResult


# ── Data Structures ──────────────────────────────────────────────

@dataclass
class PendingPosition:
    """A simulated open position with time-based capital lock."""
    trade_id: int
    zone_id: str
    entry_time: pd.Timestamp
    resolution_time: pd.Timestamp  # entry_time + ttr
    bet_size: float
    entry_odds: float
    signal_direction: str
    strike_price: float
    label: float  # known outcome (from dataset)


@dataclass
class TradeResult:
    """Resolved trade record."""
    trade_id: int
    zone_id: str
    zone_type: str
    entry_odds: float
    bet_size: float
    payout: float
    pnl: float
    outcome: str  # WIN / LOSS
    capital_before: float
    capital_after: float
    entry_time: pd.Timestamp
    resolution_time: pd.Timestamp


@dataclass
class RejectionRecord:
    """Tracks why a signal was rejected."""
    trade_id: int
    zone_id: str
    reason: str  # DEATH_ZONE, NEUTRAL_ZONE, MAX_POSITIONS, EXPOSURE_CAP, BELOW_MIN_BET
    entry_time: pd.Timestamp
    entry_odds: float
    ttr_minutes: float
    distance_usd: float


# ── Predator Simulator ──────────────────────────────────────────

class PredatorSimulator:
    """
    Full Predator Architecture simulator with concurrent position management.

    Simulates time-ordered trade execution with:
    - Zone classification gating
    - Full Kelly sizing (25% cap)
    - Concurrent positions (max 3, 60% exposure cap)
    - Intra-session compounding
    """

    def __init__(
        self,
        initial_capital: float = 100.0,
        max_positions: int = 3,
        max_exposure_pct: float = 0.60,
        max_bet_pct: float = 0.25,
        min_bet: float = 1.0,
        fee_pct: float = 0.02,
        slippage_pct: float = 0.005,
        gas_flat: float = 0.01,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_positions = max_positions
        self.max_exposure_pct = max_exposure_pct
        self.max_bet_pct = max_bet_pct
        self.min_bet = min_bet
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.gas_flat = gas_flat

        self.pending: List[PendingPosition] = []
        self.trades: List[TradeResult] = []
        self.rejections: List[RejectionRecord] = []
        self.equity_curve: List[dict] = []
        self._trade_counter = 0

    @property
    def pending_exposure(self) -> float:
        """Total capital currently locked in open positions."""
        return sum(p.bet_size for p in self.pending)

    @property
    def available_for_new_bets(self) -> float:
        """Capital available for new positions (respecting 60% exposure cap)."""
        max_total = self.capital * self.max_exposure_pct
        return max(0, max_total - self.pending_exposure)

    def _resolve_expired(self, current_time: pd.Timestamp) -> None:
        """Resolve all positions whose resolution_time <= current_time."""
        still_pending = []
        for pos in self.pending:
            if current_time >= pos.resolution_time:
                # Resolve this trade
                won = (pos.label == 1.0)
                capital_before = self.capital

                if won:
                    payout_mult = (1.0 / pos.entry_odds) * (1.0 - self.fee_pct - self.slippage_pct) - self.gas_flat
                    payout = pos.bet_size * payout_mult
                    pnl = payout - pos.bet_size
                else:
                    payout = 0.0
                    pnl = -pos.bet_size

                self.capital += pnl
                capital_after = self.capital

                self.trades.append(TradeResult(
                    trade_id=pos.trade_id,
                    zone_id=pos.zone_id,
                    zone_type="ALPHA",
                    entry_odds=pos.entry_odds,
                    bet_size=pos.bet_size,
                    payout=payout,
                    pnl=pnl,
                    outcome="WIN" if won else "LOSS",
                    capital_before=capital_before,
                    capital_after=capital_after,
                    entry_time=pos.entry_time,
                    resolution_time=pos.resolution_time,
                ))
            else:
                still_pending.append(pos)
        self.pending = still_pending

    def _compute_bet_size(self, zone: ZoneResult) -> float:
        """Full Kelly with 25% cap, using zone empirical WR."""
        if zone.kelly_fraction <= 0:
            return 0.0

        # Full Kelly from zone empirical win rate
        raw_bet = self.capital * zone.kelly_fraction

        # Cap at 25% of current capital
        max_bet = self.capital * self.max_bet_pct
        raw_bet = min(raw_bet, max_bet)

        # Cap at available exposure
        raw_bet = min(raw_bet, self.available_for_new_bets)

        # Integer rounding (Polymarket requires integers)
        bet = max(self.min_bet, math.floor(raw_bet))

        return float(bet)

    def process_signal(
        self,
        row_idx: int,
        timestamp: pd.Timestamp,
        ttr_minutes: float,
        distance_usd: float,
        entry_odds: float,
        signal_direction: str,
        strike_price: float,
        label: float,
    ) -> None:
        """Process a single signal through the Predator pipeline."""
        self._trade_counter += 1
        tid = self._trade_counter

        # Step 0: Resolve any expired positions first
        self._resolve_expired(timestamp)

        # Step 1: Zone classification
        zone = classify_zone(ttr_minutes, distance_usd, entry_odds)

        # Step 2: Gate check
        if zone.zone_type == "DEATH":
            self.rejections.append(RejectionRecord(
                trade_id=tid, zone_id=zone.zone_id, reason="DEATH_ZONE",
                entry_time=timestamp, entry_odds=entry_odds,
                ttr_minutes=ttr_minutes, distance_usd=distance_usd,
            ))
            return

        if zone.zone_type == "NEUTRAL":
            self.rejections.append(RejectionRecord(
                trade_id=tid, zone_id=zone.zone_id, reason="NEUTRAL_ZONE",
                entry_time=timestamp, entry_odds=entry_odds,
                ttr_minutes=ttr_minutes, distance_usd=distance_usd,
            ))
            return

        # Step 3: Position limit check
        if len(self.pending) >= self.max_positions:
            self.rejections.append(RejectionRecord(
                trade_id=tid, zone_id=zone.zone_id, reason="MAX_POSITIONS",
                entry_time=timestamp, entry_odds=entry_odds,
                ttr_minutes=ttr_minutes, distance_usd=distance_usd,
            ))
            return

        # Step 4: Exposure cap check
        if self.available_for_new_bets < self.min_bet:
            self.rejections.append(RejectionRecord(
                trade_id=tid, zone_id=zone.zone_id, reason="EXPOSURE_CAP",
                entry_time=timestamp, entry_odds=entry_odds,
                ttr_minutes=ttr_minutes, distance_usd=distance_usd,
            ))
            return

        # Step 5: Compute bet size
        bet_size = self._compute_bet_size(zone)
        if bet_size < self.min_bet:
            self.rejections.append(RejectionRecord(
                trade_id=tid, zone_id=zone.zone_id, reason="BELOW_MIN_BET",
                entry_time=timestamp, entry_odds=entry_odds,
                ttr_minutes=ttr_minutes, distance_usd=distance_usd,
            ))
            return

        # Step 6: Capital check
        if bet_size > self.capital:
            self.rejections.append(RejectionRecord(
                trade_id=tid, zone_id=zone.zone_id, reason="INSUFFICIENT_CAPITAL",
                entry_time=timestamp, entry_odds=entry_odds,
                ttr_minutes=ttr_minutes, distance_usd=distance_usd,
            ))
            return

        # Step 7: Execute — open position
        resolution_time = timestamp + pd.Timedelta(minutes=ttr_minutes)
        self.pending.append(PendingPosition(
            trade_id=tid,
            zone_id=zone.zone_id,
            entry_time=timestamp,
            resolution_time=resolution_time,
            bet_size=bet_size,
            entry_odds=entry_odds,
            signal_direction=signal_direction,
            strike_price=strike_price,
            label=label,
        ))

        # Record equity state
        self.equity_curve.append({
            "time": timestamp,
            "capital": self.capital,
            "pending_exposure": self.pending_exposure,
            "net_equity": self.capital - self.pending_exposure + sum(
                p.bet_size for p in self.pending  # locked capital is still ours until lost
            ),
            "open_positions": len(self.pending),
        })

    def finalize(self) -> None:
        """Resolve all remaining pending positions at end of dataset."""
        if self.pending:
            # Use a far-future timestamp to force-resolve everything
            future = pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=365)
            self._resolve_expired(future)


# ── Main Execution ───────────────────────────────────────────────

def main():
    print("=" * 90)
    print("PREDATOR ARCHITECTURE — QUARANTINE STRESS TEST")
    print("=" * 90)

    # Load dataset
    csv_path = "dataset/raw/alpha_v1_master.csv"
    if not os.path.exists(csv_path):
        print(f"Dataset not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df['label'].notnull()].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['ttr_minutes'] = df['ttr_seconds'] / 60.0
    df['distance_to_strike'] = abs(df['binance_price'] - df['strike_price'])

    # Sort by timestamp (time-ordered replay)
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"Dataset: {csv_path}")
    print(f"Labeled rows: {len(df)}")
    print(f"Time span: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Signal directions: BUY_UP={len(df[df['signal_direction']=='BUY_UP'])}, BUY_DOWN={len(df[df['signal_direction']=='BUY_DOWN'])}")
    print()

    # Initialize simulator
    sim = PredatorSimulator(initial_capital=100.0)
    print(f"Initial Capital: ${sim.initial_capital:.2f}")
    print(f"Max Positions: {sim.max_positions}")
    print(f"Max Exposure: {sim.max_exposure_pct:.0%}")
    print(f"Max Bet/Trade: {sim.max_bet_pct:.0%}")
    print(f"Kelly Mode: FULL (divisor=1)")
    print()

    # Process each signal in time order
    for idx, row in df.iterrows():
        sim.process_signal(
            row_idx=idx,
            timestamp=row['timestamp'],
            ttr_minutes=row['ttr_minutes'],
            distance_usd=row['distance_to_strike'],
            entry_odds=row['entry_odds'],
            signal_direction=row['signal_direction'],
            strike_price=row['strike_price'],
            label=row['label'],
        )

    # Finalize remaining positions
    sim.finalize()

    # ── STRESS REPORT ────────────────────────────────────────────

    trades = sim.trades
    rejections = sim.rejections

    print("=" * 90)
    print("STRESS REPORT: PREDATOR ARCHITECTURE")
    print("=" * 90)

    # 1. PnL Summary
    total_pnl = sum(t.pnl for t in trades)
    final_capital = sim.capital
    wins = [t for t in trades if t.outcome == "WIN"]
    losses = [t for t in trades if t.outcome == "LOSS"]
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    print(f"\n--- PNL SUMMARY ---")
    print(f"  Final Capital:    ${final_capital:>10.2f}")
    print(f"  Total PnL:        ${total_pnl:>+10.2f} ({total_pnl/sim.initial_capital*100:>+.1f}%)")
    print(f"  Total Trades:     {len(trades):>10}")
    print(f"  Wins:             {len(wins):>10} ({win_rate:.1f}%)")
    print(f"  Losses:           {len(losses):>10} ({100-win_rate:.1f}%)")
    if wins:
        print(f"  Avg Win PnL:      ${np.mean([t.pnl for t in wins]):>+10.2f}")
    if losses:
        print(f"  Avg Loss PnL:     ${np.mean([t.pnl for t in losses]):>+10.2f}")

    # Profit Factor
    gross_profit = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1e-8
    print(f"  Profit Factor:    {gross_profit / (gross_loss + 1e-8):>10.2f}")

    # 2. Drawdown Analysis
    if trades:
        equity = [sim.initial_capital]
        for t in trades:
            equity.append(equity[-1] + t.pnl)
        equity = np.array(equity)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)

        print(f"\n--- DRAWDOWN ANALYSIS ---")
        print(f"  Max Drawdown:     {max_dd:>10.1%}")
        print(f"  Max DD at trade:  #{max_dd_idx}")
        print(f"  Capital at Max DD: ${equity[max_dd_idx]:>10.2f}")
        print(f"  Peak before DD:   ${peak[max_dd_idx]:>10.2f}")

        # Rolling drawdown periods
        in_drawdown = drawdown < -0.05
        dd_streaks = []
        streak = 0
        for d in in_drawdown:
            if d:
                streak += 1
            else:
                if streak > 0:
                    dd_streaks.append(streak)
                streak = 0
        if streak > 0:
            dd_streaks.append(streak)
        if dd_streaks:
            print(f"  DD periods (>5%): {len(dd_streaks)}")
            print(f"  Longest DD streak:{max(dd_streaks):>10} trades")

    # 3. Zone-level Performance
    print(f"\n--- ZONE PERFORMANCE ---")
    zone_stats = {}
    for t in trades:
        if t.zone_id not in zone_stats:
            zone_stats[t.zone_id] = {"wins": 0, "losses": 0, "pnl": 0, "bets": 0, "total_bet": 0}
        zone_stats[t.zone_id]["bets"] += 1
        zone_stats[t.zone_id]["total_bet"] += t.bet_size
        zone_stats[t.zone_id]["pnl"] += t.pnl
        if t.outcome == "WIN":
            zone_stats[t.zone_id]["wins"] += 1
        else:
            zone_stats[t.zone_id]["losses"] += 1

    print(f"  {'Zone':<6} {'Trades':>7} {'WR':>7} {'PnL':>10} {'Avg Bet':>9} {'EV/Trade':>10}")
    print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*10} {'-'*9} {'-'*10}")
    for zone_id in sorted(zone_stats.keys()):
        s = zone_stats[zone_id]
        wr = s["wins"] / s["bets"] * 100 if s["bets"] > 0 else 0
        avg_bet = s["total_bet"] / s["bets"] if s["bets"] > 0 else 0
        ev_trade = s["pnl"] / s["bets"] if s["bets"] > 0 else 0
        print(f"  {zone_id:<6} {s['bets']:>7} {wr:>6.1f}% ${s['pnl']:>+9.2f} ${avg_bet:>8.2f} ${ev_trade:>+9.2f}")

    # 4. Rejection Analysis (Race Conditions)
    print(f"\n--- REJECTION ANALYSIS (Race Condition Stress) ---")
    rej_reasons = {}
    for r in rejections:
        rej_reasons[r.reason] = rej_reasons.get(r.reason, 0) + 1

    total_signals = len(trades) + len(rejections)
    print(f"  Total Signals Evaluated: {total_signals}")
    print(f"  Accepted:                {len(trades)} ({len(trades)/total_signals*100:.1f}%)")
    print(f"  Rejected:                {len(rejections)} ({len(rejections)/total_signals*100:.1f}%)")
    print()
    for reason, count in sorted(rej_reasons.items(), key=lambda x: -x[1]):
        pct = count / total_signals * 100
        marker = "!!!" if reason in ("MAX_POSITIONS", "EXPOSURE_CAP") else "   "
        print(f"  {marker} {reason:<25} {count:>6} ({pct:>5.1f}%)")

    race_conditions = rej_reasons.get("MAX_POSITIONS", 0) + rej_reasons.get("EXPOSURE_CAP", 0)
    print(f"\n  RACE CONDITION COUNT (MAX_POS + EXPOSURE_CAP): {race_conditions}")
    if race_conditions > 0:
        # Show when they happened
        rc_events = [r for r in rejections if r.reason in ("MAX_POSITIONS", "EXPOSURE_CAP")]
        print(f"  First occurrence:  {rc_events[0].entry_time}")
        print(f"  Last occurrence:   {rc_events[-1].entry_time}")

        # Zone distribution of race conditions
        rc_zones = {}
        for r in rc_events:
            rc_zones[r.zone_id] = rc_zones.get(r.zone_id, 0) + 1
        print(f"  Affected zones:")
        for z, c in sorted(rc_zones.items(), key=lambda x: -x[1]):
            print(f"    {z}: {c} rejections")

    # 5. Concurrency Analysis
    print(f"\n--- CONCURRENCY ANALYSIS ---")
    max_concurrent = 0
    concurrent_events = []
    for ec in sim.equity_curve:
        if ec["open_positions"] > max_concurrent:
            max_concurrent = ec["open_positions"]
        if ec["open_positions"] >= 2:
            concurrent_events.append(ec)

    print(f"  Max Concurrent Positions: {max_concurrent}")
    print(f"  Events with 2+ positions: {len(concurrent_events)}")
    if concurrent_events:
        max_exposure_seen = max(ec["pending_exposure"] for ec in concurrent_events)
        avg_exposure = np.mean([ec["pending_exposure"] for ec in concurrent_events])
        print(f"  Max Pending Exposure:     ${max_exposure_seen:.2f}")
        print(f"  Avg Exposure (when 2+):   ${avg_exposure:.2f}")

    # 6. Equity Curve Summary (10-point snapshot)
    if trades:
        print(f"\n--- EQUITY CURVE (10-Point Snapshot) ---")
        step = max(1, len(trades) // 10)
        print(f"  {'Trade#':>7} {'Capital':>10} {'PnL':>10} {'Zone':>6} {'Result':>6}")
        print(f"  {'-'*7} {'-'*10} {'-'*10} {'-'*6} {'-'*6}")
        for i in range(0, len(trades), step):
            t = trades[i]
            print(f"  {t.trade_id:>7} ${t.capital_after:>9.2f} ${t.pnl:>+9.2f} {t.zone_id:>6} {t.outcome:>6}")
        # Always show last trade
        t = trades[-1]
        print(f"  {t.trade_id:>7} ${t.capital_after:>9.2f} ${t.pnl:>+9.2f} {t.zone_id:>6} {t.outcome:>6}")

    # 7. Export results
    out_dir = "scripts/output"
    os.makedirs(out_dir, exist_ok=True)

    # Trades CSV
    trades_csv = os.path.join(out_dir, "predator_trades.csv")
    trades_df = pd.DataFrame([{
        "trade_id": t.trade_id, "zone_id": t.zone_id,
        "entry_odds": t.entry_odds, "bet_size": t.bet_size,
        "pnl": t.pnl, "outcome": t.outcome,
        "capital_before": t.capital_before, "capital_after": t.capital_after,
        "entry_time": t.entry_time, "resolution_time": t.resolution_time,
    } for t in trades])
    trades_df.to_csv(trades_csv, index=False)

    # Rejections CSV
    rej_csv = os.path.join(out_dir, "predator_rejections.csv")
    rej_df = pd.DataFrame([{
        "trade_id": r.trade_id, "zone_id": r.zone_id, "reason": r.reason,
        "entry_time": r.entry_time, "entry_odds": r.entry_odds,
        "ttr_minutes": r.ttr_minutes, "distance_usd": r.distance_usd,
    } for r in rejections])
    rej_df.to_csv(rej_csv, index=False)

    print(f"\n--- FILES SAVED ---")
    print(f"  Trades:     {trades_csv}")
    print(f"  Rejections: {rej_csv}")
    print(f"\n{'=' * 90}")
    print(f"VERDICT: {'EXPOSURE LOGIC VALIDATED' if race_conditions < len(trades) * 0.5 else 'EXCESSIVE RACE CONDITIONS — REVIEW NEEDED'}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
