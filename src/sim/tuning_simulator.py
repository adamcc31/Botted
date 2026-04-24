"""
tuning_simulator.py — Offline Grid Search / Historical Backtesting Engine.

Reads Railway dry-run JSON logs, reconstructs epoch-level state snapshots
(Vatic strike, Binance spot prices, TTR), then replays them through the
signal_generator edge computation under multiple tuning schemas in parallel.

Usage:
    python -m src.sim.tuning_simulator --log-file logs/logs.1775783692599.json

Outputs a comparative table of KPIs per schema to stdout and CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ─── Helpers ──────────────────────────────────────────────────

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def parse_structlog_message(raw_msg: str) -> dict[str, Any]:
    """
    Parse a structlog flat-format message into {event, key=value...}.
    Handles:
      - ANSI color codes
      - Bracket-wrapped levels like [info     ]
      - Quoted values with spaces: question='Bitcoin Up or Down - ...'
    """
    cleaned = strip_ansi(raw_msg).strip()
    result: dict[str, Any] = {}

    # 1. Extract ISO timestamp at the start
    ts_match = re.match(r"(\d{4}-\d{2}-\d{2}T[\d:.]+Z?)\s+", cleaned)
    if ts_match:
        result["_ts"] = ts_match.group(1)
        cleaned = cleaned[ts_match.end():]

    # 2. Strip bracket-wrapped log level: [info     ] or [warning  ]
    cleaned = re.sub(r"\[\s*\w+\s*\]\s*", "", cleaned).strip()

    # 3. Find the event name: everything before the first key=value pair
    first_kv = re.search(r"\s+\w+=", cleaned)
    if first_kv:
        result["_event"] = cleaned[:first_kv.start()].strip()
        kv_part = cleaned[first_kv.start():]
    else:
        result["_event"] = cleaned
        return result

    # 4. Parse key=value pairs (supports quoted values with spaces)
    kv_pattern = re.compile(r"""(\w+)=(?:'([^']*)'|"([^"]*)"|(\S+))""")
    for m in kv_pattern.finditer(kv_part):
        k = m.group(1)
        v = m.group(2) if m.group(2) is not None else (m.group(3) if m.group(3) is not None else m.group(4))
        # Try numeric conversion
        try:
            if "." in v:
                result[k] = float(v)
            else:
                result[k] = int(v)
        except (ValueError, OverflowError):
            if v == "None":
                result[k] = None
            else:
                result[k] = v

    return result


# ─── Data Structures ─────────────────────────────────────────

@dataclass
class EpochSnapshot:
    """Reconstructed state for one 5-minute epoch."""
    epoch_ts: int
    slug: str
    market_id: str
    question: str
    strike_price: float
    vatic_strike: Optional[float]
    discovery_time: str  # ISO timestamp of market_discovered
    spot_prices: list[float] = field(default_factory=list)  # 1m binance close prices during this epoch
    resolution_price: Optional[float] = None  # last known price at epoch end
    yes_prob: Optional[float] = None
    TTR_at_discovery: float = 5.0
    outcome: Optional[str] = None  # "YES" or "NO" (determined from final spot vs strike)


@dataclass
class TuningSchema:
    name: str
    margin_of_safety: float
    spread_max_bps: float
    uncertainty_multiplier: float  # multiplier on top of base_u=0.03
    description: str = ""


@dataclass
class SimulatedSignal:
    epoch_ts: int
    slug: str
    direction: str  # BUY_UP, BUY_DOWN, ABSTAIN
    edge: float
    P_model: float
    strike_price: float
    spot_at_entry: float
    resolution_price: float
    outcome: str   # WIN, LOSS
    pnl_pct: float  # ROI percentage
    clob_ask_simulated: float
    abstain_reason: Optional[str] = None


@dataclass
class SchemaResult:
    schema: TuningSchema
    total_epochs: int = 0
    signals_generated: int = 0
    signals_buy_yes: int = 0
    signals_buy_no: int = 0
    abstain_count: int = 0
    wins: int = 0
    losses: int = 0
    avg_edge: float = 0.0
    avg_pnl_pct: float = 0.0
    total_pnl_pct: float = 0.0
    win_rate: float = 0.0
    spread_rejections: int = 0
    margin_rejections: int = 0
    ttr_rejections: int = 0
    signals: list = field(default_factory=list)


# ─── Black-Scholes Digital Probability ────────────────────────

def phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_fair_prob(
    spot: float,
    strike: float,
    tau_seconds: float,
    sigma_ann: float = 0.50,  # conservative default for BTC
    mu: float = 0.0,
) -> float:
    """
    P(S_T >= K) using BS digital approximation.
    Same formula as fair_probability.py line 137.
    """
    tau_years = tau_seconds / 31557600.0
    if sigma_ann <= 0.0 or tau_years <= 0.0:
        return 0.5
    sigma_sqrt_t = sigma_ann * math.sqrt(tau_years)
    d2 = (math.log(spot / strike) + (mu - 0.5 * sigma_ann ** 2) * tau_years) / (sigma_sqrt_t + 1e-12)
    q = phi(d2)
    return max(1e-6, min(1.0 - 1e-6, q))


def estimate_rv_from_closes(closes: list[float], annualizer: float = math.sqrt(252 * 1440)) -> float:
    """Estimate annualized realized volatility from 1-minute closes."""
    if len(closes) < 2:
        return 0.50  # fallback
    log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0]
    if len(log_returns) < 2:
        return 0.50
    mean_r = sum(log_returns) / len(log_returns)
    var_r = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
    return math.sqrt(var_r) * annualizer


# ─── Log Parser ──────────────────────────────────────────────

def parse_logs(log_file: Path) -> list[EpochSnapshot]:
    """Parse Railway JSON logs into structured EpochSnapshot list."""
    with open(log_file, "r", encoding="utf-8") as f:
        raw_logs = json.load(f)

    # Extract structured data from each log entry
    events: list[dict] = []
    for entry in raw_logs:
        msg = entry.get("message", "")
        parsed = parse_structlog_message(msg)
        parsed["_raw_ts"] = entry.get("timestamp", "")
        events.append(parsed)

    # Group by epoch
    epochs: dict[int, EpochSnapshot] = {}
    # Track all binance_bar_closed prices chronologically
    all_spot_prices: list[tuple[str, float]] = []

    for ev in events:
        event_name = ev.get("_event", "")

        if event_name == "binance_bar_closed":
            close = ev.get("close")
            if close is not None:
                all_spot_prices.append((ev.get("_ts", ""), float(close)))

        elif event_name == "vatic_oracle_strike_acquired":
            epoch = ev.get("epoch")
            strike = ev.get("strike")
            if epoch and strike:
                epoch_int = int(epoch)
                if epoch_int not in epochs:
                    epochs[epoch_int] = EpochSnapshot(
                        epoch_ts=epoch_int, slug=f"btc-updown-5m-{epoch_int}",
                        market_id="", question="", strike_price=float(strike),
                        vatic_strike=float(strike), discovery_time=ev.get("_ts", ""),
                    )
                else:
                    epochs[epoch_int].vatic_strike = float(strike)
                    epochs[epoch_int].strike_price = float(strike)

        elif event_name == "dynamic_5m_candidate_found":
            slug = ev.get("slug", "")
            # Extract epoch_ts from slug
            parts = slug.rsplit("-", 1)
            if len(parts) == 2:
                try:
                    epoch_int = int(parts[1])
                except ValueError:
                    continue
                if epoch_int not in epochs:
                    epochs[epoch_int] = EpochSnapshot(
                        epoch_ts=epoch_int, slug=slug,
                        market_id=ev.get("market_id", ""),
                        question=ev.get("question", ""),
                        strike_price=ev.get("strike_price", 0.0),
                        vatic_strike=None,
                        discovery_time=ev.get("_ts", ""),
                        TTR_at_discovery=ev.get("TTR_minutes", 5.0),
                        yes_prob=ev.get("yes_prob"),
                    )
                else:
                    e = epochs[epoch_int]
                    e.market_id = ev.get("market_id", e.market_id)
                    e.question = ev.get("question", e.question)
                    e.slug = slug
                    if ev.get("strike_price"):
                        e.strike_price = float(ev["strike_price"])
                    e.TTR_at_discovery = ev.get("TTR_minutes", e.TTR_at_discovery)

    # Assign spot prices to each epoch window
    for epoch_ts, snapshot in epochs.items():
        epoch_start = epoch_ts
        epoch_end = epoch_ts + 300  # 5 minutes
        matching = [p for ts_str, p in all_spot_prices
                    if _ts_in_range(ts_str, epoch_start, epoch_end)]
        snapshot.spot_prices = matching
        if matching:
            snapshot.resolution_price = matching[-1]
        # Determine binary outcome
        if snapshot.resolution_price and snapshot.strike_price > 0:
            snapshot.outcome = "YES" if snapshot.resolution_price >= snapshot.strike_price else "NO"

    # Sort by epoch
    sorted_epochs = sorted(epochs.values(), key=lambda e: e.epoch_ts)
    # Filter epochs that have valid strike + resolution data
    valid = [e for e in sorted_epochs if e.strike_price > 0 and e.resolution_price is not None]
    return valid


def _ts_in_range(ts_str: str, epoch_start: int, epoch_end: int) -> bool:
    """Check if a structlog timestamp falls within an epoch window."""
    if not ts_str:
        return False
    try:
        ts_str = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_str)
        ts = dt.timestamp()
        return epoch_start <= ts < epoch_end
    except Exception:
        return False


# ─── Simulation Engine ───────────────────────────────────────

DEFAULT_SCHEMAS = [
    TuningSchema(
        name="A_Conservative",
        margin_of_safety=0.05,
        spread_max_bps=5.0,
        uncertainty_multiplier=1.0,
        description="Current production defaults (As-Is)",
    ),
    TuningSchema(
        name="B_Balanced_HFT",
        margin_of_safety=0.03,
        spread_max_bps=10.0,
        uncertainty_multiplier=0.80,
        description="Moderate: relax safety margin, wider spread tolerance",
    ),
    TuningSchema(
        name="C_Aggressive",
        margin_of_safety=0.02,
        spread_max_bps=15.0,
        uncertainty_multiplier=0.60,
        description="Aggressive: thin margin, high spread tolerance",
    ),
]


def simulate_schema(
    schema: TuningSchema,
    epochs: list[EpochSnapshot],
    all_spot_prices: list[float],
) -> SchemaResult:
    """
    Replay all epoch snapshots through signal computation with given schema params.
    Simulates what signal_generator.py would have produced.
    """
    result = SchemaResult(schema=schema, total_epochs=len(epochs))

    # Estimate RV from all spot prices collected during the session
    sigma_ann = estimate_rv_from_closes(all_spot_prices) if len(all_spot_prices) > 20 else 0.50
    # Apply volatility floor (same as fair_probability.py)
    sigma_ann = max(sigma_ann, 0.20)

    base_uncertainty = 0.03  # fair_prob.base_uncertainty_p default

    for epoch in epochs:
        strike = epoch.strike_price
        resolution = epoch.resolution_price
        if resolution is None or strike <= 0:
            result.abstain_count += 1
            continue

        # For each minute bar within the epoch, we check if a signal would fire
        # In practice, the bot evaluates once per bar close.
        # We simulate checking at minute 1 (TTR ~4 min) which is optimal entry.
        if not epoch.spot_prices:
            result.abstain_count += 1
            continue

        # Use first available spot price as "entry spot" (minute 1 after epoch open)
        entry_spot = epoch.spot_prices[0] if epoch.spot_prices else strike

        # TTR at evaluation: ~4 minutes remaining
        tau_seconds = 4.0 * 60.0  # 240s

        # Compute P_model (fair probability)
        P_model = compute_fair_prob(
            spot=entry_spot,
            strike=strike,
            tau_seconds=tau_seconds,
            sigma_ann=sigma_ann,
        )

        # Uncertainty buffer (same logic as fair_probability.py)
        # For ultra-short: near_w = 0 (no expiry bias)
        uncertainty_u = base_uncertainty * schema.uncertainty_multiplier

        # Simulate CLOB prices
        # In a real 5-min "Up or Down" market, the default CLOB price is around 0.50
        # (50/50 coin flip at market open). We use 0.50 as the simulated ask/bid.
        # The real edge comes from P_model ≠ 0.50.
        clob_yes_ask = 0.50
        clob_no_ask = 0.50

        # Edge calculation (same as signal_generator.py STEP 4)
        edge_yes_raw = P_model - clob_yes_ask
        edge_no_raw = (1.0 - P_model) - clob_no_ask

        # Conservative edges
        edge_yes = edge_yes_raw - uncertainty_u
        edge_no = edge_no_raw - uncertainty_u

        # Simulated spread check (we use strike distance as proxy for spread environment)
        # In reality, Binance spread is always tight — this gate rarely fires.
        # For simulation, we assume spread = 2 bps (typical for BTCUSDT)
        simulated_spread_bps = 2.0

        # Check gates
        if simulated_spread_bps > schema.spread_max_bps:
            result.spread_rejections += 1
            result.abstain_count += 1
            continue

        # Margin of Safety check (STEP 5)
        if max(edge_yes, edge_no) <= schema.margin_of_safety:
            result.margin_rejections += 1
            result.abstain_count += 1
            continue

        # Signal selection (STEP 6)
        if edge_yes > schema.margin_of_safety and edge_no > schema.margin_of_safety:
            direction = "BUY_UP" if edge_yes >= edge_no else "BUY_DOWN"
            edge = max(edge_yes, edge_no)
        elif edge_yes > schema.margin_of_safety:
            direction = "BUY_UP"
            edge = edge_yes
        else:
            direction = "BUY_DOWN"
            edge = edge_no

        result.signals_generated += 1
        if direction == "BUY_UP":
            result.signals_buy_yes += 1
        else:
            result.signals_buy_no += 1

        # Determine outcome
        actual_outcome = epoch.outcome or "UNKNOWN"
        if direction == "BUY_UP":
            won = (actual_outcome == "YES")
            # PnL: buy at 0.50, settle at 1.00 if win, 0.00 if loss
            pnl_pct = 100.0 if won else -100.0
        else:
            won = (actual_outcome == "NO")
            pnl_pct = 100.0 if won else -100.0

        if won:
            result.wins += 1
        else:
            result.losses += 1

        sig = SimulatedSignal(
            epoch_ts=epoch.epoch_ts,
            slug=epoch.slug,
            direction=direction,
            edge=edge,
            P_model=P_model,
            strike_price=strike,
            spot_at_entry=entry_spot,
            resolution_price=resolution,
            outcome="WIN" if won else "LOSS",
            pnl_pct=pnl_pct,
            clob_ask_simulated=clob_yes_ask,
        )
        result.signals.append(sig)

    # Compute aggregate metrics
    if result.signals_generated > 0:
        result.avg_edge = sum(s.edge for s in result.signals) / result.signals_generated
        result.avg_pnl_pct = sum(s.pnl_pct for s in result.signals) / result.signals_generated
        result.total_pnl_pct = sum(s.pnl_pct for s in result.signals)
        result.win_rate = result.wins / result.signals_generated

    return result


# ─── Output Formatters ───────────────────────────────────────

def print_comparison_table(results: list[SchemaResult]) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 100)
    print("  OFFLINE TUNING SIMULATOR — COMPARATIVE RESULTS")
    print("=" * 100)

    headers = [
        "Schema", "Epochs", "Signals", "Buy YES", "Buy NO",
        "Wins", "Losses", "Win%", "Avg Edge", "Total PnL%",
        "Spread Rej", "Margin Rej",
    ]
    row_fmt = "{:<20} {:>6} {:>7} {:>8} {:>7} {:>5} {:>6} {:>6} {:>9} {:>10} {:>10} {:>10}"

    print(row_fmt.format(*headers))
    print("-" * 100)

    for r in results:
        print(row_fmt.format(
            r.schema.name,
            r.total_epochs,
            r.signals_generated,
            r.signals_buy_yes,
            r.signals_buy_no,
            r.wins,
            r.losses,
            f"{r.win_rate:.1%}" if r.signals_generated > 0 else "N/A",
            f"{r.avg_edge:.4f}" if r.signals_generated > 0 else "N/A",
            f"{r.total_pnl_pct:+.1f}%" if r.signals_generated > 0 else "N/A",
            r.spread_rejections,
            r.margin_rejections,
        ))

    print("=" * 100)

    # Per-schema detail
    for r in results:
        print(f"\n── {r.schema.name}: {r.schema.description} ──")
        print(f"   margin_of_safety={r.schema.margin_of_safety}, "
              f"spread_max_bps={r.schema.spread_max_bps}, "
              f"uncertainty_mult={r.schema.uncertainty_multiplier}")
        if r.signals:
            edges = [s.edge for s in r.signals]
            pnls = [s.pnl_pct for s in r.signals]
            print(f"   Edge range: [{min(edges):.4f}, {max(edges):.4f}]")
            print(f"   Best trade: epoch {max(r.signals, key=lambda s: s.pnl_pct).epoch_ts}")
            if r.signals_generated > 0:
                print(f"   Expectancy per trade: {sum(pnls)/len(pnls):+.1f}%")
        else:
            print("   ⚠ No signals generated — all epochs filtered by gates.")

    print()


def export_signals_csv(results: list[SchemaResult], output_dir: Path) -> None:
    """Export per-signal detail to CSV for each schema."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        csv_path = output_dir / f"sim_{r.schema.name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch_ts", "slug", "direction", "edge", "P_model",
                "strike", "spot_entry", "resolution_price", "outcome", "pnl_pct",
            ])
            for s in r.signals:
                writer.writerow([
                    s.epoch_ts, s.slug, s.direction,
                    f"{s.edge:.6f}", f"{s.P_model:.6f}",
                    f"{s.strike_price:.2f}", f"{s.spot_at_entry:.2f}",
                    f"{s.resolution_price:.2f}", s.outcome, f"{s.pnl_pct:.1f}",
                ])
        print(f"  📄 Exported: {csv_path}")


def export_summary_csv(results: list[SchemaResult], output_path: Path) -> None:
    """Export summary comparison table to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "schema", "total_epochs", "signals", "buy_yes", "buy_no",
            "wins", "losses", "win_rate", "avg_edge", "total_pnl_pct",
            "spread_rejections", "margin_rejections",
            "margin_of_safety", "spread_max_bps", "uncertainty_multiplier",
        ])
        for r in results:
            writer.writerow([
                r.schema.name, r.total_epochs, r.signals_generated,
                r.signals_buy_yes, r.signals_buy_no,
                r.wins, r.losses,
                f"{r.win_rate:.4f}" if r.signals_generated > 0 else "N/A",
                f"{r.avg_edge:.6f}" if r.signals_generated > 0 else "N/A",
                f"{r.total_pnl_pct:.2f}" if r.signals_generated > 0 else "N/A",
                r.spread_rejections, r.margin_rejections,
                r.schema.margin_of_safety, r.schema.spread_max_bps,
                r.schema.uncertainty_multiplier,
            ])
    print(f"  📊 Summary CSV: {output_path}")


# ─── Main Entry Point ────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline Tuning Simulator — replay dry-run logs through multiple parameter schemas"
    )
    parser.add_argument(
        "--log-file", type=str, required=True,
        help="Path to Railway JSON log file (e.g. logs/logs.1775783692599.json)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="sim_results",
        help="Directory for CSV output files"
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"❌ Log file not found: {log_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    print(f"\n🔬 Loading logs from: {log_path}")
    epochs = parse_logs(log_path)
    print(f"   ✅ Extracted {len(epochs)} valid epochs with strike + resolution data")

    if not epochs:
        print("❌ No valid epochs found. Check log file format.")
        sys.exit(1)

    # Collect all spot prices for RV estimation
    all_spots: list[float] = []
    for ep in epochs:
        all_spots.extend(ep.spot_prices)

    if all_spots:
        rv = estimate_rv_from_closes(all_spots)
        print(f"   📈 Estimated session RV (annualized): {rv:.2%}")
        print(f"   💰 BTC price range: ${min(all_spots):,.2f} — ${max(all_spots):,.2f}")

    # Epoch summary
    outcomes = [e.outcome for e in epochs if e.outcome]
    yes_count = sum(1 for o in outcomes if o == "YES")
    no_count = sum(1 for o in outcomes if o == "NO")
    print(f"   🎯 Outcomes: {yes_count} YES / {no_count} NO ({len(outcomes)} total)")

    # Run simulations
    print(f"\n⚡ Running {len(DEFAULT_SCHEMAS)} schema simulations...")
    results: list[SchemaResult] = []
    for schema in DEFAULT_SCHEMAS:
        r = simulate_schema(schema, epochs, all_spots)
        results.append(r)
        print(f"   ✓ {schema.name}: {r.signals_generated} signals, "
              f"win rate {r.win_rate:.1%}, total PnL {r.total_pnl_pct:+.1f}%")

    # Output
    print_comparison_table(results)
    export_signals_csv(results, output_dir)
    export_summary_csv(results, output_dir / "summary.csv")

    print("\n✅ Simulation complete. Results saved to:", output_dir)


if __name__ == "__main__":
    main()
