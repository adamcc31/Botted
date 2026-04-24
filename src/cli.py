"""
cli.py — Rich terminal dashboard with 5-second auto-refresh.

Displays:
  - Market status (TTR, strike, current price, phase)
  - CLOB orderbook (YES/NO ask/bid, vig, liquidity)
  - Model output (P_model, edges, signal)
  - Session P&L (capital, win rate, trades, dry run score)
  - System health (WS metrics, CLOB freshness, mode)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.schemas import (
    ActiveMarket,
    CLOBState,
    SessionMetrics,
    SignalResult,
    WSHealthMetrics,
)

console = Console()


def build_market_panel(
    market: Optional[ActiveMarket],
    current_price: Optional[float] = None,
) -> Panel:
    """Build market status panel."""
    if not market:
        return Panel(
            Text("⏳ WAITING — No active market", style="yellow bold"),
            title="[bold cyan]MARKET STATUS[/]",
            border_style="yellow",
        )

    now = datetime.now(timezone.utc)
    ttr_seconds = (market.T_resolution - now).total_seconds()
    ttr_minutes = max(0.0, ttr_seconds / 60.0)

    # TTR phase
    if ttr_minutes > 12:
        phase = "EARLY"
        phase_style = "blue"
        phase_icon = "🔵"
    elif ttr_minutes >= 5:
        phase = "ENTRY_WINDOW"
        phase_style = "green"
        phase_icon = "🟢"
    else:
        phase = "LATE"
        phase_style = "red"
        phase_icon = "🔴"

    # Strike distance
    strike_dist = ""
    zone = ""
    if current_price and market.strike_price:
        dist_pct = (current_price - market.strike_price) / market.strike_price * 100
        strike_dist = f"{dist_pct:+.2f}%"
        if abs(dist_pct) < 0.5:
            zone = "(Contest Zone)"
        elif dist_pct > 0:
            zone = "(YES Territory)"
        else:
            zone = "(NO Territory)"

    text = (
        f"Market ID: {market.market_id[:16]}...\n"
        f"TTR: {ttr_minutes:.1f} min | Phase: {phase_icon} [{phase_style}]{phase}[/]\n"
        f"Strike: ${market.strike_price:,.2f}"
    )

    if current_price:
        text += f" | Current: ${current_price:,.2f}\n"
        text += f"Strike Distance: {strike_dist} {zone}"
    else:
        text += "\n"

    if market.resolution_source:
        text += f"\nOracle: {market.resolution_source}"

    return Panel(
        Text.from_markup(text),
        title="[bold cyan]MARKET STATUS[/]",
        border_style="cyan",
    )


def build_clob_panel(clob: Optional[CLOBState]) -> Panel:
    """Build CLOB orderbook panel."""
    if not clob:
        return Panel(
            Text("No CLOB data", style="dim"),
            title="[bold magenta]CLOB[/]",
            border_style="dim",
        )

    liquid_icon = "✓ LIQUID" if clob.is_liquid else "✗ ILLIQUID"
    liquid_style = "green" if clob.is_liquid else "red"
    stale_note = " [red]⚠ STALE[/]" if clob.is_stale else ""

    text = (
        f"YES Ask: {clob.yes_ask:.3f} | YES Bid: {clob.yes_bid:.3f}\n"
        f"NO  Ask: {clob.no_ask:.3f} | NO  Bid: {clob.no_bid:.3f}\n"
        f"YES Depth: ${clob.yes_depth_usd:.1f} | NO Depth: ${clob.no_depth_usd:.1f}\n"
        f"Market Vig: {clob.market_vig:.3f} | Status: [{liquid_style}]{liquid_icon}[/]{stale_note}"
    )

    return Panel(
        Text.from_markup(text),
        title="[bold magenta]CLOB[/]",
        border_style="magenta",
    )


def build_model_panel(signal: Optional[SignalResult]) -> Panel:
    """Build model/signal panel."""
    if not signal:
        return Panel(
            Text("Awaiting signal...", style="dim"),
            title="[bold green]MODEL[/]",
            border_style="dim",
        )

    # Edge display
    edge_yes_style = "green bold" if signal.edge_yes > 0.05 else "dim"
    edge_no_style = "green bold" if signal.edge_no > 0.05 else "dim"

    edge_yes_txt = f"[{edge_yes_style}]{signal.edge_yes:+.3f}[/]"
    edge_no_txt = f"[{edge_no_style}]{signal.edge_no:+.3f}[/]"

    if signal.edge_yes > 0.05:
        edge_yes_txt += " ■ SIGNAL"
    if signal.edge_no > 0.05:
        edge_no_txt += " ■ SIGNAL"

    # Signal direction
    if signal.signal == "BUY_YES":
        sig_txt = "[bold green]→ BUY_YES[/]"
    elif signal.signal == "BUY_NO":
        sig_txt = "[bold blue]→ BUY_NO[/]"
    else:
        reason = signal.abstain_reason or ""
        sig_txt = f"[yellow]→ ABSTAIN ({reason})[/]"

    text = (
        f"P_model: {signal.P_model:.3f}\n"
        f"edge_YES: {edge_yes_txt} | edge_NO: {edge_no_txt}\n"
        f"Signal: {sig_txt}"
    )

    return Panel(
        Text.from_markup(text),
        title="[bold green]MODEL[/]",
        border_style="green",
    )


def build_pnl_panel(metrics: Optional[SessionMetrics]) -> Panel:
    """Build session P&L panel."""
    if not metrics:
        return Panel(
            Text("No session data", style="dim"),
            title="[bold yellow]SESSION P&L[/]",
            border_style="dim",
        )

    # PnL color
    pnl_style = "green" if metrics.total_pnl_usd >= 0 else "red"
    pnl_pct = metrics.total_pnl_pct_capital

    # Pass/Fail
    pf_icon = "✓ PASS" if metrics.pass_fail == "PASS" else "✗ FAIL"
    pf_style = "green" if metrics.pass_fail == "PASS" else "red"

    text = (
        f"Session PnL: [{pnl_style}]{metrics.total_pnl_usd:+.2f}[/] "
        f"([{pnl_style}]{pnl_pct:+.1f}%[/]) | "
        f"Capital: ${metrics.capital_end:.2f}\n"
        f"Win Rate: {metrics.win_rate*100:.1f}% | "
        f"Trades: {metrics.trades_executed} "
        f"(W:{metrics.win_count}, L:{metrics.loss_count})\n"
        f"Dry Run Score: {metrics.dry_run_score:.2f} | "
        f"Status: [{pf_style}]{pf_icon}[/]"
    )

    return Panel(
        Text.from_markup(text),
        title="[bold yellow]SESSION P&L[/]",
        border_style="yellow",
    )


def build_health_panel(
    ws_health: Optional[WSHealthMetrics],
    clob_stale: bool = False,
    mode: str = "DRY RUN",
    session_id: str = "",
) -> Panel:
    """Build system health panel."""
    if not ws_health:
        ws_health = WSHealthMetrics()

    drop_ok = ws_health.drop_rate < 0.001
    drop_icon = "✓" if drop_ok else "⚠"
    drop_style = "green" if drop_ok else "red"

    lat_ok = ws_health.latency_p99_ms < 2000
    lat_icon = "✓" if lat_ok else "⚠"
    lat_style = "green" if lat_ok else "red"

    clob_icon = "✓ FRESH" if not clob_stale else "✗ STALE"
    clob_style = "green" if not clob_stale else "red"

    text = (
        f"WS Drop Rate: [{drop_style}]{ws_health.drop_rate*100:.3f}% {drop_icon}[/] | "
        f"Latency P99: [{lat_style}]{ws_health.latency_p99_ms:.0f}ms {lat_icon}[/]\n"
        f"CLOB Feed: [{clob_style}]{clob_icon}[/]\n"
        f"Mode: [bold]{mode}[/] | Session: {session_id}"
    )

    return Panel(
        Text.from_markup(text),
        title="[bold blue]SYSTEM HEALTH[/]",
        border_style="blue",
    )


def build_dashboard(
    market: Optional[ActiveMarket] = None,
    clob: Optional[CLOBState] = None,
    signal: Optional[SignalResult] = None,
    metrics: Optional[SessionMetrics] = None,
    ws_health: Optional[WSHealthMetrics] = None,
    current_price: Optional[float] = None,
    mode: str = "DRY RUN",
    session_id: str = "",
) -> Layout:
    """Build complete dashboard layout."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=1),
        Layout(name="body"),
    )

    layout["header"].update(
        Text(
            "  POLYMARKET MISPRICING BOT  ",
            style="bold white on dark_blue",
            justify="center",
        )
    )

    layout["body"].split_column(
        Layout(name="top", size=7),
        Layout(name="middle", size=7),
        Layout(name="bottom_1", size=6),
        Layout(name="bottom_2", size=5),
    )

    layout["top"].update(build_market_panel(market, current_price))
    layout["middle"].update(build_clob_panel(clob))
    layout["bottom_1"].update(build_model_panel(signal))

    # Split bottom row
    layout["bottom_2"].split_row(
        Layout(name="pnl"),
        Layout(name="health"),
    )

    layout["pnl"].update(build_pnl_panel(metrics))
    clob_stale = clob.is_stale if clob else False
    layout["health"].update(
        build_health_panel(ws_health, clob_stale, mode, session_id)
    )

    return layout
