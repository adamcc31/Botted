"""
telegram_notifier.py — Telegram notifications for engine operations.

Production behavior:
- If `TELEGRAM_BOT_TOKEN` (or legacy `TELEGRAM_TOKEN`) and `TELEGRAM_CHAT_ID`
  are not present, the notifier becomes a no-op (never crashes the bot).
- All methods are async and use small timeouts.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx

from src.config_manager import ConfigManager


class TelegramNotifier:
    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        self._enabled = bool(self._config.get("telegram.enabled", False))

        self._token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
        self._chat_id = os.getenv("TELEGRAM_CHAT_ID")

        # If Telegram env vars exist, allow sending even if config enabled=false,
        # but keep config as the default.
        if self._token and self._chat_id:
            self._enabled = True

        self._rest_timeout_s = float(self._config.get("telegram.timeout_seconds", 10.0))

    async def send_message(
        self,
        title: str,
        message: str,
        *,
        parse_mode: Optional[str] = "HTML",
        disable_notification: bool = False,
    ) -> None:
        if not self._enabled:
            return
        if not self._token or not self._chat_id:
            return

        text = f"🔥 <b>{title}</b>\n{message}"
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        payload = {
            "chat_id": str(self._chat_id),
            "text": text,
            "disable_web_page_preview": True,
            "disable_notification": disable_notification,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        async with httpx.AsyncClient(timeout=self._rest_timeout_s) as client:
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            except Exception:
                # Never fail the trading engine if Telegram errors.
                return

    async def send_document(
        self,
        file_path: str,
        caption: str = "",
        disable_notification: bool = False,
    ) -> None:
        if not self._enabled or not self._token or not self._chat_id:
            return
        
        url = f"https://api.telegram.org/bot{self._token}/sendDocument"
        data = {
            "chat_id": str(self._chat_id),
            "caption": caption,
            "disable_notification": disable_notification,
        }
        
        import os
        if not os.path.exists(file_path):
            return

        # Prepare multipart/form-data for the file
        files = {
            "document": (os.path.basename(file_path), open(file_path, "rb"))
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(url, data=data, files=files)
                resp.raise_for_status()
            except Exception:
                pass
            finally:
                files["document"][1].close()


class SlingshotAlerts:
    """Centralized alert formatter for the trading engine.

    TELEMETRY SEPARATION CONTRACT
    ─────────────────────────────────────────────────────────────────────────────
    [SLINGGER V5] — V5-exclusive methods (shadow scalp lifecycle only):
        entry()        Shadow scalp opened
        exit_hit()     Swing target reached
        emergency()    TTR < 60s decision gate
        miss()         Position closed at loss
        daily_summary() End-of-day V5 stats

    [ALPHA V1] — Called exclusively from the Alpha V1 execution path
    (_on_bar_close → _schedule_resolution → _send_telegram):
        order_execution()     Trade entry notification
        order_result()        Trade resolution (WIN/LOSS)
        paper_trade_resolved() High-fidelity paper trade resolved
        session_aborted()     Hard abort triggered
        go_live_enabled()     Dry-run gate passed
        session_report()      Periodic 2h summary
        heartbeat()           15-min market watch pulse
        system_health()       Startup health check
        session_finished()    Session end summary
        dry_run_limit()       Max duration exceeded

    Any change to a prefix MUST update this docstring.
    ─────────────────────────────────────────────────────────────────────────────
    """

    # Single source of truth for model identifiers.
    MODEL_V1 = "ALPHA V1"
    MODEL_V5 = "SLINGGER V5"

    # ─────────────────────────────────────────────────────────────────────────
    # V5-EXCLUSIVE: Shadow Scalp Lifecycle Notifications
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def entry(market_slug: str,
              entry_price: float,
              exit_target: float,
              ttr: int,
              confidence: float,
              side: str,
              stake_usd: float,
              shares: float,
              depth_available_usd: float,
              btc_vs_strike_pct: float) -> str:

        side_label = (
            "UP ↑ (Above Strike)"
            if side == 'YES'
            else "DOWN ↓ (Below Strike)"
        )

        side_emoji = (
            "📈"
            if side == 'YES'
            else "📉"
        )

        max_win = (
            (exit_target - entry_price)
            * 0.98
            * shares
            - 0.005
        )

        max_loss = -stake_usd - 0.005

        roi_pct = (
            (max_win / stake_usd) * 100
            if stake_usd > 0
            else 0.0
        )

        tier = (
            "HIGH"
            if confidence >= 0.65
            else (
                "MEDIUM"
                if confidence >= 0.55
                else "LOW"
            )
        )

        btc_sign = (
            "+"
            if btc_vs_strike_pct >= 0
            else ""
        )

        return (
            f"[SLINGGER V5] 🟢 SHADOW ENTRY {side_emoji}\n"
            f"\n"
            f"📌 {market_slug}\n"
            f"\n"
            f"🎯 Entry   : {entry_price:.3f}\n"
            f"🏁 Target  : {exit_target:.3f}\n"
            f"📊 Side    : {side_label}\n"
            f"💰 Stake   : ${stake_usd:.2f} "
            f"({shares:.2f} shares)\n"
            f"📈 Max Win : +${max_win:.2f} "
            f"(+{roi_pct:.1f}%)\n"
            f"⚠️  Max Loss: ${max_loss:.2f} (-100%)\n"
            f"\n"
            f"📉 BTC vs Strike : "
            f"{btc_sign}{btc_vs_strike_pct:.2f}%\n"
            f"💧 Depth @ Entry : "
            f"${depth_available_usd:.0f} available\n"
            f"⏱  TTR     : {ttr}s\n"
            f"🧠 Conf    : {confidence:.1%} [{tier}]"
        )

    @staticmethod
    def exit_hit(market_slug: str,
                 entry_price: float,
                 exit_price: float,
                 exit_target: float,
                 latency_s: int,
                 ttr_at_exit: int,
                 shares: float,
                 stake_usd: float,
                 daily_pnl: float,
                 daily_wins: int,
                 daily_losses: int) -> str:

        from src.utils import fmt_money

        gross_pnl = (
            (exit_price - entry_price)
            * shares
        )

        fee_cost = (
            exit_price
            * shares
            * 0.02
        )

        net_pnl = gross_pnl - fee_cost - 0.005

        roi_pct = (
            (net_pnl / stake_usd) * 100
            if stake_usd > 0
            else 0.0
        )

        overshot = exit_price - exit_target

        if overshot > 0.001:
            overshot_str = (
                f"+{overshot:.3f} above target"
            )

        elif overshot < -0.001:
            overshot_str = (
                f"{overshot:.3f} below target"
            )

        else:
            overshot_str = "exactly at target"

        return (
            f"[SLINGGER V5] 🎯 SWING TARGET HIT\n"
            f"\n"
            f"📌 {market_slug}\n"
            f"\n"
            f"📊 {entry_price:.3f} "
            f"────────────→ "
            f"{exit_price:.3f}\n"
            f"💰 PnL    : "
            f"{fmt_money(net_pnl)} "
            f"({roi_pct:+.1f}%)\n"
            f"   Gross  : "
            f"{fmt_money(gross_pnl)}\n"
            f"   Fee    : "
            f"-${fee_cost:.2f} "
            f"| Spread: -$0.005\n"
            f"📦 Shares : {shares:.2f}\n"
            f"⏱  Time   : "
            f"{latency_s}s "
            f"(TTR left: {ttr_at_exit}s)\n"
            f"📈 Exit   : {overshot_str}\n"
            f"\n"
            f"📅 Today  : "
            f"{fmt_money(daily_pnl)} "
            f"| {daily_wins}W / {daily_losses}L"
        )

    @staticmethod
    def emergency(market_slug: str,
                  ttr: int,
                  decision: str,
                  ev_exit: float,
                  ev_hold: float,
                  current_price: float,
                  exit_target: float,
                  stake_usd: float) -> str:

        gap = exit_target - current_price

        gap_str = (
            f"{gap:+.3f} remaining"
            if gap > 0
            else f"{gap:+.3f} beyond target"
        )

        d_emoji = (
            "🚪"
            if decision == 'EXIT_NOW'
            else "⏳"
        )

        return (
            f"[SLINGGER V5] ⚠️ EMERGENCY\n"
            f"\n"
            f"📌 {market_slug}\n"
            f"\n"
            f"⏱  TTR Left  : {ttr}s\n"
            f"📊 Current   : "
            f"{current_price:.3f} "
            f"(target: {exit_target:.3f})\n"
            f"📦 Gap       : {gap_str}\n"
            f"\n"
            f"EV EXIT NOW      : "
            f"{ev_exit:+.4f}/share\n"
            f"EV HOLD MATURITY : "
            f"{ev_hold:+.4f}/share\n"
            f"\n"
            f"⚡ Decision  : "
            f"{d_emoji} {decision}\n"
            f"💰 Stake at risk : "
            f"${stake_usd:.2f}"
        )

    @staticmethod
    def miss(market_slug: str,
             entry_price: float,
             exit_target: float,
             final_price: float,
             stake_usd: float,
             reason: str,
             daily_pnl: float,
             daily_wins: int,
             daily_losses: int) -> str:

        from src.utils import fmt_money

        loss = -stake_usd - 0.005

        gap_missed = (
            exit_target - final_price
        )

        return (
            f"[SLINGGER V5] ❌ MISS\n"
            f"\n"
            f"📌 {market_slug}\n"
            f"\n"
            f"🎯 Entry  : {entry_price:.3f}\n"
            f"🏁 Target : "
            f"{exit_target:.3f} "
            f"(missed by {gap_missed:.3f})\n"
            f"📊 Final  : {final_price:.3f}\n"
            f"💸 Loss   : ${loss:.2f}\n"
            f"📝 Reason : {reason}\n"
            f"\n"
            f"📅 Today  : "
            f"{fmt_money(daily_pnl)} "
            f"| {daily_wins}W / {daily_losses}L"
        )

    @staticmethod
    def daily_summary(date_str: str,
                      total: int,
                      hit: int,
                      miss: int,
                      emergency: int,
                      emergency_exit_now: int,
                      emergency_hold: int,
                      gross_pnl: float,
                      total_fees: float,
                      net_pnl: float,
                      best_trade: float,
                      worst_trade: float,
                      avg_win: float,
                      avg_loss: float,
                      avg_hold_seconds: float,
                      sharpe_1d: float,
                      current_capital: float) -> str:

        from src.utils import fmt_money

        hit_rate = (
            hit / max(total, 1)
        ) * 100

        return (
            f"[SLINGGER V5] 📊 DAILY SUMMARY\n"
            f"{date_str} UTC\n"
            f"\n"
            f"Trades   : "
            f"{total} "
            f"({hit} hit / "
            f"{miss} miss / "
            f"{emergency}E)\n"
            f"Hit Rate : {hit_rate:.1f}%\n"
            f"\n"
            f"💰 Gross PnL : "
            f"{fmt_money(gross_pnl)}\n"
            f"   Fees      : "
            f"-${total_fees:.2f}\n"
            f"   Net PnL   : "
            f"{fmt_money(net_pnl)}\n"
            f"\n"
            f"📊 Best trade  : "
            f"{fmt_money(best_trade)}\n"
            f"   Worst trade : "
            f"{fmt_money(worst_trade)}\n"
            f"   Avg win     : "
            f"{fmt_money(avg_win)}\n"
            f"   Avg loss    : "
            f"{fmt_money(avg_loss)}\n"
            f"\n"
            f"⚠️  Emergency  : "
            f"{emergency} "
            f"({emergency_exit_now} EXIT / "
            f"{emergency_hold} HOLD)\n"
            f"⏱  Avg hold   : "
            f"{avg_hold_seconds:.0f}s\n"
            f"📈 Sharpe (1D): "
            f"{sharpe_1d:.3f}\n"
            f"💼 Capital    : "
            f"${current_capital:.2f}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ALPHA V1: Order & Session Lifecycle Notifications
    # Called exclusively from: _on_bar_close → _schedule_resolution path
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def system_health(data: dict) -> str:
        return "[ALPHA V1] 🛠 SYSTEM HEALTH START\n\n" + SlingshotAlerts._tg_kv(data)

    @staticmethod
    def heartbeat(data: dict) -> str:
        return "[ALPHA V1] 💓 HEARTBEAT / MARKET WATCH\n\n" + SlingshotAlerts._tg_kv(data)

    @staticmethod
    def session_report(title: str, data: dict) -> str:
        return f"[ALPHA V1] 📊 {title}\n\n" + SlingshotAlerts._tg_kv(data)

    @staticmethod
    def order_execution(title: str, data: dict) -> str:
        return f"[ALPHA V1] 📦 {title}\n\n" + SlingshotAlerts._tg_kv(data)

    @staticmethod
    def order_result(data: dict) -> str:
        return "[ALPHA V1] ✅ ORDER RESULT\n\n" + SlingshotAlerts._tg_kv(data)

    @staticmethod
    def paper_trade_resolved(data: dict) -> str:
        return "[ALPHA V1] 📊 Paper Trade Resolved\n\n" + SlingshotAlerts._tg_kv(data)

    @staticmethod
    def session_aborted(reason: str, session_id: str) -> str:
        return (
            f"[ALPHA V1] 🛑 SESSION ABORTED\n\n"
            f"Reason: {reason}\n"
            f"Session: {session_id}"
        )

    @staticmethod
    def go_live_enabled(data: dict) -> str:
        return "[ALPHA V1] 🚀 GO LIVE ENABLED\n\n" + SlingshotAlerts._tg_kv(data)

    @staticmethod
    def dry_run_limit(max_hours: float, session_id: str) -> str:
        return (
            f"[ALPHA V1] ⏳ DRY RUN TIME LIMIT\n\n"
            f"Dry-run belum mencapai gate live dalam maksimal {max_hours} jam.\n"
            f"Session: {session_id}"
        )

    @staticmethod
    def session_finished(title: str, prefix: str, reason: str, stats_text: str) -> str:
        return (
            f"[ALPHA V1] 🏁 {title}\n\n"
            f"{prefix}\n"
            f"Reason: {reason}\n\n"
            f"{stats_text}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Shared Utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _tg_kv(data: dict) -> str:
        import html
        lines = []
        for k, v in data.items():
            key = html.escape(str(k))
            val = html.escape(str(v))
            lines.append(f"<b>{key}</b>: {val}")
        return "\n".join(lines)
