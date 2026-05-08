import asyncio
import os
from datetime import datetime, timezone
import numpy as np
from dotenv import load_dotenv
from src.telegram_notifier import TelegramNotifier, SlingshotAlerts
from src.config_manager import ConfigManager
from src.utils import fmt_money

load_dotenv()

async def test_telemetry():
    print("Starting Telemetry Upgrade Verification...")
    
    # 1. Initialize Config and Notifier
    config = ConfigManager.get_instance()
    notifier = TelegramNotifier(config)
    
    if not notifier._enabled:
        print("❌ Telegram Notifier is DISABLED. Check .env and config.")
        return

    print(f"Telegram Connected (Chat ID: {notifier._chat_id})")

    # 2. Test Data
    market_slug = "btc-price-at-2026-05-08-21-30"
    entry_price = 0.470
    exit_target = 0.800
    ttr = 120
    confidence = 0.685
    side = "YES"
    stake_usd = 2.50
    shares = 5.32
    depth_available_usd = 450.0
    btc_vs_strike_pct = 0.15

    # 3. Test Entry Alert
    print("--- Sending Entry Alert ---")
    msg_entry = SlingshotAlerts.entry(
        market_slug=market_slug,
        entry_price=entry_price,
        exit_target=exit_target,
        ttr=ttr,
        confidence=confidence,
        side=side,
        stake_usd=stake_usd,
        shares=shares,
        depth_available_usd=depth_available_usd,
        btc_vs_strike_pct=btc_vs_strike_pct
    )
    await notifier.send_message("SLINGGER V5", msg_entry)

    # 4. Test Exit Hit Alert
    print("--- Sending Exit Hit Alert ---")
    msg_hit = SlingshotAlerts.exit_hit(
        market_slug=market_slug,
        entry_price=entry_price,
        exit_price=0.812,
        exit_target=exit_target,
        latency_s=45,
        ttr_at_exit=75,
        shares=shares,
        stake_usd=stake_usd,
        daily_pnl=12.45,
        daily_wins=3,
        daily_losses=1
    )
    await notifier.send_message("SLINGGER V5", msg_hit)

    # 5. Test Emergency Alert
    print("--- Sending Emergency Alert ---")
    msg_emerg = SlingshotAlerts.emergency(
        market_slug=market_slug,
        ttr=55,
        decision="EXIT_NOW",
        ev_exit=0.015,
        ev_hold=-0.042,
        current_price=0.550,
        exit_target=exit_target,
        stake_usd=stake_usd
    )
    await notifier.send_message("SLINGGER V5", msg_emerg)

    # 6. Test Miss Alert
    print("--- Sending Miss Alert ---")
    msg_miss = SlingshotAlerts.miss(
        market_slug=market_slug,
        entry_price=entry_price,
        exit_target=exit_target,
        final_price=0.350,
        stake_usd=stake_usd,
        reason="expired",
        daily_pnl=10.15,
        daily_wins=3,
        daily_losses=2
    )
    await notifier.send_message("SLINGGER V5", msg_miss)

    # 7. Test Daily Summary
    print("--- Sending Daily Summary ---")
    msg_summary = SlingshotAlerts.daily_summary(
        date_str=datetime.now(timezone.utc).date().isoformat(),
        total=15,
        hit=10,
        miss=3,
        emergency=2,
        emergency_exit_now=1,
        emergency_hold=1,
        gross_pnl=45.20,
        total_fees=3.15,
        net_pnl=42.05,
        best_trade=8.50,
        worst_trade=-2.50,
        avg_win=4.52,
        avg_loss=-1.25,
        avg_hold_seconds=145.0,
        sharpe_1d=2.15,
        current_capital=142.05
    )
    await notifier.send_message("Daily Summary", msg_summary)

    # 8. Test Heartbeat (KV formatting)
    print("--- Sending Heartbeat ---")
    hb_data = {
        "status": "RUNNING",
        "market": market_slug,
        "btc_price": 65432.10,
        "active_tasks": 3,
        "memory_usage": "142MB"
    }
    msg_hb = SlingshotAlerts.heartbeat(hb_data)
    await notifier.send_message("HEARTBEAT", msg_hb)

    print("\nVerification Script Completed. Please check your Telegram.")

if __name__ == "__main__":
    asyncio.run(test_telemetry())
