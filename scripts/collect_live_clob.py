import asyncio
import httpx
import pandas as pd
import structlog
from pathlib import Path
from datetime import datetime, timezone
import os

# Configuration
SNAPSHOT_DIR = Path("data/raw/clob_snapshots")
POLL_INTERVAL = 300  # 5 minutes
HEARTBEAT_INTERVAL = 1800  # 30 minutes
GAP_THRESHOLD = 900  # 15 minutes

logger = structlog.get_logger(__name__)

async def get_active_btc_markets(client: httpx.AsyncClient):
    """
    Fetch active BTC Above markets from Gamma API.
    """
    url = "https://gamma-api.polymarket.com/markets"
    params = {
        "active": "true",
        "closed": "false",
        "order": "volume",
        "limit": 50
    }
    try:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        markets = resp.json()
        
        filtered = []
        for m in markets:
            q = m.get("question", "").lower()
            if "btc" in q and "above" in q and not any(x in q for x in ["dip", "below", "up", "down"]):
                # Ensure it has a token ID
                clob_rewards = m.get("clobRewards", [])
                yes_token = m.get("yesTokenId")
                if yes_token:
                    filtered.append({
                        "market_id": m["id"],
                        "condition_id": m.get("conditionId"),
                        "token_id": yes_token,
                        "question": m["question"]
                    })
        return filtered
    except Exception as e:
        logger.error("market_sync_failed", error=str(e))
        return []

async def get_book_snapshot(client: httpx.AsyncClient, token_id: str):
    """
    Fetch current Best Bid/Ask from CLOB Orderbook.
    """
    url = f"https://clob.polymarket.com/book?token_id={token_id}"
    try:
        resp = await client.get(url, timeout=10.0)
        resp.raise_for_status()
        book = resp.json()
        
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        
        if not bids or not asks:
            return None
            
        best_bid = float(bids[0]["price"])
        best_ask = float(asks[0]["price"])
        bid_size = float(bids[0]["size"])
        ask_size = float(asks[0]["size"])
        
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": (best_bid + best_ask) / 2,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "spread_bps": (best_ask - best_bid) / ((best_bid + best_ask) / 2) * 10000
        }
    except Exception as e:
        return None

def check_for_gaps():
    """
    Detect if there is a gap > 15m since the last snapshot.
    """
    if not SNAPSHOT_DIR.exists():
        return
    
    files = sorted(SNAPSHOT_DIR.glob("*.parquet"))
    if not files:
        return
        
    try:
        latest_file = files[-1]
        df = pd.read_parquet(latest_file)
        if df.empty:
            return
            
        last_ts = df["timestamp_ms"].max() / 1000
        current_ts = datetime.now(timezone.utc).timestamp()
        
        gap_sec = current_ts - last_ts
        if gap_sec > GAP_THRESHOLD:
            logger.warning("gap_detected", gap_minutes=round(gap_sec / 60, 1), last_data=datetime.fromtimestamp(last_ts).isoformat())
        else:
            logger.info("continuity_verified", last_data=datetime.fromtimestamp(last_ts).isoformat())
    except Exception as e:
        logger.error("gap_check_failed", error=str(e))

async def main():
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("service_started", interval=POLL_INTERVAL, heartbeat=HEARTBEAT_INTERVAL)
    check_for_gaps()
    
    last_heartbeat = datetime.now(timezone.utc).timestamp()
    total_snapshots = 0
    
    async with httpx.AsyncClient() as client:
        while True:
            start_time = datetime.now(timezone.utc).timestamp()
            
            # 1. Sync Markets
            active_markets = await get_active_btc_markets(client)
            if not active_markets:
                logger.warning("no_active_markets_found")
            else:
                timestamp_ms = int(start_time * 1000)
                snapshots = []
                
                # 2. Sequential Snapshots (avoid rate limits)
                for m in active_markets:
                    data = await get_book_snapshot(client, m["token_id"])
                    if data:
                        data.update({
                            "market_id": m["market_id"],
                            "token_id": m["token_id"],
                            "timestamp_ms": timestamp_ms,
                            "question": m["question"]
                        })
                        snapshots.append(data)
                    await asyncio.sleep(0.5) # Throttle to 2 req/sec
                
                # 3. Save
                if snapshots:
                    df = pd.DataFrame(snapshots)
                    # Use weekly file naming to prevent single massive file
                    current_week = datetime.now(timezone.utc).strftime("%Y-w%W")
                    file_path = SNAPSHOT_DIR / f"snapshots_{current_week}.parquet"
                    
                    if file_path.exists():
                        existing = pd.read_parquet(file_path)
                        df = pd.concat([existing, df], ignore_index=True)
                    
                    df.to_parquet(file_path, index=False)
                    total_snapshots += len(snapshots)
                    
            # 4. Heartbeat
            now = datetime.now(timezone.utc).timestamp()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                logger.info("heartbeat", total_recorded=total_snapshots, active_markets=len(active_markets))
                last_heartbeat = now
            
            # 5. Sleep until next 5m interval
            elapsed = now - start_time
            sleep_time = max(0, POLL_INTERVAL - elapsed)
            await asyncio.sleep(sleep_time)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("service_stopped_by_user")
