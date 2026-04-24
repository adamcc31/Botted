import asyncio
import httpx
import pandas as pd
import structlog
import subprocess
from pathlib import Path
from tqdm.asyncio import tqdm
from typing import Dict, Optional

# Logger setup
logger = structlog.get_logger(__name__)

# Constants
PYTH_HERMES_URL = "https://hermes.pyth.network/v2/updates/price"
BTC_FEED_ID = "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43"
RAW_DATA_DIR = Path("data/raw")
MARKETS_PATH = RAW_DATA_DIR / "polymarket_markets.parquet"
CACHE_PATH = RAW_DATA_DIR / "pyth_prices.parquet"

MAX_RETRIES = 5
BACKOFF_FACTOR = 2

def parse_pyth_binary_pnau_v2(hex_data: str, feed_id: str = BTC_FEED_ID):
    """
    Decodes Pyth PNAU binary for a specific feed_id.
    PNAU structures in Hermes V2 often wrap the PriceFeed message.
    """
    try:
        h = hex_data.replace("0x", "")
        clean_fid = feed_id.replace("0x", "")
        # Find the internal PriceFeed message start using the Feed ID
        idx = h.find(clean_fid)
        if idx == -1:
            return None
            
        price_hex = h[idx + 64 : idx + 80]
        expo_hex = h[idx + 96 : idx + 104]
        
        price_raw = int(price_hex, 16)
        if price_raw > 0x7FFFFFFFFFFFFFFF:
            price_raw -= 0x10000000000000000
            
        exponent = int(expo_hex, 16)
        if exponent > 0x7FFFFFFF:
            exponent -= 0x100000000
            
        return price_raw * (10 ** exponent)
    except Exception as e:
        logger.error("binary_decode_error", error=str(e))
        return None

async def fetch_pyth_price(client: httpx.AsyncClient, timestamp_ms: int, semaphore: asyncio.Semaphore) -> Optional[float]:
    """Fetch BTC price from Pyth Hermes V2 for a given timestamp."""
    timestamp_sec = int(timestamp_ms // 1000)
    url = f"{PYTH_HERMES_URL}/{timestamp_sec}"
    params = {
        "ids[]": BTC_FEED_ID,
        "parsed": "true"
    }

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.get(url, params=params, timeout=15.0)
                
                if resp.status_code == 429:
                    wait = BACKOFF_FACTOR ** attempt
                    await asyncio.sleep(wait)
                    continue
                
                if resp.status_code == 404:
                    return None
                    
                resp.raise_for_status()
                data = resp.json()
                
                # Check for Parsed JSON
                parsed = data.get("parsed", [])
                if parsed:
                    p_obj = parsed[0].get("price", {})
                    price = float(p_obj.get("price", 0))
                    expo = int(p_obj.get("expo", 0))
                    return price * (10 ** expo)
                
                # Fallback to Binary PNAU Decoder
                binary_list = data.get("binary", {}).get("data", [])
                if binary_list:
                    return parse_pyth_binary_pnau_v2(binary_list[0])
                
                return None

            except httpx.ConnectError as e:
                logger.warning("dns_connect_error", error=str(e), attempt=attempt)
                await asyncio.sleep(30.0) # Long sleep for DNS recovery
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error("pyth_fetch_failed", error=str(e), timestamp=timestamp_sec)
                    return None
                await asyncio.sleep(BACKOFF_FACTOR ** attempt)
    return None

async def main():
    if not MARKETS_PATH.exists():
        logger.error("markets_file_missing")
        return

    df_markets = pd.read_parquet(MARKETS_PATH)
    q_lower = df_markets["question"].str.lower()
    mask = (
        q_lower.str.contains('above') & 
        ~q_lower.str.contains('dip|below|up|down') &
        ~q_lower.str.contains(':|am-|pm-|min|5-minute|15-minute')
    )
    df_filtered = df_markets[mask].copy()
    
    df_filtered["lifespan_ms"] = df_filtered["t_resolution_epoch_ms"] - df_filtered["t_open_epoch_ms"]
    df_filtered = df_filtered[df_filtered["lifespan_ms"] >= 18 * 3600 * 1000]
    
    MIN_TS_MS = 1704067200000 
    df_filtered = df_filtered[df_filtered["t_resolution_epoch_ms"] > MIN_TS_MS]
    
    unique_timestamps = df_filtered["t_resolution_epoch_ms"].dropna().unique().astype("int64")
    logger.info("fetch_start", total_markets=len(df_filtered), unique_timestamps=len(unique_timestamps))

    cache: Dict[int, float] = {}
    if CACHE_PATH.exists():
        try:
            df_cache = pd.read_parquet(CACHE_PATH)
            cache = dict(zip(df_cache["timestamp_ms"], df_cache["oracle_price"]))
        except Exception as e:
            logger.warning("cache_load_error_resetting", error=str(e))
            cache = {}
            
    logger.info("cache_loaded", entries=len(cache))

    missing_ts = [ts for ts in unique_timestamps if ts not in cache]
    logger.info("missing_timestamps", count=len(missing_ts))

    if not missing_ts:
        logger.info("all_prices_cached")
    else:
        semaphore = asyncio.Semaphore(5)
        async with httpx.AsyncClient() as client:
            save_counter = 0
            for ts in tqdm(missing_ts, desc="Fetching Pyth Prices"):
                price = await fetch_pyth_price(client, ts, semaphore)
                if price is not None:
                    cache[ts] = price
                    save_counter += 1
                    
                    # Batch save every 10 updates to avoid constant I/O
                    if save_counter % 10 == 0:
                        df_new_cache = pd.DataFrame(list(cache.items()), columns=["timestamp_ms", "oracle_price"])
                        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
                        df_new_cache.to_parquet(CACHE_PATH, index=False)
                        
                await asyncio.sleep(0.2) # 0.2 sec delay

            # Final save
            df_new_cache = pd.DataFrame(list(cache.items()), columns=["timestamp_ms", "oracle_price"])
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            df_new_cache.to_parquet(CACHE_PATH, index=False)
            logger.info("cache_saved_final", path=str(CACHE_PATH), entries=len(df_new_cache))

    # Final Validation
    df_final = df_filtered.merge(
        pd.DataFrame(list(cache.items()), columns=["t_resolution_epoch_ms", "oracle_price"]),
        on="t_resolution_epoch_ms",
        how="left"
    )
    
    null_rate = df_final["oracle_price"].isna().mean()
    logger.info("verification_complete", null_rate=f"{null_rate:.1%}")
    
    if null_rate > 0.05: # Allow up to 5% failure without blocking
        logger.warning("null_rate_elevated", null_rate=null_rate)
        
    logger.info("triggering_build_dataset")
    subprocess.run([".\\venv\\Scripts\\python.exe", "scripts/build_dataset.py"])

if __name__ == "__main__":
    asyncio.run(main())
