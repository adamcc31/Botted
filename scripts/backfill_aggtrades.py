import asyncio
import httpx
import pandas as pd
import zipfile
import io
import os
import structlog
from pathlib import Path
from datetime import datetime, timedelta

logger = structlog.get_logger(__name__)

# Constants
START_DATE = "2026-01-01"
END_DATE = "2026-03-26" # We already have March 27 onwards locally
SYMBOL = "BTCUSDT"
BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
TARGET_DIR = Path("data/raw/aggTrades")

COLUMNS = [
    "agg_trade_id", 
    "price", 
    "quantity", 
    "first_trade_id", 
    "last_trade_id", 
    "timestamp", 
    "is_buyer_maker", 
    "is_best_match"
]

MAX_RETRIES = 3

async def download_and_process(client: httpx.AsyncClient, date_str: str, semaphore: asyncio.Semaphore):
    file_name = f"{SYMBOL}-aggTrades-{date_str}"
    zip_name = f"{file_name}.zip"
    url = f"{BASE_URL}/{SYMBOL}/{zip_name}"
    
    out_path = TARGET_DIR / f"aggTrades_{date_str}.parquet"
    if out_path.exists():
        logger.info("skip_existing", date=date_str)
        return

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                # 1. Download ZIP
                logger.info("downloading", date=date_str, url=url)
                resp = await client.get(url, timeout=30.0)
                
                if resp.status_code == 404:
                    logger.warning("missing_on_binance", date=date_str)
                    return
                resp.raise_for_status()
                
                # 2. Extract and load into Pandas from memory
                zip_file = zipfile.ZipFile(io.BytesIO(resp.content))
                csv_filename = zip_file.namelist()[0]
                
                with zip_file.open(csv_filename) as f:
                    # Binance Vision CSV can have headers sometimes. 
                    # We will read it without header first, check if first row is string.
                    # Usually it's raw data or the header matches closely.
                    # We'll use a fast engine with fallback column mapping.
                    df = pd.read_csv(f, header=None, low_memory=False)
                    
                    # If first element of first column is not numeric, it's a header row
                    if not str(df.iloc[0, 0]).isdigit():
                        df = df.iloc[1:].reset_index(drop=True)
                        
                    # Standardize columns
                    if len(df.columns) == 8:
                        df.columns = COLUMNS
                    elif len(df.columns) == 7: # sometimes is_best_match is omitted
                        df.columns = COLUMNS[:7]
                        df['is_best_match'] = True
                    else:
                        logger.error("schema_mismatch", date=date_str, cols=len(df.columns))
                        return

                    # Enforce data types
                    df['agg_trade_id'] = df['agg_trade_id'].astype('int64')
                    df['price'] = df['price'].astype('float64')
                    df['quantity'] = df['quantity'].astype('float64')
                    df['first_trade_id'] = df['first_trade_id'].astype('int64')
                    df['last_trade_id'] = df['last_trade_id'].astype('int64')
                    df['timestamp'] = df['timestamp'].astype('int64')
                    
                    # Handle boolean conversion (Binance uses strings like 'True'/'False' or bool)
                    if df['is_buyer_maker'].dtype != 'bool':
                        df['is_buyer_maker'] = df['is_buyer_maker'].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
                    if df['is_best_match'].dtype != 'bool':
                        df['is_best_match'] = df['is_best_match'].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})

                # 3. Save as Parquet
                df.to_parquet(out_path, index=False)
                logger.info("processed", date=date_str, rows=len(df), size_mb=round(out_path.stat().st_size/1024/1024, 1))
                return

            except Exception as e:
                wait = 5 * (attempt + 1)
                logger.warning("download_error", date=date_str, error=str(e), attempt=attempt, wait=wait)
                await asyncio.sleep(wait)
        
        logger.error("failed_to_process", date=date_str)

async def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(start=START_DATE, end=END_DATE).strftime("%Y-%m-%d").tolist()
    
    logger.info("start_backfill", start=START_DATE, end=END_DATE, total_days=len(dates))
    
    # Binance Vision allows multiple concurrent downloads.
    # We will use 4 to avoid overwhelming memory as extracting zips can be memory intensive.
    semaphore = asyncio.Semaphore(4)
    
    async with httpx.AsyncClient() as client:
        tasks = [download_and_process(client, d, semaphore) for d in dates]
        await asyncio.gather(*tasks)
        
    logger.info("backfill_complete")

if __name__ == "__main__":
    asyncio.run(main())
