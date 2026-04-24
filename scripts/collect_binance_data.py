"""
collect_binance_data.py — Historical data collection from Binance.

Collects TWO data types:
  1. OHLCV 15-minute bars via REST API → data/raw/ohlcv_15m.parquet
  2. aggTrades from Binance Vision (daily ZIP files) → data/raw/aggTrades/

aggTrades are ESSENTIAL for reconstructing Trade Flow Momentum (TFM) historically.
Without them, TFM features (the core alpha) would be zero during training.

Usage:
  python scripts/collect_binance_data.py                        # OHLCV only
  python scripts/collect_binance_data.py --with-aggtrades       # OHLCV + aggTrades
  python scripts/collect_binance_data.py --aggtrades-only       # aggTrades only
  python scripts/collect_binance_data.py --months 6             # 6 months
"""

from __future__ import annotations

import asyncio
import io
import os
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import click
import httpx
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

REST_URL = "https://api.binance.com/api/v3/klines"
VISION_BASE = "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
AGGTRADES_DIR = OUTPUT_DIR / "aggTrades"
MAX_PER_REQUEST = 1000
RATE_LIMIT_DELAY = 0.05


# ============================================================
# OHLCV Collection (REST API)
# ============================================================

async def fetch_klines(
    client: httpx.AsyncClient,
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int,
    limit: int = MAX_PER_REQUEST,
) -> list:
    """Fetch klines from Binance REST API."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": limit,
    }
    resp = await client.get(REST_URL, params=params)
    resp.raise_for_status()
    return resp.json()


async def collect_historical_ohlcv(
    months: int = 3,
    symbol: str = "BTCUSDT",
    interval: str = "15m",
) -> pd.DataFrame:
    """Collect historical OHLCV data from Binance REST API."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=months * 30)

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    all_klines = []
    current_start = start_ms

    async with httpx.AsyncClient(timeout=30.0) as client:
        request_count = 0
        while current_start < end_ms:
            try:
                klines = await fetch_klines(
                    client, symbol, interval,
                    start_time=current_start,
                    end_time=end_ms,
                )

                if not klines:
                    break

                all_klines.extend(klines)
                request_count += 1

                last_time = datetime.fromtimestamp(
                    klines[-1][0] / 1000, tz=timezone.utc
                )
                if request_count % 10 == 0:
                    logger.info(
                        "fetching_ohlcv",
                        request=request_count,
                        bars=len(all_klines),
                        last_time=last_time.strftime("%Y-%m-%d %H:%M"),
                    )

                current_start = klines[-1][6] + 1
                await asyncio.sleep(RATE_LIMIT_DELAY)

            except httpx.HTTPError as e:
                logger.error("binance_rest_error", error=str(e))
                await asyncio.sleep(5.0)
            except Exception as e:
                logger.error("collect_error", error=str(e))
                break

    if not all_klines:
        logger.error("no_ohlcv_data_collected")
        return pd.DataFrame()

    df = pd.DataFrame(all_klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades_count",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])

    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                 "taker_buy_base", "taker_buy_quote"]:
        df[col] = df[col].astype(float)

    for col in ["open_time", "close_time", "trades_count"]:
        df[col] = df[col].astype("int64")

    df = df.drop(columns=["ignore"])
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    logger.info("ohlcv_collection_complete", total_bars=len(df))
    return df


# ============================================================
# aggTrades Collection (Binance Vision ZIP downloads)
# ============================================================

async def download_aggtrades_day(
    client: httpx.AsyncClient,
    date_str: str,
) -> Optional[pd.DataFrame]:
    """
    Download one day of aggTrades from Binance Vision.

    Format: BTCUSDT-aggTrades-YYYY-MM-DD.zip
    Each ZIP contains a CSV with columns:
      agg_trade_id, price, quantity, first_trade_id, last_trade_id,
      timestamp, is_buyer_maker, is_best_match

    The is_buyer_maker flag is the KEY for TFM reconstruction:
      True  = taker sell (seller initiated)
      False = taker buy  (buyer initiated)
    """
    url = f"{VISION_BASE}/BTCUSDT-aggTrades-{date_str}.zip"
    try:
        resp = await client.get(url, follow_redirects=True, timeout=60.0)
        if resp.status_code == 404:
            logger.debug("aggtrades_not_found", date=date_str)
            return None
        resp.raise_for_status()

        # Extract CSV from ZIP
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as csv_file:
                df = pd.read_csv(
                    csv_file,
                    header=None,
                    names=[
                        "agg_trade_id", "price", "quantity",
                        "first_trade_id", "last_trade_id",
                        "timestamp", "is_buyer_maker", "is_best_match",
                    ],
                    dtype={
                        "agg_trade_id": "int64",
                        "price": float,
                        "quantity": float,
                        "first_trade_id": "int64",
                        "last_trade_id": "int64",
                        "timestamp": "int64",
                        "is_buyer_maker": bool,
                        "is_best_match": bool,
                    },
                )
                return df

    except httpx.HTTPError as e:
        logger.warning("aggtrades_download_error", date=date_str, error=str(e))
        return None
    except Exception as e:
        logger.error("aggtrades_parse_error", date=date_str, error=str(e))
        return None


async def collect_aggtrades(months: int = 3) -> int:
    """
    Download aggTrades from Binance Vision for dates matching Polymarket data.
    Reads polymarket_markets.parquet to determine required dates.
    """
    AGGTRADES_DIR.mkdir(parents=True, exist_ok=True)
    
    # ── Calculate required dates based on Polymarket markets ──
    required_dates = set()
    markets_file = Path(__file__).parent.parent / "data" / "raw" / "polymarket_markets.parquet"
    if markets_file.exists():
        try:
            mdf = pd.read_parquet(markets_file)
            valid_m = mdf[mdf["t_resolution_epoch_ms"].notna()]
            for _, row in valid_m.iterrows():
                # Get resolution time
                t_res = datetime.fromtimestamp(row["t_resolution_epoch_ms"]/1000, tz=timezone.utc)
                # TFM requires aggTrades from up to ~12 hours before resolution
                # So we just need the day of resolution, and the day before it.
                required_dates.add(t_res.strftime("%Y-%m-%d"))
                required_dates.add((t_res - timedelta(days=1)).strftime("%Y-%m-%d"))
            logger.info("aggtrades_date_filter_applied", required_days=len(required_dates))
        except Exception as e:
            logger.warning("failed_to_read_polymarket_for_dates", error=str(e))
    else:
        logger.warning("polymarket_data_missing_downloading_all_dates_in_range")

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=months * 30)
    current = start
    files_downloaded = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        while current < now - timedelta(days=1):  # Vision has ~1 day delay
            date_str = current.strftime("%Y-%m-%d")
            
            # If we know the required dates, skip if it's not in the set
            if required_dates and date_str not in required_dates:
                current += timedelta(days=1)
                continue

            out_file = AGGTRADES_DIR / f"aggTrades_{date_str}.parquet"

            # Skip if already downloaded
            if out_file.exists():
                logger.debug("aggtrades_already_exists", date=date_str)
                current += timedelta(days=1)
                continue

            df = await download_aggtrades_day(client, date_str)

            if df is not None and not df.empty:
                df.to_parquet(out_file, index=False)
                files_downloaded += 1

                if files_downloaded % 7 == 0:
                    logger.info(
                        "aggtrades_progress",
                        date=date_str,
                        files=files_downloaded,
                        rows=len(df),
                    )

            current += timedelta(days=1)
            await asyncio.sleep(0.3)  # Rate limit

    logger.info("aggtrades_collection_complete", total_files=files_downloaded)
    return files_downloaded


# ============================================================
# CLI
# ============================================================

@click.command()
@click.option("--months", default=3, help="Months of historical data")
@click.option("--symbol", default="BTCUSDT", help="Trading pair")
@click.option("--with-aggtrades", is_flag=True, help="Also download aggTrades from Binance Vision")
@click.option("--aggtrades-only", is_flag=True, help="Download only aggTrades")
@click.option("--output", default=None, help="Output file path for OHLCV")
def main(
    months: int,
    symbol: str,
    with_aggtrades: bool,
    aggtrades_only: bool,
    output: str | None,
) -> None:
    """Collect historical Binance data (OHLCV + aggTrades)."""

    async def _run():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if not aggtrades_only:
            click.echo(f"📊 Collecting {months} months of {symbol} 15m OHLCV data...")
            df = await collect_historical_ohlcv(months=months, symbol=symbol)

            if df.empty:
                click.echo("❌ No OHLCV data collected")
            else:
                out_path = Path(output) if output else OUTPUT_DIR / "ohlcv_15m.parquet"
                df.to_parquet(out_path, index=False)
                click.echo(f"✅ Saved {len(df)} OHLCV bars to {out_path}")

        if with_aggtrades or aggtrades_only:
            click.echo(f"\n📥 Downloading {months} months of aggTrades from Binance Vision...")
            click.echo(f"   Output: {AGGTRADES_DIR}")
            click.echo("   ⚠  This may take a while (~50-200MB per day)\n")
            n_files = await collect_aggtrades(months=months)
            click.echo(f"✅ Downloaded {n_files} daily aggTrades files")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
