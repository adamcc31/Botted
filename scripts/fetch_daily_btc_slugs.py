import httpx
import pandas as pd
from datetime import timedelta, datetime, timezone
import asyncio
import sys
from pathlib import Path

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.collect_polymarket import _parse_resolved_market

async def fetch_target_slug(client: httpx.AsyncClient, date_obj: datetime) -> list:
    """Attempt to fetch 'bitcoin-up-or-down-on-month-d-yyyy'."""
    month = date_obj.strftime("%B").lower()
    day = date_obj.day
    year = date_obj.year
    slug = f"bitcoin-up-or-down-on-{month}-{day}-{year}"

    try:
        r = await client.get(f"https://gamma-api.polymarket.com/events", params={"slug": slug})
        if r.status_code == 200:
            events = r.json()
            if events and isinstance(events, list):
                markets = events[0].get("markets", [])
                parsed_markets = []
                for m in markets:
                    parsed = _parse_resolved_market(m)
                    if parsed:
                        # Ensure resolution_date matches the slug roughly to flag its daily nature
                        parsed["is_daily"] = True
                        parsed["slug_date"] = date_obj.strftime("%Y-%m-%d")
                        parsed_markets.append(parsed)
                return parsed_markets
    except Exception as e:
        pass
    return []

async def run_discovery():
    print("Initiating slug-based chronological Daily BTC collection...")
    # Generate last 100 days
    end_date = datetime.now(timezone.utc)
    dates = [end_date - timedelta(days=d) for d in range(100)]
    
    all_markets = []
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Batch to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(dates), batch_size):
            batch_dates = dates[i:i+batch_size]
            tasks = [fetch_target_slug(client, d) for d in batch_dates]
            results = await asyncio.gather(*tasks)
            for res in results:
                all_markets.extend(res)
            print(f"Processed {i+len(batch_dates)}/{len(dates)} days. Found {len(all_markets)} valid markets.")
            await asyncio.sleep(1.0) # rate limit respect
            
    if all_markets:
        df = pd.DataFrame(all_markets)
        df = df.drop_duplicates(subset=["market_id"])
        
        # Enforce floats
        for col in ["volume_usd", "liquidity_usd", "yes_final_price", "no_final_price"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        
        print(f"\nFinal tally: {len(df)} daily markets found.")
        out_path = Path(__file__).parent.parent / "data" / "raw" / "polymarket_markets.parquet"
        df.to_parquet(out_path)
        print(f"Overwritten exactly clean daily dataset to {out_path}")
    else:
        print("No markets found.")

if __name__ == "__main__":
    asyncio.run(run_discovery())
