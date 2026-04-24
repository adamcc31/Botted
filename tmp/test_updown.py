import asyncio
import httpx
import pandas as pd
from datetime import datetime
import sys
sys.path.append('.')
from scripts.fetch_oracle_prices import fetch_pyth_price

async def test_updown_10():
    df_markets = pd.read_parquet('data/raw/polymarket_markets.parquet')
    
    # Filter for Up/Down Daily Markets
    q_lower = df_markets["question"].str.lower()
    mask = (
        q_lower.str.contains('up|down') & 
        ~q_lower.str.contains('dip|below') &
        ~q_lower.str.contains(':|am-|pm-|min|5-minute|15-minute') &
        df_markets["outcome_binary"].notna()
    )
    df_filtered = df_markets[mask].copy()
    
    # Filter lifespan >= 18h & post 2024
    df_filtered["lifespan_ms"] = df_filtered["t_resolution_epoch_ms"] - df_filtered["t_open_epoch_ms"]
    df_filtered = df_filtered[df_filtered["lifespan_ms"] >= 18 * 3600 * 1000]
    df_filtered = df_filtered[df_filtered["t_resolution_epoch_ms"] > 1704067200000]
    
    # Pick 10 random markets
    sample = df_filtered.sample(10, random_state=42)
    
    # Fetch Pyth Cache
    cache_df = pd.read_parquet('data/raw/pyth_prices.parquet')
    cache = dict(zip(cache_df["timestamp_ms"], cache_df["oracle_price"]))
    
    results = []
    
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(5)
        for _, row in sample.iterrows():
            mid = row['market_id']
            q = row['question']
            t_open = int(row['t_open_epoch_ms'])
            t_res = int(row['t_resolution_epoch_ms'])
            outcome = int(row['outcome_binary'])
            
            # 1. Fetch Pyth Open Price
            open_price = await fetch_pyth_price(client, t_open, sem)
            
            # 2. Get Resolution Price
            res_price = cache.get(t_res)
            if res_price is None:
                res_price = await fetch_pyth_price(client, t_res, sem)
            
            if open_price is not None and res_price is not None:
                computed_label = 1 if res_price > open_price else 0
                match = (computed_label == outcome)
                results.append({
                    "market_id": mid,
                    "target": "Up/Down",
                    "open_utc": datetime.fromtimestamp(t_open//1000).strftime('%m-%d %H:%M'),
                    "res_utc": datetime.fromtimestamp(t_res//1000).strftime('%m-%d %H:%M'),
                    "pyth_open": round(open_price, 2),
                    "pyth_res": round(res_price, 2),
                    "computed": computed_label,
                    "actual": outcome,
                    "match": "✅" if match else "❌"
                })
            else:
                print(f"Failed to fetch Pyth data for {mid}")

    print("\n" + "="*80)
    print("UP/DOWN STRIKE IMPUTATION TEST (10 Markets)")
    print("="*80)
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        print(df_res.to_string(index=False))
        match_rate = df_res['match'].apply(lambda x: 1 if x == '✅' else 0).mean()
        print(f"\nTarget Accuracy: {match_rate:.1%}")
    else:
        print("No valid results fetched.")

if __name__ == "__main__":
    asyncio.run(test_updown_10())
