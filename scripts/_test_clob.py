import asyncio
import httpx
import pandas as pd
from datetime import datetime, timezone

async def test():
    # Get a sample market from train dataset
    train = pd.read_parquet("data/processed/merged_training_features.parquet")
    sample_market = train.iloc[0]
    market_id = sample_market["market_id"]
    signal_ts = sample_market["signal_timestamp_ms"]
    
    print(f"Testing market_id: {market_id}")
    print(f"Target signal_timestamp: {datetime.fromtimestamp(signal_ts/1000, tz=timezone.utc)}")

    async with httpx.AsyncClient() as client:
        # Get market details to find yes_token_id
        # Polymarket uses condition_id for market endpoint
        resp = await client.get(f"https://gamma-api.polymarket.com/markets/{market_id}")
        if resp.status_code != 200:
            print("Failed to get market by ID, trying condition_id...")
            resp = await client.get(f"https://gamma-api.polymarket.com/markets?condition_id={market_id}")
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                m = data[0]
            else:
                print("Market not found")
                return
        else:
            m = resp.json()
            
        clob_ids = m.get("clobTokenIds", [])
        if clob_ids and len(clob_ids) >= 2:
            yes_token = clob_ids[0]
        else:
            print("Token ID not found")
            return
            
        print(f"Found yes_token_id: {yes_token}")
        
        # Test clob history endpoint
        resp = await client.get(
            "https://clob.polymarket.com/prices-history",
            params={"tokenID": yes_token, "interval": "1d", "fidelity": 10}
        )
        print(f"History (fidelity=10) status: {resp.status_code}")
        if resp.status_code == 200:
            hist = resp.json().get("history", [])
            print(f"Points returned: {len(hist)}")
            if hist:
                print(f"Sample data point: {hist[0]}")
                # Convert t to datetime
                dt = datetime.fromtimestamp(hist[0]['t'], tz=timezone.utc)
                print(f"Timestamp parsed: {dt}")

if __name__ == "__main__":
    asyncio.run(test())
