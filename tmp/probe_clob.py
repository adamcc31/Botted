import httpx
import pandas as pd
from datetime import datetime
import sys

def get_test_market(df, current_ts_ms, days_ago):
    target_ts = current_ts_ms - (days_ago * 24 * 3600 * 1000)
    # Find market closest to target_ts
    diffs = (df['t_resolution_epoch_ms'] - target_ts).abs()
    market = df.loc[diffs.idxmin()]
    return market

def check_history():
    df = pd.read_parquet('data/raw/polymarket_markets.parquet')
    df = df[df['t_resolution_epoch_ms'].notna()]
    
    current_time_ms = int(datetime(2026, 4, 7).timestamp() * 1000)
    
    markets_to_test = [
        (3, get_test_market(df, current_time_ms, 3)),
        (10, get_test_market(df, current_time_ms, 10)),
        (20, get_test_market(df, current_time_ms, 20))
    ]
    
    with httpx.Client() as client:
        for days, market in markets_to_test:
            # 1. Get Condition ID from Gamma API
            gamma_id = market['market_id']
            try:
                gamma_resp = client.get(f"https://gamma-api.polymarket.com/markets/{gamma_id}")
                if gamma_resp.status_code != 200:
                    print(f"[{days} days ago] Gamma API failed for {gamma_id}")
                    continue
                condition_id = gamma_resp.json().get('conditionId')
                
                # 2. Check CLOB Prices History
                r = client.get(
                    "https://clob.polymarket.com/prices-history",
                    params={"market": condition_id, "fidelity": 10} # using 10 min fidelity
                )
                print(f"[{days} days ago] Market {gamma_id} resolved on {market['t_resolution_utc']}")
                print(f" - Status: {r.status_code}")
                
                data_points = len(r.json().get('history', []))
                print(f" - Data points: {data_points}")
                print("-" * 50)
            except Exception as e:
                print(f"[{days} days ago] Error: {e}")

if __name__ == "__main__":
    check_history()
