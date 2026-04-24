import httpx
import pandas as pd
from datetime import datetime, timezone

def test_daily_lag():
    # Long-term BTC market: Will bitcoin hit $1m before GTA VI?
    # ConditionID: 0xbb57ccf5853a85487bc3d83d04d669310d28c6c810758953b9d9b91d1aee89d2
    target_id = "0xbb57ccf5853a85487bc3d83d04d669310d28c6c810758953b9d9b91d1aee89d2"
    
    with httpx.Client() as client:
        # 1. Fetch CLOB Price History (1m fidelity)
        r = client.get(
            "https://clob.polymarket.com/prices-history",
            params={"market": target_id, "fidelity": 1}
        )
        if r.status_code != 200:
            print(f"Failed to fetch CLOB history: {r.status_code}")
            return
            
        history = r.json().get('history', [])
        if not history:
            print("No history found.")
            return
            
        df_poly = pd.DataFrame(history)
        df_poly['datetime'] = pd.to_datetime(df_poly['t'], unit='s', utc=True)
        df_poly = df_poly.rename(columns={'p': 'poly_price'})
        
        # 2. Sync with Binance (OHLCV 1m)
        # We'll use the local ohlcv file for simplicity if it covers the range
        ohlcv = pd.read_parquet('data/raw/ohlcv_15m.parquet')
        ohlcv['datetime'] = pd.to_datetime(ohlcv['timestamp'], unit='ms', utc=True)
        
        # Merge on 1-min bins
        merged = pd.merge_asof(
            df_poly.sort_values('datetime'),
            ohlcv.sort_values('datetime'),
            on='datetime',
            direction='backward'
        )
        
        # 3. Calculate Cross-Correlation Lag
        # Since these are 1-min buckets, we check if poly(t) correlates better with binance(t-1)
        merged['binance_ret'] = merged['close'].pct_change()
        merged['poly_ret'] = merged['poly_price'].pct_change()
        
        print(f"\n📊 DAILY MARKET LAG PROBE: {target_id}")
        print("-" * 50)
        for shift in range(0, 6):
            corr = merged['binance_ret'].shift(shift).corr(merged['poly_ret'])
            print(f"Correlation (Binance Lag {shift}m): {corr:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    test_daily_lag()
