import httpx
import pandas as pd
from datetime import datetime, timezone

def analyze_daily_market_lag():
    # YES Token for 'Bitcoin above 69,200 on April 6'
    # Resolved market
    token_id = "31480227867141000748333313157669522563453197920987842824481193367398025467986"
    
    with httpx.Client() as client:
        print(f"Fetching history for token: {token_id}...")
        r = client.get(
            "https://clob.polymarket.com/prices-history",
            params={
                "market": token_id, 
                "interval": "all",
                "fidelity": 1
            }
        )
        
        if r.status_code != 200:
            print(f"Failed to fetch history: {r.status_code} - {r.text}")
            return
            
        history = r.json().get('history', [])
        if not history:
            print("No history returned for this token.")
            return
            
        df_poly = pd.DataFrame(history)
        df_poly['datetime'] = pd.to_datetime(df_poly['t'], unit='s', utc=True)
        df_poly = df_poly.rename(columns={'p': 'poly_price'})
        
        # Filter for April 6 ONLY (when the market was active)
        df_poly = df_poly[df_poly['datetime'] >= '2026-04-06T00:00:00Z']
        
        if len(df_poly) < 10:
            print("Too few data points for analysis.")
            return
            
        # Match with Binance OHLCV
        binance = pd.read_parquet('data/raw/ohlcv_15m.parquet')
        binance['datetime'] = pd.to_datetime(binance['open_time'], unit='ms', utc=True)
        
        merged = pd.merge_asof(
            df_poly.sort_values('datetime'),
            binance.sort_values('datetime'),
            on='datetime',
            direction='backward'
        )
        
        # Calculate Correlation vs Lags (Minutes)
        merged['binance_ret'] = merged['close'].pct_change()
        merged['poly_ret'] = merged['poly_price'].pct_change()
        
        print("\n📊 DAILY MARKET CROSS-CORRELATION (1-min bins)")
        print("-" * 50)
        for shift in range(0, 5):
            corr = merged['binance_ret'].shift(shift).corr(merged['poly_ret'])
            print(f"Lag {shift} min correlation: {corr:.4f}")
        print("-" * 50)
        
        # Heuristic: If lag 0 is highest, HFT is also here.
        # If lag 1 or 2 is highest, we have actionable retail lag.
        
if __name__ == "__main__":
    analyze_daily_market_lag()
