import httpx
import pandas as pd
import numpy as np
from datetime import datetime

def parse_pyth_binary_pnau_v2(hex_data: str, feed_id: str = "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43"):
    """
    Decodes Pyth PNAU binary for a specific feed_id.
    PNAU structures in Hermes V2 often wrap the PriceFeed message.
    """
    try:
        h = hex_data.replace("0x", "")
        # Find the internal PriceFeed message start using the Feed ID
        idx = h.find(feed_id)
        if idx == -1:
            return None
            
        # Message format (offset from Type byte):
        # [Type: 1b] [FeedID: 32b] [Price: 8b] [Conf: 8b] [Expo: 4b] [PublishTime: 8b]
        # idx is starts at FeedID. 
        # Price starts at idx + 64 hex characters (32 bytes after FeedID start)
        price_hex = h[idx + 64 : idx + 80]
        expo_hex = h[idx + 96 : idx + 104]
        
        price_raw = int(price_hex, 16)
        # Handle signed int64
        if price_raw > 0x7FFFFFFFFFFFFFFF:
            price_raw -= 0x10000000000000000
            
        exponent = int(expo_hex, 16)
        # Handle signed int32
        if exponent > 0x7FFFFFFF:
            exponent -= 0x100000000
            
        return price_raw * (10 ** exponent)
    except Exception as e:
        print(f"Decoder Error: {e}")
        return None

def test_decoder_on_january():
    # 3 Sample Timestamps (Jan 8-9 2026)
    # 1. 1767938400 (Jan 9, 1:00 AM ET)
    # 2. 1767938100 (Jan 9, 12:55 AM ET)
    # 3. 1767978000 (Jan 9, 12:00 PM ET)
    samples = [1767938400, 1767938100, 1767978000]
    
    # Load OHLCV for verification (since January aggTrades are not local)
    print("Loading Binance OHLCV 15m for January...")
    ohlcv = pd.read_parquet("data/raw/ohlcv_15m.parquet")
    
    results = []
    for ts in samples:
        ts_ms = ts * 1000
        print(f"\nProcessing TS {ts} ({datetime.fromtimestamp(ts)})...")
        
        # 1. Fetch Pyth Binary
        r = httpx.get(f"https://hermes.pyth.network/v2/updates/price/{ts}", 
                      params={"ids[]": "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43"})
        if r.status_code != 200:
            print(f"  Pyth 429/Error: {r.status_code}")
            continue
            
        binary_data = r.json().get("binary", {}).get("data", [None])[0]
        if not binary_data:
            print("  No binary data found.")
            continue
            
        # 2. Decode
        pyth_price = parse_pyth_binary_pnau_v2(binary_data)
        
        # 3. Get Binance Reference (Find the 15m candle containing this ts)
        # Candle covers [open_time, close_time]
        matches = ohlcv[(ohlcv.open_time <= ts_ms) & (ohlcv.close_time >= ts_ms)]
        
        if len(matches) > 0:
            row = matches.iloc[0]
            b_low = float(row['low'])
            b_high = float(row['high'])
            b_close = float(row['close'])
            
            # 4. Compare
            # Range check: Pyth should be within the 15m High/Low
            in_range = b_low <= pyth_price <= b_high
            diff_pct = abs(pyth_price - b_close) / b_close * 100
            # Even if slightly out of range due to 15m vs 1m, diff should be tiny
            match = in_range or (diff_pct < 0.05)
            
            results.append({
                "ts": ts,
                "pyth": round(pyth_price, 2),
                "b_range": f"[{b_low:.2f}, {b_high:.2f}]",
                "diff_pct": f"{diff_pct:.4f}%",
                "match": match
            })
            print(f"  Pyth: {pyth_price:.2f} | Binance 15m Range: [{b_low:.2f}, {b_high:.2f}]")
            print(f"  Match: {match} (Diff vs Close: {diff_pct:.4f}%)")
        else:
            print("  No Binance OHLCV candle found for this window.")

    print("\n" + "="*40)
    print("DECODER VALIDATION SUMMARY")
    print("="*40)
    print(pd.DataFrame(results))

if __name__ == "__main__":
    test_decoder_on_january()
