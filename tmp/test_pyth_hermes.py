import httpx
import json
import pandas as pd
from datetime import datetime
import time

def fetch_pyth_price(timestamp_sec):
    # Pyth V2 Hermes Endpoint: https://hermes.pyth.network/v2/updates/price/{timestamp}
    # ids[]: BTC/USD 0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43
    url = f"https://hermes.pyth.network/v2/updates/price/{timestamp_sec}"
    params = {
        "ids[]": "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43"
    }
    
    try:
        r = httpx.get(url, params=params, timeout=10.0)
        r.raise_for_status()
        data = r.json()
        
        # Parse price from parsed[0]
        parsed = data.get("parsed", [])
        if not parsed:
            return None
            
        p_obj = parsed[0].get("price", {})
        price = float(p_obj.get("price", 0))
        expo = int(p_obj.get("expo", 0))
        
        return price * (10**expo)
    except Exception as e:
        print(f"Pyth API Error for {timestamp_sec}: {e}")
        return None

def test_pyth_alignment():
    # Test cases: [market_id, start_ts_ms, res_ts_ms, outcome_binary]
    # Dates: Jan 9 2026
    test_cases = [
        ["1133976", 1767938100000, 1767938400000, 1], # 12:55 -> 1:00 (Outcome 1)
        ["1133984", 1767938400000, 1767938700000, 1], # 1:00 -> 1:05 (Outcome 1)
        ["1134039", 1767938700000, 1767939000000, 0], # 1:05 -> 1:10 (Outcome 0)
        ["1134045", 1767939000000, 1767939300000, 0], # 1:10 -> 1:15 (Outcome 0)
        ["1133981", 1767938400000, 1767939300000, 0], # 1:00 -> 1:15 (Outcome 0)
    ]
    
    results = []
    for mid, start_ms, res_ms, target in test_cases:
        print(f"\nEvaluating Market {mid}...")
        
        # Get start price (Strike)
        strike = fetch_pyth_price(start_ms // 1000)
        # Get res price (Final)
        final = fetch_pyth_price(res_ms // 1000)
        
        if strike and final:
            computed = 1 if final > strike else 0
            match = (computed == target)
            diff = final - strike
            diff_pct = (diff / strike) * 100
            
            results.append({
                "id": mid,
                "strike": round(strike, 2),
                "final": round(final, 2),
                "diff": round(diff, 2),
                "diff_pct": round(diff_pct, 4),
                "target": target,
                "computed": computed,
                "match": match
            })
            print(f"  Strike: {strike:.2f} | Final: {final:.2f}")
            print(f"  Result: {computed} | Target: {target} | Match: {match}")
        else:
            print(f"  Failed to fetch Pyth prices for {mid}")
            
    print("\n" + "="*30)
    print("FINAL PYTH ALIGNMENT RESULTS")
    print("="*30)
    matches = sum(1 for r in results if r["match"])
    print(f"Accuracy: {matches}/{len(results)} ({matches/len(results):.0%})")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    test_pyth_alignment()
