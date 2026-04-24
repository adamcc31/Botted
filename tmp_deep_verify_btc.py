import httpx
import json
from datetime import datetime, timezone

url = "https://gamma-api.polymarket.com/markets"
params = {
    "active": "true",
    "closed": "false",
    "limit": 200,  # Look deeper
    "order": "volume24hr",
    "ascending": "false"
}

print(f"--- RESEARCHING ACTIVE BTC PRICE-ACTION MARKETS (UTC: {datetime.now(timezone.utc)}) ---")

with httpx.Client(timeout=15.0) as client:
    try:
        resp = client.get(url, params=params)
        if resp.status_code == 200:
            markets = resp.json()
            btc_count = 0
            for m in markets:
                q = m.get("question", "")
                q_low = q.lower()
                
                # Check for Bitcoin + Price Action markers
                is_btc = "bitcoin" in q_low or "btc" in q_low
                is_price_action = any(p in q_low for p in ["above", "up", "down", "below"])
                
                if is_btc:
                    btc_count += 1
                    status = "[PA MATCH]" if is_price_action else "[NON-PA]"
                    print(f"{status} {q}")
                    print(f"    Vol24h: {m.get('volume24hr')} | EOB: {m.get('enableOrderBook')} | End: {m.get('endDate')}")
            
            print(f"\nTotal BTC-related markets in top 200: {btc_count}")
        else:
            print(f"API Error: {resp.status_code}")
    except Exception as e:
        print(f"Error: {e}")
