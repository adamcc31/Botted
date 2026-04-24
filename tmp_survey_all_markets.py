import httpx
import json

url = "https://gamma-api.polymarket.com/markets"
params = {
    "active": "true",
    "closed": "false",
    "limit": 100
}

with httpx.Client(timeout=15.0) as client:
    try:
        resp = client.get(url, params=params)
        if resp.status_code == 200:
            markets = resp.json()
            # print(f"Found {len(markets)} active markets")
            with open("active_markets_raw.json", "w", encoding="utf-8") as f:
                json.dump(markets, f, indent=2)
            
            btc_candidates = []
            for m in markets:
                q = m.get("question", "").lower()
                if "bitcoin" in q or "btc" in q:
                    btc_candidates.append(m)
            
            print(f"Total BTC candidates in top 100: {len(btc_candidates)}")
            for b in btc_candidates[:10]:
                print(f"  - {b.get('question')} | Vol: {b.get('volume')} | EOB: {b.get('enableOrderBook')}")
        else:
            print(f"Error Code: {resp.status_code}")
    except Exception as e:
        print(f"Exception: {e}")
