import httpx
import json

def fetch_market_detailed():
    mid = "1846005"
    r = httpx.get(f"https://gamma-api.polymarket.com/markets/{mid}")
    market = r.json()
    print(f"Outcome field: {market.get('outcome')}")
    print(f"All keys: {market.keys()}")
    for k, v in market.items():
        if "winner" in k.lower() or "result" in k.lower() or "outcome" in k.lower():
            print(f"{k}: {v}")

if __name__ == "__main__":
    fetch_market_detailed()
