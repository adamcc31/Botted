"""Survey Daily BTC market structure on Polymarket."""
import httpx, json
from datetime import datetime, timezone

def survey_daily():
    print("=== SEARCHING DAILY BTC MARKETS ===\n")
    
    # Search for active markets with 'Bitcoin' and 'Price' or 'above'
    r = httpx.get("https://gamma-api.polymarket.com/markets", params={
        "closed": "false",
        "limit": 100
    })
    
    all_markets = r.json()
    
    # Common daily patterns: "Price of Bitcoin on April 7?", "Bitcoin above $70,000 on April 7?"
    daily_candidates = [m for m in all_markets if "bitcoin" in m.get("question","").lower()]
    
    print(f"Found {len(daily_candidates)} active BTC candidates\n")
    for m in daily_candidates[:10]:
        q = m.get("question","")
        vol = m.get("volume", 0)
        liq = m.get("liquidity", 0)
        end = m.get("endDate", "")
        slug = m.get("slug", "")
        print(f"  Q: {q}")
        print(f"    slug={slug} | vol=${vol} | liq=${liq} | end={end}")
        print()

    # Search closed daily markets to see history
    print("\n=== SEARCHING CLOSED DAILY BTC MARKETS ===\n")
    r_closed = httpx.get("https://gamma-api.polymarket.com/markets", params={
        "closed": "true",
        "limit": 100,
        "order": "endDate",
        "ascending": "false"
    })
    
    closed_markets = r_closed.json()
    closed_daily = [m for m in closed_markets if "bitcoin" in m.get("question","").lower() and ("above" in m.get("question","").lower() or "price" in m.get("question","").lower())]
    
    print(f"Found {len(closed_daily)} closed Daily/Price BTC markets\n")
    for m in closed_daily[:10]:
        q = m.get("question","")
        vol = m.get("volume", 0)
        end = m.get("endDate", "")
        slug = m.get("slug", "")
        print(f"  Q: {q}")
        print(f"    slug={slug} | vol=${vol} | end={end}")

if __name__ == "__main__":
    survey_daily()
