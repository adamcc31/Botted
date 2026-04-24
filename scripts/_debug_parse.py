import httpx
import sys
from pathlib import Path
import os

# Add root to sys.path to import scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.collect_polymarket import _parse_resolved_market, MARKET_SEARCH_TERMS

def debug_parse():
    print("=== DEBUGGING PARSE ===")
    r = httpx.get("https://gamma-api.polymarket.com/markets", params={
        "closed": "true",
        "search": "bitcoin",
        "limit": 10,
        "order": "endDate",
        "ascending": "false"
    })
    
    if r.status_code != 200:
        print(f"Error: {r.status_code}")
        return
        
    markets = r.json()
    print(f"Total markets from API: {len(markets)}")
    
    for m in markets:
        q = m.get("question", "")
        print(f"\nQuestion: {q}")
        
        # 1. BTC Filter check
        is_btc = any(term in q.lower() for term in MARKET_SEARCH_TERMS)
        print(f"  Is BTC market (search terms): {is_btc}")
        
        # 2. Parse check
        parsed = _parse_resolved_market(m)
        if parsed:
            print(f"  ✅ PARSED: strike={parsed.get('strike_price')}, outcome={parsed.get('outcome_binary')}")
        else:
            print("  ❌ FAILED TO PARSE (likely duration or strike extraction)")

if __name__ == "__main__":
    debug_parse()
