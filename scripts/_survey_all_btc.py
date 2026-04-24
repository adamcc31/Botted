"""Search for ANY BTC related markets (closed)."""
import httpx, json
from datetime import datetime, timezone

def survey_all_btc_closed():
    print("=== SEARCHING CLOSED BTC MARKETS ===\n")
    
    # Try multiple searches
    queries = ["bitcoin", "btc"]
    all_found = []
    
    for q in queries:
        r = httpx.get("https://gamma-api.polymarket.com/markets", params={
            "closed": "true",
            "search": q,
            "limit": 100,
            "order": "endDate",
            "ascending": "false"
        })
        if r.status_code == 200:
            all_found.extend(r.json())
    
    print(f"Total matching results: {len(all_found)}\n")
    
    # Sort by endDate desc
    unique = {m['id']: m for m in all_found}.values()
    sorted_m = sorted(unique, key=lambda x: x.get('endDate', ''), reverse=True)
    
    for m in sorted_m[:20]:
        q = m.get("question","")
        vol = m.get("volume", 0)
        end = m.get("endDate", "")
        slug = m.get("slug", "")
        print(f"  Q: {q}")
        print(f"    slug={slug} | vol=${vol} | end={end}")
        print()

if __name__ == "__main__":
    survey_all_btc_closed()
