"""Search for CLOSED BTC EVENTS to find Daily patterns."""
import httpx, json
from datetime import datetime, timezone

def survey_closed_events():
    print("=== SEARCHING CLOSED BTC EVENTS ===\n")
    
    # Events often contain the recurring markets
    r = httpx.get("https://gamma-api.polymarket.com/events", params={
        "closed": "true",
        "limit": 100,
        "order": "endDate",
        "ascending": "false"
    })
    
    if r.status_code != 200:
        print(f"Error: {r.status_code}")
        return

    all_events = r.json()
    btc_events = [e for e in all_events if "bitcoin" in str(e).lower()]
    
    print(f"Found {len(btc_events)} BTC events\n")
    for e in btc_events[:20]:
        t = e.get("title", "")
        slug = e.get("slug", "")
        end = e.get("endDate", "")
        print(f"  Title: {t}")
        print(f"    slug={slug} | end={end}")
        
        # Check sub-markets if any
        markets = e.get("markets", [])
        if markets:
            print(f"    Markets: {len(markets)}")
            for m in markets[:2]:
                print(f"      - {m.get('question','')[:60]} (vol=${m.get('volume',0)})")
        print()

if __name__ == "__main__":
    survey_closed_events()
