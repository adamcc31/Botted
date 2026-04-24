"""Search high volume closed BTC markets."""
import httpx, json

def survey_high_vol():
    print("=== SEARCHING HIGH VOLUME CLOSED BTC MARKETS ===\n")
    
    # Polymarket API returns "volume" as a float or string sometimes.
    r = httpx.get("https://gamma-api.polymarket.com/markets", params={
        "closed": "true",
        "limit": 100,
        "order": "volume",
        "ascending": "false",
        "search": "Bitcoin"
    })
    
    if r.status_code == 200:
        found = r.json()
        print(f"Results: {len(found)}")
        for m in found[:50]:
            try:
                v = float(m.get('volume', 0) or 0)
                q = m.get('question', '')
                if "bitcoin" not in q.lower() and "btc" not in q.lower():
                    continue
                print(f"  Q: {q[:100]}")
                print(f"    slug={m.get('slug','')[:50]} | vol=${v:,.0f} | end={m.get('endDate','')[:10]}")
            except:
                pass
            
if __name__ == "__main__":
    survey_high_vol()
