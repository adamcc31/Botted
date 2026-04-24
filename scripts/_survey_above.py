"""Survey target BTC markets for pivot."""
import httpx, json

def survey_target():
    print("=== SEARCHING TARGET BTC MARKETS (Above/Below) ===\n")
    
    # Try searching for "Bitcoin above"
    r = httpx.get("https://gamma-api.polymarket.com/markets", params={
        "closed": "true",
        "search": "Bitcoin above",
        "limit": 100,
        "order": "endDate",
        "ascending": "false"
    })
    
    if r.status_code == 200:
        found = r.json()
        print(f"Results for 'Bitcoin above': {len(found)}")
        for m in found[:10]:
            print(f"  Q: {m.get('question','')[:80]}")
            print(f"    slug={m.get('slug','')[:40]} | vol=${m.get('volume',0)} | end={m.get('endDate', '')}")
            
if __name__ == "__main__":
    survey_target()
