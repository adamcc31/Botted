"""Find BTC Up or Down markets — they use a programmatic slug pattern."""
import httpx, json
from datetime import datetime, timezone

# The URL pattern from user screenshot: btc-updown-15m-1775468700
# 1775468700 is likely an epoch timestamp: 2026-04-06 09:45:00 UTC
print("Decoding epoch from slug: 1775468700")
dt = datetime.fromtimestamp(1775468700, tz=timezone.utc)
print(f"  = {dt} (market start time?)\n")

# Try searching by slug pattern
print("=== Searching by slug pattern ===\n")
patterns = [
    "btc-updown-15m",
    "btc-updown",
    "bitcoin-up-or-down",
]

for pat in patterns:
    try:
        r = httpx.get(f"https://gamma-api.polymarket.com/events?slug={pat}", timeout=10)
        print(f"  slug={pat}: status={r.status_code}, results={len(r.json()) if r.status_code == 200 else 'N/A'}")
        if r.status_code == 200 and r.json():
            for e in r.json()[:2]:
                print(f"    title: {e.get('title','')[:60]}")
    except Exception as e:
        print(f"  slug={pat}: error={e}")

# Try direct event lookup
print("\n=== Direct event lookup ===\n")
try:
    r2 = httpx.get("https://gamma-api.polymarket.com/events?slug=btc-updown-15m-1775468700", timeout=10)
    print(f"Direct lookup status: {r2.status_code}")
    if r2.status_code == 200 and r2.json():
        event = r2.json()[0] if isinstance(r2.json(), list) else r2.json()
        print(json.dumps({k: v for k, v in event.items() if k in ['title','slug','endDate','startDate','id']}, indent=2))
        markets_in_event = event.get("markets", [])
        print(f"\n  Markets in event: {len(markets_in_event)}")
        for m in markets_in_event[:3]:
            q = m.get("question","")[:60]
            vol = m.get("volume", 0)
            print(f"    {q} | vol=${vol}")
except Exception as e:
    print(f"Direct lookup error: {e}")

# Search for any 15-minute or "up or down" text
print("\n=== Tag-based search ===\n")
try:
    for tag in ["15m", "up-or-down", "btc-updown"]:
        r3 = httpx.get(f"https://gamma-api.polymarket.com/events", params={"tag": tag, "limit": 5}, timeout=10)
        if r3.status_code == 200 and r3.json():
            print(f"  tag={tag}: {len(r3.json())} events")
            for e in r3.json()[:2]:
                print(f"    {e.get('title','')[:60]} | slug={e.get('slug','')[:40]}")
        else:
            print(f"  tag={tag}: {r3.status_code}, {len(r3.json()) if r3.status_code==200 else 'error'}")
except Exception as e:
    print(f"Tag search error: {e}")

# The 15m markets may be fetched by text search
print("\n=== Text search ===\n")
try:
    r4 = httpx.get("https://gamma-api.polymarket.com/markets", params={
        "closed": "true", 
        "limit": 30, 
        "order": "endDate", 
        "ascending": "false",
        "tag": "crypto"
    })
    markets = r4.json()
    updown = [m for m in markets if "15" in m.get("question","").lower() or "up or down" in m.get("question","").lower()]
    print(f"Found {len(updown)} potential 15m/updown markets")
    for m in updown[:5]:
        print(f"  {m.get('question','')[:70]} | vol={m.get('volume',0)}")
except Exception as e:
    print(f"Text search error: {e}")

# Try the polymarket CLOB API directly for BTC events
print("\n=== CLOB Events API ===\n")
try:
    r5 = httpx.get("https://clob.polymarket.com/sampling-markets", timeout=10)
    if r5.status_code == 200:
        data = r5.json()
        print(f"  sampling-markets type: {type(data)}")
        if isinstance(data, list):
            btc_sm = [d for d in data if "bitcoin" in str(d).lower() or "btc" in str(d).lower()]
            print(f"  BTC-related: {len(btc_sm)}")
            for d in btc_sm[:3]:
                print(f"    {json.dumps(d)[:200]}")
    else:
        print(f"  status: {r5.status_code}")
except Exception as e:
    print(f"  CLOB API error: {e}")

# Check the STRK API endpoint for rewards markets
print("\n=== Rewards Markets ===\n")
try:
    r6 = httpx.get("https://gamma-api.polymarket.com/markets", params={
        "closed": "false",
        "limit": 100, 
    })
    all_active = r6.json()
    btc_active = [m for m in all_active if "bitcoin" in m.get("question","").lower()]
    crypto = [m for m in all_active if "crypto" in str(m.get("tags","")).lower() or m.get("question","").lower().startswith("bitcoin")]
    print(f"Total active markets: {len(all_active)}")
    print(f"BTC-related: {len(btc_active)}")
    print(f"Crypto-tagged: {len(crypto)}")
    for c in btc_active[:5]:
        q = c.get("question","")[:70]
        slug = c.get("slug","")[:40]
        print(f"  {q}\n    slug={slug}")
except Exception as e:
    print(f"Error: {e}")
