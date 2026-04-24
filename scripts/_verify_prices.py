import httpx

mids = ["1846005", "1831366"]
for mid in mids:
    r = httpx.get(f"https://gamma-api.polymarket.com/markets/{mid}")
    m = r.json()
    print(mid, m.get("question"))
    print(f"Prices: {m.get('outcomePrices')}")
    print(f"Outcomes: {m.get('outcomes')}")
