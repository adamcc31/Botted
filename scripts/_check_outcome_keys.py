import httpx

def check_outcome_keys():
    mid = "1846005"
    r = httpx.get(f"https://gamma-api.polymarket.com/markets", params={"id": mid})
    market = r.json()[0]
    print("Found keys in market object:")
    print(market.keys())
    for k in ["outcome", "winner_index", "status", "closed", "resolution_source", "options", "outcomes"]:
        print(f"{k}: {market.get(k)}")

if __name__ == "__main__":
    check_outcome_keys()
