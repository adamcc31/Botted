import httpx
import json

url = "https://gamma-api.polymarket.com/events"
params = {
    "active": "true",
    "closed": "false",
    "archived": "false",
    "tag_slug": "bitcoin",
    "limit": 10
}

with httpx.Client(timeout=15.0) as client:
    try:
        resp = client.get(url, params=params)
        if resp.status_code == 200:
            data = resp.json()
            with open("bitcoin_events_fixed.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print("Successfully saved to bitcoin_events_fixed.json")
        else:
            print(f"Error Code: {resp.status_code}")
    except Exception as e:
        print(f"Exception: {e}")
