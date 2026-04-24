import httpx
import json

url = "https://gamma-api.polymarket.com/events"
params = {
    "active": "true",
    "closed": "false",
    "archived": "false",
    "tag_slug": "bitcoin",
    "limit": 5
}

with httpx.Client(timeout=15.0) as client:
    try:
        resp = client.get(url, params=params)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(json.dumps(data, indent=2))
        else:
            print(f"Error Body: {resp.text}")
    except Exception as e:
        print(f"Exception: {e}")
