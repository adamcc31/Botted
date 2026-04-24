import httpx
import sys

def search_events():
    r = httpx.get("https://gamma-api.polymarket.com/events", params={
        "search": "bitcoin",
        "limit": 50,
        "closed": "true"
    })
    events = r.json()
    for e in events:
        print(f"Title: {e.get('title')} | Slug: {e.get('slug')}")

if __name__ == "__main__":
    search_events()
