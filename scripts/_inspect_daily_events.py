import httpx
import json

def survey_target_events():
    slugs = [
        "bitcoin-up-or-down-on-april-5-2026",
        "bitcoin-up-or-down-on-april-4-2026"
    ]
    
    for slug in slugs:
        print(f"=== INSPECTING EVENT: {slug} ===")
        r = httpx.get(f"https://gamma-api.polymarket.com/events", params={"slug": slug})
        if r.status_code == 200:
            events = r.json()
            if events:
                e = events[0]
                print(f"Title: {e.get('title')}")
                print(f"End Date: {e.get('endDate')}")
                markets = e.get("markets", [])
                print(f"Number of sub-markets: {len(markets)}")
                for m in markets[:3]:
                    mq = m.get('question')
                    m_id = m.get('id')
                    mv = m.get('volume')
                    print(f"  - Q: {mq} | id={m_id} | vol=${mv}")
            else:
                print("Event not found (maybe slug is different in API?)")
        else:
            print(f"Error {r.status_code}")
        print("-" * 40)

if __name__ == "__main__":
    survey_target_events()
