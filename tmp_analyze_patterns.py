import json
import re

with open("bitcoin_events_fixed.json", "r", encoding="utf-8") as f:
    events = json.load(f)

print(f"Total events: {len(events)}")

found_count = 0
for event in events:
    print(f"Event: {event.get('title')} | Vol24h: {event.get('volume24hr')}")
    markets = event.get("markets", [])
    for m in markets:
        q = m.get("question", "").lower()
        active = m.get("active")
        closed = m.get("closed")
        eob = m.get("enableOrderBook")
        vol24 = float(m.get("volume24hr") or 0.0)
        
        matches_pattern = ("bitcoin" in q or "btc" in q) and any(p in q for p in ["above", "up", "down", "hit"])
        
        if matches_pattern:
            found_count += 1
            print(f"  [MATCH] {m.get('question')} | Active: {active} | Closed: {closed} | EOB: {eob} | Vol24h: {vol24}")
        else:
            # print(f"  [SKIP ] {m.get('question')} | Pattern fail")
            pass

print(f"\nTotal matches found: {found_count}")
