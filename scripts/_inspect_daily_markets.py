import httpx
import json

def survey_target_markets():
    # Market id for April 5: 1846005
    # Market id for April 4: 1831366
    m_ids = ["1846005", "1831366"]
    
    for mid in m_ids:
        print(f"=== INSPECTING MARKET ID: {mid} ===")
        r = httpx.get(f"https://gamma-api.polymarket.com/markets", params={"id": mid})
        if r.status_code == 200:
            markets = r.json()
            if markets:
                m = markets[0]
                print(f"Q: {m.get('question')}")
                print(f"Status: Resolved? {m.get('closed')}")
                print(f"Outcomes: {m.get('outcomes')}")
                print(f"Group ID: {m.get('group_id')}")
                print(f"Outcome (Resolved): {m.get('outcome')}")
                # Check for CLOB tokens
                clob_tokens = m.get('clobTokenIds', [])
                if isinstance(clob_tokens, str):
                    import json
                    clob_tokens = json.loads(clob_tokens)
                print(f"CLOB Tokens: {clob_tokens}")
            else:
                print("Market not found")
        print("-" * 40)

if __name__ == "__main__":
    survey_target_markets()
