import httpx
import json
import pandas as pd
from pathlib import Path

def test_gamma_markets():
    ids = [1131900, 1131898, 966270, 966280, 1073840]
    results = {}
    
    with httpx.Client(timeout=30.0) as client:
        for mid in ids:
            try:
                r = client.get(f"https://gamma-api.polymarket.com/markets/{mid}")
                r.raise_for_status()
                data = r.json()
                
                # Check for settlement info
                settlement = {
                    "question": data.get("question"),
                    "resolution_price": data.get("resolutionPrice"),
                    "resolver_price": data.get("resolverPrice"),
                    "description": data.get("description"),
                    "rules": data.get("rules"),
                    "outcome": data.get("outcome")
                }
                results[mid] = settlement
                print(f"--- Market ID: {mid} ---")
                print(json.dumps(data, indent=2))
                print("\n" + "="*50 + "\n")
            except Exception as e:
                print(f"Error for {mid}: {e}")

if __name__ == "__main__":
    test_gamma_markets()
