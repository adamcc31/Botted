import httpx
import json

def diagnostic():
    ts = 1767938400  # Jan 9
    url = f"https://hermes.pyth.network/v2/updates/price/{ts}"
    params = {
        "ids[]": "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
        "parsed": "true"
    }
    
    r = httpx.get(url, params=params)
    print(f"Status: {r.status_code}")
    data = r.json()
    print("Top-level keys:", list(data.keys()))
    
    if "parsed" in data:
        print("Parsed exists. Count:", len(data["parsed"]))
        if len(data["parsed"]) > 0:
            print("First parsed element keys:", list(data["parsed"][0].keys()))
            if "price" in data["parsed"][0]:
                 print("Price object:", data["parsed"][0]["price"])
    else:
        print("Parsed is MISSING from top level.")
        # Check if it's nested or if there's an error message
        print("Full JSON snippet:", str(data)[:1000])

if __name__ == "__main__":
    diagnostic()
