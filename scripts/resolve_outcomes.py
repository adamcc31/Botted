import pandas as pd
import requests
import json
import os
import sys
import time
from datetime import datetime

# Disable insecure request warnings since we might use verify=False if certificate fails on local
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

GAMMA_API = "https://gamma-api.polymarket.com/markets"

def fetch_gamma_batch(market_ids, status="closed"):
    """Fetch market metadata from Gamma API in batches of 20."""
    # Build query params with multiple condition_ids
    params = [("condition_ids", mid) for mid in market_ids]
    params.append((status, "true"))
    
    for attempt in range(3):
        try:
            resp = requests.get(GAMMA_API, params=params, timeout=15, verify=False)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"Warning: Gamma API returned status {resp.status_code} for batch")
                time.sleep(1)
        except Exception as e:
            print(f"Error fetching batch: {e}")
            time.sleep(1)
    return []

def resolve_outcomes(input_path, output_path):
    print("=" * 60)
    print("POLYMARKET SHADOW OUTCOME RESOLVER")
    print("=" * 60)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found!")
        return
        
    df = pd.read_csv(input_path)
    total_rows = len(df)
    print(f"Total rows in shadow CSV: {total_rows:,}")
    
    unique_mids = df["market_id"].dropna().unique().tolist()
    print(f"Unique market IDs to resolve: {len(unique_mids):,}")
    
    # Query Gamma API in batches of 20
    market_resolutions = {}
    batch_size = 20
    
    # 1. Query closed markets
    print("Querying closed markets on Gamma API...")
    for i in range(0, len(unique_mids), batch_size):
        batch = unique_mids[i : i + batch_size]
        data = fetch_gamma_batch(batch, status="closed")
        for m in data:
            cid = m.get("conditionId")
            outcome_prices_str = m.get("outcomePrices")
            if cid and outcome_prices_str:
                try:
                    outcome_prices = json.loads(outcome_prices_str)
                    # outcomes = ["Up", "Down"]
                    # If outcomePrices is ["1", "0"], Up/YES won -> WIN
                    # If outcomePrices is ["0", "1"], Down/NO won -> LOSE
                    p_yes = float(outcome_prices[0])
                    p_no = float(outcome_prices[1])
                    if p_yes > 0.9:
                        market_resolutions[cid] = "WIN"
                    elif p_no > 0.9:
                        market_resolutions[cid] = "LOSE"
                except Exception as e:
                    print(f"Error parsing outcome for {cid}: {e}")
        time.sleep(0.1) # rate limit friendly
        
    # 2. Query active markets for missing IDs (just in case)
    missing_mids = [mid for mid in unique_mids if mid not in market_resolutions]
    if missing_mids:
        print(f"Querying {len(missing_mids)} remaining markets under active status...")
        for i in range(0, len(missing_mids), batch_size):
            batch = missing_mids[i : i + batch_size]
            data = fetch_gamma_batch(batch, status="active")
            for m in data:
                cid = m.get("conditionId")
                outcome_prices_str = m.get("outcomePrices")
                if cid and outcome_prices_str:
                    try:
                        outcome_prices = json.loads(outcome_prices_str)
                        p_yes = float(outcome_prices[0])
                        p_no = float(outcome_prices[1])
                        if p_yes > 0.9:
                            market_resolutions[cid] = "WIN"
                        elif p_no > 0.9:
                            market_resolutions[cid] = "LOSE"
                    except Exception as e:
                        pass
            time.sleep(0.1)
            
    print(f"Mapped {len(market_resolutions):,}/{len(unique_mids):,} markets successfully.")
    
    # Update actual_outcome in dataframe
    resolved_count = 0
    pending_count = 0
    win_count = 0
    lose_count = 0
    
    def map_row_outcome(row):
        nonlocal resolved_count, pending_count, win_count, lose_count
        mid = row["market_id"]
        if pd.isna(mid):
            return "PENDING"
        res = market_resolutions.get(mid)
        if res == "WIN":
            win_count += 1
            resolved_count += 1
            return "WIN"
        elif res == "LOSE":
            lose_count += 1
            resolved_count += 1
            return "LOSE"
        else:
            pending_count += 1
            return "PENDING"
            
    df["actual_outcome"] = df.apply(map_row_outcome, axis=1)
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved resolved shadow CSV to {output_path}")
    
    # Print summary format
    resolve_rate = (resolved_count / total_rows) * 100 if total_rows > 0 else 0
    print("\nSUMMARY:")
    print(f"Total baris: {total_rows:,}")
    print(f"Resolved WIN:     {win_count:,}")
    print(f"Resolved LOSE:    {lose_count:,}")
    print(f"Masih PENDING:    {pending_count:,}")
    print(f"Resolve rate:     {resolve_rate:.2f}%")
    
    # Check for NaN values in actual_outcome
    nan_outcome = df["actual_outcome"].isna().sum()
    if nan_outcome > 0:
        print(f"Warning: Found {nan_outcome} NaN outcomes after resolution!")
    else:
        print("Validation: No NaN values in actual_outcome columns.")

if __name__ == "__main__":
    # Default paths (can be overridden by arguments)
    inp = r"dataset/raw/dry_run_shadow_2026-06-02_073440.csv"
    out = r"dataset/raw/dry_run_shadow_resolved.csv"
    if len(sys.argv) > 1:
        inp = sys.argv[1]
    if len(sys.argv) > 2:
        out = sys.argv[2]
        
    resolve_outcomes(inp, out)
