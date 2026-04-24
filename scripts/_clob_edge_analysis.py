import asyncio
import httpx
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime, timezone
from pathlib import Path
from tqdm.asyncio import tqdm

# Setup
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_VER = "v20260406_084313"

async def get_market_tokens(client, market_id, semaphore):
    async with semaphore:
        try:
            resp = await client.get(
                f"https://gamma-api.polymarket.com/markets",
                params={"id": market_id},
                timeout=10.0
            )
            if resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data, list):
                    m = data[0]
                    clob_ids = m.get("clobTokenIds", [])
                    if isinstance(clob_ids, str):
                        try:
                            clob_ids = json.loads(clob_ids)
                        except:
                            pass
                    if clob_ids and len(clob_ids) >= 2:
                        return market_id, clob_ids[0] # yes_token
        except Exception as e:
            pass
        return market_id, None

async def get_clob_history(client, token_id, semaphore):
    async with semaphore:
        try:
            resp = await client.get(
                "https://clob.polymarket.com/prices-history",
                params={"market": token_id, "interval": "max", "fidelity": 1},
                timeout=10.0
            )
            if resp.status_code == 200:
                hist = resp.json().get("history", [])
                # Return list of (timestamp_ms, price)
                return token_id, [(h["t"] * 1000, float(h["p"])) for h in hist if "p" in h]
        except Exception as e:
            pass
        return token_id, []

async def fetch_all_data(unique_markets):
    tokens = {}
    
    # 5 concurrent requests to avoid rate limit
    sem = asyncio.Semaphore(5)
    
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=5)) as client:
        # 1. Fetch tokens
        tasks = [get_market_tokens(client, mid, sem) for mid in unique_markets]
        print(f"Fetching tokens for {len(unique_markets)} markets...")
        results = await tqdm.gather(*tasks)
        for mid, token in results:
            if token:
                tokens[mid] = token
        
        # 2. Fetch price histories
        print(f"\nFetching CLOB history for {len(tokens)} tokens...")
        valid_tokens = list(tokens.values())
        tasks2 = [get_clob_history(client, tid, sem) for tid in valid_tokens]
        results2 = await tqdm.gather(*tasks2)
        
        histories = {}
        for tid, hist in results2:
            if hist:
                histories[tid] = hist
                
    return tokens, histories

def load_models():
    model_path = DATA_DIR / "models" / f"model_lgbm_{MODEL_VER}.pkl"
    scaler_path = DATA_DIR / "models" / f"scaler_{MODEL_VER}.pkl"
    calib_path = DATA_DIR / "models" / f"calibrator_{MODEL_VER}.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file missing: {model_path}")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    try:
        calibrator = joblib.load(calib_path)
    except:
        calibrator = None
        
    return model, scaler, calibrator

def print_ascii_hist(data, bins=20):
    if len(data) == 0:
        print("No data for histogram.")
        return
        
    counts, bin_edges = np.histogram(data, bins=bins)
    max_count = max(counts)
    
    print("\n--- Plot: P(Model) - P(CLOB) Distribution ---")
    for i in range(len(counts)):
        bar_len = int(50 * counts[i] / max_count) if max_count > 0 else 0
        bin_label = f"[{bin_edges[i]:+5.2f}, {bin_edges[i+1]:+5.2f})"
        print(f"{bin_label} | {'█' * bar_len} ({counts[i]})")

def main():
    df_path = DATA_DIR / "processed" / "merged_training_features.parquet"
    print(f"Loading dataset: {df_path}")
    df = pd.read_parquet(df_path)
    print(f"Dataset samples: {len(df)}")
    
    try:
        model, scaler, calibrator = load_models()
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    feature_cols = df.columns[:24]
    X_raw = df[feature_cols].copy()
    X_scaled = scaler.transform(X_raw)
    
    if calibrator:
        p_calib = calibrator.predict_proba(X_scaled)[:, 1]
    else:
        p_calib = model.predict_proba(X_scaled)[:, 1]
        
    df["P_model"] = p_calib
    
    # We will sample 100 markets because 1600 is too API intensive for a quick test
    all_markets = df["market_id"].unique()
    sample_markets = np.random.choice(all_markets, size=min(100, len(all_markets)), replace=False)
    
    # Filter dataset to only those markets
    df = df[df["market_id"].isin(sample_markets)].copy()
    
    tokens, histories = asyncio.run(fetch_all_data(sample_markets))
    
    df["P_clob"] = np.nan
    df["clob_age_sec"] = np.nan
    
    matched = 0
    for idx, row in df.iterrows():
        mid = row["market_id"]
        ts_signal = row["signal_timestamp_ms"]
        
        token = tokens.get(mid)
        if not token: continue
        
        hist = histories.get(token)
        if not hist: continue
        
        hist_sorted = sorted(hist, key=lambda x: x[0])
        best_p = None
        best_gap = float('inf')
        
        for t_ms, p in hist_sorted:
            gap = ts_signal - t_ms
            if 0 <= gap < best_gap:
                best_gap = gap
                best_p = p
                
        # Only accept if price is from within last 15 minutes
        if best_p is not None and best_gap <= 15 * 60 * 1000:
            df.at[idx, "P_clob"] = best_p
            df.at[idx, "clob_age_sec"] = best_gap / 1000
            matched += 1

    print(f"\nMatched CLOB prices for {matched} out of {len(df)} samples in subset.")
    
    has_clob = df.dropna(subset=["P_clob"]).copy()
    if len(has_clob) == 0:
        print("No historical CLOB data could be aligned.")
        return
        
    has_clob = has_clob[(has_clob["P_clob"] >= 0.02) & (has_clob["P_clob"] <= 0.98)]
    print(f"Samples with active CLOB price (0.02 - 0.98): {len(has_clob)}")
    
    has_clob["Edge"] = has_clob["P_model"] - has_clob["P_clob"]
    has_clob["Abs_Edge"] = has_clob["Edge"].abs()
    
    print("\n--- STATISTICS ---")
    print(f"Mean P_model: {has_clob['P_model'].mean():.4f}")
    print(f"Mean P_clob:  {has_clob['P_clob'].mean():.4f}")
    print(f"Mean Edge:    {has_clob['Edge'].mean():.4f}")
    print(f"Mean Abs Edge:{has_clob['Abs_Edge'].mean():.4f}")
    
    threshold = 0.05
    bets = has_clob[has_clob["Abs_Edge"] > threshold].copy()
    print(f"\n--- TRADING SIMULATION (Threshold = {threshold*100}%) ---")
    print(f"Total theoretical trades: {len(bets)} / {len(has_clob)} ({(len(bets)/len(has_clob))*100:.1f}%)")
    
    if len(bets) > 0:
        def calc_pnl(row):
            if row["Edge"] > 0:
                cost = row["P_clob"]
                revenue = 1 if row["label"] == 1 else 0
            else:
                cost = 1 - row["P_clob"]
                revenue = 1 if row["label"] == 0 else 0
            return revenue - cost
            
        bets["PnL"] = bets.apply(calc_pnl, axis=1)
        win_rate = (bets["PnL"] > 0).mean()
        roi = bets["PnL"].sum() / sum([row["P_clob"] if row["Edge"] > 0 else (1 - row["P_clob"]) for _, row in bets.iterrows()])
        
        print(f"Win Rate: {win_rate*100:.2f}%")
        print(f"Total PnL (1 share per bet): {bets['PnL'].sum():.2f} USD")
        print(f"ROI on capital at risk: {roi*100:.2f}%")
        
    print_ascii_hist(has_clob["Edge"].values, bins=20)
    
if __name__ == "__main__":
    main()
