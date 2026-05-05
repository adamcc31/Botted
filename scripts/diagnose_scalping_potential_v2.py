import os
import sys
import pandas as pd
import numpy as np
import httpx
import asyncio
from datetime import datetime, timezone

# Config
DATA_TRADES_API = "https://data-api.polymarket.com/trades"

async def fetch_trades(market_id):
    """Fetch trades for a market ID from Polymarket Data API."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # We don't use startTs here to get all trades for the market to ensure we have the resolution phase
            resp = await client.get(DATA_TRADES_API, params={"market": market_id})
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

async def main():
    print("Loading dry run dataset...")
    dry_path = "dataset/processed/dry_run_master_clean.csv"
    if not os.path.exists(dry_path):
        print(f"Error: {dry_path} not found.")
        return
        
    df = pd.read_csv(dry_path, low_memory=False)
    df['entry_timestamp_dt'] = pd.to_datetime(df['timestamp'], utc=True)
    df['entry_timestamp_unix'] = df['entry_timestamp_dt'].view('int64') // 10**9
    
    # Filter BUY signals
    signals_mask = df['signal_direction'].isin(['BUY_UP', 'BUY_DOWN'])
    unique_mids = df.loc[signals_mask, 'market_id'].unique().tolist()
    
    print(f"Processing {len(unique_mids)} unique markets via Data API...")
    df['max_high_bid'] = np.nan
    
    processed_count = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        for mid in unique_mids:
            trades = await fetch_trades(mid)
            if not trades:
                processed_count += 1
                continue
                
            # Filter rows for this market
            m_idx = df[(df['market_id'] == mid) & signals_mask].index
            
            for idx in m_idx:
                row = df.loc[idx]
                entry_ts = row['entry_timestamp_unix']
                direction = row['signal_direction'] # BUY_UP or BUY_DOWN
                
                # Determine outcome string to match
                # BUY_UP -> Up or Yes
                # BUY_DOWN -> Down or No
                target_outcomes = []
                if direction == 'BUY_UP':
                    target_outcomes = ['Up', 'Yes', 'YES', 'UP']
                else:
                    target_outcomes = ['Down', 'No', 'NO', 'DOWN']
                
                # Filter trades: timestamp > entry_ts and outcome matches
                future_prices = [
                    t['price'] for t in trades 
                    if t.get('timestamp', 0) > entry_ts and t.get('outcome') in target_outcomes
                ]
                
                if future_prices:
                    df.at[idx, 'max_high_bid'] = max(future_prices)

            processed_count += 1
            if processed_count % 50 == 0:
                print(f"Processed {processed_count}/{len(unique_mids)} markets...")
            
            # Rate limiting
            await asyncio.sleep(0.02)

    # Save results
    out_path = "dataset_diagnostics.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved modified dataframe to {out_path}")
    
    # --- REPORTING ---
    
    # Filter for signals only for report accuracy
    report_df = df[signals_mask].copy()
    
    # A. Threshold Sweep
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    print("\n--- HIT RATE BY THRESHOLD ---")
    for t in thresholds:
        hit_rate = (report_df['max_high_bid'] >= t).mean()
        print(f"Threshold {t}: {hit_rate:.2%} hit rate")
        
    # B. Lost Alpha
    # Encode outcome
    if 'label' in report_df.columns:
        report_df['is_loss'] = (report_df['label'] == 0)
    elif 'actual_outcome' in report_df.columns:
        report_df['is_loss'] = (report_df['actual_outcome'].astype(str) == 'LOSS')
    else:
        print("Outcome column missing.")
        return

    loss_df = report_df[report_df['is_loss']]
    print("\n--- LOSS CONVERTED TO WIN (LOST ALPHA) ---")
    print(f"Total LOSS trades: {len(loss_df)}")

    print("\n  [ALL DIRECTIONS]")
    for t in thresholds:
        rate = (loss_df['max_high_bid'] >= t).mean() if len(loss_df) > 0 else 0
        print(f"  Threshold {t}: {rate:.2%} of LOSS trades could have been won")

    for direction in ['BUY_UP', 'BUY_DOWN']:
        dir_loss = loss_df[loss_df['signal_direction'] == direction]
        print(f"\n  [{direction}] ({len(dir_loss)} trades)")
        for t in thresholds:
            rate = (dir_loss['max_high_bid'] >= t).mean() if len(dir_loss) > 0 else 0
            print(f"  Threshold {t}: {rate:.2%}")
            
    # C. Hit Rate x Odds Bucket
    print("\n--- HIT RATE BY ODDS BUCKET ---")
    report_df['odds_bucket'] = pd.cut(
        report_df['entry_odds'],
        bins=[0, 0.15, 0.25, 0.40, 0.55, 0.70, 1.0]
    )
    print(report_df.groupby('odds_bucket', observed=True).agg(
        n           = ('max_high_bid', 'count'),
        hit_080     = ('max_high_bid', lambda x: (x >= 0.80).mean()),
        hit_085     = ('max_high_bid', lambda x: (x >= 0.85).mean()),
        hit_090     = ('max_high_bid', lambda x: (x >= 0.90).mean()),
        avg_max_high= ('max_high_bid', 'mean')
    ).round(3).to_string())
        
    # D. Feature Correlation Check
    print("\n--- FEATURE CORRELATION WITH SPIKE (Threshold 0.85 & 0.90) ---")
    features = [
        'obi_value', 'depth_ratio', 'spread_pct',
        'tfm_value', 'rv_value', 'odds_delta_60s'
    ]
    report_df['target_085'] = (report_df['max_high_bid'] >= 0.85).astype(int)
    report_df['target_090'] = (report_df['max_high_bid'] >= 0.90).astype(int)

    existing_features = [f for f in features if f in report_df.columns]
    print("\nCorrelation with hitting 0.85:")
    print(report_df[existing_features + ['target_085']].corr()['target_085'].drop('target_085'))
    print("\nCorrelation with hitting 0.90:")
    print(report_df[existing_features + ['target_090']].corr()['target_090'].drop('target_090'))

if __name__ == "__main__":
    asyncio.run(main())
