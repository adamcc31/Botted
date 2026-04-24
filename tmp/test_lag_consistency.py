import pandas as pd
import numpy as np
import structlog
from datetime import timedelta

logger = structlog.get_logger(__name__)

def validate_lag_consistency():
    logger.info("loading_hf_dataset", path="dataset/btc_5m_hf_ticks.parquet")
    df = pd.read_parquet('dataset/btc_5m_hf_ticks.parquet')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 1. Identify "Momentum Events"
    # We look for 30s windows where binance moves > 0.05% (0.1% might be too rare for 50 samples in a small slice)
    # Using a 30s rolling max-min diff
    df['binance_ret_30s'] = df['binance_price'].pct_change(periods=30) # Roughly 30 ticks if 1Hz
    
    # Filter for significant moves
    threshold = 0.0005 # 0.05%
    spikes = df[df['binance_ret_30s'].abs() > threshold].copy()
    
    if len(spikes) < 50:
        logger.warning("too_few_spikes_at_0.05_percent", count=len(spikes))
        threshold = 0.0002 # Relax to 0.02% if needed
        spikes = df[df['binance_ret_30s'].abs() > threshold].copy()

    logger.info("spikes_found", count=len(spikes), threshold=threshold)
    
    # To avoid overlapping spikes from the same move, we take only the first tick of each spike group
    spikes['diff_idx'] = spikes.index.to_series().diff()
    independent_spikes = spikes[spikes['diff_idx'] > 60].head(100) # Spikes at least 60 ticks apart
    
    lags = []
    
    for idx in independent_spikes.index:
        spike_time = df.loc[idx, 'datetime']
        spike_dir = np.sign(df.loc[idx, 'binance_ret_30s'])
        
        # Look at the next 120 seconds (120 ticks roughly) for Poly reaction
        lookahead = df.iloc[idx : idx + 120]
        
        # Reaction: first move in Poly mid that is > 0.001 in probability and in SAME direction
        current_poly = df.loc[idx, 'up_mid']
        
        # Find first index where poly moves significantly in spike_dir
        reaction = lookahead[
            ((lookahead['up_mid'] - current_poly) * spike_dir > 0.005) # 0.5% prob move
        ]
        
        if not reaction.empty:
            reaction_time = reaction['datetime'].iloc[0]
            lag = (reaction_time - spike_time).total_seconds()
            if lag > 0: # Ensure it's a reaction, not a lead
                lags.append(lag)
                
    if not lags:
        logger.error("no_reactions_found")
        return

    lags = pd.Series(lags)
    stats = {
        "count": len(lags),
        "mean": round(lags.mean(), 2),
        "median": round(lags.median(), 2),
        "std": round(lags.std(), 2),
        "min": round(lags.min(), 2),
        "max": round(lags.max(), 2),
        "p25": round(lags.quantile(0.25), 2),
        "p75": round(lags.quantile(0.75), 2)
    }
    
    print("\n" + "="*50)
    print("📊 LAG CONSISTENCY VALIDATION")
    print("="*50)
    for k, v in stats.items():
        print(f"{k.capitalize():<10}: {v}")
    print("="*50)
    
    if stats['std'] < 15:
        print("✅ RESULT: Consistent Lag Detected (Scenario A - Alpha Likely Real)")
    elif stats['std'] > 30:
        print("❌ RESULT: Highly Variable Lag (Scenario B - Likely Timestamp Artifact)")
    else:
        print("⚠️ RESULT: Moderate Variance - Needs further inspection")

if __name__ == "__main__":
    validate_lag_consistency()
