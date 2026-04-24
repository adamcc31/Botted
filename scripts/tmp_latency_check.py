import polars as pl
import numpy as np

def analyze_latency():
    print("Loading HF Ticks Dataset...")
    # Load a smaller slice (e.g. first 2 million rows) to be fast
    df = pl.read_parquet("dataset/btc_5m_hf_ticks.parquet").head(2_000_000)
    
    print(f"Loaded {df.height:,} rows.")
    
    # We want to measure how fast poly_up_mid reacts to binance_price.
    # Group by event_slug to ensure time series continuity, though we could just look globally.
    # Let's resample to 500ms intervals just to smooth out the tick jitter
    df = df.with_columns([
        (pl.col('poly_up_bid') + pl.col('poly_up_ask')).alias('poly_up_mid') / 2.0
    ])
    
    # Drop where mid is null
    df = df.drop_nulls(subset=["binance_price", "poly_up_mid"])
    
    # Calculate returns (log returns)
    df = df.with_columns([
        (pl.col("binance_price") / pl.col("binance_price").shift(1) - 1.0).alias("spot_ret"),
        (pl.col("poly_up_mid") / pl.col("poly_up_mid").shift(1) - 1.0).alias("clob_ret")
    ])
    
    df = df.drop_nulls(subset=["spot_ret", "clob_ret"])
    
    # Let's compute cross-correlation between spot_ret and clob_ret at different lag ticks
    # Assuming avg tick rate is ~10 ticks per second (100ms)
    spot_rets = df["spot_ret"].to_numpy()
    clob_rets = df["clob_ret"].to_numpy()
    
    # Calculate zero-mean for correlation
    spot_rets = spot_rets - np.mean(spot_rets)
    clob_rets = clob_rets - np.mean(clob_rets)
    
    spot_std = np.std(spot_rets)
    clob_std = np.std(clob_rets)
    
    if spot_std == 0 or clob_std == 0:
        print("Std is 0, cannot compute correlation.")
        return
        
    print("\n--- Cross-Correlation (Binance Spot leads CLOB) ---")
    max_lag_ticks = 50 # 5 seconds if 10 ticks/sec
    
    correlations = []
    for lag in range(0, max_lag_ticks + 1):
        if lag == 0:
            cov = np.mean(spot_rets * clob_rets)
        else:
            cov = np.mean(spot_rets[:-lag] * clob_rets[lag:])
        corr = cov / (spot_std * clob_std)
        correlations.append((lag, corr))
        
    # Find peak correlation
    best_lag = max(correlations, key=lambda x: x[1])
    
    print(f"Peak Correlation is at Lag = {best_lag[0]} ticks (approx {best_lag[0]*100}ms)")
    print(f"Correlation value: {best_lag[1]:.5f}")
    
    print("\nTop 5 correlation lags:")
    for lag, corr in sorted(correlations, key=lambda x: x[1], reverse=True)[:5]:
        print(f"Lag +{lag*100:4d}ms: {corr:.5f}")
        
    # Calculate how often mispricing exceeds 5%
    # This is a bit tricky without fair prob, but we can check spread
    spreads = df["up_spread"].to_numpy()
    mean_spread = np.nanmean(spreads)
    print(f"\nMean CLOB Spread in HF dataset: {mean_spread:.4f}")

if __name__ == "__main__":
    analyze_latency()
