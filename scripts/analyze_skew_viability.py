import polars as pl
from datetime import datetime, time
import numpy as np

def analyze_skew_viability():
    print("🚀 Starting Polars Skew Sanity Audit (Tumbling Window Debounce)...")
    
    # 1. Load Data
    hf_ticks_path = 'dataset/btc_5m_hf_ticks.parquet'
    agg_trades_path = 'data/raw/aggTrades/aggTrades_2026-03-*.parquet'
    
    poly_lazy = pl.scan_parquet(hf_ticks_path).with_columns([
        pl.col("datetime").cast(pl.Datetime("us", "UTC")).alias("dt"),
        pl.col("expires").str.to_datetime("%Y-%m-%dT%H:%M:%SZ").cast(pl.Datetime("us", "UTC")).alias("expiry_dt")
    ])
    
    binance_lazy = pl.scan_parquet(agg_trades_path).with_columns([
        pl.from_epoch(pl.col("timestamp"), time_unit="us").dt.replace_time_zone("UTC").alias("dt"),
        ((pl.col("price") * pl.col("quantity")) * 
         pl.when(pl.col("is_buyer_maker") == False).then(1.0).otherwise(-1.0)).alias("tfm_delta")
    ])

    # 2. Signal Generation (Binance Side)
    # Resample to 1-minute bins
    binance_1m = binance_lazy.group_by_dynamic("dt", every="1m").agg([
        pl.col("tfm_delta").sum().alias("tfm_1m")
    ])

    # Rolling 1m TFM and 1h Z-Score (closed='left' logic)
    binance_signals = binance_1m.with_columns([
        pl.col("tfm_1m").rolling_sum(window_size=1).alias("tfm_sig"),
        pl.col("tfm_1m").rolling_mean(window_size=60).alias("tfm_avg"),
        pl.col("tfm_1m").rolling_std(window_size=60).alias("tfm_std")
    ]).with_columns([
        ((pl.col("tfm_sig") - pl.col("tfm_avg")) / pl.col("tfm_std")).alias("z_tfm")
    ]).filter(pl.col("z_tfm").is_not_null())

    # 3. Join and Target Generation
    # join poly ticks with signals
    merged_lazy = poly_lazy.sort("dt").join_asof(
        binance_signals.sort("dt"),
        on="dt",
        strategy="backward"
    ).filter(
        (pl.col("up_spread") < 0.05) & 
        (pl.col("dt") < pl.col("expiry_dt") - pl.duration(minutes=1)) # 5-min target filter
    )

    # Future Bid Probe (T + 2m)
    merged_lazy = merged_lazy.with_columns([
        (pl.col("dt") + pl.duration(minutes=2)).alias("dt_future")
    ])
    future_bid_lazy = poly_lazy.select(["dt", "poly_up_bid"]).rename({"poly_up_bid": "poly_up_bid_future", "dt": "dt_future"})
    
    final_lazy = merged_lazy.join_asof(
        future_bid_lazy.sort("dt_future"),
        on="dt_future",
        strategy="backward"
    ).with_columns([
        (pl.col("poly_up_bid_future") - pl.col("poly_up_ask")).alias("executable_profit"),
        pl.when(pl.col("dt").dt.hour().is_between(0, 7)).then(pl.lit("Asia"))
          .when(pl.col("dt").dt.hour().is_between(8, 15)).then(pl.lit("EU"))
          .otherwise(pl.lit("US")).alias("session")
    ]).with_columns([
        (pl.col("executable_profit") > 0).alias("is_win")
    ])

    # 4. EVENT DEBOUNCING (Tumbling Windows 30m)
    # This addresses the Autocorrelation issue by only taking the first trigger per window
    # We filter for high conviction first
    high_z_lazy = final_lazy.filter(pl.col("z_tfm").abs() >= 5.0)
    
    # Apply Tumbling Window grouping
    independent_lazy = high_z_lazy.sort("dt").group_by_dynamic(
        "dt", 
        every="30m", 
        by="event_slug"
    ).agg(pl.all().first())

    print("📊 Computing Independent Opportunity Matrix...")
    
    results = independent_lazy.group_by([
        pl.col("z_tfm").round(0).alias("z_bucket"),
        pl.col("session")
    ]).agg([
        pl.len().alias("N_Independent_Events"),
        pl.col("is_win").mean().alias("Win_Rate_Theoretical"),
        pl.col("executable_profit").mean().alias("Avg_Profit_Executable"),
        # Ghost check: % of signals where price didn't move
        (pl.col("poly_up_bid_future") == pl.col("poly_up_bid")).mean().alias("Ghost_Threshold")
    ]).collect()

    print("\n" + "="*80)
    print("📈 OPTION 4: INDEPENDENT EVENTS PERFORMANCE (30m DEBOUNCED)")
    print("="*80)
    display_df = results.to_pandas().sort_values(["z_bucket", "session"])
    print(display_df.to_string(index=False))
    print("="*80)
    
    total_days = 16 # March 5-21
    total_events = results["N_Independent_Events"].sum()
    print(f"Total Unique Trading Opportunities (Z > 5.0): {total_events}")
    print(f"Average Opportunities Per Day: {total_events / total_days:.2f}")
    print(f"Estimated Monthly Yield (assuming 1 position/signal): {total_events / total_days * 30:.0f} signals/mo")
    
    results.write_csv("tmp/independent_skew_results.csv")
    print("✅ Audit complete. Results saved to tmp/independent_skew_results.csv")

if __name__ == "__main__":
    analyze_skew_viability()
