import pandas as pd
import os

def clean_shadow():
    resolved_path = r"z:\01 ADAM\00 DOKUMENTASI PROJECT\polymarket\MADE IN ABYSS V2\dataset\raw\dry_run_shadow_resolved.csv"
    output_path = r"z:\01 ADAM\00 DOKUMENTASI PROJECT\polymarket\MADE IN ABYSS V2\dataset\processed\shadow_v6_clean.csv"
    
    print("=" * 60)
    print("CLEANING SHADOW CSV FOR V6 DATASET")
    print("=" * 60)
    
    if not os.path.exists(resolved_path):
        print(f"Error: {resolved_path} not found!")
        return
        
    df = pd.read_csv(resolved_path)
    initial_rows = len(df)
    print(f"Initial rows: {initial_rows:,}")
    
    # 1. Drop PENDING rows
    df = df[df["actual_outcome"] != "PENDING"]
    after_pending = len(df)
    print(f"Rows after dropping PENDING: {after_pending:,} (Dropped {initial_rows - after_pending:,} rows)")
    
    # 2. Drop rows with btc_realized_vol_prior_30m == 0.45 (fallback)
    # Note: 0.45 is the exact fallback value
    fallback_value = 0.45
    # Since they are floats, we should check with a small tolerance or check exact value as string/float
    # In CSV, it is written as 0.45 or 0.450000 etc.
    df = df[df["btc_realized_vol_prior_30m"] != fallback_value]
    after_fallback = len(df)
    print(f"Rows after dropping fallback vol (0.45): {after_fallback:,} (Dropped {after_pending - after_fallback:,} rows)")
    
    # 3. Filter invalid prices
    df = df[
        (df['yes_price_t0'] > 0.05) &  # YES price tidak boleh < 5%
        (df['no_price_t0'] > 0.05) &   # NO price tidak boleh < 5%
        (df['yes_price_t0'] + df['no_price_t0'] > 0.80) &  # Sum harus mendekati 1.0
        (df['yes_price_t0'] + df['no_price_t0'] < 1.20)    # Tidak boleh terlalu tinggi
    ]
    after_price_filter = len(df)
    print(f"Rows after invalid price filtering: {after_price_filter:,} (Dropped {after_fallback - after_price_filter:,} rows)")
    
    # 4. Validate no NaN in critical features
    critical_features = [
        "spread_pct", "yes_price_t0", "no_price_t0", "clob_spread_t0", 
        "yes_depth_t0", "no_depth_t0", "btc_realized_vol_prior_30m"
    ]
    
    print("\nChecking for NaN in critical features:")
    for feat in critical_features:
        nans = df[feat].isna().sum()
        print(f"  {feat}: {nans} NaNs")
        
    # Drop any rows with NaN in these features
    df = df.dropna(subset=critical_features)
    final_rows = len(df)
    print(f"\nFinal rows remaining: {final_rows:,} (Dropped {after_price_filter - final_rows:,} rows with NaN)")
    
    # 4. Save to processed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Clean shadow CSV saved to: {output_path}")
    
    # Print label distribution
    print("\nLabel Distribution:")
    print(df["actual_outcome"].value_counts())

if __name__ == "__main__":
    clean_shadow()
