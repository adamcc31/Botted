import pandas as pd

clob_log_path = "dataset/raw/audit_16h/clob_log.csv"

try:
    df = pd.read_csv(clob_log_path)
    df['yes_spread_usd'] = df['yes_ask'] - df['yes_bid']
    df['yes_spread_pct'] = (df['yes_spread_usd'] / df['yes_bid']) * 100
    
    # Filter ticks that are liquid AND bids are not flat 0.01
    filtered_df = df[(df['is_liquid'] == True) & (df['yes_bid'] > 0.05) & (df['yes_ask'] < 0.95)]
    print(f"Total ticks: {len(df)}")
    print(f"Filtered (liquid, bid > 0.05, ask < 0.95) ticks: {len(filtered_df)}")
    if len(filtered_df) > 0:
        print("\n=== POLYMARKET SPREAD STATISTICS FOR FILTERED TICKS ===")
        print(filtered_df['yes_spread_pct'].describe())
        print(f"Mean spread: {filtered_df['yes_spread_pct'].mean():.4f}%")
        print(f"Median spread: {filtered_df['yes_spread_pct'].median():.4f}%")
        print(f"Min spread: {filtered_df['yes_spread_pct'].min():.4f}%")
        print(f"Max spread: {filtered_df['yes_spread_pct'].max():.4f}%")
        
        # Count within thresholds
        for th in [2.0, 5.0, 10.0, 20.0]:
            count = (filtered_df['yes_spread_pct'] <= th).sum()
            pct = count / len(filtered_df) * 100
            print(f"Spread <= {th}%: {count} ticks ({pct:.2f}%)")
    else:
        print("No ticks match filtered criteria.")
except Exception as e:
    print(f"Error: {e}")
