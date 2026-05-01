import pandas as pd
import numpy as np
import os
import sys

def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    # TASK 1: Load dataset
    datasets = [
        "dataset/raw/alpha_v1_master.csv",
        "dataset/weekend_market/dataset_ml_ready.csv",
        "dataset/weekday_market/dataset_ml_ready.csv"
    ]
    
    file_path = None
    for ds in datasets:
        if os.path.exists(ds):
            file_path = ds
            break
            
    if not file_path:
        print("No dataset found.")
        return
        
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    
    print(f"Total rows: {len(df)}")
    if 'label' in df.columns:
        valid_labels = df['label'].notnull().sum()
        print(f"Rows with non-null label: {valid_labels}")
        df = df[df['label'].notnull()].copy()
    else:
        print("No 'label' column found.")
        return

    # TASK 6 Validation part 1
    assert df['label'].isin([0, 1]).all(), "Label column contains values other than 0 and 1"

    # Reconstruct Physical Features
    btc_col = None
    if 'btc_binance' in df.columns:
        btc_col = 'btc_binance'
    elif 'binance_price' in df.columns:
        btc_col = 'binance_price'
    else:
        for col in df.columns:
            if 'btc' in col.lower() and 'price' in col.lower() and 'chainlink' not in col.lower():
                btc_col = col
                break
        if not btc_col:
            for col in df.columns:
                if 'price' in col.lower() and 'chainlink' not in col.lower() and 'strike' not in col.lower() and 'resolution' not in col.lower() and 'settlement' not in col.lower():
                    btc_col = col
                    break

    strike_col = None
    if 'strike_price' in df.columns:
        strike_col = 'strike_price'
    else:
        for col in df.columns:
            if 'strike' in col.lower():
                strike_col = col
                break

    if not btc_col or not strike_col:
        print(f"Could not find BTC or Strike column. BTC: {btc_col}, Strike: {strike_col}")
        print("Columns available:", df.columns.tolist())
        return

    print(f"Using BTC column: {btc_col}")
    print(f"Using Strike column: {strike_col}")

    # Convert to float
    df[strike_col] = pd.to_numeric(df[strike_col], errors='coerce')
    df[btc_col] = pd.to_numeric(df[btc_col], errors='coerce')
    df = df.dropna(subset=[strike_col, btc_col]).copy()

    df['distance_to_strike'] = abs(df[strike_col] - df[btc_col])
    
    # Removed entry_odds < 0.40 filter
    filtered_rows_count = len(df)
    print(f"Rows after removing NaNs for price/strike: {filtered_rows_count}")

    # TASK 2: High-Resolution Buckets
    if 'ttr_minutes' in df.columns:
        ttr = df['ttr_minutes']
    elif 'ttr_seconds' in df.columns:
        ttr = df['ttr_seconds'] / 60
    elif 'ttr' in df.columns:
        ttr = df['ttr']
    else:
        print("Could not find TTR column.")
        return

    def get_ttr_bucket(x):
        if x < 1.5: return "< 1.5 min"
        elif x <= 3: return "1.5 - 3 min"
        elif x <= 5: return "3 - 5 min"
        else: return "> 5 min"

    def get_dist_bucket(x):
        if x < 15: return "< $15"
        elif x <= 30: return "$15 - $30"
        elif x <= 60: return "$30 - $60"
        elif x <= 100: return "$60 - $100"
        else: return "> $100"

    def get_odds_bucket(x):
        if x < 0.15: return "< 0.15"
        elif x <= 0.25: return "0.15 - 0.25"
        elif x <= 0.40: return "0.25 - 0.40"
        elif x <= 0.55: return "0.40 - 0.55"
        elif x <= 0.70: return "0.55 - 0.70"
        else: return "> 0.70"

    df['ttr_bucket'] = ttr.apply(get_ttr_bucket)
    df['dist_bucket'] = df['distance_to_strike'].apply(get_dist_bucket)
    df['odds_bucket'] = df['entry_odds'].apply(get_odds_bucket)

    # TASK 3: Standardisasi Kalkulasi NetEV
    df['payout_multiplier'] = (1.0 / df['entry_odds']) * (1.0 - 0.02 - 0.005) - 0.01
    df['net_ev'] = np.where(df['label'] == 1, df['payout_multiplier'] - 1.0, -1.0)

    # Group by buckets
    grouped = df.groupby(['ttr_bucket', 'dist_bucket', 'odds_bucket']).agg(
        n=('label', 'count'),
        win_rate=('label', 'mean'),
        avg_net_ev=('net_ev', 'mean')
    ).reset_index()

    def get_status(row):
        if row['n'] < 15:
            return "⚠️ LOW_N"
        elif row['avg_net_ev'] > 0:
            return "✅ +EV"
        else:
            return "❌ -EV"

    grouped['status'] = grouped.apply(get_status, axis=1)
    
    # Save CSV (all records)
    out_dir = "scripts/output"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, "forensic_full_spectrum.csv")
    grouped.to_csv(out_path, index=False, encoding='utf-8')
    
    # TASK 4: Cetak Laporan
    valid_grouped = grouped[grouped['n'] >= 15].copy()
    
    alpha_zones = valid_grouped.sort_values(by='avg_net_ev', ascending=False).head(10)
    death_zones = valid_grouped.sort_values(by='avg_net_ev', ascending=True).head(10)

    print("\n\U0001f525 THE ALPHA ZONES (Top 10 Kombinasi dengan NetEV Tertinggi)")
    print("  TTR          | Distance    | Odds        |   N | WinRate | NetEV  | Status")
    print("  -------------|-------------|-------------|-----|---------|--------|-------")
    for _, row in alpha_zones.iterrows():
        print(f"  {row['ttr_bucket']:<12} | {row['dist_bucket']:<11} | {row['odds_bucket']:<11} | {row['n']:>3} | {row['win_rate']*100:>6.1f}% | {row['avg_net_ev']:>+6.3f} | {row['status']}")

    print("\n\u2620\ufe0f THE DEATH ZONES (Top 10 Kombinasi dengan NetEV Terburuk)")
    print("  TTR          | Distance    | Odds        |   N | WinRate | NetEV  | Status")
    print("  -------------|-------------|-------------|-----|---------|--------|-------")
    for _, row in death_zones.iterrows():
        print(f"  {row['ttr_bucket']:<12} | {row['dist_bucket']:<11} | {row['odds_bucket']:<11} | {row['n']:>3} | {row['win_rate']*100:>6.1f}% | {row['avg_net_ev']:>+6.3f} | {row['status']}")

    print(f"\nMatrix lengkap (Full Spectrum) tersimpan di: {out_path}")

if __name__ == "__main__":
    main()
