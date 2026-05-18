import pandas as pd
import numpy as np

# Load data hasil tarikan terbaru (PowerShell redirection uses utf-16)
df = pd.read_csv('dataset/audit_zombie/clob_log.csv', low_memory=False, encoding='utf-16')


# Pastikan kolom timestamp dikonversi ke datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

print(f"Rentang Waktu Log Audit: {df['timestamp'].min()} s/d {df['timestamp'].max()}")
print(f"Total Baris Data (Ticks): {len(df)}")

# 1. Hitung statistik spread_pct riil di pasar
df['calculated_spread_pct'] = ((df['yes_ask'] - df['yes_bid']) / df['yes_bid']) * 100

print("\n--- STATISTIK SPREAD RIEL ---")
print(df['calculated_spread_pct'].describe())

# 2. Hitung berapa persen data yang lolos regulasi 0.03%
below_threshold = df[df['calculated_spread_pct'] <= 0.03]
pct_allowed = (len(below_threshold) / len(df)) * 100
print(f"\nPersentase Spread <= 0.03% (Seharusnya Lolos): {pct_allowed:.2f}%")

# 3. Cek apakah data membeku (Stale Data Check)
unique_bids = df['yes_bid'].nunique()
print(f"Jumlah Perubahan Harga Unik (Unique Bids): {unique_bids} (Jika angka sangat kecil/1, data FREEZE!)")

print("\n" + "="*50)
print("ANALISIS SPREAD ORACLE (BINANCE VS CHAINLINK)")
print("="*50)

df_dry = pd.read_csv('dataset/audit_zombie/dry_run.csv', low_memory=False, encoding='utf-16')
if 'spread_pct' in df_dry.columns:
    oracle_spreads = df_dry['spread_pct'].dropna()
    print("\n--- STATISTIK SPREAD ORACLE RIEL ---")
    print(oracle_spreads.describe())
    
    below_oracle = oracle_spreads[oracle_spreads <= 0.03]
    pct_oracle_allowed = (len(below_oracle) / len(oracle_spreads)) * 100
    print(f"\nPersentase Spread Oracle <= 0.03% (Seharusnya Lolos): {pct_oracle_allowed:.2f}%")
else:
    print("Kolom spread_pct tidak ditemukan di dry_run.csv")

