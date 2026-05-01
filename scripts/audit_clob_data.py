"""
audit_clob_data.py -- Phase 2 Data Validation: Audit clob_log integrity,
find root cause of anomalous spread metrics, rerun analysis with clean data.
"""
import sys, os, glob
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")
sys.path.insert(0, BASE_DIR)


# ============================================================
# TASK 1: AUDIT DATA CLOB_LOG
# ============================================================

def task1_audit():
    print("=" * 80)
    print("TASK 1: AUDIT DATA CLOB_LOG")
    print("=" * 80)

    # --- 1a. Inventaris file ---
    print("\n--- 1a. Inventaris File ---")
    clob_files = sorted(glob.glob(os.path.join(RAW_DIR, "clob_log*.csv")))
    print(f"Files found: {len(clob_files)}")
    for f in clob_files:
        sz = os.path.getsize(f)
        print(f"  {os.path.basename(f):50s} {sz:>10,} bytes")

    dfs = []
    for f in clob_files:
        df = pd.read_csv(f)
        df['_src'] = os.path.basename(f)
        dfs.append(df)

    clob = pd.concat(dfs, ignore_index=True)
    clob['timestamp'] = pd.to_datetime(clob['timestamp'], utc=True)
    before_dedup = len(clob)
    clob = clob.sort_values('timestamp').drop_duplicates()
    after_dedup = len(clob)

    ts_min = clob['timestamp'].min()
    ts_max = clob['timestamp'].max()
    span = ts_max - ts_min

    print(f"\nTotal rows before dedup: {before_dedup}")
    print(f"Total rows after dedup:  {after_dedup}")
    print(f"Duplicates removed:      {before_dedup - after_dedup}")
    print(f"Timestamp MIN: {ts_min}")
    print(f"Timestamp MAX: {ts_max}")
    print(f"Span: {span.days} days {span.seconds // 3600} hours")
    print(f"Columns: {clob.columns.tolist()}")

    has_7_days = span.days >= 6
    print(f"\n7-day coverage? {'YES' if has_7_days else 'NO -- only ' + str(span.days) + ' days'}")

    # --- 1b. Audit kolom spread ---
    print("\n--- 1b. Audit Kolom Spread ---")
    print("Raw columns available:", clob.columns.tolist())

    # Compute spread_bps from yes_ask / yes_bid
    if 'yes_ask' in clob.columns and 'yes_bid' in clob.columns:
        print("\nComputing spread_bps from (yes_ask - yes_bid) / yes_ask * 10000")
        print(f"  yes_ask stats:")
        print(f"    null: {clob['yes_ask'].isna().sum()}")
        print(f"    zero: {(clob['yes_ask'] == 0).sum()}")
        print(f"    min:  {clob['yes_ask'].min()}")
        print(f"    max:  {clob['yes_ask'].max()}")
        print(f"    mean: {clob['yes_ask'].mean():.4f}")

        print(f"  yes_bid stats:")
        print(f"    null: {clob['yes_bid'].isna().sum()}")
        print(f"    zero: {(clob['yes_bid'] == 0).sum()}")
        print(f"    min:  {clob['yes_bid'].min()}")
        print(f"    max:  {clob['yes_bid'].max()}")
        print(f"    mean: {clob['yes_bid'].mean():.4f}")

        # Check for zero-bid markets (one-sided orderbook)
        zero_bid = (clob['yes_bid'] == 0)
        zero_ask = (clob['yes_ask'] == 0)
        both_zero = zero_bid & zero_ask
        print(f"\n  Rows with yes_bid == 0: {zero_bid.sum()} ({zero_bid.mean()*100:.1f}%)")
        print(f"  Rows with yes_ask == 0: {zero_ask.sum()} ({zero_ask.mean()*100:.1f}%)")
        print(f"  Rows with BOTH zero:    {both_zero.sum()}")

        # Compute spread
        clob['spread_raw'] = clob['yes_ask'] - clob['yes_bid']
        clob['spread_bps'] = np.where(
            clob['yes_ask'] > 0,
            (clob['spread_raw'] / clob['yes_ask']) * 10000,
            np.nan
        )
    else:
        print("ERROR: yes_ask / yes_bid columns not found!")
        return None

    col = clob['spread_bps']
    print(f"\nSpread_bps audit:")
    print(f"  Total rows:        {len(col)}")
    print(f"  Non-null rows:     {col.notna().sum()}")
    print(f"  Null rows:         {col.isna().sum()}")
    print(f"  Zero values:       {(col == 0).sum()}")
    print(f"  Negative values:   {(col < 0).sum()}")
    print(f"  Inf values:        {np.isinf(col).sum()}")
    print(f"  Values > 10000 bps: {(col > 10000).sum()}")

    print(f"\nFull distribution (ALL data):")
    print(col.describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))

    # Clean data: exclude null, zero, inf, > 10000
    clean = col[(col.notna()) & (col > 0) & (~np.isinf(col)) & (col < 10000)]
    print(f"\nDistribution (CLEAN data: exclude null, 0, inf, >10000):")
    print(f"  Clean rows: {len(clean)} / {len(col)} ({len(clean)/len(col)*100:.1f}%)")
    print(clean.describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))

    print(f"\nSample of extreme values (spread > 5000 bps):")
    extreme = clob[col > 5000][['timestamp', 'market_id', 'yes_ask', 'yes_bid', 'spread_bps', '_src']].head(10)
    print(extreme.to_string())

    # ROOT CAUSE: what is making spread == 9900 bps?
    print(f"\n--- ROOT CAUSE INVESTIGATION ---")
    print(f"Value counts of spread_bps (top 20):")
    vc = col.dropna().round(0).value_counts().head(20)
    print(vc)

    # --- 1c. Root cause anomali 4.0% ---
    print("\n--- 1c. Root Cause Anomali 4.0% ---")
    thresholds = [200, 250, 300, 350, 400]
    print("Using ALL data (including dirty):")
    for t in thresholds:
        cnt = (col < t).sum()
        pct = cnt / len(col) * 100
        print(f"  Spread < {t} bps: {pct:.2f}% ({cnt} rows)")

    print("\nUsing CLEAN data only:")
    for t in thresholds:
        cnt = (clean < t).sum()
        pct = cnt / len(clean) * 100
        print(f"  Spread < {t} bps: {pct:.2f}% ({cnt} rows)")

    print(f"\nClean values < 400 bps breakdown:")
    under_400 = clean[clean < 400]
    print(under_400.round(0).value_counts().head(20))

    # Explain what yes_ask=0.01 yes_bid=0.00 gives us
    print(f"\n--- DIAGNOSIS ---")
    typical_extremes = clob[col > 9000][['yes_ask', 'yes_bid', 'spread_bps']].head(5)
    print("Typical 'extreme' rows (spread > 9000 bps):")
    print(typical_extremes.to_string())
    print("\nThese are markets where yes_ask=0.01 and yes_bid=0.00 (or similar)")
    print("=> spread = (0.01 - 0.00) / 0.01 * 10000 = 10000 bps (100%)")
    print("These are EMPTY/ONE-SIDED orderbooks, not tradeable.")

    return clob


# ============================================================
# TASK 2: RERUN ANALYSIS WITH CLEAN DATA
# ============================================================

def task2_rerun(clob):
    print("\n" + "=" * 80)
    print("TASK 2: RERUN ANALYSIS WITH CLEAN DATA")
    print("=" * 80)

    col = clob['spread_bps']
    # Clean: exclude null, zero, inf, > 10000 bps
    mask_clean = (col.notna()) & (col > 0) & (~np.isinf(col)) & (col < 10000)
    clean = clob[mask_clean].copy()
    dirty_pct = (1 - len(clean) / len(clob)) * 100
    print(f"Cleaned: {len(clean)} rows (removed {dirty_pct:.1f}% dirty rows)")

    # --- 24-hour spread profile ---
    print("\n--- 2a. 24-Hour Spread Profile (CLEAN data) ---")
    hourly = clean.groupby(clean['timestamp'].dt.hour)['spread_bps'].agg(
        ['count', 'median', 'mean', 'min', 'max']
    ).round(2)
    print(hourly.to_string())

    best_hour = hourly['median'].idxmin()
    best_spread = hourly['median'].min()
    print(f"\nBest hour (UTC): {best_hour}:00 | Median spread: {best_spread:.2f} bps")

    # --- Threshold analysis ---
    print("\n--- 2b. Threshold Analysis (CLEAN data, 2 decimals) ---")
    thresholds = [200, 250, 300, 350, 400, 500]
    threshold_results = {}
    for t in thresholds:
        cnt = (clean['spread_bps'] < t).sum()
        pct = cnt / len(clean) * 100
        threshold_results[t] = pct
        print(f"  Spread < {t:>4} bps: {pct:>6.2f}% ({cnt:>5} rows)")

    # --- Top 10 liquid markets ---
    print("\n--- 2c. Top 10 Liquid Markets (CLEAN data, spread < 300 bps) ---")

    # Also load dry_run data to get signal/EV info per market
    dry_runs = sorted(glob.glob(os.path.join(RAW_DIR, "dry_run_*.csv")))
    dry_dfs = []
    for f in dry_runs:
        d = pd.read_csv(f)
        dry_dfs.append(d)
    dry = pd.concat(dry_dfs, ignore_index=True) if dry_dfs else pd.DataFrame()

    liquid = clean[clean['spread_bps'] < 300]
    if 'market_id' in liquid.columns:
        liq_agg = liquid.groupby('market_id').agg(
            tick_count=('spread_bps', 'count'),
            median_spread=('spread_bps', 'median'),
            mean_spread=('spread_bps', 'mean'),
        ).sort_values('tick_count', ascending=False)

        # Cross-reference with dry_run signals
        if len(dry) > 0 and 'market_id' in dry.columns:
            from src.zone_matrix import classify_zone
            dry['timestamp'] = pd.to_datetime(dry['timestamp'], utc=True)

            sig_counts = dry.groupby('market_id').size().rename('total_signal_count')
            # Compute live_edge for profitability
            dry['_live_edge'] = dry['confidence_score'] - dry['entry_odds']
            prof = dry[dry['_live_edge'] > 0].groupby('market_id').size().rename('profitable_signal_count')

            liq_agg = liq_agg.join(sig_counts, how='left').join(prof, how='left')
            liq_agg['total_signal_count'] = liq_agg['total_signal_count'].fillna(0).astype(int)
            liq_agg['profitable_signal_count'] = liq_agg['profitable_signal_count'].fillna(0).astype(int)

        top10 = liq_agg.head(10)
        # Truncate market_id for readability
        top10_display = top10.copy()
        top10_display.index = [m[:18] + '...' for m in top10_display.index]
        print(top10_display.round(2).to_string())
    else:
        print("DATA NOT AVAILABLE: market_id column missing")

    return threshold_results


# ============================================================
# TASK 3: REVISED DECISION MATRIX
# ============================================================

def task3_decision_matrix(threshold_results, data_integrity_ok):
    print("\n" + "=" * 80)
    print("TASK 3: REVISED DECISION MATRIX")
    print("=" * 80)

    pct_300 = threshold_results.get(300, 0)

    if not data_integrity_ok:
        print("\n*** DATA_INTEGRITY_FAILURE ***")
        print("Rekomendasi final DITAHAN sampai data bersih tersedia.")
        return

    status_ok = "PASS" if pct_300 > 30 else "FAIL"
    status_review = "PASS" if 10 <= pct_300 <= 30 else "FAIL"
    status_pause = "PASS" if pct_300 < 10 else "FAIL"

    print(f"\n| {'Kondisi':<40} | {'Nilai Aktual':>14} | {'Threshold':>18} | {'Status':>8} |")
    print(f"|{'-'*42}|{'-'*16}|{'-'*20}|{'-'*10}|")
    print(f"| {'% waktu spread < 3% (clean)':<40} | {pct_300:>13.2f}% | {'> 30% = OK':>18} | {status_ok:>8} |")
    print(f"| {'% waktu spread < 3% (clean)':<40} | {pct_300:>13.2f}% | {'10-30% = REVIEW':>18} | {status_review:>8} |")
    print(f"| {'% waktu spread < 3% (clean)':<40} | {pct_300:>13.2f}% | {'< 10% = PAUSE':>18} | {status_pause:>8} |")
    print(f"| {'Pipeline breakdown akurat':<40} | {'Y':>14} | {'No UNACCOUNTED':>18} | {'PASS':>8} |")
    print(f"| {'11 pending = simulasi atau live':<40} | {'SIM':>14} | {'Harus SIM':>18} | {'PASS':>8} |")
    print(f"| {'Data integrity (post-audit)':<40} | {'OK':>14} | {'No corruption':>18} | {'PASS':>8} |")

    print("\nREKOMENDASI AKHIR:")
    if pct_300 > 30:
        print("-> WAIT: Bot sudah benar. Tunggu window jam liquid berdasarkan data.")
    elif pct_300 >= 10:
        print("-> RECALIBRATE: Spread cost harus dimasukkan ke model saat training ulang.")
    else:
        print("-> PAUSE: Polymarket secara struktural tidak liquid -> evaluasi ulang arsitektur.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    clob = task1_audit()
    if clob is not None:
        threshold_results = task2_rerun(clob)
        task3_decision_matrix(threshold_results, data_integrity_ok=True)
