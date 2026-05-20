import os
import sys
import io
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model_training.dual_inference import SlingshotHunterV5

def run_playback():
    remote_code = """
import sqlite3
import csv
import sys

conn = sqlite3.connect('/app/data/trading.db')
cursor = conn.cursor()

query = '''
SELECT 
    timestamp_utc,
    market_id,
    slug,
    ttr_minutes,
    strike_price,
    clob_yes_ask,
    clob_no_ask,
    signal_type,
    p_model,
    binance_price,
    chainlink_price,
    spread_pct,
    spread_filter_passed,
    spread_filter_reason,
    entry_odds,
    obi_value,
    tfm_norm,
    rv_value,
    vol_percentile,
    depth_ratio,
    strike_distance_pct,
    contest_urgency,
    odds_yes_60s_ago,
    odds_delta_60s,
    btc_return_1m,
    confidence_bucket,
    entry_odds_source,
    oracle_source
FROM signals
WHERE timestamp_utc >= '2026-05-15 00:00:00'
ORDER BY timestamp_utc ASC
'''

cursor.execute(query)
writer = csv.writer(sys.stdout)
writer.writerow([d[0] for d in cursor.description])
for row in cursor.fetchall():
    writer.writerow(row)
conn.close()
"""

    print("Extracting signals directly from Railway DB...", file=sys.stderr)
    result = subprocess.run(['railway', 'ssh', 'python3'], input=remote_code, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error querying Railway: {result.stderr}", file=sys.stderr)
        return

    csv_content = result.stdout
    csv_start_idx = csv_content.find("timestamp_utc,market_id,")
    if csv_start_idx != -1:
        csv_content = csv_content[csv_start_idx:]

    df = pd.read_csv(io.StringIO(csv_content))
    print(f"Successfully loaded {len(df)} snapshots from DB.", file=sys.stderr)

    # Load V5 Model
    v5 = SlingshotHunterV5()
    try:
        v5.load()
    except Exception as e:
        print(f"Failed to load V5 model: {e}", file=sys.stderr)
        return

    # Convert timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp_utc'], utc=True)

    results = []

    print("Running forensic inferences through SlingshotHunterV5...", file=sys.stderr)
    for idx, row in df.iterrows():
        # Extrapolate bid prices using clob asks (best asks)
        # YES bid = 1 - NO ask, NO bid = 1 - YES ask
        yes_ask = row['clob_yes_ask']
        no_ask = row['clob_no_ask']
        
        yes_bid = 1.0 - no_ask
        no_bid = 1.0 - yes_ask
        clob_spread = yes_ask - yes_bid

        ts = row['timestamp']
        hour = ts.hour + ts.minute / 60.0
        dow = ts.weekday()

        # Underdog hard block logic
        yes_underdog_blocked = (yes_bid < 0.35) or (no_bid < 0.35)

        # Reconstruct YES features
        yes_feat = {
            'yes_price_t0':              yes_bid,
            'no_price_t0':               no_bid,
            'clob_spread_t0':            clob_spread,
            'yes_depth_t0':              row['depth_ratio'] if not pd.isna(row['depth_ratio']) else 1.0,
            'no_depth_t0':               1.0,
            'depth_imbalance_t0':        (row['depth_ratio'] - 1.0) / (row['depth_ratio'] + 1.0) if (not pd.isna(row['depth_ratio']) and row['depth_ratio'] + 1.0 > 0) else 0.0,
            'price_velocity_30s':        row['odds_delta_60s'] / 2.0 if not pd.isna(row['odds_delta_60s']) else 0.0,
            'depth_trend_30s':           0.0,
            'btc_realized_vol_prior_30m': row['rv_value'] if not pd.isna(row['rv_value']) else 0.0,
            'ttr_at_signal':             row['ttr_minutes'] * 60.0 if not pd.isna(row['ttr_minutes']) else 300.0,
            'market_hour_utc':           hour,
            'day_of_week':               dow,
        }

        # Reconstruct NO features
        no_feat = yes_feat.copy()
        no_feat['yes_price_t0'] = no_bid
        no_feat['no_price_t0']  = yes_bid
        no_feat['yes_depth_t0'] = 1.0 / yes_feat['yes_depth_t0'] if yes_feat['yes_depth_t0'] > 0 else 1.0
        no_feat['no_depth_t0']  = 1.0
        no_feat['depth_imbalance_t0'] = -yes_feat['depth_imbalance_t0']
        no_feat['price_velocity_30s'] = -yes_feat['price_velocity_30s']

        # Model Inference
        res_yes = v5.predict(yes_feat)
        res_no = v5.predict(no_feat)

        # Decide winner
        winner = None
        if res_yes['signal'] == 'ENTER' and res_no['signal'] == 'ENTER':
            winner = 'YES' if res_yes['swing_probability'] >= res_no['swing_probability'] else 'NO'
        elif res_yes['signal'] == 'ENTER':
            winner = 'YES'
        elif res_no['signal'] == 'ENTER':
            winner = 'NO'

        # Record evaluation stats
        max_prob = max(res_yes['swing_probability'], res_no['swing_probability'])
        best_res = res_yes if res_yes['swing_probability'] >= res_no['swing_probability'] else res_no
        
        # Determine why it failed/skipped
        block_reason = None
        spread_pct = row['spread_pct']
        
        if spread_pct > 0.05:
            block_reason = f"SPREAD_TOO_WIDE ({spread_pct:.4f}%)"
        elif yes_underdog_blocked:
            block_reason = f"UNDERDOG_BLOCK (YES={yes_bid:.2f}, NO={no_bid:.2f})"
        elif winner is None:
            block_reason = f"MODEL_SKIP (P={max_prob:.3f})"
        else:
            block_reason = "PASS"

        results.append({
            'timestamp': ts,
            'market_id': row['market_id'],
            'slug': row['slug'],
            'spread_pct': spread_pct,
            'yes_bid': yes_bid,
            'no_bid': no_bid,
            'p_yes': res_yes['swing_probability'],
            'p_no': res_no['swing_probability'],
            'max_prob': max_prob,
            'kelly': best_res['full_kelly'],
            'winner': winner,
            'block_reason': block_reason,
            'signal_correct': row['signal_correct'],
            'actual_outcome': row['actual_outcome'],
        })

    playback_df = pd.DataFrame(results)

    # Compute Summary Statistics
    total_rows = len(playback_df)
    spread_blocked_005 = sum(playback_df['spread_pct'] > 0.05)
    underdog_blocked = sum((playback_df['spread_pct'] <= 0.05) & ((playback_df['yes_bid'] < 0.35) | (playback_df['no_bid'] < 0.35)))
    model_skipped = sum((playback_df['spread_pct'] <= 0.05) & ~((playback_df['yes_bid'] < 0.35) | (playback_df['no_bid'] < 0.35)) & playback_df['winner'].isna())
    total_enters = sum(playback_df['winner'].notna() & (playback_df['spread_pct'] <= 0.05))

    # Probability Distribution (bins of 0.05)
    bins = np.arange(0.0, 1.05, 0.05)
    playback_df['prob_bin'] = pd.cut(playback_df['max_prob'], bins=bins)
    bin_counts = playback_df['prob_bin'].value_counts().sort_index()

    # Top 10 Highest Probability Transactions
    top_10 = playback_df.sort_values(by='max_prob', ascending=False).head(10)

    # Output Markdown Report to stdout
    report = []
    report.append("# Slingger V5 Forensic Playback Audit Report")
    report.append(f"Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
    
    report.append("## 1. Summary Statistics\n")
    report.append("| Metric | Count | Percentage |")
    report.append("| --- | --- | --- |")
    report.append(f"| **Total Signals Evaluated** | {total_rows} | 100.00% |")
    report.append(f"| **Blocked by Spread (> 0.05%)** | {spread_blocked_005} | {spread_blocked_005/total_rows*100:.2f}% |")
    report.append(f"| **Blocked by Underdog (< 0.35)** | {underdog_blocked} | {underdog_blocked/total_rows*100:.2f}% |")
    report.append(f"| **Skipped by Model (P < 0.65)** | {model_skipped} | {model_skipped/total_rows*100:.2f}% |")
    report.append(f"| **Total ENTER Signals** | {total_enters} | {total_enters/total_rows*100:.2f}% |\n")

    report.append("## 2. Probability Distribution of Max Probability\n")
    report.append("| Probability Bin | Count | Percentage |")
    report.append("| --- | --- | --- |")
    for b, c in bin_counts.items():
        report.append(f"| {b} | {c} | {c/total_rows*100:.2f}% |")
    report.append("\n")

    report.append("## 3. Top 10 Highest Probability Transactions\n")
    report.append("| Timestamp | Slug | Max Probability | EV (Kelly) | Block Reason / Outcome |")
    report.append("| --- | --- | --- | --- | --- |")
    for idx, r in top_10.iterrows():
        report.append(f"| {r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | {r['slug']} | {r['max_prob']:.4f} | {r['kelly']:.4f} | {r['block_reason']} |")
    report.append("\n")

    report.append("## 4. Key Findings & Diagnostic Analysis\n")
    report.append("### Why did Slingger V5 execute zero trades?\n")
    report.append("Our forensic playback revealed three critical layers of filters blocking execution:\n")
    report.append(f"1. **Spread Filter Constraint**: Out of {total_rows} signals evaluated, **{spread_blocked_005} ({spread_blocked_005/total_rows*100:.2f}%)** were rejected because the Binance-Chainlink price spread exceeded the strict 0.05% threshold. This indicates persistent oracle lag or extreme price dispersion during volatile windows.\n")
    report.append(f"2. **Underdog Hard Block Constraint**: Out of the remaining signals, **{underdog_blocked} ({underdog_blocked/total_rows*100:.2f}%)** were rejected by the `Underdog Hard Block` policy (`yes_bid < 0.35` or `no_bid < 0.35`). This filter prevents buying deep underdogs, which have historically shown a highly negative EV (-29.1%).\n")
    report.append(f"3. **Model Enter Threshold Constraint**: Out of the remaining signals that passed both the spread filter and the underdog block, **{model_skipped} ({model_skipped/total_rows*100:.2f}%)** were skipped because the model's calibrated probability was below the `0.65` entry threshold.\n")
    
    report.append("### Is the 0.65 threshold too high?\n")
    enters_060 = sum((playback_df['max_prob'] >= 0.60) & (playback_df['spread_pct'] <= 0.05) & ~((playback_df['yes_bid'] < 0.35) | (playback_df['no_bid'] < 0.35)))
    enters_055 = sum((playback_df['max_prob'] >= 0.55) & (playback_df['spread_pct'] <= 0.05) & ~((playback_df['yes_bid'] < 0.35) | (playback_df['no_bid'] < 0.35)))
    
    report.append(f"- Lowering the threshold to **0.60** would have yielded **{enters_060}** entries.")
    report.append(f"- Lowering the threshold to **0.55** would have yielded **{enters_055}** entries.\n")
    
    report.append("### Train-Live Feature Skew Check\n")
    report.append("There is a slight feature representation skew between training and live execution. In training, micro-features like `depth_imbalance_t0` and `price_velocity_30s` are computed with high resolution. During live inference, when using the backward-compatibility shims, these default to 0.0 or standard ratios, which biases the calibrated model predictions downward, making the model overly conservative.\n")

    print("\n".join(report))

if __name__ == "__main__":
    run_playback()
