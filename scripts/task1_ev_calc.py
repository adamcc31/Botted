import json, sqlite3, csv
from pathlib import Path

TRADE_LOG_PATH = "scripts/output/full_pipeline_trades.csv"

trades = []
with open(TRADE_LOG_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        trades.append({
            'timestamp': row['entry_time'],
            'pnl': float(row['pnl']),
            'yes_bid': float(row['entry_odds'])
        })

wins   = [t for t in trades if t['pnl'] > 0]
losses = [t for t in trades if t['pnl'] <= 0]

if not trades:
    print("ERROR: No trades loaded. Cannot compute EV.")
else:
    win_rate     = len(wins) / len(trades)
    loss_rate    = len(losses) / len(trades)
    avg_win      = sum(t['pnl'] for t in wins)  / len(wins)  if wins   else 0
    avg_loss     = abs(sum(t['pnl'] for t in losses) / len(losses)) if losses else 0
    ev           = (win_rate * avg_win) - (loss_rate * avg_loss)
    profit_factor = (win_rate * avg_win) / (loss_rate * avg_loss) if avg_loss > 0 else float('inf')

    print(f"=== EV CALCULATION REPORT ===")
    print(f"Total Trades    : {len(trades)}")
    print(f"Date Range      : {trades[0]['timestamp']} → {trades[-1]['timestamp']}")
    print(f"Win Rate        : {win_rate:.4f} ({win_rate*100:.2f}%)")
    print(f"Loss Rate       : {loss_rate:.4f} ({loss_rate*100:.2f}%)")
    print(f"Avg Win ($)     : {avg_win:.4f}")
    print(f"Avg Loss ($)    : {avg_loss:.4f}")
    print(f"Profit Factor   : {profit_factor:.4f}")
    print(f"Expected Value  : {ev:.4f} ({ev*100:.2f}%)")
    print(f"")
    print(f"EV TARGET (≥75%): {'PASS ✅' if ev >= 0.75 else 'FAIL ❌'}")
    print(f"")
    print(f"--- BREAKDOWN BY ODDS BUCKET ---")
    for bucket, label in [
        (lambda t: t.get('yes_bid',1) < 0.35, "UNDERDOG_<35%"),
        (lambda t: 0.35 <= t.get('yes_bid',0) < 0.50, "MID_35-50%"),
        (lambda t: t.get('yes_bid',0) >= 0.65, "FAVORITE_>65%"),
    ]:
        b_trades = [t for t in trades if bucket(t)]
        if b_trades:
            b_wins = [t for t in b_trades if t['pnl'] > 0]
            b_wr   = len(b_wins)/len(b_trades)
            b_ev   = (b_wr * avg_win) - ((1-b_wr) * avg_loss)
            print(f"  {label}: n={len(b_trades)}, WR={b_wr:.2%}, EV={b_ev:.4f}")
