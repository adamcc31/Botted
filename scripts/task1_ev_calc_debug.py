import glob, csv, json

files = glob.glob('dataset/raw/trades_*.csv')
trades = []
source_counts = {}

for fpath in files:
    fname = fpath.split('\\')[-1].split('/')[-1]
    count = 0
    with open(fpath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Reconstruct exactly what we need
            # pnl_usd column is usually the $ profit
            trades.append(row)
            count += 1
    source_counts[fname] = count

print("=== EV SCRIPT DEBUG INFO ===")
if trades:
    print(f"trades[0] full row:\n{json.dumps(trades[0], indent=2)}")
    print("\npnl unit:")
    print("The CSV has 'pnl_usd' which represents dollar profit, and 'pnl_pct_capital'.")
    
    print("\nTrades per source file:")
    for k, v in source_counts.items():
        print(f"  {k}: {v} trades")
else:
    print("No trades found in raw files.")

# Re-evaluate EV on THESE trades instead of full_pipeline_trades.csv
wins   = [t for t in trades if float(t.get('pnl_usd', 0)) > 0]
losses = [t for t in trades if float(t.get('pnl_usd', 0)) <= 0]

if trades:
    win_rate = len(wins) / len(trades)
    loss_rate = len(losses) / len(trades)
    avg_win = sum(float(t['pnl_usd']) for t in wins) / len(wins) if wins else 0
    avg_loss = abs(sum(float(t['pnl_usd']) for t in losses) / len(losses)) if losses else 0
    ev = (win_rate * avg_win) - (loss_rate * avg_loss)
    print(f"\nRe-evaluated EV on actual trades: {ev:.4f} (${ev:.4f} per trade)")
