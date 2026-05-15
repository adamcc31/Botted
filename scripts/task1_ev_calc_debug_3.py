import glob, csv, json, statistics

files = glob.glob('dataset/raw/trades_*.csv')
trades = []

for fpath in files:
    with open(fpath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(row)

ev_values = []
for t in trades:
    pnl = float(t.get('pnl_usd', 0))
    bet = float(t.get('bet_size_usd', 1))
    if bet > 0:
        ev_values.append(pnl / bet)

print(f"Total trades live (dataset/raw/trades_*.csv): {len(trades)}")
if ev_values:
    print(f"EV per trade (mean pnl/bet) : {statistics.mean(ev_values):.4f}")
    print(f"Min (pnl/bet)               : {min(ev_values):.4f}")
    print(f"Max (pnl/bet)               : {max(ev_values):.4f}")
    print(f"Std Deviation (pnl/bet)     : {statistics.stdev(ev_values):.4f}")
else:
    print("No trades with valid bet size found.")
