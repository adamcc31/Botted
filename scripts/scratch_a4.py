import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

df = pd.read_csv('scripts/output/predator_trades.csv')
a4 = df[df['zone_id'] == 'A4'].copy()

print("A4 TRADES DEEP DIVE")
wins = a4[a4['outcome'] == 'WIN']
losses = a4[a4['outcome'] == 'LOSS']
print(f"Total: {len(a4)}, Wins: {len(wins)}, Losses: {len(losses)}")
print(f"WR: {len(wins)/len(a4)*100:.1f}%")
print(f"Avg bet WIN: ${wins['bet_size'].mean():.2f}")
print(f"Avg bet LOSS: ${losses['bet_size'].mean():.2f}")
print(f"Total profit from wins: ${wins['pnl'].sum():.2f}")
print(f"Total loss from losses: ${losses['pnl'].sum():.2f}")
print()
print("KEY PROBLEM: Losses have LARGER bets than wins (Kelly scales with capital)")
print("When capital is high -> big bets. A loss wipes the gains from many small wins.")
print()
print("A4 trades chronological:")
for _, r in a4.iterrows():
    print(f"  #{int(r['trade_id']):>4} | Bet ${r['bet_size']:>6.0f} | Odds {r['entry_odds']:.3f} | {r['outcome']:>4} | PnL ${r['pnl']:>+8.2f} | Cap ${r['capital_after']:>8.2f}")

print()
print("=" * 70)
print("ALL ZONES - Bet size distribution")
for zone in sorted(df['zone_id'].unique()):
    z = df[df['zone_id'] == zone]
    zw = z[z['outcome'] == 'WIN']
    zl = z[z['outcome'] == 'LOSS']
    avg_w = zw['bet_size'].mean() if len(zw) > 0 else 0
    avg_l = zl['bet_size'].mean() if len(zl) > 0 else 0
    ratio = avg_l / avg_w if avg_w > 0 else 0
    print(f"  {zone}: AvgWinBet=${avg_w:.1f}, AvgLossBet=${avg_l:.1f}, LossRatio={ratio:.2f}x")
