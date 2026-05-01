import sys
sys.stdout.reconfigure(encoding='utf-8')

zones = {
    'A1': {'wr': 0.276, 'odds': 0.20, 'n': 29, 'name': 'TTR 3-5m | $30-60 | 0.15-0.25'},
    'A2': {'wr': 0.297, 'odds': 0.20, 'n': 37, 'name': 'TTR 1.5-3m | $15-30 | 0.15-0.25'},
    'A3': {'wr': 0.400, 'odds': 0.33, 'n': 25, 'name': 'TTR 3-5m | $30-60 | 0.25-0.40'},
    'A4': {'wr': 0.750, 'odds': 0.625, 'n': 44, 'name': 'TTR 1.5-3m | $15-30 | 0.55-0.70'},
    'A5': {'wr': 0.722, 'odds': 0.625, 'n': 18, 'name': 'TTR 1.5-3m | <$15 | 0.55-0.70'},
    'A6': {'wr': 1.000, 'odds': 0.80, 'n': 30, 'name': 'TTR 3-5m | >$100 | >0.70'},
    'A7': {'wr': 0.387, 'odds': 0.33, 'n': 75, 'name': 'TTR 1.5-3m | <$15 | 0.25-0.40'},
    'A8': {'wr': 0.708, 'odds': 0.625, 'n': 65, 'name': 'TTR 3-5m | $30-60 | 0.55-0.70'},
}

print("=" * 100)
print("FULL KELLY FRACTION + EV PER DOLLAR ANALYSIS")
print("=" * 100)

header = f"{'Zone':<5} {'N':>4} {'WR':>7} {'Odds':>6} {'Payout':>8} {'FullKelly':>10} {'EV/$1':>8} {'$5 Win':>8} {'$5 EV':>8}"
print(header)
print("-" * 100)

total_weighted_ev = 0
total_n = 0

for z, d in zones.items():
    b = (1.0 / d['odds']) - 1
    p = d['wr']
    kelly = (p * b - (1 - p)) / b
    ev_per_dollar = p * b - (1 - p)
    bet = 5
    win_profit = bet * b
    ev_bet = bet * ev_per_dollar
    total_weighted_ev += ev_per_dollar * d['n']
    total_n += d['n']

    print(f"{z:<5} {d['n']:>4} {p:>6.1%} {d['odds']:>5.2f} {b:>7.2f}x {kelly:>9.1%} {ev_per_dollar:>+7.3f} {win_profit:>+7.2f} {ev_bet:>+7.2f}")

blended_ev = total_weighted_ev / total_n
print(f"\nBlended EV per $1 (weighted by N): +{blended_ev:.3f}")

print("\n" + "=" * 100)
print("AGGRESSIVE COMPOUND PROJECTION")
print("Full Kelly | Multi-Zone | 12-18 trades/day | Intra-day compounding")
print("=" * 100)

# Conservative: 15% daily, Moderate: 25%, Aggressive: 35%
for scenario, daily_pct in [("Conservative (15%/day)", 0.15), ("Moderate (25%/day)", 0.25), ("Aggressive (35%/day)", 0.35)]:
    print(f"\n--- {scenario} ---")
    c = 100
    for day in range(1, 22):
        c *= (1 + daily_pct)
        if day in [1, 2, 3, 5, 7, 10, 14, 21]:
            print(f"  Day {day:>2}: ${c:>10,.2f}")

print("\n" + "=" * 100)
print("ZONE A4 DEEP DIVE (The Crown Jewel)")
print("=" * 100)
p = 0.75
odds = 0.625
b = (1.0 / odds) - 1
kelly = (p * b - (1 - p)) / b
print(f"Win Rate: {p:.0%}")
print(f"Entry Odds: {odds}")
print(f"Payout multiplier: {b:.3f}x (win ${1/odds:.2f} per $1 risked)")
print(f"Full Kelly: {kelly:.1%}")
print(f"At $100 capital, Full Kelly bet = ${100 * kelly:.2f}")
print(f"Win scenario: +${100 * kelly * b:.2f}")
print(f"Lose scenario: -${100 * kelly:.2f}")
print(f"EV per trade: +${100 * kelly * (p * b - (1-p)):.2f}")
print(f"\nWith 5 trades/day in this zone alone:")
daily_ev = 5 * 100 * kelly * (p * b - (1 - p))
print(f"Daily EV (Day 1): +${daily_ev:.2f} ({daily_ev:.0f}% return!)")

# But we need to account for compounding within the day
print("\n--- Intra-day compound simulation (Zone A4 only, 5 trades) ---")
import random
random.seed(42)

sims = 10000
results = []
for _ in range(sims):
    cap = 100
    for trade in range(5):
        bet = cap * min(kelly, 0.30)  # cap at 30% per trade
        if random.random() < p:
            cap += bet * b
        else:
            cap -= bet
    results.append(cap)

results.sort()
print(f"  Median end-of-day capital: ${results[len(results)//2]:.2f}")
print(f"  25th percentile: ${results[len(results)//4]:.2f}")
print(f"  75th percentile: ${results[3*len(results)//4]:.2f}")
print(f"  Worst 5%: ${results[len(results)//20]:.2f}")
print(f"  Best 5%: ${results[19*len(results)//20]:.2f}")
print(f"  Ruin (<$10): {sum(1 for r in results if r < 10)/sims:.1%}")

print("\n--- Full multi-zone 7-day Monte Carlo (10k sims) ---")
all_zones_for_sim = [
    {'p': 0.276, 'odds': 0.20, 'freq': 2},  # A1
    {'p': 0.297, 'odds': 0.20, 'freq': 3},  # A2
    {'p': 0.400, 'odds': 0.33, 'freq': 2},  # A3
    {'p': 0.750, 'odds': 0.625, 'freq': 4}, # A4
    {'p': 0.722, 'odds': 0.625, 'freq': 2}, # A5
    {'p': 0.708, 'odds': 0.625, 'freq': 3}, # A8
]

sims = 10000
day7_results = []
for _ in range(sims):
    cap = 100
    for day in range(7):
        for zone in all_zones_for_sim:
            for _ in range(zone['freq']):
                b_z = (1.0 / zone['odds']) - 1
                k_z = (zone['p'] * b_z - (1 - zone['p'])) / b_z
                bet = cap * min(k_z, 0.25)  # 25% cap per trade
                bet = max(1, min(bet, cap * 0.25))
                if random.random() < zone['p']:
                    cap += bet * b_z
                else:
                    cap -= bet
                if cap < 1:
                    cap = 0
                    break
            if cap < 1:
                break
        if cap < 1:
            break
    day7_results.append(cap)

day7_results.sort()
print(f"  Starting capital: $100")
print(f"  Median Day 7: ${day7_results[len(day7_results)//2]:,.2f}")
print(f"  25th percentile: ${day7_results[len(day7_results)//4]:,.2f}")
print(f"  75th percentile: ${day7_results[3*len(day7_results)//4]:,.2f}")
print(f"  Mean: ${sum(day7_results)/len(day7_results):,.2f}")
print(f"  Best 10%: ${day7_results[9*len(day7_results)//10]:,.2f}")
print(f"  Ruin (<$5): {sum(1 for r in day7_results if r < 5)/sims:.1%}")
print(f"  > $500: {sum(1 for r in day7_results if r > 500)/sims:.1%}")
print(f"  > $1000: {sum(1 for r in day7_results if r > 1000)/sims:.1%}")
print(f"  > $5000: {sum(1 for r in day7_results if r > 5000)/sims:.1%}")
