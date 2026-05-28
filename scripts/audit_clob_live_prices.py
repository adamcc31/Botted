import pandas as pd

dry_run_path = "dataset/raw/audit_16h/dry_run_2026-05-18_094841.csv"

try:
    df = pd.read_csv(dry_run_path)
    clob_live = df[df['entry_odds_source'] == 'CLOB_LIVE']
    print(f"Total CLOB_LIVE rows: {len(clob_live)}")
    print("\nOdds Yes statistics:")
    print(clob_live['odds_yes'].describe())
    print("\nOdds No statistics:")
    print(clob_live['odds_no'].describe())
    
    # Calculate orderbook spread from dry run (since entry_odds is stored)
    # entry_odds is the bid/ask odds the bot was looking at
    print("\nOdds Yes value counts:")
    print(clob_live['odds_yes'].value_counts().head(10))
    print("\nOdds No value counts:")
    print(clob_live['odds_no'].value_counts().head(10))
except Exception as e:
    print(f"Error: {e}")
