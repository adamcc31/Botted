import pandas as pd

dry_run_path = "dataset/raw/audit_16h/dry_run_2026-05-18_094841.csv"

try:
    df = pd.read_csv(dry_run_path)
    print("=== ENTRY ODDS SOURCE FREQUENCY ===")
    if 'entry_odds_source' in df.columns:
        print(df['entry_odds_source'].value_counts(dropna=False))
    else:
        print("Column 'entry_odds_source' not found!")
except Exception as e:
    print(f"Error: {e}")
