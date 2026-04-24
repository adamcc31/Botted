import pandas as pd

df = pd.read_parquet('data/processed/merged_training_features.parquet')
df_markets = pd.read_parquet('data/raw/polymarket_markets.parquet')

# Isolate mismatches
mismatches = df[df['label_match'] == False]
print(f"Total Mismatches: {len(mismatches)}")

# Pick 10 random mismatches
sample_mismatches = mismatches.sample(10, random_state=42)

for idx, row in sample_mismatches.iterrows():
    mid = row['market_id']
    market_row = df_markets[df_markets['market_id'] == mid].iloc[0]
    
    print("\n" + "="*50)
    print(f"Market ID: {mid}")
    print(f"Question: {market_row.get('question', 'N/A')}")
    print(f"Strike Input: {market_row.get('strike_price', 'N/A')}")
    print(f"Computed Strike (feature): {row['strike_distance_pct']}? (Hard to reverse)")
    
    # What was computed?
    # Computed label logic: label_match = (computed == outcome_binary)
    # If outcome_binary == 1 and label_match == 0 -> computed was 0 (pyth < strike)
    print(f"Polymarket Outcome (YES/1 or NO/0): {row['label']}")
    print(f"Is Imputed: {row['is_imputed_strike']}")
    print("="*50)

