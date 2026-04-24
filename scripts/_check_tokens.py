import pandas as pd

mkt = pd.read_parquet("data/raw/polymarket_markets.parquet")
valid = mkt[mkt["strike_price"].notna() & mkt["outcome_binary"].notna() & mkt["t_resolution_epoch_ms"].notna()]

empty_tokens = (valid["yes_token_id"] == "") | valid["yes_token_id"].isna()
print(f"Valid markets: {len(valid)}")
print(f"With empty yes_token_id: {empty_tokens.sum()}")
print(f"With valid yes_token_id: {(~empty_tokens).sum()}")

has_tokens = valid[~empty_tokens]
if len(has_tokens) > 0:
    sample = has_tokens.iloc[0]
    token = sample["yes_token_id"]
    mid = sample["market_id"]
    print(f"Sample token ID: {token[:80]}")
    print(f"Sample market_id: {mid}")
    print(f"Token ID length: {len(token)}")
else:
    print("NO markets have token IDs!")
    print(f"All yes_token_id values: {valid['yes_token_id'].unique()[:5]}")

# Check training data market IDs overlap
train = pd.read_parquet("data/processed/merged_training_features.parquet")
train_mids = set(train["market_id"].unique())
mkt_mids = set(valid["market_id"].unique())
overlap = train_mids & mkt_mids
print(f"\nTraining market IDs: {len(train_mids)}")
print(f"Overlap with market DB: {len(overlap)}")

# For the overlap, check token availability
overlap_markets = valid[valid["market_id"].isin(overlap)]
ol_empty = (overlap_markets["yes_token_id"] == "") | overlap_markets["yes_token_id"].isna()
print(f"Overlap markets with tokens: {(~ol_empty).sum()} / {len(overlap_markets)}")
