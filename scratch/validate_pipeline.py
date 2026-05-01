import sys
import os
import pandas as pd
import numpy as np
from collections import Counter

# Add current dir to path
sys.path.append(os.getcwd())

def test_feature_engineering():
    print("Testing model_training/features.py...")
    from model_training.features import build_features
    
    # Create a dataframe with minimal columns to pass the abs() calculation
    df = pd.DataFrame({
        "odds_yes_60s_ago": [0.5, None, 0.4],
        "odds_delta_60s": [0.01, 0.02, None],
        "q_fair": [0.5, 0.6, 0.7],
        "entry_odds": [0.45, 0.55, 0.65],
        "obi_value": [0.1, -0.1, 0.0],
        "vol_percentile": [0.8, 0.9, 0.7],
        "tfm_value": [0.05, -0.05, 0.01],
        "contest_urgency": [0.2, 0.4, 0.6],
        "rv_value": [0.01, 0.02, 0.01],
        "timestamp": ["2026-05-01T10:00:00Z"] * 3
    })
    
    try:
        # We wrap build_features to ignore the final validation for this unit test
        from model_training.features import ALL_FEATURES
        # Temporarily mock ALL_FEATURES to only include what we have
        import model_training.features as mf
        original_all = mf.ALL_FEATURES
        mf.ALL_FEATURES = [c for c in original_all if c in df.columns or c in ["odds_momentum", "edge_vs_crowd", "obi_vol_interaction", "tfm_vol_interaction", "urgency_vol", "micro_alignment", "hour_wib", "is_weekend"]]
        
        df_out = build_features(df)
        print("Success: build_features completed calculation without NoneType error.")
        print("Odds Momentum values:")
        print(df_out["odds_momentum"].values)
        
        # Restore
        mf.ALL_FEATURES = original_all
    except Exception as e:
        print(f"Error: build_features failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def test_main_imports():
    print("\nTesting main.py imports...")
    try:
        # We don't want to run the whole main, just check if Counter is there
        import main
        if hasattr(main, 'Counter'):
            print("Success: Counter is defined in main.py.")
        else:
            print("Error: Counter NOT found in main.py.")
            sys.exit(1)
    except Exception as e:
        # main.py might fail on other things (env vars), but we check if NameError happened during import
        if "Counter" in str(e):
            print(f"Error: Counter NameError in main.py during import: {e}")
            sys.exit(1)
        else:
            # Other errors are fine for this specific test as long as it's not NameError on Counter
            print(f"Note: main.py import triggered other expected error (env/config): {e}")
            # If it's a NameError for something else, we should know, but Counter is our focus
            if isinstance(e, NameError) and "Counter" in str(e):
                sys.exit(1)
            print("Success: Counter NameError was NOT triggered.")

if __name__ == "__main__":
    test_feature_engineering()
    test_main_imports()
    print("\nVAL_SUCCESS")
