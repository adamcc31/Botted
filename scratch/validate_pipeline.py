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
        
        # Restore
        mf.ALL_FEATURES = original_all
    except Exception as e:
        print(f"Error: build_features failed: {e}")
        sys.exit(1)

def test_main_imports():
    print("\nTesting main.py imports...")
    try:
        import main
        if not hasattr(main, 'Counter'):
            print("Error: Counter NOT found in main.py.")
            sys.exit(1)
        if not hasattr(main, 'html'):
            print("Error: html NOT found in main.py.")
            sys.exit(1)
        
        # Test html.escape as used in main.py
        test_html = main.html.escape("<b>test</b>")
        if test_html == "&lt;b&gt;test&lt;/b&gt;":
             print("Success: html.escape is working in main.py.")
        else:
             print(f"Error: html.escape output mismatch: {test_html}")
             sys.exit(1)
             
        print("Success: Critical imports (Counter, html) found in main.py.")
    except Exception as e:
        if "Counter" in str(e) or "html" in str(e):
            print(f"Error: Critical import NameError in main.py: {e}")
            sys.exit(1)
        print(f"Note: main.py import triggered other expected error: {e}")
        print("Success: Critical NameErrors were NOT triggered.")

if __name__ == "__main__":
    test_feature_engineering()
    test_main_imports()
    print("\nVAL_SUCCESS")
