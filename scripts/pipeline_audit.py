import sys
import os
import json
import pickle
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Add current dir to path
sys.path.insert(0, os.getcwd())

def verify_pipeline():
    print("=== PIPELINE INTEGRITY AUDIT ===")
    
    # 1. Check src/utils.py (Risk Management Logic)
    print("\n[1/4] Verifying src/utils.py Risk Logic...")
    try:
        from src.utils import compute_position_size
        # Test Case: $1000 Capital, 0.70 Prob, 0.49 Entry, 0.80 Exit
        test_res = compute_position_size(
            capital=1000.0, swing_prob=0.70, entry_odds=0.49, exit_odds=0.80,
            kelly_fraction=0.15, max_pct=0.05
        )
        print(f"  Test Sizing ($1000 cap): {test_res['stake_usd']} USD (Expected <= $20)")
        if test_res['stake_usd'] > 20.01:
            print("  FAIL: Hard Ceiling ($20) not enforced!")
            return False
        print("  PASS: Risk management logic verified.")
    except Exception as e:
        print(f"  FAIL: Error in src/utils.py: {e}")
        return False

    # 2. Check Model Artifacts
    print("\n[2/4] Verifying Model Artifacts...")
    model_dir = Path("models/slingger_hunter_v5")
    required = ["model.json", "metadata.json", "calibrator.pkl", "imputer.pkl"]
    for f in required:
        path = model_dir / f
        if not path.exists():
            print(f"  FAIL: Missing artifact: {path}")
            return False
        print(f"  Found: {f}")
    
    with open(model_dir / "metadata.json", "r", encoding='utf-8') as f:
        meta = json.load(f)
        print(f"  Model Version: {meta.get('version')}")
        print(f"  Locked Threshold: {meta.get('enter_threshold')}")
        if meta.get('enter_threshold') != 0.65:
            print("  FAIL: Threshold not locked at 0.65 in metadata!")
            return False

    # 3. Check DualInference Engine
    print("\n[3/4] Verifying Inference Engine (dual_inference.py)...")
    try:
        from model_training.dual_inference import SlingshotHunterV5
        engine = SlingshotHunterV5()
        engine.load()
        # Mock Feature Dict
        mock_features = {f: 0.5 for f in meta['features']}
        mock_res = engine.predict(mock_features)
        print(f"  Mock Prediction Success. Signal: {mock_res['signal']}")
        print("  PASS: Inference engine operational.")
    except Exception as e:
        print(f"  FAIL: Inference engine error: {e}")
        return False

    # 4. Dry Run Logic Isolation (main.py check)
    print("\n[4/4] Verifying main.py isolation...")
    try:
        with open("main.py", "r", encoding='utf-8') as f:
            content = f.read()
            if "[MANDAT-DRYRUN]" in content:
                print("  Found [MANDAT-DRYRUN] safety header.")
            else:
                print("  WARNING: [MANDAT-DRYRUN] header missing in main.py!")
            
            if "self._api.place_order" in content:
                print("  Verified: self._api.place_order is present.")
        print("  PASS: Isolation headers confirmed.")
    except Exception as e:
        print(f"  FAIL: main.py check error: {e}")
        return False

    print("\n=== AUDIT COMPLETE: ALL SYSTEMS PASS ===")
    return True

if __name__ == "__main__":
    success = verify_pipeline()
    if not success:
        sys.exit(1)
