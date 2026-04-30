"""
smoke_test_inference.py — Smoke test for model_training/inference.py
"""

import sys
import logging
from pathlib import Path

# Add project root to sys.path so we can import model_training
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.INFO)

from model_training.inference import XGBoostGate

def test_inference_gate():
    gate = XGBoostGate()
    model_dir = BASE_DIR / "models" / "alpha_v1"
    
    print(f"Loading model from {model_dir}...")
    gate.load_model(model_dir)
    
    assert gate.is_loaded, "Gate failed to load model"
    print(f"Model version: {gate.model_version}")
    
    # Mock RAW features as output by the bot
    raw_features = {
        "OBI": 0.5,
        "VAM": 0.1,
        "RV": 0.05,
        "depth_ratio": 2.5,
        "price_vs_ema20": 0.01,
        "hour_sin": 0.5,
        "hour_cos": 0.5,
        "dow_sin": 0.5,
        "dow_cos": 0.5,
        "clob_yes_mid": 0.25,
        "clob_yes_spread": 0.02,
        "clob_no_spread": 0.02,
        "market_vig": 0.04,
        "current_btc_price": 60000,
        "strike_price": 59000,
        "TTR_minutes": 30,
        # TFM_raw features if used
        "taker_buy_vol_60s": 1000,
        "taker_sell_vol_60s": 500,
        "total_vol_60s": 1500,
    }
    
    print("\nEvaluating signal...")
    result = gate.evaluate_signal(raw_features, entry_odds=0.25)
    
    print("Decision:", result["decision"])
    print("Reason:", result["reason"])
    print("P(WIN):", result["p_win"])
    print("EV:", result["ev"])
    print("Confidence:", result["confidence"])
    
    assert result["decision"] in ["PASS", "REJECT"], "Invalid decision"
    print("\nSmoke test passed!")

if __name__ == "__main__":
    test_inference_gate()
