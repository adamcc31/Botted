"""
smoke_test_xgboost_gate.py — Verify XGBoost gate works in production.

Tests:
  1. Model loads from models/alpha_v1/model.pkl
  2. Feature vector alignment (5 features in correct order)
  3. Isotonic calibration API (no AttributeError)
  4. EV calculation and PASS/REJECT decision
  5. Edge cases: NaN features, missing keys
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Mock structlog for test environment
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

class MockLogger:
    def info(self, *a, **kw): logging.info(str(a[0]) if a else str(kw))
    def warning(self, *a, **kw): logging.warning(str(a[0]) if a else str(kw))
    def error(self, *a, **kw): logging.error(str(a[0]) if a else str(kw))
    def debug(self, *a, **kw): pass

import src.xgboost_gate as xg_module
xg_module.logger = MockLogger()

from src.xgboost_gate import XGBoostGate, GateResult

print("=" * 60)
print("SMOKE TEST: XGBoost Gate Production Inference")
print("=" * 60)

# ── Test 1: Model Load ──
print("\n[1/5] Model Load...")
gate = XGBoostGate()
loaded = gate.load()
assert loaded, "FAIL: Model could not be loaded"
print(f"  PASS — version={gate.version}, loaded={gate.is_loaded}")

# ── Test 2: Feature Alignment ──
print("\n[2/5] Feature Alignment...")
assert gate._feature_names == gate.FEATURE_ORDER, \
    f"FAIL: Feature mismatch: {gate._feature_names} != {gate.FEATURE_ORDER}"
print(f"  PASS — {len(gate.FEATURE_ORDER)} features in correct order")

# ── Test 3: Normal Signal (should PASS or REJECT cleanly) ──
print("\n[3/5] Normal Signal Evaluation...")
mock_signal = {
    "entry_odds": 0.25,          # underdog bet
    "depth_ratio": 2.5,          # healthy order book
    "contest_urgency": 0.03,     # moderate urgency
    "tfm_value": 0.15,           # slight buy pressure
    "obi_vol_interaction": 0.05, # OBI * vol
}
result = gate.evaluate_signal(mock_signal)
assert isinstance(result, GateResult), "FAIL: Result is not GateResult"
assert result.decision in ("PASS", "REJECT"), f"FAIL: Bad decision: {result.decision}"
assert 0.0 <= result.p_win <= 1.0, f"FAIL: p_win out of range: {result.p_win}"
print(f"  PASS — decision={result.decision}, p_win={result.p_win:.4f}, "
      f"p_win_adj={result.p_win_adjusted:.4f}, ev={result.ev:.4f}")
print(f"         reason: {result.reason}")
print(f"         features: {result.feature_vector}")

# ── Test 4: High-odds signal (likely REJECT — no edge on favorites) ──
print("\n[4/5] High-Odds Signal (Favorite)...")
fav_signal = {
    "entry_odds": 0.75,
    "depth_ratio": 1.0,
    "contest_urgency": 0.01,
    "tfm_value": 0.0,
    "obi_vol_interaction": 0.0,
}
result2 = gate.evaluate_signal(fav_signal)
print(f"  decision={result2.decision}, p_win={result2.p_win:.4f}, ev={result2.ev:.4f}")
print(f"  reason: {result2.reason}")

# ── Test 5: Edge Cases (NaN/missing) ──
print("\n[5/5] Edge Cases...")
# Missing keys
missing_signal = {"entry_odds": 0.30}
result3 = gate.evaluate_signal(missing_signal)
assert result3.decision in ("PASS", "REJECT"), "FAIL: Crashed on missing keys"
print(f"  Missing keys: decision={result3.decision}, p_win={result3.p_win:.4f}")

# NaN values
nan_signal = {
    "entry_odds": float("nan"),
    "depth_ratio": float("inf"),
    "contest_urgency": 0.02,
    "tfm_value": None,
    "obi_vol_interaction": 0.0,
}
result4 = gate.evaluate_signal(nan_signal)
assert result4.decision in ("PASS", "REJECT"), "FAIL: Crashed on NaN/inf"
print(f"  NaN/Inf:      decision={result4.decision}, p_win={result4.p_win:.4f}")

print("\n" + "=" * 60)
print("ALL 5 TESTS PASSED — Gate is production-ready")
print("=" * 60)
