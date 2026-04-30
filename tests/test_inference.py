"""
test_inference.py — Tests for model_training/inference.py XGBoostGate.

Tests:
  1. Missing features → HARD REJECT, p_win=0.0
  2. NaN entry_odds → HARD REJECT, p_win=0.0
  3. Complete features → normal PASS/REJECT with valid p_win
  4. No p_win=0.5 fallback ever returned
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pytest

from model_training.inference import XGBoostGate


MODEL_DIR = BASE_DIR / "models" / "alpha_v1"


@pytest.fixture(scope="module")
def gate():
    g = XGBoostGate()
    g.load_model(MODEL_DIR)
    return g


def _complete_features() -> dict:
    """Return a dict with ALL RAW_FEATURES populated."""
    return {
        "obi_value": 0.15,
        "tfm_value": 0.10,
        "depth_ratio": 2.5,
        "obi_tfm_product": 0.015,
        "obi_tfm_alignment": 1.0,
        "rv_value": 0.05,
        "vol_percentile": 0.60,
        "strike_distance_pct": 0.02,
        "contest_urgency": 0.03,
        "ttr_seconds": 180,
        "odds_yes": 0.25,
        "odds_no": 0.75,
        "entry_odds": 0.25,
        "odds_yes_60s_ago": 0.24,
        "odds_delta_60s": 0.01,
        "spread_pct": 0.005,
        "btc_return_1m": 0.001,
        "confidence_score": 0.55,
        "timestamp": "2026-04-30T12:00:00+00:00",
    }


class TestHardRejectMissingFeatures:
    """TASK 2: Missing critical features → REJECT, p_win=0.0"""

    def test_missing_entry_odds(self, gate):
        feats = _complete_features()
        del feats["entry_odds"]
        result = gate.evaluate_signal(feats, entry_odds=0.25)
        assert result["decision"] == "REJECT"
        assert result["p_win"] == 0.0
        assert "entry_odds" in result["missing_features"]

    def test_missing_obi_value(self, gate):
        feats = _complete_features()
        del feats["obi_value"]
        result = gate.evaluate_signal(feats, entry_odds=0.25)
        assert result["decision"] == "REJECT"
        assert result["p_win"] == 0.0

    def test_missing_contest_urgency(self, gate):
        feats = _complete_features()
        del feats["contest_urgency"]
        result = gate.evaluate_signal(feats, entry_odds=0.25)
        assert result["decision"] == "REJECT"
        assert result["p_win"] == 0.0

    def test_missing_depth_ratio(self, gate):
        feats = _complete_features()
        del feats["depth_ratio"]
        result = gate.evaluate_signal(feats, entry_odds=0.25)
        assert result["decision"] == "REJECT"
        assert result["p_win"] == 0.0

    def test_missing_tfm_value(self, gate):
        feats = _complete_features()
        del feats["tfm_value"]
        result = gate.evaluate_signal(feats, entry_odds=0.25)
        assert result["decision"] == "REJECT"
        assert result["p_win"] == 0.0

    def test_missing_vol_percentile(self, gate):
        feats = _complete_features()
        del feats["vol_percentile"]
        result = gate.evaluate_signal(feats, entry_odds=0.25)
        assert result["decision"] == "REJECT"
        assert result["p_win"] == 0.0

    def test_missing_all_critical(self, gate):
        result = gate.evaluate_signal({}, entry_odds=0.25)
        assert result["decision"] == "REJECT"
        assert result["p_win"] == 0.0
        assert len(result["missing_features"]) == 6  # 6 raw required features


class TestHardRejectNaN:
    """NaN values in critical features → REJECT, p_win=0.0"""

    def test_nan_entry_odds(self, gate):
        feats = _complete_features()
        feats["entry_odds"] = float("nan")
        result = gate.evaluate_signal(feats, entry_odds=0.25)
        assert result["decision"] == "REJECT"
        assert result["p_win"] == 0.0

    def test_nan_tfm_value(self, gate):
        feats = _complete_features()
        feats["tfm_value"] = float("nan")
        result = gate.evaluate_signal(feats, entry_odds=0.25)
        assert result["decision"] == "REJECT"
        assert result["p_win"] == 0.0

    def test_none_depth_ratio(self, gate):
        feats = _complete_features()
        feats["depth_ratio"] = None
        result = gate.evaluate_signal(feats, entry_odds=0.25)
        assert result["decision"] == "REJECT"
        assert result["p_win"] == 0.0


class TestCompleteFeatures:
    """Complete features → valid prediction (no p=0.5 fallback)"""

    def test_complete_features_returns_valid_p_win(self, gate):
        feats = _complete_features()
        result = gate.evaluate_signal(feats, entry_odds=0.25)
        assert result["decision"] in ("PASS", "REJECT")
        assert result["p_win"] != 0.5, "p_win=0.5 fallback detected — FATAL"
        assert 0.0 <= result["p_win"] <= 1.0
        assert np.isfinite(result["ev"])

    def test_no_fallback_anywhere(self, gate):
        """Sweep multiple entry_odds — should never hit FALLBACK path"""
        feats = _complete_features()
        for odds in [0.10, 0.25, 0.50, 0.75, 0.90]:
            feats["entry_odds"] = odds
            feats["odds_yes"] = odds
            result = gate.evaluate_signal(feats, entry_odds=odds)
            assert result["confidence"] != "FALLBACK", (
                f"FALLBACK at entry_odds={odds} — blind execution risk"
            )
            assert result["confidence"] != "REJECTED", (
                f"REJECTED at entry_odds={odds} — feature missing"
            )


class TestEVThreshold:
    """EV threshold enforcement"""

    def test_high_odds_should_reject(self, gate):
        """Favorite odds (0.90) should have negative EV → REJECT"""
        feats = _complete_features()
        feats["entry_odds"] = 0.90
        feats["odds_yes"] = 0.90
        result = gate.evaluate_signal(feats, entry_odds=0.90)
        # With p_win likely < 0.90, EV should be negative
        if result["p_win"] < 0.90:
            assert result["decision"] == "REJECT"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
