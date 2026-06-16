"""
model_training/dual_inference.py
===============================
Slingger Hunter V5 — Intramarket Swing Trading Inference Engine.
Replaces the legacy DualXGBoostGate / ShadowPredatorV4.

Also preserves the legacy DualXGBoostGate class for backward compatibility
with callers that have not yet been migrated.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import joblib

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# NEW: Slingger Hunter V6 (Standalone Microstructure Model)
# ═══════════════════════════════════════════════════════════════

class SlingshotHunterV6:
    """
    Slingger Hunter V6 Standalone Classifier.
    Directly analyzes raw microstructure features and outputs trading decisions.
    No dependencies on the poisoned V5 model or shadow_prob/shadow_kelly features.
    """

    V6_MODEL_DIR = Path("slingger/models")

    def __init__(self):
        self.v6_model = None
        self.v6_metadata: Optional[dict] = None
        self.v6_features: Optional[list] = None
        self._loaded: bool = False

    def load(self) -> "SlingshotHunterV6":
        """Load the V6 production model and metadata."""
        model_path = self.V6_MODEL_DIR / "v6_production.pkl"
        meta_path = self.V6_MODEL_DIR / "v6_production_meta.json"

        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"V6 model or metadata missing in {self.V6_MODEL_DIR}. Run training pipeline."
            )

        self.v6_model = joblib.load(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.v6_metadata = json.load(f)

        self.v6_features = self.v6_metadata["features"]
        
        # Verify that V5 features are NOT present in V6 features list
        for feat in ["shadow_prob_yes", "shadow_prob_no", "shadow_kelly_yes", "shadow_kelly_no"]:
            if feat in self.v6_features:
                logger.warning(f"[SlingshotHunterV6] Warning: V6 features list still contains V5 feature '{feat}'!")

        self._loaded = True
        logger.info(
            "[SlingshotHunterV6] Loaded Standalone | V6 OOF AUC: %.4f | Features: %d",
            self.v6_metadata.get("oof_auc_mean", 0.0),
            len(self.v6_features),
        )
        return self

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, feature_dict: dict, live_entry_odds: float = 0.50) -> dict:
        """
        Runs prediction directly using the raw features.
        """
        if not self._loaded:
            raise RuntimeError("V6 Model not loaded. Call .load() first.")

        try:
            # Ensure all features align to expected V6 schema (excluding shadow_* features)
            X = np.array(
                [[feature_dict.get(f, 0.0) for f in self.v6_features]],
                dtype=np.float32,
            )

            # Impute any missing or NaN values
            X = np.nan_to_num(X, nan=0.0)

            # Run V6 classifier prediction
            v6_prob = float(self.v6_model.predict_proba(X)[0][1])

            # compute Kelly
            enter_threshold = self.v6_metadata.get("enter_threshold", 0.65)
            entry_odds = live_entry_odds
            exit_odds = 0.80
            fee = self.v6_metadata.get("polymarket_fee", 0.02)

            gross_return = (exit_odds - entry_odds) / entry_odds if entry_odds > 0 else 0.0
            b_adj = gross_return * (1 - fee)
            q = 1 - v6_prob
            if b_adj > 0:
                full_kelly = (b_adj * v6_prob - q) / b_adj
            else:
                full_kelly = -1.0

            above_threshold = v6_prob >= enter_threshold
            kelly_positive = full_kelly > 0.0
            signal = "ENTER" if (above_threshold and kelly_positive) else "SKIP"

            if v6_prob >= 0.65 and kelly_positive:
                tier = "HIGH"
            elif v6_prob >= 0.55:
                tier = "MEDIUM"
            else:
                tier = "LOW"

            return {
                "swing_probability": v6_prob,
                "entry_odds": entry_odds,
                "exit_odds": exit_odds,
                "signal": signal,
                "confidence_tier": tier,
                "full_kelly": round(full_kelly, 4),
            }

        except Exception as e:
            logger.error("[SlingshotHunterV6] PREDICT_EXCEPTION: %s", str(e), exc_info=True)
            return {
                "swing_probability": 0.0,
                "entry_odds": live_entry_odds,
                "exit_odds": 0.8,
                "signal": "SKIP",
                "confidence_tier": "ERROR",
                "full_kelly": -1.0,
            }

    def evaluate_signal(self, raw_features: dict, entry_odds: float) -> dict:
        """Shim for backward compatibility."""
        v6_input = {
            'yes_price_t0':              raw_features.get('entry_odds', entry_odds),
            'no_price_t0':               raw_features.get('odds_no', 1.0 - entry_odds),
            'clob_spread_t0':            raw_features.get('spread_pct', 0.005),
            'yes_depth_t0':              raw_features.get('depth_ratio', 1.0),
            'no_depth_t0':               1.0,
            'depth_imbalance_t0':        0.0,
            'price_velocity_30s':        0.0,
            'depth_trend_30s':           0.0,
            'btc_realized_vol_prior_30m': raw_features.get('rv_value', 0.0),
            'ttr_at_signal':             raw_features.get('ttr_seconds', 300.0),
            'market_hour_utc':           12.0,
            'day_of_week':               0.0
        }
        res = self.predict(v6_input, live_entry_odds=entry_odds)
        return {
            "decision": "PASS" if res['signal'] == "ENTER" else "REJECT",
            "p_win": res['swing_probability'],
            "ev": res['full_kelly'],
            "reason": "V6_SHIM_REJECT" if res['signal'] == "SKIP" else "V6_SHIM_PASS"
        }


