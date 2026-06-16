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
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import joblib
from xgboost import XGBClassifier

from .inference import XGBoostGate

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Slingger Hunter V5 (Killed / Poisoned — meta-features source)
# ═══════════════════════════════════════════════════════════════

class SlingshotHunterV5:
    """
    Legacy SlingshotHunterV5.
    Loaded strictly from models/killed_v5_poisoned/ to serve as the
    meta-features input for SlingshotHunterV6.
    """

    MODEL_DIR = Path("models/killed_v5_poisoned")

    def __init__(self):
        self.model: Optional[XGBClassifier] = None
        self.calibrator = None
        self.imputer = None
        self.metadata: Optional[dict] = None
        self.features: Optional[list] = None
        self._loaded: bool = False

    def load(self) -> "SlingshotHunterV5":
        """Load all artifacts from models/killed_v5_poisoned/."""
        if not (self.MODEL_DIR / "metadata.json").exists():
            raise FileNotFoundError(
                f"Killed V5 metadata not found in {self.MODEL_DIR}. "
                "Ensure models/slingger_hunter_v5/ was successfully moved to models/killed_v5_poisoned/."
            )
            
        with open(self.MODEL_DIR / "metadata.json") as f:
            self.metadata = json.load(f)

        self.features = self.metadata["features"]

        self.model = XGBClassifier()
        self.model.load_model(str(self.MODEL_DIR / "model.json"))

        with open(self.MODEL_DIR / "calibrator.pkl", "rb") as f:
            self.calibrator = pickle.load(f)

        with open(self.MODEL_DIR / "imputer.pkl", "rb") as f:
            self.imputer = pickle.load(f)

        self._loaded = True
        logger.info(
            "[SlingshotHunterV5] Loaded | AUC: %.4f | Features: %d",
            self.metadata["oof_roc_auc"],
            len(self.features),
        )
        return self

    def predict(self, feature_dict: dict, live_entry_odds: float = 0.50) -> dict:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        try:
            X = np.array(
                [[feature_dict.get(f, np.nan) for f in self.features]],
                dtype=np.float32,
            )
            X = self.imputer.transform(X)

            raw_prob = self.model.predict_proba(X)[0][1]
            cal_prob = float(
                self.calibrator.predict_proba(np.array([[raw_prob]]))[0][1]
            )

            enter_threshold = self.metadata.get("enter_threshold", 0.65)
            entry_odds = live_entry_odds
            exit_odds = 0.80
            fee = self.metadata.get("polymarket_fee", 0.02)

            gross_return = (exit_odds - entry_odds) / entry_odds if entry_odds > 0 else 0.0
            b_adj = gross_return * (1 - fee)
            q = 1 - cal_prob
            if b_adj > 0:
                full_kelly = (b_adj * cal_prob - q) / b_adj
            else:
                full_kelly = -1.0

            kelly_positive = full_kelly > 0.0
            above_threshold = cal_prob >= enter_threshold
            signal = "ENTER" if (above_threshold and kelly_positive) else "SKIP"

            if cal_prob >= 0.65 and kelly_positive:
                tier = "HIGH"
            elif cal_prob >= 0.55:
                tier = "MEDIUM"
            else:
                tier = "LOW"

            return {
                "swing_probability": cal_prob,
                "entry_odds": entry_odds,
                "exit_odds": exit_odds,
                "signal": signal,
                "confidence_tier": tier,
                "full_kelly": round(full_kelly, 4),
            }
        except Exception as e:
            logger.error("[SlingshotHunterV5] PREDICT_EXCEPTION: %s", str(e), exc_info=True)
            return {
                "swing_probability": 0.0,
                "entry_odds": 0.5,
                "exit_odds": 0.8,
                "signal": "SKIP",
                "confidence_tier": "ERROR",
                "full_kelly": -1.0,
            }

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ═══════════════════════════════════════════════════════════════
# NEW: Slingger Hunter V6 (Production Stacking Model)
# ═══════════════════════════════════════════════════════════════

class SlingshotHunterV6:
    """
    Slingger Hunter V6 Stacking Classifier.
    Uses V5 model outputs as meta-features combined with market indicators.
    """

    V6_MODEL_DIR = Path("slingger/models")

    def __init__(self):
        self.v5_hunter = SlingshotHunterV5()
        self.v6_model = None
        self.v6_metadata: Optional[dict] = None
        self.v6_features: Optional[list] = None
        self._loaded: bool = False

    def load(self) -> "SlingshotHunterV6":
        """Load both V5 (meta-features) and V6 models."""
        self.v5_hunter.load()

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
        self._loaded = True
        logger.info(
            "[SlingshotHunterV6] Loaded | V6 OOF AUC: %.4f | Features: %d",
            self.v6_metadata.get("oof_auc_mean", 0.0),
            len(self.v6_features),
        )
        return self

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, feature_dict: dict, live_entry_odds: float = 0.50) -> dict:
        """
        Runs the dual-stage stacked prediction.
        """
        if not self._loaded:
            raise RuntimeError("V6 Model not loaded. Call .load() first.")

        try:
            # 1. Run V5 for target side
            res_target = self.v5_hunter.predict(feature_dict, live_entry_odds=live_entry_odds)
            shadow_prob_yes = res_target["swing_probability"]
            shadow_kelly_yes = res_target["full_kelly"]

            # Compute opposite side's V5 prediction
            opp_feat = feature_dict.copy()
            opp_feat['yes_price_t0'] = feature_dict['no_price_t0']
            opp_feat['no_price_t0'] = feature_dict['yes_price_t0']
            opp_feat['yes_depth_t0'] = feature_dict['no_depth_t0']
            opp_feat['no_depth_t0'] = feature_dict['yes_depth_t0']
            total_d = opp_feat['yes_depth_t0'] + opp_feat['no_depth_t0']
            opp_feat['depth_imbalance_t0'] = (opp_feat['yes_depth_t0'] - opp_feat['no_depth_t0']) / max(total_d, 1.0)
            
            opp_entry_odds = feature_dict.get('no_price_t0', 1.0 - live_entry_odds)
            res_opp = self.v5_hunter.predict(opp_feat, live_entry_odds=opp_entry_odds)
            shadow_prob_no = res_opp["swing_probability"]
            shadow_kelly_no = res_opp["full_kelly"]

            # 2. Construct V6 features
            v6_input_dict = feature_dict.copy()
            v6_input_dict["shadow_prob_yes"] = shadow_prob_yes
            v6_input_dict["shadow_kelly_yes"] = shadow_kelly_yes
            v6_input_dict["shadow_prob_no"] = shadow_prob_no
            v6_input_dict["shadow_kelly_no"] = shadow_kelly_no

            # Ensure all features align to expected V6 schema
            X = np.array(
                [[v6_input_dict.get(f, 0.0) for f in self.v6_features]],
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
        v5_input = {
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
        res = self.predict(v5_input, live_entry_odds=entry_odds)
        return {
            "decision": "PASS" if res['signal'] == "ENTER" else "REJECT",
            "p_win": res['swing_probability'],
            "ev": res['full_kelly'],
            "reason": "V6_SHIM_REJECT" if res['signal'] == "SKIP" else "V6_SHIM_PASS"
        }


