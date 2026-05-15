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
from xgboost import XGBClassifier

from .inference import XGBoostGate

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# NEW: Slingger Hunter V5
# ═══════════════════════════════════════════════════════════════

class SlingshotHunterV5:
    """
    Replaces DualXGBoostGate / ShadowPredatorV4.
    Strategy: intramarket swing trading on Polymarket YES/NO tokens.
    Predicts probability that a dip-then-rip (or pump-then-dump) pattern
    completes with sufficient time remaining before resolution.
    """

    MODEL_DIR = Path("models/slingger_hunter_v5")

    def __init__(self):
        self.model: Optional[XGBClassifier] = None
        self.calibrator = None
        self.imputer = None
        self.metadata: Optional[dict] = None
        self.features: Optional[list] = None
        self._loaded: bool = False

    def load(self) -> "SlingshotHunterV5":
        """Load all artifacts from models/slingger_hunter_v5/."""
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

    def predict(self, feature_dict: dict) -> dict:
        """
        Input:  dict of feature_name -> value (matches self.features list)
        Output: dict with keys:
                  swing_probability  : float [0,1]
                  entry_odds         : float
                  exit_odds          : float
                  signal             : str   'ENTER' | 'SKIP'
                  confidence_tier    : str   'HIGH' | 'MEDIUM' | 'LOW'
                  full_kelly         : float (Kelly criterion, negative = skip)

        [FIX-THRESHOLD] ENTER threshold raised 0.55 → 0.65.
        Rationale: With entry=0.49, exit=0.80, fee=0.02, Kelly formula gives:
          gross_return = (0.80-0.49)/0.49 = 0.633
          b_adj = 0.633 * 0.98 = 0.620
          breakeven p = q / (b_adj + q) = (1-p) / (0.620 + (1-p))
          => breakeven p ≈ 0.617
        Previous threshold 0.55 was BELOW breakeven → guaranteed negative EV.
        New threshold 0.65 gives Kelly=+0.08, margin above breakeven.

        [FIX-EV-1] Inline Kelly check: returns SKIP if full_kelly < 0 even
        if cal_prob >= threshold, as final safety gate. This is the V1 gold
        standard: RiskManager rejects any bet where Kelly <= 0.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        X = np.array(
            [[feature_dict.get(f, np.nan) for f in self.features]],
            dtype=np.float32,
        )
        X = self.imputer.transform(X)

        raw_prob = self.model.predict_proba(X)[0][1]
        cal_prob = float(
            self.calibrator.predict_proba(np.array([[raw_prob]]))[0][1]
        )

        # [FIX-THRESHOLD] Raised from 0.55 to 0.65 based on Kelly breakeven analysis.
        # Threshold from metadata if available, else hardcoded 0.65.
        enter_threshold = self.metadata.get("enter_threshold", 0.65)

        # [FIX-EV-1] Inline Kelly validation as final gate
        entry_odds = self.metadata["optimal_entry_odds"]
        exit_odds = self.metadata["optimal_exit_odds"]
        fee = self.metadata.get("polymarket_fee", 0.02)

        gross_return = (exit_odds - entry_odds) / entry_odds if entry_odds > 0 else 0.0
        b_adj = gross_return * (1 - fee)
        q = 1 - cal_prob
        if b_adj > 0:
            full_kelly = (b_adj * cal_prob - q) / b_adj
        else:
            full_kelly = -1.0

        # Signal: ENTER only if prob meets threshold AND Kelly is positive
        kelly_positive = full_kelly > 0.0
        above_threshold = cal_prob >= enter_threshold
        signal = "ENTER" if (above_threshold and kelly_positive) else "SKIP"

        if cal_prob >= 0.65 and kelly_positive:
            tier = "HIGH"
        elif cal_prob >= 0.55:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        if signal == "SKIP" and above_threshold and not kelly_positive:
            logger.debug(
                "[SlingshotHunterV5] SKIP_NEGATIVE_KELLY",
                cal_prob=round(cal_prob, 4),
                full_kelly=round(full_kelly, 4),
            )

        return {
            "swing_probability": cal_prob,
            "entry_odds": entry_odds,
            "exit_odds": exit_odds,
            "signal": signal,
            "confidence_tier": tier,
            "full_kelly": round(full_kelly, 4),
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ── Backward compatibility aliases ─────────────────────────────
ShadowPredatorV4 = SlingshotHunterV5
DualXGBoostGate = SlingshotHunterV5
