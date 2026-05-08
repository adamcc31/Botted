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

        signal = "ENTER" if cal_prob >= 0.55 else "SKIP"

        if cal_prob >= 0.65:
            tier = "HIGH"
        elif cal_prob >= 0.55:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        return {
            "swing_probability": cal_prob,
            "entry_odds": self.metadata["optimal_entry_odds"],
            "exit_odds": self.metadata["optimal_exit_odds"],
            "signal": signal,
            "confidence_tier": tier,
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ── Backward compatibility aliases ─────────────────────────────
ShadowPredatorV4 = SlingshotHunterV5
DualXGBoostGate = SlingshotHunterV5
