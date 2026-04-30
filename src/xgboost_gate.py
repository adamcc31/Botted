"""
xgboost_gate.py — XGBoost Alpha V1 Quality Gate for production inference.

Loads the trained model bundle (model.pkl) and provides a signal quality
assessment via calibrated win probability and EV calculation.

Feature mapping (CRITICAL):
  The training pipeline uses columns from the exporter CSV schema.
  The live bot uses a 24-element FeatureVector from feature_engine.py.
  This module bridges the gap by extracting the 5 SELECTED_FEATURES
  from live data sources (FeatureVector, CLOBState, ActiveMarket).

Model architecture:
  - Base: XGBClassifier (5 features)
  - Calibrator: IsotonicRegression (NOT LogisticRegression / Platt)
  - API: calibrator.predict(x), NOT calibrator.predict_proba(x)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "alpha_v1" / "model.pkl"

# EV constants (V2 environment)
LATENCY_PENALTY = 0.015  # -1.5% absolute
EV_THRESHOLD = 0.02      # 2% edge minimum


@dataclass
class GateResult:
    """Result from XGBoost quality gate evaluation."""
    decision: str           # "PASS" or "REJECT"
    p_win: float            # Calibrated P(WIN) from Isotonic
    p_win_adjusted: float   # P(WIN) after latency penalty
    ev: float               # Post-penalty expected value
    reason: str             # Human-readable reason
    feature_vector: list    # The 5-element feature vector used


class XGBoostGate:
    """
    Production inference gate using Alpha V1 XGBoost model.

    Usage:
        gate = XGBoostGate()
        result = gate.evaluate_signal(signal_data)
        if result.decision == "PASS":
            # Execute trade
    """

    # Feature order MUST match training: SELECTED_FEATURES in config.py
    FEATURE_ORDER = [
        "entry_odds",
        "depth_ratio",
        "contest_urgency",
        "tfm_value",
        "obi_vol_interaction",
    ]

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._model_path = model_path or MODEL_PATH
        self._base_model = None
        self._calibrator = None  # IsotonicRegression
        self._imputer_vals = None
        self._feature_names = None
        self._version = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def version(self) -> str:
        return self._version or "none"

    def load(self) -> bool:
        """Load model bundle from disk."""
        try:
            import joblib
            if not self._model_path.exists():
                logger.warning(
                    "xgboost_gate_model_not_found",
                    path=str(self._model_path),
                )
                return False

            bundle = joblib.load(self._model_path)
            self._base_model = bundle["base_model"]
            self._calibrator = bundle["platt"]  # IsotonicRegression (key kept for compat)
            self._imputer_vals = bundle.get("imputer_vals")
            self._feature_names = bundle.get("feature_names", self.FEATURE_ORDER)
            self._version = bundle.get("version", "unknown")
            self._is_loaded = True

            # Sanity: verify feature alignment
            if self._feature_names != self.FEATURE_ORDER:
                logger.error(
                    "xgboost_gate_feature_mismatch",
                    expected=self.FEATURE_ORDER,
                    got=self._feature_names,
                )
                self._is_loaded = False
                return False

            # Verify calibrator API (Isotonic vs Platt)
            if hasattr(self._calibrator, "predict_proba"):
                logger.warning(
                    "xgboost_gate_calibrator_is_platt",
                    note="Expected IsotonicRegression. Using predict_proba fallback.",
                )

            logger.info(
                "xgboost_gate_loaded",
                version=self._version,
                features=self._feature_names,
                calibrator_type=type(self._calibrator).__name__,
            )
            return True

        except Exception as e:
            logger.error("xgboost_gate_load_error", error=str(e))
            return False

    def _extract_features(self, signal_data: dict) -> np.ndarray:
        """
        Extract the 5 SELECTED_FEATURES from live signal data.

        Maps from the live bot's data sources to the training feature names:
          - entry_odds: from signal_data["entry_odds"] or CLOB mid price
          - depth_ratio: from signal_data["depth_ratio"] or feature vector
          - contest_urgency: from signal_data["contest_urgency"] or feature vector
          - tfm_value: from signal_data["tfm_value"] or feature vector
          - obi_vol_interaction: from signal_data["obi_vol_interaction"]
                                  = obi_value * vol_percentile
        """
        values = []
        for feat in self.FEATURE_ORDER:
            val = signal_data.get(feat)
            if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                # Impute with training median
                if self._imputer_vals is not None and feat in self._imputer_vals:
                    val = float(self._imputer_vals[feat])
                else:
                    val = 0.0
            values.append(float(val))

        return np.array(values, dtype=np.float32).reshape(1, -1)

    def predict_calibrated(self, X: np.ndarray) -> float:
        """
        Get calibrated P(WIN) from the model.

        CRITICAL: IsotonicRegression uses .predict(), NOT .predict_proba().
        """
        raw_proba = self._base_model.predict_proba(X)[:, 1]

        # Route based on calibrator type
        if hasattr(self._calibrator, "predict_proba"):
            # Platt / LogisticRegression fallback
            p_cal = self._calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
        else:
            # IsotonicRegression (correct path)
            p_cal = self._calibrator.predict(raw_proba)
            p_cal = np.clip(p_cal, 0.001, 0.999)

        return float(p_cal[0])

    def evaluate_signal(self, signal_data: dict) -> GateResult:
        """
        Evaluate a live signal through the XGBoost quality gate.

        Args:
            signal_data: dict with keys matching FEATURE_ORDER.
                Required: entry_odds, depth_ratio, contest_urgency,
                          tfm_value, obi_vol_interaction

        Returns:
            GateResult with PASS/REJECT decision and EV calculation.
        """
        if not self._is_loaded:
            return GateResult(
                decision="REJECT",
                p_win=0.0,
                p_win_adjusted=0.0,
                ev=0.0,
                reason="MODEL_NOT_LOADED",
                feature_vector=[],
            )

        try:
            X = self._extract_features(signal_data)
            feature_list = X.flatten().tolist()

            # Calibrated probability
            p_win = self.predict_calibrated(X)

            # Apply latency penalty
            p_win_adj = max(p_win - LATENCY_PENALTY, 0.0)

            # Calculate EV
            entry_odds = signal_data.get("entry_odds", 0.5)
            if entry_odds > 0:
                payout = (1.0 / entry_odds) - 1.0
                ev = p_win_adj * payout - (1.0 - p_win_adj)
            else:
                ev = -1.0

            # Decision
            if ev > EV_THRESHOLD:
                decision = "PASS"
                reason = f"EV={ev:.4f} > threshold={EV_THRESHOLD}"
            else:
                decision = "REJECT"
                reason = f"EV={ev:.4f} <= threshold={EV_THRESHOLD}"

            logger.info(
                "xgboost_gate_result",
                decision=decision,
                p_win=round(p_win, 4),
                p_win_adj=round(p_win_adj, 4),
                ev=round(ev, 4),
                entry_odds=entry_odds,
            )

            return GateResult(
                decision=decision,
                p_win=p_win,
                p_win_adjusted=p_win_adj,
                ev=ev,
                reason=reason,
                feature_vector=feature_list,
            )

        except Exception as e:
            logger.error("xgboost_gate_evaluate_error", error=str(e))
            return GateResult(
                decision="REJECT",
                p_win=0.0,
                p_win_adjusted=0.0,
                ev=0.0,
                reason=f"EXCEPTION: {str(e)}",
                feature_vector=[],
            )
