"""
model.py — LightGBM + Logistic Regression ensemble with calibration.

Architecture:
  Primary   : LightGBM Classifier
  Secondary : Logistic Regression (baseline, overfit detection)
  Ensemble  : P_model = 0.7 × LGBM_calibrated + 0.3 × LogReg_prob

CRITICAL (from validation):
  LightGBM outputs are NOT well-calibrated probabilities by default.
  Isotonic Regression calibration is MANDATORY for Kelly sizing accuracy.
  Without calibration, edge calculations and Kelly fractions will be destructive.
"""

from __future__ import annotations

import glob
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from src.config_manager import ConfigManager

logger = structlog.get_logger(__name__)

_MODELS_DIR = Path(__file__).parent.parent / "models"
_DATA_DIR = Path(__file__).parent.parent / "data" / "models"


def _get_model_dir() -> Path:
    """Get model directory — fallback to data/models for Railway."""
    for d in [_DATA_DIR, _MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR if _DATA_DIR.exists() else _MODELS_DIR


class ModelEnsemble:
    """
    LightGBM + LogReg ensemble with Isotonic Regression calibration.

    The ensemble outputs calibrated P(YES outcome) ∈ [0, 1].
    Calibration is performed post-training using isotonic regression
    because tree-based models produce biased probability estimates.
    """

    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        self._lgbm_model = None
        self._logreg_model = None
        self._scaler = None  # StandardScaler for LogReg
        self._calibrator = None  # CalibratedClassifierCV wrapper or standalone
        self._lgbm_weight = config.get("model.ensemble_lgbm_weight", 0.70)
        self._logreg_weight = config.get("model.ensemble_logreg_weight", 0.30)
        self._calibration_method = config.get("model.calibration_method", "isotonic")
        self._version: Optional[str] = None
        self._is_loaded = False

    # ── Public Interface ──────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def version(self) -> str:
        return self._version or "none"

    def predict(self, feature_vector: np.ndarray) -> float:
        """
        Predict P(YES outcome) from feature vector.

        Args:
            feature_vector: numpy array of shape (1, 24)

        Returns:
            P_model: calibrated probability ∈ [0, 1]

        If model not loaded, returns 0.5 (neutral — will ABSTAIN due to no edge).
        """
        if not self._is_loaded:
            logger.warning("model_not_loaded_returning_neutral")
            return 0.5

        try:
            X = np.array(feature_vector).reshape(1, -1)

            # LightGBM prediction (calibrated if calibrator available)
            if self._calibrator is not None:
                lgbm_prob = self._calibrator.predict_proba(X)[0, 1]
            else:
                lgbm_prob = self._lgbm_model.predict_proba(X)[0, 1]

            # LogReg prediction (inherently better calibrated)
            if self._logreg_model is not None and self._scaler is not None:
                X_scaled = self._scaler.transform(X)
                logreg_prob = self._logreg_model.predict_proba(X_scaled)[0, 1]
            else:
                logreg_prob = lgbm_prob  # Fallback to LGBM only

            # Weighted ensemble
            P_model = (
                self._lgbm_weight * lgbm_prob
                + self._logreg_weight * logreg_prob
            )

            # Clamp to [0, 1]
            P_model = max(0.0, min(1.0, P_model))

            logger.debug(
                "model_prediction",
                P_model=round(P_model, 4),
                lgbm_prob=round(lgbm_prob, 4),
                logreg_prob=round(logreg_prob, 4),
            )

            return float(P_model)

        except Exception as e:
            logger.error("model_predict_error", error=str(e))
            return 0.5

    # ── Model Loading ─────────────────────────────────────────

    def load_latest(self) -> bool:
        """Load the latest model version from disk."""
        # V2_MIGRATION: AI Amnesia Mode - Model loading disabled for Rule Engine deterministic execution
        logger.warning("AI_AMNESIA_MODE_ACTIVE_MODEL_LOADING_DISABLED")
        return False

    def load_version(self, version: str) -> bool:
        """Load a specific model version."""
        model_dir = _get_model_dir()

        lgbm_path = model_dir / f"model_lgbm_{version}.pkl"
        logreg_path = model_dir / f"model_logreg_{version}.pkl"
        scaler_path = model_dir / f"scaler_{version}.pkl"
        calibrator_path = model_dir / f"calibrator_{version}.pkl"

        try:
            # Load LightGBM (required)
            if not lgbm_path.exists():
                logger.error("lgbm_model_not_found", path=str(lgbm_path))
                return False

            with open(lgbm_path, "rb") as f:
                self._lgbm_model = pickle.load(f)

            # Load LogReg (optional)
            if logreg_path.exists():
                with open(logreg_path, "rb") as f:
                    self._logreg_model = pickle.load(f)

            # Load Scaler (for LogReg)
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self._scaler = pickle.load(f)

            # Load Calibrator (for LightGBM — CRITICAL)
            if calibrator_path.exists():
                with open(calibrator_path, "rb") as f:
                    self._calibrator = pickle.load(f)
                logger.info("calibrator_loaded", method=self._calibration_method)
            else:
                logger.warning(
                    "calibrator_not_found_lgbm_uncalibrated",
                    path=str(calibrator_path),
                )

            self._version = version
            self._is_loaded = True

            logger.info(
                "model_loaded",
                version=version,
                has_logreg=self._logreg_model is not None,
                has_calibrator=self._calibrator is not None,
            )
            return True

        except Exception as e:
            logger.error("model_load_error", version=version, error=str(e))
            return False

    # ── Model Saving ──────────────────────────────────────────

    @staticmethod
    def save_models(
        lgbm_model: Any,
        logreg_model: Any,
        scaler: Any,
        calibrator: Any,
        version_tag: str,
        metrics: Optional[Dict] = None,
    ) -> Path:
        """Save model artifacts to disk."""
        model_dir = _get_model_dir()

        with open(model_dir / f"model_lgbm_{version_tag}.pkl", "wb") as f:
            pickle.dump(lgbm_model, f)

        with open(model_dir / f"model_logreg_{version_tag}.pkl", "wb") as f:
            pickle.dump(logreg_model, f)

        with open(model_dir / f"scaler_{version_tag}.pkl", "wb") as f:
            pickle.dump(scaler, f)

        if calibrator is not None:
            with open(model_dir / f"calibrator_{version_tag}.pkl", "wb") as f:
                pickle.dump(calibrator, f)

        if metrics:
            import json
            with open(model_dir / f"training_metrics_{version_tag}.json", "w") as f:
                json.dump(metrics, f, indent=2, default=str)

        logger.info("models_saved", version=version_tag, dir=str(model_dir))

        # Cleanup: keep only N latest versions
        _cleanup_old_versions(model_dir, keep_n=3)

        return model_dir

    # ── Rollback ──────────────────────────────────────────────

    def rollback(self) -> bool:
        """Rollback to previous model version."""
        model_dir = _get_model_dir()
        lgbm_files = sorted(
            glob.glob(str(model_dir / "model_lgbm_v*.pkl")),
            reverse=True,
        )

        if len(lgbm_files) < 2:
            logger.error("no_previous_version_for_rollback")
            return False

        prev_file = Path(lgbm_files[1])
        prev_version = prev_file.stem.replace("model_lgbm_", "")
        logger.info("rolling_back_model", to_version=prev_version)
        return self.load_version(prev_version)

    # ── Overfit Detection ─────────────────────────────────────

    def check_overfit_divergence(self, X: np.ndarray) -> Optional[float]:
        """
        Check divergence between LGBM and LogReg predictions.
        Large divergence suggests LGBM may be overfitting.
        """
        if not self._is_loaded or self._logreg_model is None:
            return None

        try:
            lgbm_probs = self._lgbm_model.predict_proba(X)[:, 1]
            X_scaled = self._scaler.transform(X)
            logreg_probs = self._logreg_model.predict_proba(X_scaled)[:, 1]
            mae = float(np.mean(np.abs(lgbm_probs - logreg_probs)))
            return mae
        except Exception:
            return None


def _cleanup_old_versions(model_dir: Path, keep_n: int = 3) -> None:
    """Keep only the N latest model versions, delete older ones."""
    versions = set()
    for f in model_dir.glob("model_lgbm_v*.pkl"):
        version = f.stem.replace("model_lgbm_", "")
        versions.add(version)

    sorted_versions = sorted(versions, reverse=True)
    to_delete = sorted_versions[keep_n:]

    for v in to_delete:
        for pattern in [
            f"model_lgbm_{v}.pkl",
            f"model_logreg_{v}.pkl",
            f"scaler_{v}.pkl",
            f"calibrator_{v}.pkl",
            f"training_metrics_{v}.json",
        ]:
            path = model_dir / pattern
            if path.exists():
                path.unlink()
                logger.info("old_model_deleted", file=str(path))
