"""
ml/trainer.py
=============
Training pipeline utama:
  1. GroupKFold CV pada training set → estimasi generalisasi
  2. Final XGBoost fit pada seluruh training set
  3. Platt calibration (LogisticRegression sigmoid) pada calibration set
  4. Quality gates: AUC, ECE, Brier — harus semua PASS sebelum save
  5. Simpan artefak model + metadata versi
"""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from .config import (
    XGB_CFG, QUALITY_GATES, SPLIT_CFG,
    ALL_FEATURES, SELECTED_FEATURES, TARGET_COL,
)
from .features import build_features, get_feature_matrix
from .dataset import get_market_groups, chronological_split, impute_features, apply_imputer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: ECE (Expected Calibration Error)
# ---------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray,
                n_bins: int = 10) -> float:
    """
    ECE = mean |fraction_of_positives - mean_predicted_probability|
    per bin. Nilai < 0.05 menandakan kalibrasi yang baik.
    """
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return float(np.mean(np.abs(fraction_pos - mean_pred)))


# ---------------------------------------------------------------------------
# GroupKFold Cross-Validation
# ---------------------------------------------------------------------------

def cross_validate(X_train: np.ndarray,
                   y_train: np.ndarray,
                   groups: np.ndarray,
                   xgb_cfg=None) -> dict:
    """
    GroupKFold CV — tidak ada sinyal dari market yang sama
    di training dan validation sekaligus.

    Returns dict dengan metrics per fold dan agregat.
    """
    if xgb_cfg is None:
        xgb_cfg = XGB_CFG

    gkf = GroupKFold(n_splits=SPLIT_CFG.n_cv_folds)
    fold_results = []
    oof_preds  = np.zeros(len(y_train))
    oof_labels = np.zeros(len(y_train))

    params = asdict(xgb_cfg)
    # early_stopping_rounds passes to init in xgboost >= 2.0

    logger.info("Mulai %d-fold GroupKFold CV...", SPLIT_CFG.n_cv_folds)

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        proba = model.predict_proba(X_va)[:, 1]
        auc   = roc_auc_score(y_va, proba)
        brier = brier_score_loss(y_va, proba)
        best_iter = model.best_iteration

        oof_preds[va_idx]  = proba
        oof_labels[va_idx] = y_va

        fold_results.append({
            "fold": fold_idx + 1,
            "auc": round(auc, 4),
            "brier": round(brier, 4),
            "best_iteration": best_iter,
            "n_train": len(tr_idx),
            "n_val": len(va_idx),
        })
        logger.info(
            "  Fold %d: AUC=%.4f  Brier=%.4f  best_iter=%d",
            fold_idx + 1, auc, brier, best_iter
        )

    mean_auc   = float(np.mean([f["auc"] for f in fold_results]))
    std_auc    = float(np.std([f["auc"] for f in fold_results]))
    mean_brier = float(np.mean([f["brier"] for f in fold_results]))
    oof_ece    = compute_ece(oof_labels, oof_preds)
    oof_auc    = float(roc_auc_score(oof_labels, oof_preds))

    cv_result = {
        "folds": fold_results,
        "mean_auc": round(mean_auc, 4),
        "std_auc": round(std_auc, 4),
        "mean_brier": round(mean_brier, 4),
        "oof_auc": round(oof_auc, 4),
        "oof_ece_raw": round(oof_ece, 4),   # sebelum Platt — biasanya lebih besar
        "n_estimators_optimal": int(np.mean([f["best_iteration"] for f in fold_results])),
    }

    logger.info(
        "CV selesai: mean_AUC=%.4f ±%.4f | OOF_AUC=%.4f | mean_Brier=%.4f",
        mean_auc, std_auc, oof_auc, mean_brier
    )
    return cv_result


# ---------------------------------------------------------------------------
# Final Training + Platt Calibration
# ---------------------------------------------------------------------------

def train_and_calibrate(df_train: pd.DataFrame,
                         df_calib: pd.DataFrame,
                         xgb_cfg=None,
                         imputer_vals: Optional[pd.Series] = None) -> dict:
    """
    1. Fit XGBoost final pada df_train.
    2. Fit Platt sigmoid pada df_calib.
    3. Evaluasi kalibrasi pada df_calib.
    4. Return artefak lengkap.

    Returns dict berisi model, platt, metrics, feature_names, dan metadata.
    """
    if xgb_cfg is None:
        xgb_cfg = XGB_CFG

    # Build features
    df_train = build_features(df_train)
    df_calib = build_features(df_calib)

    X_tr = df_train[SELECTED_FEATURES].values.astype(np.float32)
    y_tr = df_train[TARGET_COL].values.astype(np.int32)
    X_ca = df_calib[SELECTED_FEATURES].values.astype(np.float32)
    y_ca = df_calib[TARGET_COL].values.astype(np.int32)

    # ------------------------------------------------------------------
    # XGBoost final fit
    # ------------------------------------------------------------------
    params = asdict(xgb_cfg)

    # Split kecil dari train untuk early stopping internal
    n_es = max(int(len(X_tr) * 0.15), 50)
    X_es_tr, X_es_va = X_tr[:-n_es], X_tr[-n_es:]
    y_es_tr, y_es_va = y_tr[:-n_es], y_tr[-n_es:]

    logger.info("Training final XGBoost: n_train=%d | n_es_val=%d", len(X_es_tr), n_es)
    t0 = time.time()

    base_model = XGBClassifier(**params)
    base_model.fit(
        X_es_tr, y_es_tr,
        eval_set=[(X_es_va, y_es_va)],
        verbose=False,
    )
    logger.info("XGBoost training selesai dalam %.1fs | best_iter=%d",
                time.time() - t0, base_model.best_iteration)

    # ------------------------------------------------------------------
    # Isotonic Calibration (non-parametric — no sigmoid assumption)
    # ------------------------------------------------------------------
    raw_calib = base_model.predict_proba(X_ca)[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(raw_calib, y_ca)

    p_cal = isotonic.predict(raw_calib)
    p_cal = np.clip(p_cal, 0.001, 0.999)  # prevent log(0)

    # ------------------------------------------------------------------
    # Evaluasi kalibrasi
    # ------------------------------------------------------------------
    calib_auc   = float(roc_auc_score(y_ca, p_cal))
    calib_brier = float(brier_score_loss(y_ca, p_cal))
    calib_ece   = compute_ece(y_ca, p_cal)

    logger.info(
        "Hold-out calibration: AUC=%.4f | Brier=%.4f | ECE=%.4f",
        calib_auc, calib_brier, calib_ece
    )

    calib_curve_frac, calib_curve_pred = calibration_curve(y_ca, p_cal, n_bins=10)

    return {
        "base_model": base_model,
        "platt": isotonic,  # key kept as 'platt' for backward compat
        "imputer_vals": imputer_vals,
        "feature_names": SELECTED_FEATURES,
        "metrics": {
            "calib_auc": round(calib_auc, 4),
            "calib_brier": round(calib_brier, 4),
            "calib_ece": round(calib_ece, 4),
        },
        "calib_curve": {
            "fraction_of_positives": calib_curve_frac.tolist(),
            "mean_predicted": calib_curve_pred.tolist(),
        },
        "best_iteration": base_model.best_iteration,
        "n_train": len(X_tr),
        "n_calib": len(X_ca),
    }


# ---------------------------------------------------------------------------
# Quality Gates
# ---------------------------------------------------------------------------

def run_quality_gates(cv_result: dict, train_result: dict,
                       gates=None, eval_result=None) -> dict:
    """
    Evaluasi semua quality gates. Return dict status tiap gate.

    Args:
        cv_result: hasil cross_validate()
        train_result: hasil train_and_calibrate()
        gates: QualityGates config (default: QUALITY_GATES)
        eval_result: hasil run_full_evaluation() — untuk test ECE (OOS)
    """
    if gates is None:
        gates = QUALITY_GATES

    checks = {}

    # Gate 1: AUC CV mean
    checks["auc_cv_min"] = {
        "value": cv_result["mean_auc"],
        "threshold": gates.min_auc_cv,
        "pass": cv_result["mean_auc"] >= gates.min_auc_cv,
    }

    # Gate 2: AUC CV std (konsistensi antar fold)
    checks["auc_cv_std"] = {
        "value": cv_result["std_auc"],
        "threshold": gates.max_auc_std,
        "pass": cv_result["std_auc"] <= gates.max_auc_std,
    }

    # Gate 3: ECE kalibrasi (in-sample — fit set)
    checks["ece_calib"] = {
        "value": train_result["metrics"]["calib_ece"],
        "threshold": gates.max_ece,
        "pass": train_result["metrics"]["calib_ece"] <= gates.max_ece,
    }

    # Gate 4: Brier score
    checks["brier"] = {
        "value": train_result["metrics"]["calib_brier"],
        "threshold": gates.max_brier,
        "pass": train_result["metrics"]["calib_brier"] <= gates.max_brier,
    }

    # Gate 5: ECE Test (out-of-sample) — mencegah Isotonic bypass
    if eval_result is not None and "calibration" in eval_result:
        test_ece = eval_result["calibration"].get("ece", None)
        if test_ece is not None:
            checks["ece_test"] = {
                "value": test_ece,
                "threshold": gates.max_ece_test,
                "pass": test_ece <= gates.max_ece_test,
            }
        else:
            logger.warning("Test ECE tidak tersedia dari eval_result. Gate ece_test di-skip.")
    else:
        logger.warning(
            "eval_result belum tersedia saat quality gates dijalankan. "
            "Gate ece_test di-skip."
        )

    all_pass = all(c["pass"] for c in checks.values())

    gate_results = {
        "all_pass": all_pass,
        "checks": checks,
    }

    if all_pass:
        logger.info("SEMUA QUALITY GATES PASS — model siap disimpan.")
    else:
        failed = [name for name, c in checks.items() if not c["pass"]]
        logger.warning("QUALITY GATES GAGAL: %s — model TIDAK disimpan.", failed)

    return gate_results


# ---------------------------------------------------------------------------
# Save / Load artefak
# ---------------------------------------------------------------------------

def save_model(train_result: dict,
               cv_result: dict,
               gate_result: dict,
               output_dir: str | Path,
               model_version: str = None) -> Path:
    """
    Simpan semua artefak ke output_dir:
      - model.pkl       : XGBClassifier + Platt + imputer
      - metadata.json   : versi, metrics, gate status, feature list
      - cv_results.json : detail per-fold CV
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_version is None:
        model_version = time.strftime("%Y%m%d_%H%M%S")

    if not gate_result["all_pass"]:
        raise RuntimeError(
            "Model tidak disimpan karena quality gates tidak terpenuhi. "
            f"Detail: {gate_result['checks']}"
        )

    # Model bundle
    bundle = {
        "base_model": train_result["base_model"],
        "platt": train_result["platt"],
        "imputer_vals": train_result.get("imputer_vals"),
        "feature_names": train_result["feature_names"],
        "version": model_version,
    }
    model_path = output_dir / "model.pkl"
    joblib.dump(bundle, model_path, compress=3)
    logger.info("Model tersimpan: %s", model_path)

    # Metadata
    metadata = {
        "version": model_version,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature_names": train_result["feature_names"],
        "n_features": len(train_result["feature_names"]),
        "n_train": train_result["n_train"],
        "n_calib": train_result["n_calib"],
        "best_iteration": train_result["best_iteration"],
        "metrics": train_result["metrics"],
        "cv": {
            "mean_auc": cv_result["mean_auc"],
            "std_auc": cv_result["std_auc"],
            "mean_brier": cv_result["mean_brier"],
            "oof_auc": cv_result["oof_auc"],
            "n_estimators_optimal": cv_result["n_estimators_optimal"],
        },
        "quality_gates": gate_result,
        "calib_curve": train_result["calib_curve"],
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # CV detail
    cv_path = output_dir / "cv_results.json"
    with open(cv_path, "w") as f:
        json.dump(cv_result, f, indent=2)

    logger.info("Metadata tersimpan: %s", meta_path)
    return model_path


def load_model(model_dir: str | Path) -> dict:
    """Load model bundle dari direktori artefak."""
    model_dir = Path(model_dir)
    bundle = joblib.load(model_dir / "model.pkl")
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)
    bundle["metadata"] = metadata
    logger.info(
        "Model dimuat: version=%s | AUC_CV=%.4f | ECE=%.4f",
        metadata["version"],
        metadata["cv"]["mean_auc"],
        metadata["metrics"]["calib_ece"],
    )
    return bundle
