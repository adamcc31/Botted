"""
scripts/run_v5_ablation.py
==========================
V5 Ablation Study Orchestrator — runs three experiments sequentially:
  A) Baseline V4 (5 original features)
  B) Phase 1 (+ odds_delta_signed, clob_alignment, odds_conviction)
  C) Full V5 (+ depth_ratio_std, clob_tfm_confluence, dual_confirmation) + Optuna HPO

Entry point:
    python scripts/run_v5_ablation.py

Prerequisites:
    - Master dataset already built at dataset/dataset_master_final.csv
    - Run scripts/build_master_dataset.py first if not available
"""

import glob
import json
import logging
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# --- Add project root to path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model_training.config import (
    SELECTED_FEATURES,
    V5_FEATURES_PHASE1,
    V5_FEATURES_FULL,
    TARGET_COL,
    SPLIT_CFG,
    XGB_CFG,
    XGBConfig,
)
from model_training.dataset import (
    load_from_csv,
    validate_dataset,
    deduplicate_per_market,
    chronological_split,
    get_market_groups,
    impute_features,
    apply_imputer,
    merge_datasets,
    sanitize_dataset,
)
from model_training.features import build_features, validate_no_leakage
from model_training.trainer import (
    cross_validate,
    train_and_calibrate,
    TemperatureScaling,
    compute_ece,
)
from model_training.evaluate import (
    run_full_evaluation,
    simulate_ev_strategy_v5,
    generate_metrics_report,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

# Fix Windows cp1252 encoding for Unicode output
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Experiment Configs
# ---------------------------------------------------------------------------

EXPERIMENT_A = {
    "name": "baseline_v4",
    "description": "Reproduksi baseline — 5 fitur original, max_depth=3",
    "features": list(SELECTED_FEATURES),
    "model_params": {
        "max_depth": 3,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_lambda": 1.0,
        "early_stopping_rounds": 50,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "eval_metric": "logloss",
    },
    "calibration": "temperature",
    "use_optuna": False,
}

EXPERIMENT_B = {
    "name": "phase1_odds_delta",
    "description": "Baseline + odds_delta_signed — isolasi kontribusi fitur kritis",
    "features": list(V5_FEATURES_PHASE1),
    "model_params": {
        "max_depth": 4,
        "n_estimators": 400,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "min_child_weight": 5,
        "reg_lambda": 2.0,
        "early_stopping_rounds": 50,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "eval_metric": "logloss",
    },
    "calibration": "temperature",
    "use_optuna": False,
}

EXPERIMENT_C = {
    "name": "full_v5_optuna",
    "description": "Full feature set + Optuna HPO — target AUC >= 0.80",
    "features": list(V5_FEATURES_FULL),
    "model_params": None,  # will be filled by Optuna
    "optuna_config": {
        "n_trials": 150,
        "timeout": 900,  # 15 minutes max
        "param_space": {
            "max_depth":        ("int",   3, 6),
            "n_estimators":     ("int",   200, 800),
            "learning_rate":    ("float", 0.01, 0.1, True),    # log scale
            "subsample":        ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.5, 1.0),
            "min_child_weight": ("int",   3, 15),
            "reg_alpha":        ("float", 1e-4, 5.0, True),
            "reg_lambda":       ("float", 0.5, 5.0),
            "gamma":            ("float", 0.0, 0.5),
        },
    },
    "calibration": "temperature",
    "use_optuna": True,
}


# ---------------------------------------------------------------------------
# Optuna HPO
# ---------------------------------------------------------------------------

def run_optuna_hpo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    param_space: Dict[str, tuple],
    n_trials: int = 150,
    timeout: int = 900,
) -> Dict[str, Any]:
    """
    Run Optuna HPO for XGBoost hyperparameters.
    Optimizes for GroupKFold CV AUC.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("Optuna not installed. Install with: pip install optuna")
        raise

    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    def objective(trial: optuna.Trial) -> float:
        params = {}
        for param_name, spec in param_space.items():
            if spec[0] == "int":
                params[param_name] = trial.suggest_int(param_name, spec[1], spec[2])
            elif spec[0] == "float":
                log_scale = len(spec) > 3 and spec[3]
                params[param_name] = trial.suggest_float(
                    param_name, spec[1], spec[2], log=log_scale
                )

        # Fixed params
        params["early_stopping_rounds"] = 50
        params["random_state"] = 42
        params["n_jobs"] = -1
        params["verbosity"] = 0
        params["eval_metric"] = "logloss"

        gkf = GroupKFold(n_splits=5)
        fold_aucs = []

        for tr_idx, va_idx in gkf.split(X_train, y_train, groups):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            model = XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )
            proba = model.predict_proba(X_va)[:, 1]
            fold_aucs.append(roc_auc_score(y_va, proba))

        return float(np.mean(fold_aucs))

    study = optuna.create_study(
        direction="maximize",
        study_name="v5_xgb_hpo",
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    logger.info("Optuna best AUC: %.4f", study.best_value)
    logger.info("Optuna best params: %s", study.best_params)

    best_params = study.best_params.copy()
    best_params["early_stopping_rounds"] = 50
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    best_params["verbosity"] = 0
    best_params["eval_metric"] = "logloss"

    return {
        "best_params": best_params,
        "best_auc": study.best_value,
        "n_trials_completed": len(study.trials),
    }


# ---------------------------------------------------------------------------
# Run Single Experiment
# ---------------------------------------------------------------------------

def run_experiment(
    experiment: Dict[str, Any],
    df_train: pd.DataFrame,
    df_calib: pd.DataFrame,
    df_test: pd.DataFrame,
    imputer_vals: pd.Series,
) -> Dict[str, Any]:
    """Run a single ablation experiment and return metrics."""
    exp_name = experiment["name"]
    features = experiment["features"]

    logger.info("")
    logger.info("=" * 60)
    logger.info("  EXPERIMENT: %s", exp_name.upper())
    logger.info("  Features: %s", str(len(features)))
    logger.info("=" * 60)

    # Build features
    df_train_feat = build_features(df_train)
    df_test_feat = build_features(df_test)

    X_train = df_train_feat[features].values.astype(np.float32)
    y_train = df_train_feat[TARGET_COL].values.astype(np.int32)
    groups = get_market_groups(df_train_feat)

    # --- Determine model params ---
    if experiment["use_optuna"]:
        logger.info("Running Optuna HPO...")
        optuna_result = run_optuna_hpo(
            X_train, y_train, groups,
            param_space=experiment["optuna_config"]["param_space"],
            n_trials=experiment["optuna_config"]["n_trials"],
            timeout=experiment["optuna_config"]["timeout"],
        )
        model_params = optuna_result["best_params"]
        logger.info("Optuna completed: %d trials, best AUC=%.4f",
                     optuna_result["n_trials_completed"],
                     optuna_result["best_auc"])
    else:
        model_params = experiment["model_params"]
        optuna_result = None

    # --- Create XGBConfig from params ---
    xgb_cfg = XGBConfig(**{
        k: v for k, v in model_params.items()
        if k in XGBConfig.__dataclass_fields__
    })

    # --- Cross-Validate ---
    cv_result = cross_validate(
        X_train, y_train, groups,
        xgb_cfg=xgb_cfg,
        calibration_method=experiment["calibration"],
    )

    # --- Train & Calibrate ---
    train_result = train_and_calibrate(
        df_train, df_calib,
        xgb_cfg=xgb_cfg,
        imputer_vals=imputer_vals,
        feature_list=features,
        calibration_method=experiment["calibration"],
    )

    # --- Evaluate on test set ---
    X_test = df_test_feat[features].values.astype(np.float32)
    y_test = df_test_feat[TARGET_COL].values.astype(np.int32)

    base_model = train_result["base_model"]
    calibrator = train_result["platt"]

    raw_proba = base_model.predict_proba(X_test)[:, 1]
    if hasattr(calibrator, "predict_proba") and isinstance(calibrator, TemperatureScaling):
        y_prob = calibrator.predict_proba(raw_proba)
    elif hasattr(calibrator, "predict"):
        y_prob = calibrator.predict(raw_proba)
    else:
        y_prob = raw_proba

    y_prob = np.clip(y_prob, 0.001, 0.999)

    # --- V5 Simulation ---
    sim_result = simulate_ev_strategy_v5(
        df=df_test_feat,
        y_prob=y_prob,
    )

    return {
        "name": exp_name,
        "description": experiment["description"],
        "features": features,
        "n_features": len(features),
        "model_params": model_params,
        "cv_result": cv_result,
        "train_result": train_result,
        "sim_result": sim_result,
        "optuna_result": optuna_result,
        "cv_auc": cv_result["mean_auc"],
        "cv_std": cv_result["std_auc"],
        "ece_before": cv_result.get("ece_before", 0),
        "ece_after": cv_result.get("ece_after", 0),
        "temperature": cv_result.get("temperature", None),
    }


# ---------------------------------------------------------------------------
# Ablation Table
# ---------------------------------------------------------------------------

def print_ablation_table(results: List[Dict[str, Any]]) -> str:
    """Print formatted ablation study results table."""
    baseline_auc = results[0]["cv_auc"] if results else 0.0
    best_auc = max(r["cv_auc"] for r in results) if results else 0.0

    table = "\n"
    table += "=" * 70 + "\n"
    table += " ABLATION STUDY RESULTS\n"
    table += "=" * 70 + "\n"
    table += f" {'Exp':<4} {'Name':<25} {'CV AUC':>8} {'Std':>8} {'Delta vs A':>12}\n"
    table += "-" * 70 + "\n"

    for i, r in enumerate(results):
        letter = chr(ord("A") + i)
        delta = f"+{r['cv_auc'] - baseline_auc:.4f}" if i > 0 else "--"
        table += (
            f" {letter:<4} {r['name']:<25} "
            f"{r['cv_auc']:>8.4f} ±{r['cv_std']:>6.4f} "
            f"{delta:>12}\n"
        )

    table += "=" * 70 + "\n"
    table += f" Target: >= 0.80\n"
    table += f" Best AUC: {best_auc:.4f}\n"
    table += f" Status: {'ACHIEVED' if best_auc >= 0.80 else 'BELUM TERCAPAI'}\n"
    table += "=" * 70 + "\n"

    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("  POLYMARKET V5 ABLATION STUDY")
    logger.info("  Target: CV AUC >= 0.80")
    logger.info("=" * 60)

    # --- Step 1: Build master dataset if not exists ---
    MASTER_CSV = PROJECT_ROOT / "dataset" / "dataset_master_final.csv"

    if not MASTER_CSV.exists():
        logger.info("Master dataset not found. Building from scratch...")

        V4_TRAIN = PROJECT_ROOT / "dataset" / "processed" / "predator_v4_train.csv"
        RAW_DIR = PROJECT_ROOT / "dataset" / "raw"

        if not V4_TRAIN.exists():
            raise FileNotFoundError(f"V4 training data not found: {V4_TRAIN}")

        df_v4 = pd.read_csv(V4_TRAIN, low_memory=False)
        logger.info("V4 Training data: %d rows", len(df_v4))

        dry_run_files = sorted(glob.glob(str(RAW_DIR / "dry_run_*.csv")))
        logger.info("Found %d dry-run files", len(dry_run_files))

        df_master = merge_datasets(df_v4, dry_run_files)
        df_master = sanitize_dataset(df_master)

        # Save master
        MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_master.to_csv(MASTER_CSV, index=False)
        logger.info("Master dataset saved: %s (%d rows)", MASTER_CSV, len(df_master))
    else:
        logger.info("Loading existing master dataset: %s", MASTER_CSV)
        df_master = pd.read_csv(MASTER_CSV, low_memory=False)
        df_master["timestamp"] = pd.to_datetime(df_master["timestamp"], utc=True)

    logger.info("Master dataset: %d rows", len(df_master))
    logger.info("Label dist: WIN=%.1f%% LOSS=%.1f%%",
                df_master["label"].mean() * 100,
                (1 - df_master["label"].mean()) * 100)

    # --- Step 2: Validate and split ---
    validate_no_leakage(df_master)

    df_train, df_calib, df_test = chronological_split(
        df_master,
        calib_ratio=SPLIT_CFG.calib_holdout_ratio,
        test_ratio=0.15,
    )
    logger.info("Split: train=%d | calib=%d | test=%d",
                len(df_train), len(df_calib), len(df_test))

    # Impute
    df_train, imputer_vals = impute_features(df_train, strategy="median")
    df_calib = apply_imputer(df_calib, imputer_vals)
    df_test = apply_imputer(df_test, imputer_vals)

    # --- Step 3: Run experiments ---
    experiments = [EXPERIMENT_A, EXPERIMENT_B, EXPERIMENT_C]
    all_results = []

    for exp in experiments:
        result = run_experiment(exp, df_train, df_calib, df_test, imputer_vals)
        all_results.append(result)

    # --- Step 4: Print ablation table ---
    table = print_ablation_table(all_results)
    print(table)
    logger.info(table)

    # --- Step 5: Find best experiment ---
    best = max(all_results, key=lambda r: r["cv_auc"])
    logger.info("Best experiment: %s (AUC=%.4f)", best["name"], best["cv_auc"])

    # --- Step 5b: Check if Experiment D needed ---
    if best["cv_auc"] < 0.775:
        logger.warning(
            "Best AUC %.4f < 0.775 — Experiment D (LightGBM ensemble) "
            "would be needed. Skipping auto-execution.",
            best["cv_auc"],
        )

    # --- Step 6: Save final model artifacts ---
    OUTPUT_DIR = PROJECT_ROOT / "models" / "v5_final"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    final_model = best["train_result"]["base_model"]
    final_calibrator = best["train_result"]["platt"]
    final_cv = best["cv_result"]

    # Save model
    with open(OUTPUT_DIR / "model.pkl", "wb") as f:
        pickle.dump(final_model, f)

    # Save calibrator
    with open(OUTPUT_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(final_calibrator, f)

    # Save metadata
    metadata = {
        "model_version": "v5_final",
        "created_at": datetime.now().isoformat(),
        "features": best["features"],
        "n_features": len(best["features"]),
        "cv_auc_baseline": 0.7145,
        "cv_auc_v5": best["cv_auc"],
        "cv_auc_std": best["cv_std"],
        "auc_delta": round(best["cv_auc"] - 0.7145, 4),
        "calibration_method": "temperature_scaling",
        "temperature": best.get("temperature"),
        "ece_before": best.get("ece_before"),
        "ece_after": best.get("ece_after"),
        "model_params": best["model_params"],
        "execution_gates": {
            "edge_threshold": 0.05,
            "vol_percentile_range": [0.15, 0.80],
            "spread_max_pct": 0.0008,
            "ttr_range_seconds": [60, 250],
            "clob_alignment_required": True,
        },
        "kelly_fraction": 0.25,
        "kelly_upgrade_condition": "ece_after < 0.08",
        "training_data": "dataset_master_final.csv",
        "n_samples": len(df_master),
        "label_win_rate": float(df_master["label"].mean()),
        "best_experiment": best["name"],
        "ablation_results": [
            {
                "name": r["name"],
                "cv_auc": r["cv_auc"],
                "cv_std": r["cv_std"],
                "n_features": r["n_features"],
            }
            for r in all_results
        ],
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # --- Step 7: Generate and save report ---
    report = generate_metrics_report(
        sim_result=best["sim_result"],
        cv_auc=best["cv_auc"],
        cv_std=best["cv_std"],
        ece_before=best.get("ece_before", 0),
        ece_after=best.get("ece_after", 0),
        temperature=best.get("temperature", 1.0),
    )
    print(report)

    with open(OUTPUT_DIR / "report.txt", "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n" + table)

    elapsed = time.time() - start_time
    logger.info("")
    logger.info("[OK] Artifacts saved to: %s", OUTPUT_DIR)
    logger.info("   model.pkl       -- XGBoost v5 final")
    logger.info("   calibrator.pkl  -- %s", repr(final_calibrator))
    logger.info("   metadata.json   -- Full experiment record")
    logger.info("   report.txt      -- Comprehensive metrics report")
    logger.info("")
    logger.info("Total elapsed: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
