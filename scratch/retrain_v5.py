"""
Retrain V5 model using DATASET_MASTER_08-05-2026.csv
Full ablation study: Experiment A/B/C + Optuna HPO
"""
import sys, glob, json, pickle, time
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("retrain_v5")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

from model_training.config import (
    SELECTED_FEATURES, V5_FEATURES_PHASE1, V5_FEATURES_FULL,
    TARGET_COL, SPLIT_CFG, XGB_CFG, XGBConfig,
)
from model_training.dataset import (
    chronological_split, get_market_groups,
    impute_features, apply_imputer,
)
from model_training.features import build_features, validate_no_leakage
from model_training.trainer import (
    cross_validate, train_and_calibrate, TemperatureScaling, compute_ece,
)
from model_training.evaluate import (
    simulate_ev_strategy_v5, generate_metrics_report,
)

# === Import Optuna HPO from ablation script ===
sys.path.insert(0, 'scripts')
from run_v5_ablation import run_optuna_hpo

PROJECT_ROOT = Path('.')
MASTER_CSV = PROJECT_ROOT / "dataset" / "DATASET_MASTER_08-05-2026.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "v5_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# EXPERIMENT CONFIGS
# ============================================================================
EXPERIMENTS = [
    {
        "name": "A_baseline_v4",
        "features": list(SELECTED_FEATURES),
        "params": {
            "max_depth": 3, "n_estimators": 300, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
            "reg_lambda": 1.0, "early_stopping_rounds": 50,
            "random_state": 42, "n_jobs": -1, "verbosity": 0,
            "eval_metric": "logloss",
        },
        "use_optuna": False,
    },
    {
        "name": "B_phase1_odds_delta",
        "features": list(V5_FEATURES_PHASE1),
        "params": {
            "max_depth": 4, "n_estimators": 400, "learning_rate": 0.04,
            "subsample": 0.8, "colsample_bytree": 0.75, "min_child_weight": 5,
            "reg_lambda": 2.0, "early_stopping_rounds": 50,
            "random_state": 42, "n_jobs": -1, "verbosity": 0,
            "eval_metric": "logloss",
        },
        "use_optuna": False,
    },
    {
        "name": "C_full_v5_optuna",
        "features": list(V5_FEATURES_FULL),
        "params": None,
        "use_optuna": True,
        "optuna_config": {
            "n_trials": 200,
            "timeout": 1200,
            "param_space": {
                "max_depth":        ("int",   3, 6),
                "n_estimators":     ("int",   200, 800),
                "learning_rate":    ("float", 0.01, 0.1, True),
                "subsample":        ("float", 0.6, 1.0),
                "colsample_bytree": ("float", 0.5, 1.0),
                "min_child_weight": ("int",   3, 15),
                "reg_alpha":        ("float", 1e-4, 5.0, True),
                "reg_lambda":       ("float", 0.5, 5.0),
                "gamma":            ("float", 0.0, 0.5),
            },
        },
    },
]

# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("  RETRAIN V5 -- DATASET_MASTER_08-05-2026.csv")
    logger.info("=" * 60)

    # --- Load dataset ---
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    logger.info("Master dataset: %d rows", len(df))
    logger.info("Label: WIN=%.1f%% LOSS=%.1f%%", df["label"].mean()*100, (1-df["label"].mean())*100)

    # --- Validate ---
    validate_no_leakage(df)

    # --- Split ---
    df_train, df_calib, df_test = chronological_split(
        df, calib_ratio=SPLIT_CFG.calib_holdout_ratio, test_ratio=0.15,
    )
    logger.info("Split: train=%d | calib=%d | test=%d", len(df_train), len(df_calib), len(df_test))

    # --- Impute ---
    df_train, imputer_vals = impute_features(df_train, strategy="median")
    df_calib = apply_imputer(df_calib, imputer_vals)
    df_test = apply_imputer(df_test, imputer_vals)

    # --- Run experiments ---
    all_results = []
    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        features = exp["features"]
        logger.info("")
        logger.info("=" * 60)
        logger.info("  EXPERIMENT: %s (%d features)", exp_name, len(features))
        logger.info("=" * 60)

        df_train_feat = build_features(df_train)
        df_test_feat = build_features(df_test)

        X_train = df_train_feat[features].values.astype(np.float32)
        y_train = df_train_feat[TARGET_COL].values.astype(np.int32)
        groups = get_market_groups(df_train_feat)

        # --- Optuna or fixed params ---
        if exp.get("use_optuna"):
            logger.info("Running Optuna HPO (%d trials)...", exp["optuna_config"]["n_trials"])
            optuna_result = run_optuna_hpo(
                X_train, y_train, groups,
                param_space=exp["optuna_config"]["param_space"],
                n_trials=exp["optuna_config"]["n_trials"],
                timeout=exp["optuna_config"]["timeout"],
            )
            model_params = optuna_result["best_params"]
            logger.info("Optuna best AUC: %.4f (%d trials)", optuna_result["best_auc"], optuna_result["n_trials_completed"])
        else:
            model_params = exp["params"]
            optuna_result = None

        # --- XGBConfig ---
        xgb_cfg = XGBConfig(**{k: v for k, v in model_params.items() if k in XGBConfig.__dataclass_fields__})

        # --- Cross-validate ---
        cv = cross_validate(X_train, y_train, groups, xgb_cfg=xgb_cfg, calibration_method="temperature")

        # --- Train & Calibrate ---
        tr = train_and_calibrate(
            df_train, df_calib, xgb_cfg=xgb_cfg,
            imputer_vals=imputer_vals, feature_list=features,
            calibration_method="temperature",
        )

        # --- Test eval ---
        X_test = df_test_feat[features].values.astype(np.float32)
        base_model = tr["base_model"]
        calibrator = tr["platt"]

        raw_proba = base_model.predict_proba(X_test)[:, 1]
        if isinstance(calibrator, TemperatureScaling):
            y_prob = calibrator.predict_proba(raw_proba)
        elif hasattr(calibrator, "predict"):
            y_prob = calibrator.predict(raw_proba)
        else:
            y_prob = raw_proba
        y_prob = np.clip(y_prob, 0.001, 0.999)

        # --- V5 Sim ---
        sim = simulate_ev_strategy_v5(df=df_test_feat, y_prob=y_prob)

        r = {
            "name": exp_name,
            "features": features,
            "n_features": len(features),
            "params": model_params,
            "cv_auc": cv["mean_auc"],
            "cv_std": cv["std_auc"],
            "ece_before": cv.get("ece_before", 0),
            "ece_after": cv.get("ece_after", 0),
            "temperature": cv.get("temperature"),
            "cv_result": cv,
            "train_result": tr,
            "sim_result": sim,
            "optuna_result": optuna_result,
        }
        all_results.append(r)

    # --- Ablation table ---
    print("\n" + "=" * 70)
    print(" ABLATION STUDY RESULTS -- DATASET_MASTER_08-05-2026")
    print("=" * 70)
    baseline_auc = all_results[0]["cv_auc"]
    for i, r in enumerate(all_results):
        letter = chr(ord("A") + i)
        delta = "--" if i == 0 else "%+.4f" % (r["cv_auc"] - baseline_auc)
        print(" %s  %-30s  AUC=%.4f +/-%.4f  delta=%s" % (letter, r["name"], r["cv_auc"], r["cv_std"], delta))
    best = max(all_results, key=lambda x: x["cv_auc"])
    print("=" * 70)
    print(" Best: %s (AUC=%.4f)" % (best["name"], best["cv_auc"]))
    print(" Target: >= 0.80 | Status: %s" % ("ACHIEVED" if best["cv_auc"] >= 0.80 else "BELUM TERCAPAI"))
    print("=" * 70)

    # --- Save best model ---
    final_model = best["train_result"]["base_model"]
    final_cal = best["train_result"]["platt"]

    with open(OUTPUT_DIR / "model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    with open(OUTPUT_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(final_cal, f)

    metadata = {
        "model_version": "v5_final",
        "created_at": datetime.now().isoformat(),
        "training_data": "DATASET_MASTER_08-05-2026.csv",
        "n_samples": len(df),
        "label_win_rate": float(df["label"].mean()),
        "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
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
        "model_params": best["params"],
        "execution_gates": {
            "edge_threshold": 0.05,
            "ttr_range_seconds": [60, 250],
            "clob_alignment_required": True,
        },
        "kelly_fraction": 0.25,
        "best_experiment": best["name"],
        "ablation_results": [
            {"name": r["name"], "cv_auc": r["cv_auc"], "cv_std": r["cv_std"], "n_features": r["n_features"]}
            for r in all_results
        ],
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # --- Report ---
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

    elapsed = time.time() - t0
    logger.info("[OK] Artifacts saved to: %s", OUTPUT_DIR)
    logger.info("Total elapsed: %.1f seconds", elapsed)

if __name__ == "__main__":
    main()
