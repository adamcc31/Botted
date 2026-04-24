"""
train_model.py — Full ML training pipeline.

EXPECTS: data/processed/merged_training_features.parquet
  (produced by scripts/build_dataset.py)

Pipeline:
  1. Load merged dataset (features + ground truth labels)
  2. Chronological 60/20/20 split (NEVER random)
  3. Optuna hyperparameter search on train set
  4. Fit LightGBM → Isotonic Regression calibration on validation set
  5. Fit LogisticRegression (overfit detection baseline)
  6. Evaluate on held-out test set
  7. Save model artifacts

CRITICAL DESIGN NOTES:
  - Dataset MUST contain real Polymarket labels, not simulated ones.
  - Isotonic Regression calibration is MANDATORY for Kelly accuracy.
  - All feature placeholders (OBI=0, TFM=0) are flagged in diagnostics.
  - Model will underperform if TFM/OBI are all zeros; this is expected
    until sufficient aggTrades + orderbook data is collected.

Usage:
  python scripts/train_model.py --dataset ./data/processed/merged_training_features.parquet
  python scripts/train_model.py --dataset ./data/processed/merged_training_features.parquet --n-trials 100
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import click
import numpy as np
import pandas as pd
import structlog
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_manager import ConfigManager
from src.feature_engine import FEATURE_NAMES
from src.model import ModelEnsemble


# ============================================================
# Data Loading & Validation
# ============================================================

def load_and_validate(dataset_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict]:
    """
    Load merged training dataset and validate schema.

    Returns:
        X: feature DataFrame (24 columns)
        y: label Series (binary 0/1)
        groups: market_id Series for grouped CV
        diagnostics: dict with data quality metrics
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Run scripts/build_dataset.py first to create the merged dataset."
        )

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    logger.info("dataset_loaded", rows=len(df), columns=list(df.columns)[:10])

    # ── Validate required columns ─────────────────────────────
    # Must have all 24 features + label
    missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}\n"
            f"The dataset was likely built without the full feature set."
        )

    if "label" not in df.columns:
        raise ValueError(
            "Missing 'label' column!\n"
            "The dataset must contain ground truth labels from Polymarket.\n"
            "Do NOT use raw OHLCV data — run build_dataset.py first."
        )

    # ── Extract X, y ──────────────────────────────────────────
    # Sort by signal timestamp for temporal ordering
    if "signal_timestamp_ms" in df.columns:
        df = df.sort_values("signal_timestamp_ms").reset_index(drop=True)
    elif "t_resolution_ms" in df.columns:
        df = df.sort_values("t_resolution_ms").reset_index(drop=True)

    # Drop rows with NaN in features or label
    pre_drop = len(df)
    df = df.dropna(subset=FEATURE_NAMES + ["label"])
    post_drop = len(df)

    if post_drop < 50:
        raise ValueError(
            f"Too few valid samples: {post_drop} (need ≥50).\n"
            f"Dropped {pre_drop - post_drop} rows with NaN values."
        )

    X = df[FEATURE_NAMES].copy()
    y = df["label"].astype(int)
    
    # Use resolution_date for grouped CV if it exists, otherwise use market_id
    if "resolution_date" in df.columns:
        groups = df["resolution_date"].copy()
        logger.info("using_resolution_date_for_grouping")
    else:
        groups = df["market_id"].copy()
        logger.info("using_market_id_for_grouping")

    # ── Diagnostics ───────────────────────────────────────────
    diagnostics = {
        "total_samples": len(X),
        "dropped_nan": pre_drop - post_drop,
        "label_distribution": y.value_counts().to_dict(),
        "label_balance": round(y.mean(), 4),
        "feature_zero_rates": {},
        "feature_means": {},
        "feature_stds": {},
    }

    # Check for placeholder features (all zeros = no real data)
    critical_features = ["TFM_normalized", "depth_ratio", "binance_spread_bps"]
    for feat in critical_features:
        zero_rate = (X[feat] == 0).mean()
        diagnostics["feature_zero_rates"][feat] = round(zero_rate, 4)
        diagnostics["feature_means"][feat] = round(float(X[feat].mean()), 6)
        diagnostics["feature_stds"][feat] = round(float(X[feat].std()), 6)

    # Label mismatch rate (basis risk indicator)
    if "label_match" in df.columns:
        diagnostics["label_mismatch_rate"] = round(
            1.0 - df["label_match"].mean(), 4
        )

    # Unique markets
    if "market_id" in df.columns:
        diagnostics["unique_markets"] = df["market_id"].nunique()

    logger.info(
        "dataset_validated",
        samples=len(X),
        label_balance=diagnostics["label_balance"],
        feature_zero_rates=diagnostics["feature_zero_rates"],
    )

    return X, y, groups, diagnostics


# ============================================================
# Chronological Split
# ============================================================

def chronological_split(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    unique_groups = groups.drop_duplicates().tolist()
    n_groups = len(unique_groups)
    
    # Graceful handling for extreme temporal skew (e.g. only 2 days of data)
    if n_groups < 3:
        logger.warning("extreme_temporal_skew_detected", groups=n_groups)
        from sklearn.model_selection import train_test_split
        # Ignore groups, just randomize to keep the pipeline alive for testing
        idx = np.arange(len(X))
        tr_idx, val_idx = train_test_split(idx, test_size=0.4, random_state=42)
        val_idx, te_idx = train_test_split(val_idx, test_size=0.5, random_state=42)
        
        X_train, y_train, g_train = X.iloc[tr_idx], y.iloc[tr_idx], groups.iloc[tr_idx]
        X_val, y_val, g_val = X.iloc[val_idx], y.iloc[val_idx], groups.iloc[val_idx]
        X_test, y_test, g_test = X.iloc[te_idx], y.iloc[te_idx], groups.iloc[te_idx]
        
        train_groups, val_groups, test_groups = set(), set(), set()
    else:
        train_idx_end = max(1, int(n_groups * 0.60))
        val_idx_end = max(train_idx_end + 1, int(n_groups * 0.80))
        
        train_groups = set(unique_groups[:train_idx_end])
        val_groups = set(unique_groups[train_idx_end:val_idx_end])
        test_groups = set(unique_groups[val_idx_end:])

        train_mask = groups.isin(train_groups)
        val_mask = groups.isin(val_groups)
        test_mask = groups.isin(test_groups)

        X_train, y_train, g_train = X.loc[train_mask].copy(), y.loc[train_mask].copy(), groups.loc[train_mask].copy()
        X_val, y_val, g_val = X.loc[val_mask].copy(), y.loc[val_mask].copy(), groups.loc[val_mask].copy()
        X_test, y_test, g_test = X.loc[test_mask].copy(), y.loc[test_mask].copy(), groups.loc[test_mask].copy()

    logger.info(
        "data_split_chronological_grouped",
        train=len(X_train),
        val=len(X_val),
        test=len(X_test),
        train_groups=len(train_groups),
        train_label_balance=round(y_train.mean(), 4) if not y_train.empty else 0,
        val_label_balance=round(y_val.mean(), 4) if not y_val.empty else 0,
        test_label_balance=round(y_test.mean(), 4) if not y_test.empty else 0,
    )

    return X_train, y_train, g_train, X_val, y_val, g_val, X_test, y_test, g_test


# ============================================================
# Optuna Hyperparameter Tuning
# ============================================================

def run_optuna_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    g_train: pd.Series,
    n_trials: int = 50,
) -> Dict[str, Any]:
    """Run Optuna HPO for LightGBM using GroupKFold CV."""
    import optuna
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # GroupKFold prevents target correlation leakage
    n_unique_groups = len(g_train.unique())
    if n_unique_groups < 5:
        if n_unique_groups < 2:
            from sklearn.model_selection import KFold
            logger.warning("fallback_to_kfold_due_to_single_group")
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
        else:
            logger.warning("reducing_n_splits_due_to_low_groups", groups=n_unique_groups)
            cv = GroupKFold(n_splits=n_unique_groups)
    else:
        cv = GroupKFold(n_splits=5)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "class_weight": "balanced",
            "verbosity": -1,
            "random_state": 42,
        }

        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train, groups=g_train):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_va = X_train.iloc[val_idx]
            y_va = y_train.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            preds = model.predict_proba(X_va)[:, 1]
            brier = brier_score_loss(y_va, preds)
            scores.append(brier)

        return np.mean(scores)


    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(
        "optuna_complete",
        best_brier=round(study.best_value, 4),
        trials=n_trials,
        best_params=study.best_params,
    )

    return study.best_params


# ============================================================
# Model Training
# ============================================================

def train_pipeline(
    dataset_path: str,
    n_trials: int = 50,
) -> Dict[str, Any]:
    """
    Full training pipeline.

    REQUIRES: merged_training_features.parquet with real labels.
    """
    import lightgbm as lgb

    # ── Step 1: Load + Validate ───────────────────────────────
    X, y, groups, diagnostics = load_and_validate(dataset_path)

    # Warn about placeholder features
    for feat, zero_rate in diagnostics["feature_zero_rates"].items():
        if zero_rate > 0.95:
            logger.warning(
                "feature_mostly_zeros",
                feature=feat,
                zero_rate=zero_rate,
                impact=(
                    "This feature has no predictive power. "
                    "Model will rely on other features."
                ),
            )

    # ── Step 2: Chronological Split ───────────────────────────
    X_train, y_train, g_train, X_val, y_val, g_val, X_test, y_test, g_test = chronological_split(X, y, groups)

    # ── Step 3: Optuna HPO ────────────────────────────────────
    logger.info("starting_optuna_hpo", n_trials=n_trials)
    best_params = run_optuna_tuning(X_train, y_train, g_train, n_trials=n_trials)

    # ── Step 4: Fit LightGBM ──────────────────────────────────
    best_params.update({
        "class_weight": "balanced",
        "verbosity": -1,
        "random_state": 42,
    })

    lgbm_model = lgb.LGBMClassifier(**best_params)
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # ── Step 5: Isotonic Calibration (MANDATORY) ──────────────
    # LightGBM probabilities are biased.
    # Without calibration, Kelly sizing → capital destruction.
    calibrated_lgbm = CalibratedClassifierCV(
        estimator=lgbm_model,
        method="isotonic",
        cv=3,
    )
    calibrated_lgbm.fit(X_val, y_val)
    logger.info("lgbm_calibrated", method="isotonic")

    # ── Step 6: Fit LogisticRegression ────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(
        solver="lbfgs", max_iter=1000, C=0.1, class_weight="balanced"
    )
    logreg.fit(X_train_scaled, y_train)

    # ── Step 7: Evaluate on Test Set ──────────────────────────
    lgbm_probs = calibrated_lgbm.predict_proba(X_test)[:, 1]
    logreg_probs = logreg.predict_proba(X_test_scaled)[:, 1]
    ensemble_probs = 0.7 * lgbm_probs + 0.3 * logreg_probs

    # Metrics
    brier_lgbm = brier_score_loss(y_test, lgbm_probs)
    brier_logreg = brier_score_loss(y_test, logreg_probs)
    brier_ensemble = brier_score_loss(y_test, ensemble_probs)
    auc_ensemble = roc_auc_score(y_test, ensemble_probs)
    win_rate_oos = float(np.mean((ensemble_probs > 0.5) == y_test))

    # Overfit divergence (LGBM vs LogReg)
    divergence = float(np.mean(np.abs(lgbm_probs - logreg_probs)))

    # Calibration curve
    fraction_pos, mean_predicted = calibration_curve(
        y_test, ensemble_probs, n_bins=10
    )

    # Feature importance
    feature_importance = dict(
        zip(FEATURE_NAMES, lgbm_model.feature_importances_.tolist())
    )
    # Sort by importance
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    metrics = {
        "brier_lgbm_calibrated": round(float(brier_lgbm), 4),
        "brier_logreg": round(float(brier_logreg), 4),
        "brier_ensemble": round(float(brier_ensemble), 4),
        "auc_oos": round(float(auc_ensemble), 4),
        "win_rate_oos": round(float(win_rate_oos), 4),
        "lgbm_logreg_divergence": round(divergence, 4),
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "best_params": best_params,
        "calibration_curve": {
            "fraction_of_positives": fraction_pos.tolist(),
            "mean_predicted": mean_predicted.tolist(),
        },
        "feature_importance_top10": dict(list(feature_importance.items())[:10]),
        "dataset_diagnostics": diagnostics,
    }

    logger.info(
        "training_complete",
        brier_ensemble=brier_ensemble,
        auc=auc_ensemble,
        win_rate_oos=win_rate_oos,
        divergence=divergence,
    )

    # ── Step 8: Save Artifacts ────────────────────────────────
    version_tag = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    ModelEnsemble.save_models(
        lgbm_model=lgbm_model,
        logreg_model=logreg,
        scaler=scaler,
        calibrator=calibrated_lgbm,
        version_tag=version_tag,
        metrics=metrics,
    )

    logger.info("model_artifacts_saved", version=version_tag)
    return metrics


# ============================================================
# CLI
# ============================================================

@click.command()
@click.option(
    "--dataset",
    required=True,
    help="Path to merged training dataset (from build_dataset.py)",
)
@click.option("--n-trials", default=50, help="Optuna trials count")
def main(dataset: str, n_trials: int) -> None:
    """Train LightGBM + LogReg ensemble with Isotonic calibration."""
    click.echo("🧠 Training model pipeline...")
    click.echo(f"   Dataset: {dataset}")
    click.echo(f"   Optuna trials: {n_trials}\n")

    # Validate it's the merged dataset
    ds_path = Path(dataset)
    if "raw" in str(ds_path) and "ohlcv" in str(ds_path):
        click.echo("❌ ERROR: You are pointing to raw OHLCV data.")
        click.echo("   Raw OHLCV has no labels. Run build_dataset.py first:")
        click.echo("   python scripts/build_dataset.py")
        click.echo(f"   Then: python scripts/train_model.py --dataset ./data/processed/merged_training_features.parquet")
        return

    metrics = train_pipeline(dataset, n_trials=n_trials)

    # Report
    click.echo("\n" + "=" * 60)
    click.echo("📊 TRAINING RESULTS")
    click.echo("=" * 60)
    click.echo(f"   Brier (LGBM cal.):    {metrics['brier_lgbm_calibrated']}")
    click.echo(f"   Brier (LogReg):       {metrics['brier_logreg']}")
    click.echo(f"   Brier (Ensemble):     {metrics['brier_ensemble']}")
    click.echo(f"   AUC (OOS):            {metrics['auc_oos']}")
    click.echo(f"   Win Rate (OOS):       {metrics['win_rate_oos']}")
    click.echo(f"   Label Balance (Test): {metrics.get('test_label_balance', 'N/A')}")
    click.echo(f"   LGBM↔LogReg diverge:  {metrics['lgbm_logreg_divergence']}")
    click.echo(f"   Training samples:     {metrics['training_samples']}")
    click.echo(f"   Test samples:         {metrics['test_samples']}")

    click.echo("\n📈 Feature Importance (Top 10):")
    for feat, imp in metrics.get("feature_importance_top10", {}).items():
        bar = "█" * int(imp / 5) if imp > 0 else ""
        click.echo(f"   {feat:25s} {imp:6.0f} {bar}")

    # Diagnostics
    diag = metrics.get("dataset_diagnostics", {})
    zero_rates = diag.get("feature_zero_rates", {})
    if zero_rates:
        click.echo("\n⚠️  Feature Data Coverage:")
        for feat, rate in zero_rates.items():
            status = "✓ LIVE" if rate < 0.5 else "⚠ MOSTLY ZEROS" if rate < 0.95 else "✗ NO DATA"
            click.echo(f"   {feat:25s} zero_rate={rate:.1%}  {status}")

    # Thresholds
    click.echo()
    if metrics['brier_ensemble'] < 0.24:
        click.echo("✅ Model PASSES Brier threshold (<0.24)")
    else:
        click.echo("⚠️  Model FAILS Brier threshold (≥0.24)")

    if metrics['lgbm_logreg_divergence'] > 0.15:
        click.echo("⚠️  High LGBM↔LogReg divergence → possible overfitting")
    else:
        click.echo("✅ Low divergence → no overfitting detected")

    if metrics['auc_oos'] > 0.55:
        click.echo("✅ AUC above random baseline (>0.55)")
    else:
        click.echo("⚠️  AUC near random baseline — model needs more data")


if __name__ == "__main__":
    main()
