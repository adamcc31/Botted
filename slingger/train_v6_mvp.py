"""
train_v6_mvp.py — Slingger V6 MVP Training Script
===================================================
Train the first production XGBoost model for Slingger V6.
Source of truth: data/dry_run_shadow_2026-06-06_154404_resolved.csv (179 rows)

Outputs:
  models/v6_production.pkl          — Trained XGBoost model
  models/v6_production_meta.json    — Training metadata + OOF AUC

Usage:
    python train_v6_mvp.py
    python train_v6_mvp.py --dry-run    # Validate only, no file writes

ABSOLUTE CONSTRAINTS:
  - Dataset must come from dry_run_shadow_2026-06-06_154404_resolved.csv
  - ZERO rows from before 2026-06-06 (data poisoning boundary)
  - Model only saves if OOF AUC > 0.50
  - All secrets via .env, none hardcoded
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────────
# Paths — all relative to this script's location (slingger/)
# ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR  # slingger/ is the self-contained root

DATA_SOURCE_FILENAME = "dry_run_shadow_2026-06-06_154404_resolved.csv"
DATA_PATH = ROOT_DIR / "data" / DATA_SOURCE_FILENAME
MODELS_DIR = ROOT_DIR / "models"
CANDIDATES_DIR = MODELS_DIR / "candidates"
LOGS_DIR = ROOT_DIR / "logs"
PRODUCTION_MODEL_PATH = MODELS_DIR / "v6_production.pkl"
PRODUCTION_META_PATH = MODELS_DIR / "v6_production_meta.json"
ENV_PATH = ROOT_DIR.parent / ".env"  # project root .env

# Hard boundary: REJECT any data before this date
CUTOFF_DATE = pd.Timestamp("2026-06-06", tz="UTC")

# ─────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(dotenv_path=ENV_PATH, override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        RotatingFileHandler(
            LOGS_DIR / "slingger_pipeline.log",
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        ),
        logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        ),
    ],
)
logger = logging.getLogger("train_v6_mvp")

# ─────────────────────────────────────────────────────────────────
# Feature Configuration
# ─────────────────────────────────────────────────────────────────
# Features to drop (metadata / identifiers / leakage)
DROP_COLUMNS = [
    "market_id",
    "timestamp",
    "slug",
    "spread_blocked_reason",
    "shadow_signal_yes",
    "shadow_signal_no",
    "shadow_tier_yes",
    "shadow_tier_no",
    "signal_direction",
    "actual_outcome",  # TARGET — must not be a feature
    "label",          # alias if present
    "session_date",   # derived from timestamp, not predictive at inference time
    "session_id",     # metadata
    "raw_clob_response",  # raw blob
]

# NOTE on StandardScaler:
# XGBoost is a tree-based ensemble — it is invariant to monotonic feature
# transformations (including scaling). StandardScaler is deliberately OMITTED
# here. This keeps inference simple: raw features → model → probability.
# If ever migrating to a neural net / logistic regression baseline, add scaling.

# ─────────────────────────────────────────────────────────────────
# Hyperparameters — EXTREME REGULARIZATION for 179-row dataset
# ─────────────────────────────────────────────────────────────────
MODEL_PARAMS = {
    # Architecture — shallow trees to prevent memorization
    "n_estimators": 100,
    "max_depth": 3,          # Very shallow — 3 levels max
    "learning_rate": 0.05,   # Slow learner = better generalization

    # Regularization — aggressive for tiny dataset
    "reg_alpha": 5.0,        # L1: promotes sparsity in leaf weights
    "reg_lambda": 10.0,      # L2: weight decay
    "min_child_weight": 10,  # Min ~10 samples to form any leaf
    "gamma": 1.0,            # Min loss reduction needed to split a node

    # Subsampling — each tree sees a random 65% of data + 65% of features
    "subsample": 0.65,
    "colsample_bytree": 0.65,
    "colsample_bylevel": 0.8,

    # Objective
    "objective": "binary:logistic",
    "eval_metric": "auc",

    # Reproducibility
    "random_state": 42,
    "n_jobs": -1,
}


# ─────────────────────────────────────────────────────────────────
# Step 1 — Data Loading & Validation
# ─────────────────────────────────────────────────────────────────
def load_and_validate_data(path: Path) -> pd.DataFrame:
    """Load CSV and run all pre-training assertions."""
    logger.info(f"[LOAD] Reading dataset: {path}")

    # ── Assert correct source file ──────────────────────────────
    assert path.name == DATA_SOURCE_FILENAME, (
        f"WRONG SOURCE FILE. Expected '{DATA_SOURCE_FILENAME}', got '{path.name}'. "
        f"ABORT — refusing to train on unauthorized data source."
    )

    df = pd.read_csv(path)
    logger.info(f"[LOAD] Raw shape: {df.shape}")

    # ── Parse timestamps ────────────────────────────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # ── Cutoff gate: HARD BOUNDARY — reject pre-06-Jun-2026 ─────
    pre_cutoff = df[df["timestamp"] < CUTOFF_DATE]
    if len(pre_cutoff) > 0:
        logger.error(
            f"[LOAD] ❌ POISON DATA DETECTED: {len(pre_cutoff)} rows before {CUTOFF_DATE}. ABORT."
        )
        raise AssertionError(
            f"DATA BERACUN PRE-06-JUN TERDETEKSI: {len(pre_cutoff)} baris. "
            f"PERMANENT BOUNDARY VIOLATED."
        )
    logger.info(f"[LOAD] OK: Cutoff gate passed - all {len(df)} rows >= {CUTOFF_DATE}")

    # ── Label column: dataset uses 'actual_outcome', normalize to 'label' ──
    # The production dataset uses 'actual_outcome' with values WIN/LOSE
    if "actual_outcome" in df.columns and "label" not in df.columns:
        df["label"] = df["actual_outcome"]
        logger.info("[LOAD] Mapped 'actual_outcome' → 'label'")

    # ── Validate labels ─────────────────────────────────────────
    assert "label" in df.columns, "No label column found (expected 'label' or 'actual_outcome')"
    invalid_labels = ~df["label"].isin(["WIN", "LOSE"])
    if invalid_labels.any():
        bad = df.loc[invalid_labels, "label"].unique()
        raise AssertionError(f"Label tidak bersih — nilai tidak valid: {bad}")
    logger.info(f"[LOAD] OK: Labels valid: {df['label'].value_counts().to_dict()}")

    # ── Minimum size check ──────────────────────────────────────
    assert len(df) >= 100, f"Dataset terlalu kecil: {len(df)} baris (minimum 100)"
    logger.info(f"[LOAD] OK: Size check passed: {len(df)} rows")

    # ── Class balance check ─────────────────────────────────────
    win_ratio = (df["label"] == "WIN").mean()
    logger.info(f"[LOAD] Class balance — WIN: {win_ratio:.2%}, LOSE: {1 - win_ratio:.2%}")
    if not (0.35 <= win_ratio <= 0.65):
        logger.warning(
            f"[LOAD] ⚠️  Class imbalance: WIN ratio {win_ratio:.2%} outside 35%-65% range. "
            f"Training proceeds with warning (MVP baseline exception)."
        )

    return df


# ─────────────────────────────────────────────────────────────────
# Step 2 — Feature Engineering
# ─────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Prepare features X and target y.

    Returns:
        X          — Feature DataFrame (numeric only)
        y          — Binary target Series (1=WIN, 0=LOSE)
        feat_names — List of feature column names
    """
    logger.info("[FEAT] Starting feature engineering...")

    # ── Encode label ─────────────────────────────────────────────
    y = (df["label"] == "WIN").astype(int)
    logger.info(f"[FEAT] Target encoded: WIN=1, LOSE=0. Positive rate: {y.mean():.2%}")

    # ── Drop non-feature columns ─────────────────────────────────
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    logger.info(f"[FEAT] Dropped {len(cols_to_drop)} metadata columns: {cols_to_drop}")

    # ── Keep only numeric columns (XGBoost requires numeric input) ──
    non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        logger.warning(f"[FEAT] Dropping non-numeric columns: {non_numeric}")
        X = X.drop(columns=non_numeric)

    # ── Handle missing values (simple median imputation) ─────────
    null_counts = X.isnull().sum()
    if null_counts.any():
        logger.info(f"[FEAT] Null values found — applying median imputation:\n{null_counts[null_counts > 0]}")
        X = X.fillna(X.median(numeric_only=True))
    else:
        logger.info("[FEAT] ✅ No null values — skipping imputation")

    feat_names = list(X.columns)
    logger.info(f"[FEAT] Final feature set ({len(feat_names)}): {feat_names}")
    logger.info(f"[FEAT] X shape: {X.shape}, y shape: {y.shape}")

    return X, y, feat_names


# ─────────────────────────────────────────────────────────────────
# Step 3 — Train + Validate
# ─────────────────────────────────────────────────────────────────
def train_and_validate(X: pd.DataFrame, y: pd.Series) -> tuple[XGBClassifier, float, float]:
    """
    Train XGBoost with Stratified K-Fold OOF AUC validation.

    Returns:
        model        — Fully trained XGBClassifier (fit on ALL data)
        oof_auc_mean — Mean OOF AUC across folds
        oof_auc_std  — Std of OOF AUC across folds
    """
    logger.info("[TRAIN] Initializing XGBClassifier with extreme regularization params...")
    logger.info(f"[TRAIN] Hyperparams: {MODEL_PARAMS}")

    model = XGBClassifier(**MODEL_PARAMS, verbosity=0)

    # ── Stratified K-Fold OOF AUC ────────────────────────────────
    logger.info("[TRAIN] Running 5-fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    oof_auc_mean = oof_scores.mean()
    oof_auc_std = oof_scores.std()

    logger.info(f"[TRAIN] OOF AUC scores per fold: {[f'{s:.4f}' for s in oof_scores]}")
    logger.info(f"[TRAIN] OOF AUC Mean: {oof_auc_mean:.4f} +/- {oof_auc_std:.4f}")

    # ── Guard: reject random-or-worse models ─────────────────────
    if oof_auc_mean <= 0.50:
        raise AssertionError(
            f"[TRAIN] FAIL: Model tidak lebih baik dari random! "
            f"OOF AUC = {oof_auc_mean:.4f} <= 0.50. Training ABORTED."
        )
    logger.info(f"[TRAIN] OK: AUC guard passed: {oof_auc_mean:.4f} > 0.50")

    # ── Final fit on ALL data ─────────────────────────────────────
    logger.info("[TRAIN] Fitting final model on full training set...")
    model.fit(X, y)
    logger.info("[TRAIN] OK: Final model fitted on all data.")

    return model, oof_auc_mean, oof_auc_std


# ─────────────────────────────────────────────────────────────────
# Step 4 — Save Artifacts
# ─────────────────────────────────────────────────────────────────
def save_artifacts(
    model: XGBClassifier,
    df: pd.DataFrame,
    feat_names: list[str],
    oof_auc_mean: float,
    oof_auc_std: float,
    dry_run: bool = False,
) -> None:
    """Save model pickle and metadata JSON."""

    meta = {
        "version": "6.0.0-mvp",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATA_SOURCE_FILENAME,
        "n_rows": len(df),
        "features": feat_names,
        "n_features": len(feat_names),
        "oof_auc_mean": round(float(oof_auc_mean), 6),
        "oof_auc_std": round(float(oof_auc_std), 6),
        "hyperparams": MODEL_PARAMS,
        "label_encoding": {"WIN": 1, "LOSE": 0},
        "label_column_source": "actual_outcome",
        "win_count": int((df["label"] == "WIN").sum()),
        "lose_count": int((df["label"] == "LOSE").sum()),
        "win_ratio": round(float((df["label"] == "WIN").mean()), 6),
        "cutoff_date": "2026-06-06",
        "scaler": None,  # Deliberately None — XGBoost does not need scaling
        "notes": (
            "V6 MVP baseline. Trained on 179 rows post-cache-bug fix. "
            "Extreme regularization applied for small dataset. "
            "StandardScaler omitted (XGBoost is scale-invariant). "
            "AUC > 0.50 guard enforced."
        ),
    }

    if dry_run:
        logger.info("[SAVE] DRY RUN — no files written.")
        logger.info(f"[SAVE] Would write model to: {PRODUCTION_MODEL_PATH}")
        logger.info(f"[SAVE] Would write meta to:  {PRODUCTION_META_PATH}")
        logger.info(f"[SAVE] Meta preview:\n{json.dumps(meta, indent=2)}")
        return

    # ── Atomic-safe save: write then rename ─────────────────────
    tmp_model_path = PRODUCTION_MODEL_PATH.with_suffix(".pkl.tmp")
    tmp_meta_path = PRODUCTION_META_PATH.with_suffix(".json.tmp")

    logger.info(f"[SAVE] Writing model to temp: {tmp_model_path}")
    joblib.dump(model, tmp_model_path)
    os.replace(tmp_model_path, PRODUCTION_MODEL_PATH)
    logger.info(f"[SAVE] OK: Model atomically saved: {PRODUCTION_MODEL_PATH}")

    logger.info(f"[SAVE] Writing metadata to temp: {tmp_meta_path}")
    with open(tmp_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_meta_path, PRODUCTION_META_PATH)
    logger.info(f"[SAVE] OK: Metadata atomically saved: {PRODUCTION_META_PATH}")


# ─────────────────────────────────────────────────────────────────
# Step 5 — Post-train Housekeeping
# ─────────────────────────────────────────────────────────────────
def update_active_model_config(oof_auc_mean: float) -> None:
    """Write ACTIVE_MODEL to .env (or log it for manual update)."""
    active_model_value = str(PRODUCTION_MODEL_PATH)

    # Log the event
    logger.info(
        f"[DEPLOY] V6 MVP trained. AUC={oof_auc_mean:.4f}. "
        f"ACTIVE_MODEL={active_model_value}"
    )

    # Attempt to update .env ACTIVE_MODEL line if it exists
    try:
        env_content = ""
        if ENV_PATH.exists():
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                env_content = f.read()

        if "ACTIVE_MODEL=" in env_content:
            lines = env_content.splitlines()
            new_lines = [
                f"ACTIVE_MODEL={active_model_value}" if line.startswith("ACTIVE_MODEL=") else line
                for line in lines
            ]
            new_content = "\n".join(new_lines) + "\n"
            with open(ENV_PATH, "w", encoding="utf-8") as f:
                f.write(new_content)
            logger.info(f"[DEPLOY] Updated ACTIVE_MODEL in {ENV_PATH}")
        else:
            # Append to .env
            with open(ENV_PATH, "a", encoding="utf-8") as f:
                f.write(f"\n# --- Slingger V6 Active Model ---\n")
                f.write(f"ACTIVE_MODEL={active_model_value}\n")
            logger.info(f"[DEPLOY] Appended ACTIVE_MODEL to {ENV_PATH}")
    except Exception as e:
        logger.warning(f"[DEPLOY] Could not update .env: {e}. Manual update required.")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main(dry_run: bool = False) -> None:
    logger.info("=" * 60)
    logger.info("SLINGGER V6 MVP — TRAINING PIPELINE START")
    logger.info(f"  Source of truth : {DATA_PATH}")
    logger.info(f"  Dry run         : {dry_run}")
    logger.info(f"  Cutoff date     : {CUTOFF_DATE}")
    logger.info("=" * 60)

    try:
        # Import notifier here to avoid circular issues at module level
        from notifier import send_telegram
    except ImportError:
        logger.warning("[MAIN] notifier.py not found — Telegram alerts disabled.")
        send_telegram = lambda msg: None  # noqa: E731

    try:
        # ── Step 1: Load + Validate ───────────────────────────────
        df = load_and_validate_data(DATA_PATH)

        # ── Step 2: Feature Engineering ───────────────────────────
        X, y, feat_names = engineer_features(df)

        # ── Step 3: Train + Validate ──────────────────────────────
        model, oof_auc_mean, oof_auc_std = train_and_validate(X, y)

        # ── Step 4: Save Artifacts ────────────────────────────────
        save_artifacts(model, df, feat_names, oof_auc_mean, oof_auc_std, dry_run=dry_run)

        # ── Step 5: Post-train ────────────────────────────────────
        if not dry_run:
            update_active_model_config(oof_auc_mean)

        # ── Summary ───────────────────────────────────────────────
        summary = (
            f"[OK] *V6 MVP Training Complete*\n"
            f"Dataset: {len(df)} rows\n"
            f"Features: {len(feat_names)}\n"
            f"OOF AUC: {oof_auc_mean:.4f} +/- {oof_auc_std:.4f}\n"
            f"Model: `v6_production.pkl`"
        )
        logger.info(f"\n{'=' * 60}\n{summary.replace('*', '').replace('`', '')}\n{'=' * 60}")

        if not dry_run:
            send_telegram(summary)

        sys.stdout.write(f"\nV6 MVP deployed. OOF AUC: {oof_auc_mean:.4f}\n")
        return 0

    except AssertionError as e:
        logger.error(f"[MAIN] ASSERTION FAILED: {e}")
        send_telegram(f"[FAIL] *V6 Training FAILED*\n`{e}`")
        return 1
    except Exception as e:
        logger.exception(f"[MAIN] UNEXPECTED ERROR: {e}")
        send_telegram(f"[CRASH] *V6 Training CRASHED*\n`{type(e).__name__}: {e}`")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slingger V6 MVP Training Script")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and train model but do NOT write any files",
    )
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run))
