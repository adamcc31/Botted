"""
scripts/auto_retrain.py — Slingger V6 CI/CD Auto-Retrain Pipeline
===================================================================
Automatically retrains V6 when new resolved data crosses batch thresholds.
Three mathematical guardrails protect production from regression.

GUARDRAILS (all must pass for atomic production swap):
  1. Class Balance:    WIN ratio must be 35%-65%
  2. AUC Improvement: Candidate OOF AUC > Production OOF AUC
  3. Atomic Swap:     os.replace() — single syscall, corruption-proof

ABSOLUTE CONSTRAINTS:
  - ZERO data before 2026-06-06 (hard cutoff, never change)
  - NEVER overwrite v6_production.pkl without all 3 guardrails passing
  - All secrets from .env — never hardcoded
  - Always notify Telegram on success OR failure

Usage:
    python scripts/auto_retrain.py             # Check trigger, retrain if needed
    python scripts/auto_retrain.py --force     # Bypass row-count trigger, retrain now
    python scripts/auto_retrain.py --dry-run   # Simulate without writing
    python scripts/auto_retrain.py --daemon    # Run as scheduled daemon
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import schedule
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
SLINGGER_DIR = SCRIPT_DIR.parent          # slingger/
ROOT_DIR = SLINGGER_DIR.parent            # project root

if Path("/app/data").exists():
    DATA_DIR = Path("/app/data")
elif (ROOT_DIR / "data").exists():
    DATA_DIR = ROOT_DIR / "data"
else:
    DATA_DIR = SLINGGER_DIR / "data"

MODELS_DIR = SLINGGER_DIR / "models"
CANDIDATES_DIR = MODELS_DIR / "candidates"
LOGS_DIR = SLINGGER_DIR / "logs"
ENV_PATH = ROOT_DIR / ".env"

PRODUCTION_MODEL_PATH = MODELS_DIR / "v6_production.pkl"
PRODUCTION_META_PATH = MODELS_DIR / "v6_production_meta.json"
CANDIDATE_MODEL_PATH = CANDIDATES_DIR / "v6_candidate.pkl"
CANDIDATE_META_PATH = CANDIDATES_DIR / "v6_candidate_meta.json"

# ─────────────────────────────────────────────────────────────────
# HARD BOUNDARY — NEVER CHANGE THIS VALUE
# ─────────────────────────────────────────────────────────────────
CUTOFF_DATE = date(2026, 6, 6)  # PERMANENT IGNORE — data sebelum ini beracun

# ─────────────────────────────────────────────────────────────────
# Trigger Configuration
# ─────────────────────────────────────────────────────────────────
RETRAIN_BATCH_SIZE = 250   # Retrain at: 250, 500, 750, ... resolved rows
MIN_DATASET_ROWS = 179     # Absolute minimum — equals actual V6 MVP baseline (179 rows)

# ─────────────────────────────────────────────────────────────────
# GUARDRAIL 1 Thresholds
# ─────────────────────────────────────────────────────────────────
GUARDRAIL_WIN_MIN = 0.35   # WIN ratio lower bound
GUARDRAIL_WIN_MAX = 0.65   # WIN ratio upper bound

# ─────────────────────────────────────────────────────────────────
# XGBoost Hyperparameters (IDENTICAL to train_v6_mvp.py — do not diverge)
# ─────────────────────────────────────────────────────────────────
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "reg_alpha": 5.0,
    "reg_lambda": 10.0,
    "min_child_weight": 10,
    "gamma": 1.0,
    "subsample": 0.65,
    "colsample_bytree": 0.65,
    "colsample_bylevel": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
}

# Feature columns to drop (same policy as train_v6_mvp.py)
DROP_COLUMNS = [
    "market_id", "timestamp", "slug", "spread_blocked_reason",
    "shadow_signal_yes", "shadow_signal_no", "shadow_tier_yes", "shadow_tier_no",
    "signal_direction", "actual_outcome", "label", "session_date",
    "session_id", "raw_clob_response",
]

# ─────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)
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
logger = logging.getLogger("auto_retrain")


def _get_notifier():
    """Lazily import notifier to avoid circular imports."""
    try:
        sys.path.insert(0, str(SLINGGER_DIR))
        from notifier import send_telegram
        return send_telegram
    except ImportError:
        logger.warning("[RETRAIN] notifier.py not found — Telegram alerts disabled")
        return lambda msg: None


# ─────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────
def bump_version(version_str: str) -> str:
    """
    Increment the patch version: "6.0.0-mvp" → "6.0.1", "6.0.7" → "6.0.8"
    """
    clean = version_str.replace("-mvp", "").replace("-candidate", "")
    parts = clean.split(".")
    try:
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        return f"{major}.{minor}.{patch + 1}"
    except (ValueError, IndexError):
        return f"{version_str}.1"


def load_production_meta() -> Optional[dict]:
    """Load production model metadata. Returns None if file missing."""
    if not PRODUCTION_META_PATH.exists():
        logger.error(f"[RETRAIN] Production meta not found: {PRODUCTION_META_PATH}")
        return None
    with open(PRODUCTION_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────
# Trigger Check
# ─────────────────────────────────────────────────────────────────
def get_total_clob_rows() -> int:
    """Scan all clob_log*.csv files recursively and count total rows."""
    total = 0
    for path in DATA_DIR.rglob("clob_log*.csv"):
        try:
            with open(path, "rb") as f:
                total += sum(1 for _ in f) - 1
        except Exception:
            pass
    return total


def should_trigger_retrain(total_resolved_rows: int, current_clob_rows: int, production_meta: dict) -> bool:
    """
    Returns True if:
      1. Total resolved row count crosses a batch threshold (250, 500, ...)
      2. OR clob_delta >= 50,000 since last training
    """
    # 1. Row count trigger
    row_trigger = total_resolved_rows > 0 and (total_resolved_rows % RETRAIN_BATCH_SIZE == 0)

    # 2. Clob delta trigger
    last_clob_rows = production_meta.get("clob_rows_trained", 0)
    clob_delta = current_clob_rows - last_clob_rows
    clob_trigger = clob_delta >= 50_000

    if clob_trigger:
        logger.info(
            f"[TRIGGER] Clob delta trigger activated: {clob_delta:,} new rows >= 50,000 "
            f"(current: {current_clob_rows:,}, last: {last_clob_rows:,})"
        )

    return row_trigger or clob_trigger


# ─────────────────────────────────────────────────────────────────
# Dataset Assembly
# ─────────────────────────────────────────────────────────────────
def assemble_dataset() -> Optional[pd.DataFrame]:
    """
    Combine ALL resolved CSVs from data/ that:
      - Have session date >= CUTOFF_DATE
      - Contain only WIN/LOSE labels (drop PENDING/SKIP)
    
    Returns assembled DataFrame, or None if insufficient data.
    """
    pattern = "dry_run_shadow_*.csv"
    all_dfs = []

    for csv_path in sorted(DATA_DIR.rglob(pattern)):
        if "combined" in csv_path.name:
            continue
        # ── CRITICAL GATE — PERMANENT IGNORE ─────────────────────
        try:
            parts = csv_path.stem.split("_")
            date_str = None
            for part in parts:
                if len(part) == 10 and part.count("-") == 2:
                    date_str = part
                    break

            if date_str is None:
                continue

            file_date = date.fromisoformat(date_str)

            if file_date < CUTOFF_DATE:
                logger.warning(
                    f"[DATASET] SKIPPED pre-cutoff session: {csv_path.name} "
                    f"(date={file_date} < cutoff={CUTOFF_DATE})"
                )
                continue  # PERMANENT IGNORE — data beracun

        except Exception as e:
            logger.warning(f"[DATASET] Cannot parse date from {csv_path.name}: {e} — skip")
            continue

        try:
            df_chunk = pd.read_csv(csv_path)

            # Normalize label column
            if "actual_outcome" in df_chunk.columns and "label" not in df_chunk.columns:
                df_chunk["label"] = df_chunk["actual_outcome"]

            if "label" not in df_chunk.columns:
                logger.warning(f"[DATASET] No label column in {csv_path.name} — skip")
                continue

            # Keep only WIN/LOSE (drop PENDING, SKIP, null)
            labeled = df_chunk[df_chunk["label"].isin(["WIN", "LOSE"])].copy()

            if len(labeled) > 0:
                all_dfs.append(labeled)
                logger.debug(f"[DATASET] {csv_path.name}: {len(labeled)} labeled rows included")

        except Exception as e:
            logger.error(f"[DATASET] Failed to read {csv_path.name}: {e}")

    if not all_dfs:
        logger.error("[DATASET] No eligible data found in data/ directory")
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Drop duplicates by (timestamp, market_id)
    if "timestamp" in combined.columns and "market_id" in combined.columns:
        before_dedup = len(combined)
        combined = combined.drop_duplicates(subset=["timestamp", "market_id"])
        logger.info(
            f"[DATASET] Assembled dataset: {len(combined)} rows (removed {before_dedup - len(combined)} duplicates) "
            f"from {len(all_dfs)} files"
        )
    else:
        logger.info(f"[DATASET] Assembled dataset: {len(combined)} rows from {len(all_dfs)} files")

    # ── Minimum row check ─────────────────────────────────────────
    if len(combined) < MIN_DATASET_ROWS:
        logger.error(
            f"[DATASET] Dataset too small: {len(combined)} rows < minimum {MIN_DATASET_ROWS}"
        )
        return None

    return combined


# ─────────────────────────────────────────────────────────────────
# Feature Engineering (mirrors train_v6_mvp.py)
# ─────────────────────────────────────────────────────────────────
def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare X, y from assembled DataFrame."""
    y = (df["label"] == "WIN").astype(int)

    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)

    null_counts = X.isnull().sum()
    if null_counts.any():
        X = X.fillna(X.median(numeric_only=True))

    return X, y, list(X.columns)


# ─────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────
def train_candidate(X: pd.DataFrame, y: pd.Series) -> tuple[XGBClassifier, float, float]:
    """Train a candidate model. Returns (model, oof_auc_mean, oof_auc_std)."""
    logger.info(f"[TRAIN] Training V6 candidate on {len(X)} rows...")
    model = XGBClassifier(**MODEL_PARAMS, verbosity=0)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    oof_mean = float(oof_scores.mean())
    oof_std = float(oof_scores.std())

    logger.info(f"[TRAIN] Candidate OOF AUC: {oof_mean:.4f} ± {oof_std:.4f}")
    logger.info(f"[TRAIN] Per-fold: {[f'{s:.4f}' for s in oof_scores]}")

    # Fit on full data
    model.fit(X, y)
    return model, oof_mean, oof_std


# ─────────────────────────────────────────────────────────────────
# Core Retrain Pipeline
# ─────────────────────────────────────────────────────────────────
def run_retrain(force: bool = False, dry_run: bool = False) -> bool:
    """
    Execute the full retrain pipeline with 3 guardrails.

    Returns:
        True if production was updated, False otherwise.
    """
    send_telegram = _get_notifier()
    logger.info("=" * 60)
    logger.info("SLINGGER V6 - AUTO RETRAIN PIPELINE START")
    logger.info(f"  Force      : {force}")
    logger.info(f"  Dry Run    : {dry_run}")
    logger.info("=" * 60)

    # ── Load production metadata ──────────────────────────────────
    production_meta = load_production_meta()
    if production_meta is None:
        msg = (
            "[ABORT] V6 Retrain ABORTED\n"
            "Production meta file missing. Run `train_v6_mvp.py` first."
        )
        logger.error(f"[RETRAIN] Production meta not found.")
        send_telegram(msg)
        return False

    production_auc = production_meta.get("oof_auc_mean", 0.0)
    logger.info(f"[RETRAIN] Production AUC baseline: {production_auc:.4f}")

    # ── Assemble dataset ──────────────────────────────────────────
    df = assemble_dataset()
    if df is None:
        msg = (
            "[ABORT] V6 Retrain ABORTED\n"
            "Insufficient data after assembly."
        )
        logger.error(f"[RETRAIN] Insufficient data after assembly.")
        send_telegram(msg)
        return False

    total_rows = len(df)
    current_clob_rows = get_total_clob_rows()
    logger.info(f"[RETRAIN] Current CLOB log rows: {current_clob_rows:,}")

    # ── Trigger check (skip if --force) ──────────────────────────
    if not force and not should_trigger_retrain(total_rows, current_clob_rows, production_meta):
        logger.info(
            f"[RETRAIN] Trigger condition NOT met: "
            f"{total_rows} rows (next trigger at {((total_rows // RETRAIN_BATCH_SIZE) + 1) * RETRAIN_BATCH_SIZE}). "
            f"CLOB log delta is below 50,000. No retrain needed."
        )
        return False

    logger.info(f"[RETRAIN] Trigger {'FORCED' if force else 'MET'} at {total_rows} rows.")

    # ═══════════════════════════════════════════════════════════════
    # GUARDRAIL 1 — Class Balance Check
    # ═══════════════════════════════════════════════════════════════
    win_ratio = float((df["label"] == "WIN").mean())
    logger.info(f"[GUARDRAIL-1] WIN ratio: {win_ratio:.2%} (required: 35%-65%)")

    if not (GUARDRAIL_WIN_MIN <= win_ratio <= GUARDRAIL_WIN_MAX):
        msg = (
            f"[ABORT] V6 Retrain ABORTED\n"
            f"Guardrail 1: Class imbalance detected.\n"
            f"WIN ratio: {win_ratio:.2%} (outside 35%-65%)\n"
            f"Action: Collect more balanced data."
        )
        logger.error(
            f"[GUARDRAIL-1 FAIL] Class imbalance. WIN ratio: {win_ratio:.2%}. Retrain CANCELLED."
        )
        send_telegram(msg)
        return False

    logger.info(f"[GUARDRAIL-1] PASSED - WIN ratio {win_ratio:.2%} within bounds")

    # ── Prepare features ──────────────────────────────────────────
    X, y, feat_names = prepare_features(df)
    logger.info(f"[RETRAIN] Feature set ({len(feat_names)}): {feat_names}")

    # ═══════════════════════════════════════════════════════════════
    # GUARDRAIL 2 — Candidate Must Beat Production
    # ═══════════════════════════════════════════════════════════════
    candidate_model, candidate_auc, candidate_std = train_candidate(X, y)

    logger.info(
        f"[GUARDRAIL-2] Candidate AUC: {candidate_auc:.4f} vs Production AUC: {production_auc:.4f}"
    )

    if candidate_auc <= production_auc:
        msg = (
            f"[REJECTED] V6 Retrain REJECTED\n"
            f"Guardrail 2: Candidate not better than production.\n"
            f"Candidate AUC: {candidate_auc:.4f}\n"
            f"Production AUC: {production_auc:.4f}\n"
            f"Action: Candidate discarded. Production unchanged."
        )
        logger.warning(
            f"[GUARDRAIL-2 FAIL] Candidate AUC {candidate_auc:.4f} "
            f"<= Production AUC {production_auc:.4f}. Candidate DISCARDED."
        )
        send_telegram(msg)
        return False

    delta_auc = candidate_auc - production_auc
    logger.info(f"[GUARDRAIL-2] PASSED - Candidate AUC {candidate_auc:.4f} > {production_auc:.4f} (+{delta_auc:.4f})")

    # ═══════════════════════════════════════════════════════════════
    # GUARDRAIL 3 — Atomic Swap
    # ═══════════════════════════════════════════════════════════════
    if dry_run:
        logger.info(
            f"[GUARDRAIL-3] DRY RUN — would atomically swap candidate → production. "
            f"No files written."
        )
        logger.info(f"[RETRAIN] DRY RUN complete. All 3 guardrails would pass.")
        return True

    logger.info("[GUARDRAIL-3] Executing atomic production swap...")

    # Build new metadata
    new_version = bump_version(production_meta.get("version", "6.0.0"))
    new_meta = {
        **production_meta,
        "version": new_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "assembled_post_cutoff",
        "n_rows": total_rows,
        "features": feat_names,
        "n_features": len(feat_names),
        "oof_auc_mean": round(candidate_auc, 6),
        "oof_auc_std": round(candidate_std, 6),
        "hyperparams": MODEL_PARAMS,
        "previous_version": production_meta.get("version"),
        "previous_auc": production_auc,
        "delta_auc": round(delta_auc, 6),
        "win_ratio": round(win_ratio, 6),
        "retrain_trigger_rows": total_rows,
        "clob_rows_trained": current_clob_rows,
    }

    # Save candidate to staging area
    tmp_model = CANDIDATE_MODEL_PATH.with_suffix(".pkl.tmp")
    tmp_meta = CANDIDATE_META_PATH.with_suffix(".json.tmp")

    try:
        # Write candidate model
        joblib.dump(candidate_model, tmp_model)
        os.replace(tmp_model, CANDIDATE_MODEL_PATH)
        logger.info(f"[GUARDRAIL-3] Candidate model saved: {CANDIDATE_MODEL_PATH}")

        # Atomic swap: candidate → production
        os.replace(CANDIDATE_MODEL_PATH, PRODUCTION_MODEL_PATH)
        logger.info(f"[GUARDRAIL-3] ✅ Atomic swap complete: candidate → {PRODUCTION_MODEL_PATH}")

        # Write new metadata
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(new_meta, f, indent=2)
        os.replace(tmp_meta, PRODUCTION_META_PATH)
        logger.info(f"[GUARDRAIL-3] ✅ Metadata updated: {PRODUCTION_META_PATH}")

    except Exception as e:
        logger.exception(f"[GUARDRAIL-3] ❌ CRITICAL: Atomic swap FAILED: {e}")
        # Clean up temp files if they exist
        for tmp_file in (tmp_model, tmp_meta):
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)
        send_telegram(f"❌ *V6 Retrain CRITICAL FAILURE*\nAtomic swap failed: `{e}`")
        return False

    # ── Success! Log and notify ───────────────────────────────────
    logger.info(
        f"[DEPLOY] V6 updated. "
        f"AUC {production_auc:.4f} -> {candidate_auc:.4f} (+{delta_auc:.4f}). "
        f"Version: {production_meta.get('version')} -> {new_version}. "
        f"Rows: {total_rows}."
    )

    success_msg = (
        f"[SUCCESS] V6 Auto-Retrain SUCCESS\n"
        f"Version: {production_meta.get('version')} -> {new_version}\n"
        f"Dataset: {total_rows:,} rows\n"
        f"New AUC: {candidate_auc:.4f} (was {production_auc:.4f})\n"
        f"D AUC: +{delta_auc:.4f}\n"
        f"WIN ratio: {win_ratio:.2%}"
    )
    send_telegram(success_msg)
    return True


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Slingger V6 CI/CD Auto-Retrain Pipeline")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass row-count trigger and retrain immediately",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate all checks without writing any files",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as a scheduled daemon (checks every 6 hours)",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=6.0,
        help="Daemon check interval in hours (default: 6)",
    )
    args = parser.parse_args()

    if args.daemon:
        logger.info(f"[RETRAIN] Starting daemon mode (check every {args.interval_hours}h)")

        # Immediate check on start
        run_retrain(force=args.force, dry_run=args.dry_run)

        # Schedule recurring checks
        schedule.every(args.interval_hours).hours.do(
            run_retrain, force=False, dry_run=args.dry_run
        )

        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        success = run_retrain(force=args.force, dry_run=args.dry_run)
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
