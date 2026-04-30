"""
gladiator_train.py — Train 3 XGBoost variants, evaluate with Post-Penalty EV, pick winner.

Models:
  A) Sniper   — high precision, deep regularization, abstains a lot
  B) Brawler  — high recall, more trades, relies on Law of Large Numbers
  C) Calibrator — Brier-optimized, balanced params, best probability estimates

EV Hard Rules (V2 Environment):
  1. Spread/Vig: uses actual odds_yes/odds_no from data
  2. Latency Penalty: -1.5% absolute on predicted P(WIN) before EV calc
  3. EV Threshold: execute only if adjusted_EV > 0.02 (2% edge minimum)
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from model_training.config import ALL_FEATURES, TARGET_COL, SPLIT_CFG
from model_training.features import build_features
from model_training.dataset import (
    load_from_csv, validate_dataset, deduplicate_per_market,
    chronological_split, get_market_groups, impute_features, apply_imputer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("gladiator")

# ---------------------------------------------------------------------------
# EV Constants (V2 Hard Rules)
# ---------------------------------------------------------------------------
LATENCY_PENALTY = 0.015   # -1.5% absolute on P(WIN)
EV_THRESHOLD    = 0.02    # minimum edge to execute
BET_SIZE        = 1.0     # flat 1 unit per trade

# ---------------------------------------------------------------------------
# Model Configs
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    name: str
    n_estimators: int = 500
    max_depth: int = 4
    learning_rate: float = 0.04
    subsample: float = 0.80
    colsample_bytree: float = 0.75
    min_child_weight: int = 8
    gamma: float = 0.15
    reg_alpha: float = 0.1
    reg_lambda: float = 1.5
    early_stopping_rounds: int = 40
    random_state: int = 42
    n_jobs: int = -1
    verbosity: int = 0
    eval_metric: str = "logloss"

    def to_xgb_params(self) -> dict:
        d = asdict(self)
        d.pop("name")
        return d

GLADIATORS = {
    "A_Sniper": ModelConfig(
        name="A_Sniper",
        max_depth=3,
        learning_rate=0.01,
        n_estimators=1200,
        min_child_weight=15,
        gamma=0.30,
        reg_alpha=1.0,
        reg_lambda=5.0,
        subsample=0.70,
        colsample_bytree=0.60,
        early_stopping_rounds=60,
    ),
    "B_Brawler": ModelConfig(
        name="B_Brawler",
        max_depth=5,
        learning_rate=0.05,
        n_estimators=600,
        min_child_weight=5,
        gamma=0.05,
        reg_alpha=0.01,
        reg_lambda=0.5,
        subsample=0.85,
        colsample_bytree=0.85,
        early_stopping_rounds=30,
    ),
    "C_Calibrator": ModelConfig(
        name="C_Calibrator",
        max_depth=4,
        learning_rate=0.03,
        n_estimators=800,
        min_child_weight=10,
        gamma=0.10,
        reg_alpha=0.1,
        reg_lambda=1.5,
        subsample=0.80,
        colsample_bytree=0.75,
        early_stopping_rounds=50,
    ),
}

# ---------------------------------------------------------------------------
# Post-Penalty EV Calculator
# ---------------------------------------------------------------------------
def calculate_post_penalty_ev(p_win: float, entry_odds: float) -> float:
    """
    EV with V2 latency penalty.
    adjusted_p = p_win - LATENCY_PENALTY
    EV = adjusted_p * profit_if_win - (1 - adjusted_p) * stake
    profit_if_win = (1/entry_odds) - 1
    """
    if np.isnan(p_win) or np.isnan(entry_odds) or entry_odds <= 0:
        return float("nan")
    adj_p = max(p_win - LATENCY_PENALTY, 0.0)
    payout = (1.0 / entry_odds) - 1.0
    ev = adj_p * payout - (1.0 - adj_p)
    return ev


def simulate_ev_post_penalty(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    entry_odds: np.ndarray,
) -> dict:
    """
    Full EV simulation with latency penalty and threshold.
    Only executes trades where post-penalty EV > EV_THRESHOLD.
    """
    n = len(y_true)
    ev_vals = np.array([
        calculate_post_penalty_ev(p, o)
        for p, o in zip(y_prob, entry_odds)
    ])

    # Filter: only execute if EV > threshold
    execute_mask = np.isfinite(ev_vals) & (ev_vals > EV_THRESHOLD)
    n_exec = int(execute_mask.sum())
    n_abstain = n - n_exec
    abstain_rate = n_abstain / max(n, 1)

    if n_exec == 0:
        return {
            "n_total": n, "n_executed": 0, "n_abstain": n,
            "abstain_rate": 1.0, "win_rate": 0.0, "brier": float("nan"),
            "total_pnl": 0.0, "avg_pnl": 0.0, "net_ev_per_trade": 0.0,
            "avg_ev": 0.0, "sharpe": float("nan"),
        }

    exec_true = y_true[execute_mask]
    exec_prob = y_prob[execute_mask]
    exec_odds = entry_odds[execute_mask]
    exec_ev = ev_vals[execute_mask]

    win_rate = float(exec_true.mean())

    # PnL: win = bet * ((1/odds)-1), lose = -bet
    pnl = np.where(
        exec_true == 1,
        BET_SIZE * ((1.0 / np.clip(exec_odds, 1e-6, None)) - 1.0),
        -BET_SIZE,
    )
    total_pnl = float(pnl.sum())
    avg_pnl = float(pnl.mean())
    net_ev = float(exec_ev.mean())

    sharpe = float(pnl.mean() / pnl.std()) if pnl.std() > 0 else float("nan")

    # Brier on executed trades only
    try:
        brier = float(brier_score_loss(exec_true, exec_prob))
    except Exception:
        brier = float("nan")

    return {
        "n_total": n,
        "n_executed": n_exec,
        "n_abstain": n_abstain,
        "abstain_rate": round(abstain_rate, 4),
        "win_rate": round(win_rate, 4),
        "brier": round(brier, 4),
        "total_pnl": round(total_pnl, 4),
        "avg_pnl": round(avg_pnl, 4),
        "net_ev_per_trade": round(net_ev, 4),
        "avg_ev": round(net_ev, 4),
        "sharpe": round(sharpe, 4) if np.isfinite(sharpe) else None,
    }


# ---------------------------------------------------------------------------
# Train one gladiator
# ---------------------------------------------------------------------------
def train_gladiator(
    name: str,
    cfg: ModelConfig,
    df_train: pd.DataFrame,
    df_calib: pd.DataFrame,
    df_test: pd.DataFrame,
    imputer_vals: pd.Series,
) -> dict:
    """Train one XGBoost variant, calibrate, evaluate with post-penalty EV."""
    logger.info("=" * 60)
    logger.info("TRAINING: %s", name)
    logger.info("=" * 60)

    params = cfg.to_xgb_params()
    logger.info("  Params: max_depth=%d lr=%.3f reg_alpha=%.2f reg_lambda=%.2f",
                cfg.max_depth, cfg.learning_rate, cfg.reg_alpha, cfg.reg_lambda)

    # Build features
    df_tr = build_features(df_train.copy())
    df_ca = build_features(df_calib.copy())
    df_te = build_features(df_test.copy())

    X_tr = df_tr[ALL_FEATURES].values.astype(np.float32)
    y_tr = df_tr[TARGET_COL].values.astype(np.int32)
    X_ca = df_ca[ALL_FEATURES].values.astype(np.float32)
    y_ca = df_ca[TARGET_COL].values.astype(np.int32)
    X_te = df_te[ALL_FEATURES].values.astype(np.float32)
    y_te = df_te[TARGET_COL].values.astype(np.int32)

    # Early stopping split from train
    n_es = max(int(len(X_tr) * 0.15), 30)
    X_es_tr, X_es_va = X_tr[:-n_es], X_tr[-n_es:]
    y_es_tr, y_es_va = y_tr[:-n_es], y_tr[-n_es:]

    # Train
    t0 = time.time()
    model = XGBClassifier(**params)
    model.fit(X_es_tr, y_es_tr, eval_set=[(X_es_va, y_es_va)], verbose=False)
    elapsed = time.time() - t0
    logger.info("  Trained in %.1fs | best_iter=%d", elapsed, model.best_iteration)

    # Platt calibration
    raw_ca = model.predict_proba(X_ca)[:, 1].reshape(-1, 1)
    platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    platt.fit(raw_ca, y_ca)

    # Predict on TEST (never seen during training)
    raw_te = model.predict_proba(X_te)[:, 1].reshape(-1, 1)
    p_cal = platt.predict_proba(raw_te)[:, 1]

    # Classification metrics on test
    try:
        auc = float(roc_auc_score(y_te, p_cal))
    except ValueError:
        auc = float("nan")
    brier = float(brier_score_loss(y_te, p_cal))

    # ECE
    try:
        frac, mpred = calibration_curve(y_te, p_cal, n_bins=10, strategy="uniform")
        ece = float(np.mean(np.abs(frac - mpred)))
    except Exception:
        ece = float("nan")

    logger.info("  Test AUC=%.4f | Brier=%.4f | ECE=%.4f", auc, brier, ece)

    # EV simulation on test set
    entry_odds_te = df_te["entry_odds"].values.astype(np.float64)
    ev_result = simulate_ev_post_penalty(y_te, p_cal, entry_odds_te)

    logger.info("  EV Sim: exec=%d/%d | abstain=%.1f%% | win=%.1f%% | PnL=%.4f | netEV=%.4f",
                ev_result["n_executed"], ev_result["n_total"],
                ev_result["abstain_rate"]*100, ev_result["win_rate"]*100,
                ev_result["total_pnl"], ev_result["net_ev_per_trade"])

    # Feature importance (gain)
    imp = model.get_booster().get_score(importance_type="gain")
    feat_imp = {}
    for i, fname in enumerate(ALL_FEATURES):
        key = f"f{i}"
        feat_imp[fname] = imp.get(key, 0.0)
    feat_imp = dict(sorted(feat_imp.items(), key=lambda x: -x[1]))

    top5 = list(feat_imp.items())[:5]
    logger.info("  Top-5 Features:")
    for rank, (fn, val) in enumerate(top5, 1):
        logger.info("    #%d: %-25s gain=%.4f", rank, fn, val)

    # GroupKFold CV for quality gates
    groups = get_market_groups(df_tr)
    n_folds = min(SPLIT_CFG.n_cv_folds, len(np.unique(groups)))
    if n_folds >= 2:
        gkf = GroupKFold(n_splits=n_folds)
        cv_aucs = []
        for tr_idx, va_idx in gkf.split(X_tr, y_tr, groups):
            m = XGBClassifier(**params)
            m.fit(X_tr[tr_idx], y_tr[tr_idx],
                  eval_set=[(X_tr[va_idx], y_tr[va_idx])], verbose=False)
            p = m.predict_proba(X_tr[va_idx])[:, 1]
            try:
                cv_aucs.append(roc_auc_score(y_tr[va_idx], p))
            except ValueError:
                pass
        cv_mean_auc = float(np.mean(cv_aucs)) if cv_aucs else float("nan")
        cv_std_auc = float(np.std(cv_aucs)) if cv_aucs else float("nan")
    else:
        cv_mean_auc = auc
        cv_std_auc = 0.0

    logger.info("  CV AUC: %.4f +/- %.4f", cv_mean_auc, cv_std_auc)

    return {
        "name": name,
        "config": asdict(cfg),
        "base_model": model,
        "platt": platt,
        "imputer_vals": imputer_vals,
        "feature_names": ALL_FEATURES,
        "feature_importance": feat_imp,
        "metrics": {
            "test_auc": round(auc, 4),
            "test_brier": round(brier, 4),
            "test_ece": round(ece, 4),
            "cv_mean_auc": round(cv_mean_auc, 4),
            "cv_std_auc": round(cv_std_auc, 4),
        },
        "ev_result": ev_result,
        "best_iteration": model.best_iteration,
        "elapsed": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    DATA_PATH = BASE_DIR / "dataset" / "raw" / "alpha_v1_master.csv"
    OUTPUT_DIR = BASE_DIR / "models" / "alpha_v1"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GLADIATOR ARENA — Alpha V1 XGBoost Tournament")
    logger.info("=" * 60)
    logger.info("Data: %s", DATA_PATH)
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("Latency Penalty: %.3f | EV Threshold: %.3f", LATENCY_PENALTY, EV_THRESHOLD)

    # Load data
    df = load_from_csv(DATA_PATH)
    df = df[df[TARGET_COL].notna()].copy()
    logger.info("Labeled rows: %d", len(df))

    # Validate
    val = validate_dataset(df)
    logger.info("Validation: sample_gate=%s market_gate=%s", val["sample_gate"], val["market_gate"])

    # Deduplicate
    df = deduplicate_per_market(df)

    # Split: train(65%) / calib(20%) / test(15%)
    df_train, df_calib, df_test = chronological_split(
        df, calib_ratio=0.20, test_ratio=0.15
    )
    logger.info("Split: train=%d | calib=%d | test=%d", len(df_train), len(df_calib), len(df_test))

    # Impute
    df_train, imputer_vals = impute_features(df_train, strategy="median")
    df_calib = apply_imputer(df_calib, imputer_vals)
    df_test = apply_imputer(df_test, imputer_vals)

    # Train all gladiators
    results = {}
    for name, cfg in GLADIATORS.items():
        results[name] = train_gladiator(
            name, cfg, df_train, df_calib, df_test, imputer_vals
        )

    # ---------------------------------------------------------------------------
    # Comparison Table
    # ---------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("GLADIATOR COMPARISON TABLE (Post-Penalty EV)")
    logger.info("=" * 80)

    header = f"{'Model':<16} {'WinRate':>8} {'Abstain':>9} {'Brier':>7} {'PnL':>10} {'NetEV/Trade':>12} {'AUC':>6}"
    logger.info(header)
    logger.info("-" * 80)

    for name, r in results.items():
        ev = r["ev_result"]
        m = r["metrics"]
        logger.info(
            f"{name:<16} {ev['win_rate']*100:>7.1f}% {ev['abstain_rate']*100:>8.1f}% "
            f"{m['test_brier']:>7.4f} {ev['total_pnl']:>+10.4f} "
            f"{ev['net_ev_per_trade']:>+12.4f} {m['test_auc']:>6.4f}"
        )

    # ---------------------------------------------------------------------------
    # Pick Winner — best Net EV per Trade (post-penalty)
    # ---------------------------------------------------------------------------
    winner_name = max(
        results.keys(),
        key=lambda k: results[k]["ev_result"]["net_ev_per_trade"]
            if results[k]["ev_result"]["n_executed"] > 0
            else -999
    )
    winner = results[winner_name]

    logger.info("\n" + "=" * 60)
    logger.info("WINNER: %s", winner_name)
    logger.info("  Net EV/Trade: %+.4f", winner["ev_result"]["net_ev_per_trade"])
    logger.info("  Total PnL:    %+.4f units", winner["ev_result"]["total_pnl"])
    logger.info("  Win Rate:     %.1f%%", winner["ev_result"]["win_rate"]*100)
    logger.info("  Abstain Rate: %.1f%%", winner["ev_result"]["abstain_rate"]*100)
    logger.info("  Test Brier:   %.4f", winner["metrics"]["test_brier"])
    logger.info("=" * 60)

    # Top 5 features
    top5 = list(winner["feature_importance"].items())[:5]
    logger.info("\nTop 5 Feature Importance (%s):", winner_name)
    for rank, (fn, val) in enumerate(top5, 1):
        logger.info("  #%d: %-28s gain=%.4f", rank, fn, val)

    # ---------------------------------------------------------------------------
    # Save Winner
    # ---------------------------------------------------------------------------
    bundle = {
        "base_model": winner["base_model"],
        "platt": winner["platt"],
        "imputer_vals": winner["imputer_vals"],
        "feature_names": winner["feature_names"],
        "version": f"alpha_v1_{winner_name}",
        "gladiator_name": winner_name,
    }
    model_path = OUTPUT_DIR / "model.pkl"
    joblib.dump(bundle, model_path, compress=3)
    logger.info("\nModel saved: %s", model_path)

    # Save metadata
    def _safe(obj):
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return obj

    # All results JSON
    all_results = {}
    for name, r in results.items():
        all_results[name] = {
            "config": r["config"],
            "metrics": r["metrics"],
            "ev_result": r["ev_result"],
            "feature_importance_top10": dict(list(r["feature_importance"].items())[:10]),
            "best_iteration": r["best_iteration"],
            "elapsed": r["elapsed"],
        }
    all_results["_winner"] = winner_name
    all_results["_ev_rules"] = {
        "latency_penalty": LATENCY_PENALTY,
        "ev_threshold": EV_THRESHOLD,
        "bet_size": BET_SIZE,
    }

    results_path = OUTPUT_DIR / "gladiator_results.json"
    with open(results_path, "w") as f:
        json.dump(_safe(all_results), f, indent=2)
    logger.info("Results saved: %s", results_path)

    # Save imputer
    imp_path = OUTPUT_DIR / "imputer_vals.json"
    with open(imp_path, "w") as f:
        json.dump(_safe(imputer_vals), f, indent=2)

    logger.info("\nDONE. All artifacts in: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
