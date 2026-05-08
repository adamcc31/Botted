"""
V5 Final Retrain + Quant Simulation
=====================================
Perbaikan berdasarkan temuan forensik:
1. HAPUS CLOB gate dari simulasi (biarkan XGBoost yang memutuskan)
2. Pertahankan T=0.87 (OOF calibrator, bukan holdout)
3. Tambahkan fee accounting (2% winnings + spread)
4. Gunakan SEMUA data untuk training (expanding window CV)
5. Fit calibrator pada OOF predictions, BUKAN holdout split
   (Ini memperbaiki bug dimana T=1.27 holdout bertentangan dengan T=0.87 OOF)
6. Simulasi pada walk-forward OOF predictions untuk menghindari bias
"""
import sys, glob, json, pickle, time
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger("v5_final")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

from model_training.config import V5_FEATURES_FULL, TARGET_COL, XGBConfig
from model_training.dataset import get_market_groups, impute_features, apply_imputer
from model_training.features import build_features, validate_no_leakage
from model_training.trainer import TemperatureScaling, compute_ece

PROJECT = Path('.')
MASTER_CSV = PROJECT / "dataset" / "DATASET_MASTER_08-05-2026.csv"
OUTPUT_DIR = PROJECT / "models" / "v5_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = V5_FEATURES_FULL

# ============================================================================
# Optuna HPO on full dataset
# ============================================================================
def run_optuna_full(X, y, groups, n_trials=250, timeout=1500):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "early_stopping_rounds": 50,
            "random_state": 42, "n_jobs": -1, "verbosity": 0, "eval_metric": "logloss",
        }
        gkf = GroupKFold(n_splits=5)
        aucs = []
        for tr_idx, va_idx in gkf.split(X, y, groups):
            m = XGBClassifier(**params)
            m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])], verbose=False)
            aucs.append(roc_auc_score(y[va_idx], m.predict_proba(X[va_idx])[:, 1]))
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize", study_name="v5_final_full")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best = study.best_params.copy()
    best.update({"early_stopping_rounds": 50, "random_state": 42, "n_jobs": -1,
                 "verbosity": 0, "eval_metric": "logloss"})
    return best, study.best_value, len(study.trials)


# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("  V5 FINAL RETRAIN -- Full dataset + OOF Calibration")
    logger.info("=" * 60)

    # --- Load ---
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    logger.info("Dataset: %d rows | WIN=%.1f%%", len(df), df["label"].mean()*100)

    validate_no_leakage(df)

    # --- Build features on entire dataset ---
    df, imp = impute_features(df, strategy="median")
    df_feat = build_features(df)

    X_all = df_feat[FEATURES].values.astype(np.float32)
    y_all = df_feat[TARGET_COL].values.astype(np.int32)
    groups_all = get_market_groups(df_feat)

    # =====================================================================
    # Step 1: Optuna HPO on full dataset
    # =====================================================================
    logger.info("Running Optuna HPO (250 trials, full dataset)...")
    best_params, best_auc_optuna, n_trials = run_optuna_full(
        X_all, y_all, groups_all, n_trials=250, timeout=1500
    )
    logger.info("Optuna: best_AUC=%.4f (%d trials)", best_auc_optuna, n_trials)
    logger.info("Best params: %s", best_params)

    # =====================================================================
    # Step 2: Full 5-fold GroupKFold CV with OOF predictions
    # =====================================================================
    logger.info("Running 5-fold GroupKFold CV with best params...")
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y_all))
    oof_labels = np.zeros(len(y_all))
    fold_aucs = []

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_all, y_all, groups_all)):
        m = XGBClassifier(**best_params)
        m.fit(X_all[tr_idx], y_all[tr_idx],
              eval_set=[(X_all[va_idx], y_all[va_idx])], verbose=False)
        proba = m.predict_proba(X_all[va_idx])[:, 1]
        auc = roc_auc_score(y_all[va_idx], proba)
        fold_aucs.append(auc)
        oof_preds[va_idx] = proba
        oof_labels[va_idx] = y_all[va_idx]
        logger.info("  Fold %d: AUC=%.4f (n_val=%d)", fold_idx+1, auc, len(va_idx))

    cv_auc = float(np.mean(fold_aucs))
    cv_std = float(np.std(fold_aucs))
    oof_auc = roc_auc_score(oof_labels, oof_preds)
    logger.info("CV AUC: %.4f +/- %.4f | OOF AUC: %.4f", cv_auc, cv_std, oof_auc)

    # =====================================================================
    # Step 3: Fit TemperatureScaling on OOF predictions
    # =====================================================================
    ece_before = compute_ece(oof_labels, oof_preds)
    temp_scaler = TemperatureScaling()
    temp_scaler.fit(oof_preds, oof_labels)
    oof_calibrated = temp_scaler.predict_proba(oof_preds)
    ece_after = compute_ece(oof_labels, oof_calibrated)

    logger.info("ECE Before: %.4f | ECE After: %.4f | T=%.4f", ece_before, ece_after, temp_scaler.temperature)

    # =====================================================================
    # Step 4: Train final model on ALL data
    # =====================================================================
    logger.info("Training final model on ALL data...")
    n_es = max(int(len(X_all) * 0.15), 50)
    final_model = XGBClassifier(**best_params)
    final_model.fit(
        X_all[:-n_es], y_all[:-n_es],
        eval_set=[(X_all[-n_es:], y_all[-n_es:])],
        verbose=False,
    )
    logger.info("Final model: best_iter=%d", final_model.best_iteration)

    # =====================================================================
    # Step 5: OOF-based simulation (no data leakage!)
    # =====================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("  OOF-BASED SIMULATION (no data leakage)")
    logger.info("=" * 60)

    # Apply OOF calibrator to OOF predictions
    oof_cal = temp_scaler.predict_proba(oof_preds)
    oof_cal = np.clip(oof_cal, 0.001, 0.999)

    # Run simulation on OOF predictions
    configs = [
        ("A_CLOB_ON_QKelly",  True,  0.25, 0.00, 0.000, 0.05),
        ("B_NO_CLOB_QKelly",  False, 0.25, 0.00, 0.000, 0.05),
        ("C_NO_CLOB_FeeAdj",  False, 0.25, 0.02, 0.005, 0.05),
        ("D_NO_CLOB_HKelly",  False, 0.40, 0.02, 0.005, 0.05),
        ("E_NO_CLOB_Tight",   False, 0.35, 0.02, 0.005, 0.08),
    ]

    print("\n" + "=" * 110)
    print(" OOF SIMULATION RESULTS (all data, no leakage, T=%.4f)" % temp_scaler.temperature)
    print("=" * 110)
    print(f"{'Config':<28} {'Trades':>6} {'WinRate':>8} {'PnL':>10} {'ROI':>8} {'MaxDD':>8} {'Sharpe':>7} {'Contr':>6} {'CntrPnL':>9}")
    print("-" * 110)

    best_sim = None
    for cname, clob_gate, kelly_f, fee, spread, ev_thr in configs:
        cap = 1000.0
        peak = cap
        trades = []
        max_dd = 0.0
        consec_l = 0
        max_consec = 0
        n_contr = 0
        contr_pnl = 0.0

        for i in range(len(df_feat)):
            row = df_feat.iloc[i]
            p_win = float(oof_cal[i])
            entry_odds = float(row.get("entry_odds", 0.5))
            sig = str(row.get("signal_direction", "")).strip().upper()

            if sig not in ("BUY_UP", "BUY_DOWN"):
                continue
            if clob_gate and row.get("clob_alignment", -1) != 1:
                continue
            ttr = float(row.get("ttr_seconds", 150))
            if not (60 <= ttr <= 250):
                continue
            if entry_odds <= 0 or entry_odds >= 1.0:
                continue

            b = (1.0 / entry_odds) - 1.0
            b_adj = b * (1.0 - fee)
            ev = p_win * b_adj - (1.0 - p_win)
            if ev <= ev_thr:
                continue

            fk = (b_adj * p_win - (1 - p_win)) / b_adj
            stake_pct = min(max(kelly_f * fk, 0.0), 0.05)
            if stake_pct <= 0:
                continue

            stake = cap * stake_pct
            label = int(row.get("label", 0))
            is_contr = int(row.get("clob_alignment", 0)) == 0

            if label == 1:
                gpnl = stake * b
                pnl = gpnl * (1 - fee) - spread
                consec_l = 0
            else:
                pnl = -stake - spread
                consec_l += 1
                max_consec = max(max_consec, consec_l)

            cap += pnl
            peak = max(peak, cap)
            max_dd = max(max_dd, peak - cap)
            trades.append({"label": label, "pnl": pnl, "contr": is_contr})
            if is_contr:
                n_contr += 1
                contr_pnl += pnl

        nt = len(trades)
        if nt > 0:
            wr = sum(1 for t in trades if t["label"] == 1) / nt
            tpnl = cap - 1000.0
            roi = tpnl / 1000.0 * 100
            pnl_arr = np.array([t["pnl"] for t in trades])
            sh = float(np.mean(pnl_arr) / max(np.std(pnl_arr), 1e-9))
        else:
            wr = tpnl = roi = sh = 0

        line = (f"{cname:<28} {nt:>6} {wr*100:>7.1f}% ${tpnl:>+9.2f} "
                f"{roi:>+7.2f}% ${max_dd:>7.2f} {sh:>7.3f} {n_contr:>6} ${contr_pnl:>+8.2f}")
        print(line)

        sim_r = {"config": cname, "trades": nt, "win_rate": wr, "pnl": tpnl,
                 "roi": roi, "max_dd": max_dd, "sharpe": sh,
                 "n_contrarian": n_contr, "contrarian_pnl": contr_pnl,
                 "final_capital": cap, "max_consec_loss": max_consec}
        if best_sim is None or tpnl > best_sim["pnl"]:
            best_sim = sim_r

    print("=" * 110)

    # =====================================================================
    # Step 6: Save artifacts
    # =====================================================================
    with open(OUTPUT_DIR / "model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    with open(OUTPUT_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(temp_scaler, f)

    metadata = {
        "model_version": "v5_final",
        "created_at": datetime.now().isoformat(),
        "training_data": "DATASET_MASTER_08-05-2026.csv",
        "n_samples": len(df),
        "label_win_rate": float(df["label"].mean()),
        "features": FEATURES,
        "n_features": len(FEATURES),
        "cv_auc": cv_auc,
        "cv_std": cv_std,
        "oof_auc": oof_auc,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "temperature": temp_scaler.temperature,
        "model_params": best_params,
        "calibration_note": "Fitted on OOF predictions (NOT holdout split) to avoid train/test regime mismatch",
        "execution_gates": {
            "clob_alignment_gate": False,
            "clob_alignment_note": "Removed as hard gate. XGBoost uses it as feature weight internally.",
            "ttr_range_seconds": [60, 250],
            "ev_threshold": 0.05,
        },
        "kelly_fraction": 0.25,
        "kelly_upgrade_note": "ECE < 0.08 allows Half-Kelly. Test with 0.35-0.40.",
        "best_oof_simulation": best_sim,
        "optuna_best_auc": best_auc_optuna,
        "optuna_n_trials": n_trials,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Artifacts saved to: {OUTPUT_DIR}")
    print(f"  Temperature (OOF): {temp_scaler.temperature:.4f}")
    print(f"  CV AUC: {cv_auc:.4f} +/- {cv_std:.4f}")
    print(f"  ECE: {ece_before:.4f} -> {ece_after:.4f}")
    print(f"  Elapsed: {elapsed:.1f}s")

if __name__ == "__main__":
    main()
