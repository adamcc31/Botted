"""
build_training_dataset_v2.py
============================
Part 2 + Part 3: Build dataset bersih dan retrain kandidat V6+.

CONTEXT:
  - V5 (AUC 0.7278) DIBUNUH — metrik palsu karena Static Feature Cache bug 15-menit.
  - Baseline yang sah: V6 MVP (AUC 0.6213) di slingger/models/v6_production_meta.json
  - Data sah: HANYA sesi SETELAH 2026-06-06 15:44:04 UTC (cutoff session 154404)
  - SPREAD_BLOCKED valid sebagai training signal jika post-cutoff

Pipeline:
  1. Load & merge semua source shadow files
  2. HARD FILTER: buang semua baris sebelum CUTOFF_UTC = 2026-06-06T15:44:04Z
  3. Flat-feature guard: laporkan fitur dengan std=0 (indikator data beracun residual)
  4. Label: WIN=1, LOSE=0 (dari actual_outcome)
  5. Train XGBoost (arsitektur sama dengan V6 MVP) dengan 5-fold OOF AUC
  6. AUC Gate: new_auc > baseline_auc (0.6213) -> DEPLOY, else REJECT
  7. Deploy ke slingger/models/v6_production.pkl + v6_production_meta.json

Usage:
    python scripts/build_training_dataset_v2.py           # dry-run
    python scripts/build_training_dataset_v2.py --deploy  # deploy jika AUC gate pass
"""

import argparse
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).parent.parent

# ── !!CRITICAL CUTOFF!! — buang data dari era cache bug ────────────────────
# Session 2026-06-06_154404 dimulai 15:44:04 UTC = first session post-fix
CUTOFF_UTC = pd.Timestamp("2026-06-06T15:44:04Z", tz="UTC")

# ── Baseline: V6 MVP yang sah ─────────────────────────────────────────────
V6_META_PATH    = ROOT / "slingger/models/v6_production_meta.json"
V6_MODEL_PATH   = ROOT / "slingger/models/v6_production.pkl"

# ── Output ────────────────────────────────────────────────────────────────
DATASET_OUT = ROOT / "dataset/processed/SHADOW_TRAINING_v2.csv"

# ── V6 feature set (18 features dari v6_production_meta.json) ─────────────
# Catatan: shadow_prob_yes/no dan shadow_kelly_yes/no adalah fitur dari model shadow
# yang menyertakan sinyal dari alpha bot sebagai meta-features
V6_FEATURES = [
    'ttr_seconds',
    'spread_pct',
    'yes_price_t0',
    'no_price_t0',
    'clob_spread_t0',
    'yes_depth_t0',
    'no_depth_t0',
    'depth_imbalance_t0',
    'price_velocity_30s',
    'depth_trend_30s',
    'btc_realized_vol_prior_30m',
    'ttr_at_signal',
    'market_hour_utc',
    'day_of_week',
]

# ── Source shadow files ──────────────────────────────────────────────────
SOURCES = [
    ROOT / "dataset/raw/dry_run_shadow_combined_resolved.csv",
    ROOT / "slingger/data/dry_run_shadow_2026-06-06_154404_resolved.csv",
    # Tambahkan session baru di sini saat data Railway tersedia:
    # ROOT / "dataset/raw/sessions/dry_run_shadow_2026-06-11_170804_resolved.csv",
]

N_SPLITS     = 5
RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════════════════════

def load_baseline_auc() -> float:
    """Baca AUC dari V6 production meta. Satu-satunya baseline yang sah."""
    if not V6_META_PATH.exists():
        raise FileNotFoundError(
            f"V6 meta tidak ditemukan: {V6_META_PATH}\n"
            "Pastikan slingger/models/v6_production_meta.json ada."
        )
    with open(V6_META_PATH) as f:
        meta = json.load(f)
    # Support both key names
    auc = meta.get('oof_auc_mean') or meta.get('oof_auc') or meta.get('oof_roc_auc')
    if auc is None:
        raise ValueError(f"Tidak ada kunci AUC di {V6_META_PATH}: {list(meta.keys())}")
    return float(auc)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD & MERGE
# ══════════════════════════════════════════════════════════════════════════════

def load_and_merge() -> pd.DataFrame:
    frames = []
    for src in SOURCES:
        if src.exists():
            df = pd.read_csv(src)
            df['_source_file'] = src.name
            frames.append(df)
            print(f"  [LOAD] {src.name}: {len(df):,} rows")
        else:
            print(f"  [SKIP] {src.name} — file tidak ditemukan")

    if not frames:
        raise ValueError("Tidak ada source file yang ditemukan!")

    master = pd.concat(frames, ignore_index=True, sort=False)
    before = len(master)

    # Dedup by (timestamp, market_id)
    if 'timestamp' in master.columns and 'market_id' in master.columns:
        master = master.drop_duplicates(subset=['timestamp', 'market_id'])
        print(f"  [DEDUP] {before:,} -> {len(master):,} ({before-len(master):,} dupes removed)")

    return master


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: HARD DATE FILTER (ANTI-POISON)
# ══════════════════════════════════════════════════════════════════════════════

def apply_cutoff_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    BUANG semua baris sebelum CUTOFF_UTC.
    Ini adalah filter absolut untuk menghindari data era Static Cache Bug.
    """
    if 'timestamp' not in df.columns:
        print("  [WARN] Kolom 'timestamp' tidak ditemukan — cutoff filter tidak bisa diterapkan!")
        return df

    df = df.copy()
    df['_ts'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

    before = len(df)
    df_valid = df[df['_ts'] >= CUTOFF_UTC]
    df_poison = df[df['_ts'] < CUTOFF_UTC]

    print(f"  [CUTOFF] Batas: {CUTOFF_UTC}")
    print(f"  [CUTOFF] DIBUANG (pre-cutoff/beracun): {len(df_poison):,} baris")
    print(f"  [CUTOFF] DIPERTAHANKAN (post-cutoff): {len(df_valid):,} baris")

    if len(df_valid) == 0:
        raise ValueError(
            "TIDAK ADA DATA setelah cutoff filter! "
            "Pastikan source files berisi data post-2026-06-06T15:44:04Z."
        )

    return df_valid.drop(columns=['_ts'])


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: LABEL + FEATURE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_labels_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Buat label, filter hanya WIN/LOSE, dan laporkan flat features."""
    df = df.copy()

    # Filter hanya actual outcomes (bukan PENDING/SKIP)
    if 'actual_outcome' not in df.columns:
        raise ValueError("'actual_outcome' column missing!")

    before = len(df)
    df = df[df['actual_outcome'].isin(['WIN', 'LOSE'])].copy()
    print(f"  [LABEL] WIN/LOSE filter: {before:,} -> {len(df):,} rows retained")
    print(f"  [LABEL] WIN: {(df['actual_outcome']=='WIN').sum()} | LOSE: {(df['actual_outcome']=='LOSE').sum()}")

    df['label'] = (df['actual_outcome'] == 'WIN').astype(int)

    # ── Flat feature guard ─────────────────────────────────────────────
    print("\n  [GUARD] Flat feature check (std=0 = indikasi data beracun residual):")
    has_poison = False
    available_features = [f for f in V6_FEATURES if f in df.columns]
    for feat in available_features:
        std = df[feat].std()
        if std == 0.0 or (pd.isna(std) and df[feat].isna().all()):
            print(f"  [GUARD] !! FLAT: {feat} std={std} — kemungkinan data beracun residual!")
            has_poison = True
        else:
            null_pct = df[feat].isna().mean() * 100
            print(f"  [GUARD]    OK: {feat:<30} std={std:8.4f}  null={null_pct:.1f}%")

    if has_poison:
        print("\n  [GUARD] WARNING: Fitur flat terdeteksi. Periksa data source atau tambah filter.")
    else:
        print("\n  [GUARD] Semua fitur memiliki variance — tidak ada indikasi poison residual.")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build X, y dari V6_FEATURES. Fitur yang tidak tersedia diisi 0
    (konsisten dengan V6 training).
    """
    available = [f for f in V6_FEATURES if f in df.columns]
    missing   = [f for f in V6_FEATURES if f not in df.columns]

    if missing:
        print(f"\n  [FEAT] Fitur hilang (akan diisi 0): {missing}")
        for m in missing:
            df[m] = 0.0

    # Impute median untuk nulls
    df_feat = df[V6_FEATURES].copy()
    for col in V6_FEATURES:
        if df_feat[col].isna().any():
            median_val = df_feat[col].median()
            df_feat[col] = df_feat[col].fillna(median_val)

    X = df_feat.values
    y = df['label'].values
    return X, y, V6_FEATURES


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: TRAIN (XGBoost, same as V6 MVP)
# ══════════════════════════════════════════════════════════════════════════════

def train_xgboost_oof(X: np.ndarray, y: np.ndarray) -> tuple[object, float, np.ndarray]:
    """
    Train XGBoost dengan hiperparameter konservatif (sama dengan V6 MVP).
    Returns (final_model, oof_auc, oof_probs).
    """
    # Hiperparameter V6 MVP — extreme regularization untuk small dataset
    params = {
        'n_estimators':       100,
        'max_depth':          3,
        'learning_rate':      0.05,
        'reg_alpha':          5.0,
        'reg_lambda':         10.0,
        'min_child_weight':   10,
        'gamma':              1.0,
        'subsample':          0.65,
        'colsample_bytree':   0.65,
        'colsample_bylevel':  0.8,
        'objective':          'binary:logistic',
        'eval_metric':        'auc',
        'random_state':       RANDOM_STATE,
        'n_jobs':             -1,
        'use_label_encoder':  False,
    }

    if HAS_XGB:
        ModelClass = xgb.XGBClassifier
    else:
        print("  [WARN] XGBoost tidak tersedia, fallback ke GradientBoostingClassifier")
        ModelClass = None

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(y))
    fold_aucs = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if HAS_XGB:
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, verbose=False)
            probs = model.predict_proba(X_val)[:, 1]
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3,
                learning_rate=0.05, random_state=RANDOM_STATE
            )
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_val)[:, 1]

        oof_probs[val_idx] = probs
        fold_auc = roc_auc_score(y_val, probs)
        fold_aucs.append(fold_auc)
        print(f"    Fold {fold_idx+1}: AUC={fold_auc:.4f} | n={len(y_val)} | pos={y_val.mean()*100:.1f}%")

    oof_auc = roc_auc_score(y, oof_probs)
    print(f"\n  [OOF] AUC = {oof_auc:.4f}")
    print(f"  [OOF] Mean fold = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    # Train final model on all data
    if HAS_XGB:
        final_model = xgb.XGBClassifier(**params)
        final_model.fit(X, y, verbose=False)
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        final_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3,
            learning_rate=0.05, random_state=RANDOM_STATE
        )
        final_model.fit(X, y)

    return final_model, oof_auc, oof_probs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: AUC GATE + DEPLOY
# ══════════════════════════════════════════════════════════════════════════════

def gate_and_deploy(
    model,
    oof_auc: float,
    baseline_auc: float,
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    dry_run: bool,
) -> bool:
    """Gate AUC dan deploy jika lolos. Satu-satunya baseline = V6 production."""

    print(f"\n  [GATE] New AUC  : {oof_auc:.4f}")
    print(f"  [GATE] Baseline : {baseline_auc:.4f} (V6 MVP di {V6_META_PATH})")
    print(f"  [GATE] Delta    : {oof_auc - baseline_auc:+.4f}")

    if oof_auc <= baseline_auc:
        print(f"  [GATE] REJECTED — {oof_auc:.4f} tidak mengalahkan {baseline_auc:.4f}")
        return False

    print(f"  [GATE] PASSED — model baru lebih baik!")

    if dry_run:
        print("  [DRY-RUN] Tidak disimpan. Jalankan dengan --deploy untuk deploy.")
        return True

    # ── Backup current V6 ───────────────────────────────────────────────
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = V6_META_PATH.parent / f"candidates/candidate_{ts}_{oof_auc:.4f}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if V6_MODEL_PATH.exists():
        shutil.copy2(V6_MODEL_PATH, backup_dir / "v6_production.pkl")
    if V6_META_PATH.exists():
        shutil.copy2(V6_META_PATH, backup_dir / "v6_production_meta.json")
    print(f"  [BACKUP] Current V6 backed up -> {backup_dir}")

    # ── Save new model ────────────────────────────────────────────────
    with open(V6_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    # ── Update production meta ────────────────────────────────────────
    # Read existing meta for structure
    with open(V6_META_PATH) as f:
        old_meta = json.load(f)

    timestamp_range = {}
    if 'timestamp' in df.columns:
        ts_series = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        timestamp_range = {
            'min': str(ts_series.min()),
            'max': str(ts_series.max()),
        }

    new_meta = {
        "version":          f"6.1.0-retrain-{ts}",
        "trained_at":       datetime.utcnow().isoformat() + "Z",
        "prev_version":     old_meta.get("version", "6.0.0-mvp"),
        "prev_oof_auc":     baseline_auc,
        "dataset":          "SHADOW_TRAINING_v2.csv",
        "n_rows":           len(df),
        "win_count":        int(y.sum()),
        "lose_count":       int((y == 0).sum()),
        "win_ratio":        float(y.mean()),
        "features":         V6_FEATURES,
        "n_features":       len(V6_FEATURES),
        "oof_auc_mean":     round(oof_auc, 5),
        "cutoff_date":      str(CUTOFF_UTC.date()),
        "cutoff_utc":       str(CUTOFF_UTC),
        "data_range":       timestamp_range,
        "hyperparams":      old_meta.get("hyperparams", {}),
        "label_encoding":   {"WIN": 1, "LOSE": 0},
        "label_column_source": "actual_outcome",
        "scaler":           None,
        "notes": (
            f"Retrained with hard cutoff {CUTOFF_UTC} to exclude Static Cache Bug era. "
            f"Previous baseline AUC: {baseline_auc:.4f}. "
            f"New AUC: {oof_auc:.4f} (+{oof_auc-baseline_auc:.4f})."
        ),
    }

    with open(V6_META_PATH, 'w') as f:
        json.dump(new_meta, f, indent=2)

    print(f"  [DEPLOY] Model  : {V6_MODEL_PATH}")
    print(f"  [DEPLOY] Meta   : {V6_META_PATH}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Build V6 training dataset v2 dan retrain. Baseline = V6 MVP (0.6213)."
    )
    parser.add_argument(
        '--deploy', action='store_true',
        help='Deploy model ke slingger/models/ jika AUC gate pass'
    )
    args = parser.parse_args()
    dry_run = not args.deploy

    print("=" * 70)
    print("SLINGGER V6+ — RETRAIN PIPELINE v2")
    print(f"Mode: {'DEPLOY' if args.deploy else 'DRY-RUN (--deploy untuk save)'}")
    print(f"Cutoff: {CUTOFF_UTC} (buang data era cache bug)")
    print("=" * 70)

    # Load baseline AUC dari V6 production
    baseline_auc = load_baseline_auc()
    print(f"\n[BASELINE] V6 MVP AUC = {baseline_auc:.4f} (dari {V6_META_PATH})")

    # Step 1: Load & merge
    print("\n[STEP 1] Load & Merge Sources")
    df_raw = load_and_merge()

    # Step 2: Hard cutoff filter
    print("\n[STEP 2] Hard Cutoff Filter (anti-poison)")
    df_filtered = apply_cutoff_filter(df_raw)

    # Step 3: Labels + flat feature guard
    print("\n[STEP 3] Labels + Flat Feature Guard")
    df = prepare_labels_and_validate(df_filtered)

    # Save processed dataset
    DATASET_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_OUT, index=False)
    print(f"\n[SAVE] Processed dataset: {DATASET_OUT} ({len(df):,} rows)")

    # Step 4: Build features
    print("\n[STEP 4] Build Feature Matrix")
    X, y, used_features = build_feature_matrix(df)
    print(f"  X.shape = {X.shape} | y positive = {y.sum()} ({y.mean()*100:.1f}%)")

    # Step 5: Train
    print("\n[STEP 5] Train XGBoost (V6 hyperparams) — 5-fold OOF")
    model, oof_auc, oof_probs = train_xgboost_oof(X, y)

    # Step 6: Gate + deploy
    print("\n[STEP 6] AUC Gate")
    deployed = gate_and_deploy(model, oof_auc, baseline_auc, X, y, df, dry_run)

    # ── Final report ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RETRAIN SUMMARY")
    print("=" * 70)
    print(f"Cutoff           : {CUTOFF_UTC}")
    print(f"Training rows    : {len(df):,}")
    print(f"Positive labels  : {int(y.sum())} ({y.mean()*100:.1f}%)")
    print(f"OOF AUC (new)    : {oof_auc:.4f}")
    print(f"Baseline AUC (V6): {baseline_auc:.4f}")
    print(f"Delta            : {oof_auc - baseline_auc:+.4f}")

    if not deployed:
        verdict = "REJECTED (AUC tidak mengalahkan V6 baseline)"
    elif dry_run:
        verdict = "WOULD DEPLOY — jalankan --deploy untuk simpan"
    else:
        verdict = "DEPLOYED ke slingger/models/"

    print(f"Verdict          : {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
