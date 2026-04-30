"""
ml/config.py
============
Satu sumber kebenaran untuk semua konstanta pipeline.
Ubah di sini saja — tidak ada magic number tersebar di file lain.
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Kolom yang DILARANG masuk sebagai fitur (post-hoc / leakage)
# ---------------------------------------------------------------------------
LEAKAGE_COLS: List[str] = [
    "theoretical_pnl",        # dihitung dari outcome — leakage sempurna (corr 0.796)
    "theoretical_exit_odds",   # harga setelah settlement
    "resolution_price",        # harga settlement final
    "retrofix_delta",          # koreksi retroaktif post-settlement
    "actual_outcome",          # proxy label langsung
    "signal_correct",          # proxy label langsung
    "label",                   # target itu sendiri
]

# ---------------------------------------------------------------------------
# Fitur mentah dari dataset (tersedia saat sinyal dibuat)
# ---------------------------------------------------------------------------
RAW_FEATURES: List[str] = [
    # Microstructure Binance
    "obi_value",
    "tfm_value",
    "depth_ratio",
    "obi_tfm_product",
    "obi_tfm_alignment",
    # Volatility
    "rv_value",
    "vol_percentile",
    # Strike-relative (Chainlink-aligned)
    "strike_distance_pct",
    "contest_urgency",
    "ttr_seconds",
    # CLOB / Market odds
    "odds_yes",
    "odds_no",
    "entry_odds",
    "odds_yes_60s_ago",
    "odds_delta_60s",
    # Price / Spread
    "spread_pct",
    "btc_return_1m",
    # Signal engine output
    "confidence_score",
]

# Fitur rekayasa baru (dibuat di features.py)
ENGINEERED_FEATURES: List[str] = [
    "edge_vs_crowd",        # q_fair - entry_odds  → inti mispricing detection
    "obi_vol_interaction",  # obi_value × vol_percentile
    "tfm_vol_interaction",  # tfm_value × vol_percentile
    "odds_momentum",        # odds_delta_60s / (odds_yes_60s_ago + 1e-6)
    "urgency_vol",          # contest_urgency × rv_value
    "micro_alignment",      # obi_value × tfm_value (arah sama = kuat)
    "hour_wib",             # jam WIB (UTC+7) → pola volume per jam
    "is_weekend",           # 1 jika Sabtu/Minggu → likuiditas tipis
]

ALL_FEATURES: List[str] = RAW_FEATURES + ENGINEERED_FEATURES

# ---------------------------------------------------------------------------
# Fitur terpilih untuk model Alpha V1 (hasil feature pruning 160 eksperimen)
# ---------------------------------------------------------------------------
# SHAP + grid search menunjukkan hanya 5 fitur ini yang stabil dan prediktif
# setelah data leakage ditutup. Fitur lain hanya menambah noise dan varians.
# Backward compat: ALL_FEATURES tetap tersedia untuk analisis.
SELECTED_FEATURES: List[str] = [
    "entry_odds",           # #1 SHAP — harga pasar itu sendiri
    "depth_ratio",          # #2 SHAP — kedalaman order book
    "contest_urgency",      # #3 SHAP — seberapa dekat ke expiry
    "tfm_value",            # #4 SHAP — trade flow momentum
    "obi_vol_interaction",  # #5 SHAP — OBI x volatilitas (engineered)
]

# ---------------------------------------------------------------------------
# Kolom metadata (bukan fitur, tapi dibutuhkan untuk splitting & evaluasi)
# ---------------------------------------------------------------------------
META_COLS: List[str] = [
    "timestamp",
    "market_id",
    "signal_direction",
    "entry_odds",       # duplikat untuk akses mudah di EV engine
    "retrofix_status",
]

TARGET_COL: str = "label"

# ---------------------------------------------------------------------------
# Konfigurasi split & validasi
# ---------------------------------------------------------------------------
@dataclass
class SplitConfig:
    n_cv_folds: int = 5             # GroupKFold folds
    calib_holdout_ratio: float = 0.30  # 30% untuk Isotonic calibration (butuh lebih banyak data)
    min_samples_gate: int = 800     # minimum settled bets sebelum ML aktif
    min_markets_gate: int = 200     # minimum unique markets
    dedup_ttr_cutoff: int = 120     # detik — ambil sinyal terakhir sebelum TTR < cutoff


# ---------------------------------------------------------------------------
# Hyperparameter XGBoost
# ---------------------------------------------------------------------------
@dataclass
class XGBConfig:
    n_estimators: int = 800
    max_depth: int = 3
    learning_rate: float = 0.025
    subsample: float = 0.70
    colsample_bytree: float = 0.60
    min_child_weight: int = 50      # tinggi — mencegah overfit pada tick-level data
    gamma: float = 0.20             # regularisasi minimum split gain
    reg_alpha: float = 1.0          # L1 regularization (agresif)
    reg_lambda: float = 4.0         # L2 regularization (agresif)
    early_stopping_rounds: int = 50
    random_state: int = 42
    n_jobs: int = -1
    verbosity: int = 0
    eval_metric: str = "logloss"


# ---------------------------------------------------------------------------
# Gate kualitas model (harus SEMUA terpenuhi sebelum model di-deploy)
# ---------------------------------------------------------------------------
@dataclass
class QualityGates:
    min_auc_cv: float = 0.60        # AUC rata-rata GroupKFold
    max_auc_std: float = 0.08       # standar deviasi AUC antar fold
    max_ece: float = 0.05           # Expected Calibration Error (in-sample calib set)
    max_ece_test: float = 0.20      # ECE Test (OOS) — mencegah Isotonic bypass
    max_brier: float = 0.26         # Brier score (coinflip = 0.25)


# ---------------------------------------------------------------------------
# Konfigurasi EV & Kelly
# ---------------------------------------------------------------------------
@dataclass
class EVConfig:
    ev_threshold: float = 0.04          # EV minimum untuk eksekusi
    kelly_fraction: float = 0.25        # fractional Kelly (25% of full Kelly)
    max_kelly_pct: float = 0.10         # batas atas bet size (10% bankroll)
    uncertainty_buffer_multiplier: float = 0.5  # EV_thr += 0.5 × std(P_model)


# ---------------------------------------------------------------------------
# Konfigurasi drift detection
# ---------------------------------------------------------------------------
@dataclass
class DriftConfig:
    psi_threshold: float = 0.20         # Population Stability Index — regime shift
    brier_rolling_window: int = 50      # window rolling untuk Brier monitoring
    brier_degradation_threshold: float = 0.30  # downgrade ke deterministic jika lewat
    top_features_to_monitor: int = 5    # monitor N fitur teratas dari SHAP


# ---------------------------------------------------------------------------
# Singleton config — import langsung dari modul lain
# ---------------------------------------------------------------------------
SPLIT_CFG = SplitConfig()
XGB_CFG = XGBConfig()
QUALITY_GATES = QualityGates()
EV_CFG = EVConfig()
DRIFT_CFG = DriftConfig()
