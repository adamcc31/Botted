"""
ml/features.py
==============
Rekayasa fitur baru di atas raw features.

Masalah utama yang diperbaiki di sini:
  1. entry_odds mendominasi SHAP 10× — kita buat fitur yang mengukur
     *divergensi* antara model dan crowd, bukan level odds itu sendiri.
  2. obi_value hampir tidak prediktif sendirian — dikombinasikan dengan
     volatility context agar bermakna.
"""

import numpy as np
import pandas as pd
from typing import Optional

from .config import RAW_FEATURES, ENGINEERED_FEATURES, ALL_FEATURES, LEAKAGE_COLS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan semua ENGINEERED_FEATURES ke dataframe.
    Input: dataframe yang sudah berisi RAW_FEATURES.
    Output: dataframe dengan ALL_FEATURES tersedia.

    PENTING: fungsi ini hanya boleh menggunakan kolom yang tersedia
    pada saat sinyal dibuat (t=0). Tidak ada look-ahead.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. edge_vs_crowd  →  q_fair - entry_odds
    #
    # Fitur terpenting untuk mispricing bot. Mengukur seberapa jauh
    # model Black-Scholes meyakini probabilitas yang lebih tinggi dari
    # yang ditawarkan pasar.
    #   > 0  : model lebih optimis dari crowd → potensi undervalued
    #   < 0  : crowd lebih optimis → bot mungkin tidak mau masuk
    #
    # q_fair di sini diestimasi dari confidence_score yang sudah
    # mencerminkan q_fair = Φ(d2) dari Black-Scholes digital.
    # Jika kolom 'q_fair' tersedia langsung dari feature_engine,
    # gunakan itu. Fallback ke confidence_score.
    # ------------------------------------------------------------------
    q_fair_col = "q_fair" if "q_fair" in df.columns else "confidence_score"
    df["edge_vs_crowd"] = df[q_fair_col] - df["entry_odds"]

    # ------------------------------------------------------------------
    # 2. obi_vol_interaction  →  OBI × vol_percentile
    #
    # OBI sendirian lemah karena noise tinggi saat volatilitas rendah.
    # Saat vol_percentile tinggi, OBI signal lebih bermakna karena
    # ada tekanan nyata di order book. Produk ini menangkap itu.
    # ------------------------------------------------------------------
    df["obi_vol_interaction"] = df["obi_value"] * df["vol_percentile"]

    # ------------------------------------------------------------------
    # 3. tfm_vol_interaction  →  TFM × vol_percentile
    #
    # Trade Flow Momentum juga lebih bermakna saat vol tinggi.
    # ------------------------------------------------------------------
    df["tfm_vol_interaction"] = df["tfm_value"] * df["vol_percentile"]

    # ------------------------------------------------------------------
    # 4. odds_momentum  →  perubahan relatif odds dalam 60 detik
    #
    # odds_delta_60s / odds_yes_60s_ago mengukur kecepatan pergerakan
    # crowd. Crowd yang bergerak cepat ke satu arah → FOMO signal.
    # ------------------------------------------------------------------
    # Ensure numeric and fill NaN to prevent abs() NoneType error
    odds_yes_60s = pd.to_numeric(df["odds_yes_60s_ago"], errors='coerce').fillna(0.0)
    odds_delta_60s = pd.to_numeric(df["odds_delta_60s"], errors='coerce').fillna(0.0)
    
    df["odds_momentum"] = odds_delta_60s / (odds_yes_60s.abs() + 1e-6)
    # clip ekstrem
    df["odds_momentum"] = df["odds_momentum"].clip(-5.0, 5.0)

    # ------------------------------------------------------------------
    # 5. urgency_vol  →  contest_urgency × rv_value
    #
    # Pasar yang hampir expired (urgency tinggi) + volatilitas tinggi
    # = situasi paling menarik untuk asymmetric bet.
    # ------------------------------------------------------------------
    df["urgency_vol"] = df["contest_urgency"] * df["rv_value"]

    # ------------------------------------------------------------------
    # 6. micro_alignment  →  obi_value × tfm_value
    #
    # Ketika OBI dan TFM searah (keduanya positif atau keduanya negatif),
    # produknya positif → konfirmasi ganda microstructure.
    # Ketika berlawanan arah → sinyal ambigu → produk negatif.
    # ------------------------------------------------------------------
    df["micro_alignment"] = df["obi_value"] * df["tfm_value"]

    # ------------------------------------------------------------------
    # 7. hour_wib  →  jam dalam zona waktu WIB (UTC+7)
    #
    # Polymarket BTC 5-minute markets memiliki pola volume dan
    # volatilitas yang berbeda di setiap jam. Hour dalam WIB (UTC+7)
    # lebih relevan karena bot dioperasikan dari timezone Indonesia.
    # Dikonversi ke integer 0-23 — XGBoost bisa split natural.
    # ------------------------------------------------------------------
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        # UTC+7 = WIB (Waktu Indonesia Barat)
        ts_wib = ts.dt.tz_convert("Asia/Jakarta")
        df["hour_wib"] = ts_wib.dt.hour
    else:
        df["hour_wib"] = 0  # fallback — seharusnya tidak terjadi

    # ------------------------------------------------------------------
    # 8. is_weekend  →  1 jika Sabtu/Minggu (UTC), 0 jika weekday
    #
    # Market behavior berbeda signifikan di weekend:
    # - Volume lebih rendah → spread lebih lebar
    # - Volatilitas Chainlink cenderung lebih rendah
    # - OBI signal mungkin lebih noisy karena likuiditas tipis
    # ------------------------------------------------------------------
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
    else:
        df["is_weekend"] = 0

    # Validasi: pastikan semua fitur yang dibutuhkan tersedia
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"Fitur berikut tidak tersedia setelah build_features(): {missing}\n"
            f"Pastikan raw features sudah ada di dataframe input."
        )

    return df


def get_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Ekstrak X (feature matrix), y (labels), dan meta dataframe.

    Menggunakan SELECTED_FEATURES (top-5 hasil feature pruning),
    bukan ALL_FEATURES, untuk mengurangi noise dan varians model.

    Returns:
        X       : np.ndarray shape (n, len(SELECTED_FEATURES))
        y       : np.ndarray shape (n,) — binary labels
        meta_df : kolom non-fitur untuk keperluan evaluasi & splitting
    """
    from .config import TARGET_COL, META_COLS, SELECTED_FEATURES

    df = build_features(df)

    X = df[SELECTED_FEATURES].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)

    # meta: kolom yang dibutuhkan untuk GroupKFold dan EV engine
    available_meta = [c for c in META_COLS if c in df.columns]
    meta_df = df[available_meta].copy()

    return X, y, meta_df


def validate_no_leakage(df: pd.DataFrame) -> None:
    """
    Periksa bahwa tidak ada kolom leakage digunakan sebagai fitur.
    Raise ValueError jika ada irisan.
    """
    used = set(ALL_FEATURES)
    leaked = used & set(LEAKAGE_COLS)
    if leaked:
        raise ValueError(
            f"LEAKAGE DETECTED: fitur berikut ada di ALL_FEATURES "
            f"tapi juga di LEAKAGE_COLS: {leaked}"
        )
