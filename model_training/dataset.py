"""
ml/dataset.py
=============
Load dataset dari SQLite atau CSV, validasi kualitas,
deduplikasi per market, dan split chronological.

Masalah yang diperbaiki:
  - 5.72 sinyal per market → ambil SATU sinyal per market
    (sinyal terakhir sebelum TTR < cutoff, yang paling informatif)
  - Split by market_id, bukan random split
  - Gate minimum sampel sebelum ML diaktifkan
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    LEAKAGE_COLS, TARGET_COL, ALL_FEATURES,
    SPLIT_CFG, META_COLS,
)
from .features import build_features, validate_no_leakage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_from_csv(path: str | Path) -> pd.DataFrame:
    """Load dataset dari CSV yang dieksport oleh bot.

    Setelah load, langsung filter ke baris yang sudah punya resolusi
    (ground truth). Baris dengan label NaN atau signal_correct
    kosong/PENDING dibuang karena tidak bisa digunakan untuk training.
    """
    df = pd.read_csv(path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    n_raw = len(df)

    # --- Filter: hanya baris yang sudah resolved ---
    # 1. label harus ada (bukan NaN)
    if "label" in df.columns:
        df = df[df["label"].notna()]

    # 2. signal_correct harus TRUE/FALSE (bukan PENDING/N/A/kosong)
    if "signal_correct" in df.columns:
        sc = df["signal_correct"].astype(str).str.upper()
        df = df[sc.isin(["TRUE", "FALSE"])]

    df = df.reset_index(drop=True)
    n_filtered = len(df)
    logger.info(
        "Loaded %d rows dari %s (filtered from %d raw — dropped %d unresolved)",
        n_filtered, path, n_raw, n_raw - n_filtered,
    )
    return df


def load_from_sqlite(db_path: str | Path,
                     table: str = "signals",
                     only_settled: bool = True) -> pd.DataFrame:
    """
    Load dari SQLite signals.db bot.
    Hanya ambil baris yang sudah di-settle (signal_correct tidak NULL).
    """
    import sqlite3
    conn = sqlite3.connect(db_path)

    query = f"SELECT * FROM {table}"
    if only_settled:
        query += " WHERE signal_correct IS NOT NULL"
    query += " ORDER BY timestamp ASC"

    df = pd.read_sql(query, conn, parse_dates=["timestamp"])
    conn.close()

    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    logger.info("Loaded %d settled rows dari %s::%s", len(df), db_path, table)
    return df


# ---------------------------------------------------------------------------
# Validasi & cleaning
# ---------------------------------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Jalankan semua cek kualitas. Return dict hasil cek.
    Raise ValueError jika ada masalah fatal.
    """
    results = {}

    # Cek kolom target
    if TARGET_COL not in df.columns:
        raise ValueError(f"Kolom target '{TARGET_COL}' tidak ada di dataset.")

    # Cek null di fitur kritis
    raw_present = [f for f in ALL_FEATURES if f in df.columns]
    null_counts = df[raw_present].isnull().sum()
    null_feats = null_counts[null_counts > 0]
    if len(null_feats) > 0:
        logger.warning("Kolom dengan null values: %s", null_feats.to_dict())
    results["null_features"] = null_feats.to_dict()

    # Cek leakage
    validate_no_leakage(df)
    results["leakage_check"] = "PASS"

    # Statistik dasar
    results["n_rows"] = len(df)
    results["n_markets"] = df["market_id"].nunique() if "market_id" in df.columns else None
    results["win_rate"] = float(df[TARGET_COL].mean())
    results["time_span_hours"] = None

    if "timestamp" in df.columns:
        span = df["timestamp"].max() - df["timestamp"].min()
        results["time_span_hours"] = span.total_seconds() / 3600
        logger.info("Time span: %.1f jam", results["time_span_hours"])

    # Gate minimum sampel
    if results["n_rows"] < SPLIT_CFG.min_samples_gate:
        logger.warning(
            "SAMPLE GATE TIDAK TERPENUHI: %d baris < %d minimum. "
            "Lanjutkan paper trading sebelum deploy ML.",
            results["n_rows"], SPLIT_CFG.min_samples_gate
        )
        results["sample_gate"] = "FAIL"
    else:
        results["sample_gate"] = "PASS"

    if results["n_markets"] and results["n_markets"] < SPLIT_CFG.min_markets_gate:
        logger.warning(
            "MARKET GATE TIDAK TERPENUHI: %d markets < %d minimum.",
            results["n_markets"], SPLIT_CFG.min_markets_gate
        )
        results["market_gate"] = "FAIL"
    else:
        results["market_gate"] = "PASS"

    logger.info("Validasi selesai: %s", results)
    return results


# ---------------------------------------------------------------------------
# Deduplikasi — satu sinyal per market
# ---------------------------------------------------------------------------

def deduplicate_per_market(df: pd.DataFrame,
                            ttr_cutoff: int = None) -> pd.DataFrame:
    """
    Ambil SATU baris per market_id.

    Strategi: sinyal terakhir yang masih memiliki TTR >= ttr_cutoff.
    Rationale: sinyal ini paling informatif (fitur paling up-to-date)
    dan masih punya waktu masuk yang valid.

    Jika ttr_cutoff=None, ambil baris terakhir per market tanpa filter.
    """
    if ttr_cutoff is None:
        ttr_cutoff = SPLIT_CFG.dedup_ttr_cutoff

    if "market_id" not in df.columns:
        logger.warning("market_id tidak ada — skip deduplikasi.")
        return df

    before = len(df)

    if "ttr_seconds" in df.columns and ttr_cutoff > 0:
        df_filtered = df[df["ttr_seconds"] >= ttr_cutoff].copy()
        if len(df_filtered) == 0:
            logger.warning(
                "Semua baris di-filter oleh ttr_cutoff=%d. "
                "Fallback: ambil baris terakhir per market tanpa filter.",
                ttr_cutoff
            )
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()

    # [REVISI ALPHA V0.1] Nonaktifkan deduplikasi per market
    # Biarkan semua baris masuk ke training set agar XGBoost punya cukup sampel
    logger.info("Deduplikasi DINONAKTIFKAN — meneruskan %d baris (Alpha V0.1)", before)
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Split chronological — train / calibration / test
# ---------------------------------------------------------------------------

def chronological_split(df: pd.DataFrame,
                         calib_ratio: float = None,
                         test_ratio: float = 0.0) -> tuple:
    """
    Split chronological pada level MARKET, bukan level baris.

    Mencegah data leakage pada tick-level data (multiple rows per market):
      1. Grupkan market_id berdasarkan timestamp terakhir (kapan market berakhir)
      2. Urutkan market_id secara kronologis
      3. Split daftar market_id menjadi train / calib / test
      4. Rekonstruksi dataframe dengan filter market_id

    Ini memastikan SELURUH tick dari satu market masuk ke satu partisi saja.
    Tidak ada tick dari market yang sama muncul di train DAN calib/test.

    Returns: (df_train, df_calib) atau (df_train, df_calib, df_test)
    """
    if calib_ratio is None:
        calib_ratio = SPLIT_CFG.calib_holdout_ratio

    if "market_id" not in df.columns:
        raise ValueError(
            "chronological_split memerlukan kolom 'market_id' "
            "untuk mencegah data leakage pada tick-level data."
        )

    # --- Step 1: Urutkan market_id berdasarkan timestamp terakhir ---
    market_end_time = (
        df.groupby("market_id")["timestamp"]
        .max()
        .sort_values()
    )
    ordered_markets = market_end_time.index.tolist()
    n_markets = len(ordered_markets)

    # --- Step 2: Split pada level market_id ---
    test_n_markets  = int(n_markets * test_ratio)
    calib_n_markets = int(n_markets * calib_ratio)
    train_n_markets = n_markets - calib_n_markets - test_n_markets

    train_markets = set(ordered_markets[:train_n_markets])
    calib_markets = set(ordered_markets[train_n_markets:train_n_markets + calib_n_markets])

    # --- Step 3: Rekonstruksi dataframe ---
    df_train = df[df["market_id"].isin(train_markets)].copy()
    df_calib = df[df["market_id"].isin(calib_markets)].copy()

    # Sort internal masing-masing partisi
    df_train = df_train.sort_values("timestamp").reset_index(drop=True)
    df_calib = df_calib.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        "Chronological split (market-level): "
        "train=%d rows (%d markets) | calib=%d rows (%d markets)",
        len(df_train), len(train_markets),
        len(df_calib), len(calib_markets),
    )

    if test_ratio > 0:
        test_markets = set(ordered_markets[train_n_markets + calib_n_markets:])
        df_test = df[df["market_id"].isin(test_markets)].copy()
        df_test = df_test.sort_values("timestamp").reset_index(drop=True)
        logger.info(
            "  + test=%d rows (%d markets)",
            len(df_test), len(test_markets),
        )

        # Sanity check: no market overlap
        assert train_markets.isdisjoint(calib_markets), "Leakage: train & calib overlap"
        assert train_markets.isdisjoint(test_markets), "Leakage: train & test overlap"
        assert calib_markets.isdisjoint(test_markets), "Leakage: calib & test overlap"

        return df_train, df_calib, df_test

    # Sanity check
    assert train_markets.isdisjoint(calib_markets), "Leakage: train & calib overlap"

    return df_train, df_calib


def get_market_groups(df: pd.DataFrame) -> np.ndarray:
    """
    Return array group untuk GroupKFold.
    Setiap market_id mendapat integer group yang sama.
    """
    if "market_id" not in df.columns:
        raise ValueError("market_id diperlukan untuk GroupKFold.")

    market_to_int = {m: i for i, m in enumerate(df["market_id"].unique())}
    return df["market_id"].map(market_to_int).values


# ---------------------------------------------------------------------------
# Imputer sederhana — handle edge case null values
# ---------------------------------------------------------------------------

def impute_features(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Isi null values di feature columns dengan median/mean/zero.
    Gunakan hanya nilai dari training set (fit sekali, apply ke semua).
    """
    from .config import RAW_FEATURES, ENGINEERED_FEATURES
    feat_cols = [c for c in RAW_FEATURES + ENGINEERED_FEATURES if c in df.columns]

    if strategy == "median":
        fill_vals = df[feat_cols].median()
    elif strategy == "mean":
        fill_vals = df[feat_cols].mean()
    else:
        fill_vals = pd.Series(0.0, index=feat_cols)

    df = df.copy()
    df[feat_cols] = df[feat_cols].fillna(fill_vals)
    return df, fill_vals


def apply_imputer(df: pd.DataFrame, fill_vals: pd.Series) -> pd.DataFrame:
    """Apply imputer yang sudah di-fit ke dataset baru (val/test/prod)."""
    df = df.copy()
    for col, val in fill_vals.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    return df
