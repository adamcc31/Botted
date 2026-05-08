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
# V5: Load dry-run CSVs & merge into master dataset
# ---------------------------------------------------------------------------

# Kolom numerik yang mungkin menggunakan format angka Indonesia/Eropa
# (titik sebagai ribuan separator, koma sebagai desimal)
_NUMERIC_COLS_TO_NORMALIZE: list[str] = [
    "ttr_seconds", "strike_price", "odds_yes", "odds_no",
    "binance_price", "chainlink_price", "obi_value", "tfm_value",
    "rv_value", "depth_ratio", "odds_delta_60s", "btc_return_1m",
    "entry_odds", "confidence_score", "spread_pct",
    "strike_distance_pct", "contest_urgency", "vol_percentile",
    "odds_yes_60s_ago",
]


def _normalize_numeric_col(series: pd.Series) -> pd.Series:
    """
    Normalisasi kolom numerik yang mungkin menggunakan format angka
    Indonesia/Eropa (titik sebagai ribuan, koma sebagai desimal).

    Heuristik: jika ada koma DAN titik dalam satu value, asumsikan
    titik = ribuan, koma = desimal → hapus titik, ganti koma → titik.
    Jika hanya koma, asumsikan desimal.
    """
    if series.dtype in (np.float64, np.float32, np.int64, np.int32, float, int):
        return series  # sudah numerik

    s = series.astype(str)

    def _fix_number(val: str) -> str:
        val = val.strip()
        if val in ("", "nan", "None", "N/A", "NaN"):
            return "NaN"
        has_comma = "," in val
        has_dot = "." in val
        if has_comma and has_dot:
            # Cek posisi terakhir: jika koma setelah titik → format EN
            last_comma = val.rfind(",")
            last_dot = val.rfind(".")
            if last_comma > last_dot:
                # Format ID/EU: 1.234,56 → 1234.56
                val = val.replace(".", "").replace(",", ".")
            # else: format EN: 1,234.56 → 1234.56
            else:
                val = val.replace(",", "")
        elif has_comma and not has_dot:
            # Hanya koma → asumsikan desimal
            val = val.replace(",", ".")
        return val

    return pd.to_numeric(s.apply(_fix_number), errors="coerce")


def load_dry_run_data(file_paths: list[str | Path]) -> pd.DataFrame:
    """
    Load dan gabungkan beberapa file dry-run CSV.

    Dry-run CSVs menggunakan comma delimiter dan tidak memiliki kolom
    'label'. Label diderivasi dari signal_direction == actual_outcome.
    Baris tanpa resolusi (actual_outcome PENDING/N/A/kosong) dibuang.
    """
    frames = []
    for fpath in file_paths:
        fpath = Path(fpath)
        if not fpath.exists():
            logger.warning("File tidak ditemukan, skip: %s", fpath)
            continue

        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            logger.warning("Gagal baca %s: %s — skip", fpath, e)
            continue

        if len(df) == 0:
            logger.warning("File kosong: %s — skip", fpath)
            continue

        df["_source_file"] = fpath.name
        frames.append(df)
        logger.info("  Loaded dry-run: %s → %d rows", fpath.name, len(df))

    if not frames:
        logger.warning("Tidak ada dry-run file yang berhasil dimuat.")
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    logger.info("Total dry-run rows (raw): %d dari %d files", len(df_all), len(frames))
    return df_all


def merge_datasets(
    df_train: pd.DataFrame,
    dry_run_files: list[str | Path],
) -> pd.DataFrame:
    """
    Gabungkan dataset training V4 dengan dry-run CSVs menjadi satu
    master dataset.

    Kriteria merge:
      1. Load dan concat semua dry-run CSVs
      2. Drop rows dengan actual_outcome == 'PENDING' atau kosong
      3. Derive label: 1 jika signal_direction == actual_outcome, else 0
         (hanya untuk rows yang memiliki signal_direction BUY_UP/BUY_DOWN)
      4. Dedup berdasarkan (timestamp, market_id) — keep first
      5. Sort ascending by timestamp
      6. Normalisasi kolom numerik (format ID/EU)
      7. Hitung depth_ratio_std per market_id

    Args:
        df_train:       DataFrame training v4 (sudah memiliki kolom 'label')
        dry_run_files:  List path ke dry-run CSVs

    Returns:
        DataFrame gabungan dengan kolom 'label' yang konsisten.
    """
    logger.info("=" * 60)
    logger.info("MERGE DATASETS — V4 Training + Dry-Run")
    logger.info("=" * 60)

    # --- Load dry-run ---
    df_dry = load_dry_run_data(dry_run_files)

    if len(df_dry) == 0:
        logger.warning("Tidak ada dry-run data. Return training v4 saja.")
        return df_train.copy()

    # --- Filter dry-run: buang PENDING dan baris tanpa resolusi ---
    if "actual_outcome" in df_dry.columns:
        ao = df_dry["actual_outcome"].astype(str).str.strip().str.upper()
        valid_mask = ao.isin(["BUY_UP", "BUY_DOWN"])
        n_before = len(df_dry)
        df_dry = df_dry[valid_mask].copy()
        logger.info(
            "Dry-run: dropped %d unresolved rows (PENDING/N/A/kosong), kept %d",
            n_before - len(df_dry), len(df_dry),
        )

    # --- Derive label untuk dry-run ---
    if "label" not in df_dry.columns:
        df_dry["label"] = np.nan

    # Label: 1 jika signal_direction == actual_outcome, else 0
    # Hanya untuk rows dengan signal_direction yang valid (BUY_UP/BUY_DOWN)
    sd = df_dry["signal_direction"].astype(str).str.strip().str.upper()
    ao = df_dry["actual_outcome"].astype(str).str.strip().str.upper()
    valid_signal = sd.isin(["BUY_UP", "BUY_DOWN"])

    df_dry.loc[valid_signal, "label"] = (
        (sd[valid_signal] == ao[valid_signal]).astype(int)
    )

    logger.info(
        "Dry-run label derivation: %d labeled (%d valid signals), "
        "%d unlabeled (ABSTAIN/SKIP)",
        valid_signal.sum(),
        int(df_dry["label"].notna().sum()),
        int(df_dry["label"].isna().sum()),
    )

    # --- Normalisasi kolom numerik di dry-run ---
    for col in _NUMERIC_COLS_TO_NORMALIZE:
        if col in df_dry.columns:
            df_dry[col] = _normalize_numeric_col(df_dry[col])

    # --- Pastikan kolom timestamp konsisten ---
    df_dry["timestamp"] = pd.to_datetime(df_dry["timestamp"], utc=True)

    if "timestamp" in df_train.columns:
        df_train = df_train.copy()
        df_train["timestamp"] = pd.to_datetime(df_train["timestamp"], utc=True)

    # --- Concat ---
    # Align kolom — gunakan union
    df_merged = pd.concat([df_train, df_dry], ignore_index=True, sort=False)
    logger.info(
        "Merged: %d (v4) + %d (dry-run) = %d total",
        len(df_train), len(df_dry), len(df_merged),
    )

    # --- Dedup berdasarkan (timestamp, market_id) — keep first ---
    if "market_id" in df_merged.columns and "timestamp" in df_merged.columns:
        n_before = len(df_merged)
        df_merged = df_merged.drop_duplicates(
            subset=["timestamp", "market_id"], keep="first"
        )
        logger.info("Dedup: %d → %d (dropped %d dupes)",
                     n_before, len(df_merged), n_before - len(df_merged))

    # --- Sort by timestamp ---
    df_merged = df_merged.sort_values("timestamp").reset_index(drop=True)

    return df_merged


def sanitize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitasi master dataset:
      1. Drop rows tanpa label (NaN / PENDING)
      2. Pastikan label adalah int 0/1
      3. Normalisasi kolom numerik
      4. Hitung depth_ratio_std per market_id (fitur batch)
      5. Drop rows dengan entry_odds invalid (NaN, 0, >1)

    Returns:
        DataFrame yang sudah bersih dan siap untuk training.
    """
    logger.info("Sanitizing master dataset: %d rows input", len(df))
    df = df.copy()

    # --- Drop rows tanpa label ---
    if "label" in df.columns:
        n_before = len(df)
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df[df["label"].notna()].copy()
        df["label"] = df["label"].astype(int)
        logger.info("Dropped %d rows tanpa label, kept %d",
                     n_before - len(df), len(df))

    # --- Normalisasi kolom numerik ---
    for col in _NUMERIC_COLS_TO_NORMALIZE:
        if col in df.columns:
            df[col] = _normalize_numeric_col(df[col])

    # --- Drop entry_odds invalid ---
    if "entry_odds" in df.columns:
        n_before = len(df)
        df = df[
            df["entry_odds"].notna() &
            (df["entry_odds"] > 0) &
            (df["entry_odds"] <= 1.0)
        ].copy()
        logger.info("Dropped %d rows dengan entry_odds invalid",
                     n_before - len(df))

    # --- Hitung depth_ratio_std per market_id (batch feature) ---
    if "depth_ratio" in df.columns and "market_id" in df.columns:
        df_std = (
            df.groupby("market_id")["depth_ratio"]
            .std()
            .reset_index()
        )
        df_std.columns = ["market_id", "depth_ratio_std"]
        df = df.merge(df_std, on="market_id", how="left")
        df["depth_ratio_std"] = df["depth_ratio_std"].fillna(0.0)
        logger.info("Computed depth_ratio_std per market_id")

    # --- Pastikan kolom yang dibutuhkan ada ---
    # Tambahkan kolom kosong untuk fitur yang mungkin tidak tersedia
    for col in ["clob_spread_vel", "clob_depth_delta",
                "obi_tfm_product", "obi_tfm_alignment"]:
        if col not in df.columns:
            df[col] = 0.0
            logger.info("  Added missing column '%s' with default 0.0", col)

    df = df.reset_index(drop=True)
    logger.info("Sanitized dataset: %d rows, label dist: WIN=%.1f%%",
                len(df), df["label"].mean() * 100)
    return df


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
