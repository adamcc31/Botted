"""
consolidate_alpha_v1.py
========================
Consolidation pipeline untuk Alpha V1 training dataset.

Langkah:
  1. Retrofix strikes pada semua 3 file chop (via retrofix_strikes.py)
  2. Recompute labels pada file chop yang sudah di-retrofix (via recompute_labels.py)
  3. Schema alignment — normalize ttr_minutes → ttr_seconds, align semua kolom
  4. Concatenate semua 5 file
  5. Sort chronological by timestamp, drop_duplicates
  6. Generate label column, simpan alpha_v1_master.csv

Penggunaan:
    python scripts/consolidate_alpha_v1.py

    Untuk skip retrofix/recompute (jika sudah pernah dijalankan):
    python scripts/consolidate_alpha_v1.py --skip-retrofix --skip-recompute
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "dataset" / "raw"

CHOP_FILES = [
    "dataset_weekday_monday_chop_v1.csv",
    "dataset_weekend_chop_v1.csv",
    "dataset_weekday_monday-0100-0230AM ET_chop_v1.csv",
]

DRY_RUN_FILES = [
    "dry_run_2026-04-28_132421.csv",
    "dry_run_2026-04-28_155909.csv",
]

OUTPUT_FILE = RAW_DIR / "alpha_v1_master.csv"


# ─────────────────────────────────────────────
# STEP 1: RETROFIX STRIKES
# ─────────────────────────────────────────────

def run_retrofix(chop_files: list[str], skip: bool = False) -> None:
    """Jalankan retrofix_strikes.py pada setiap file chop."""
    if skip:
        print("\n[STEP 1] SKIP retrofix (--skip-retrofix flag)")
        return

    print("\n" + "=" * 60)
    print("STEP 1: RETROFIX STRIKES (all chop files)")
    print("=" * 60)

    retrofix_script = BASE_DIR / "scripts" / "retrofix_strikes.py"

    for fname in chop_files:
        fpath = RAW_DIR / fname
        print(f"\n  → Retrofix: {fname}")

        cmd = [
            sys.executable, str(retrofix_script),
            "--input", str(fpath),
            "--output", str(fpath),        # overwrite in-place
            "--skip-health-check",          # gunakan Binance fallback jika Vatic down
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"  ⚠️  Retrofix WARNING (exit code {result.returncode}):")
            print(result.stderr[-500:] if result.stderr else "(no stderr)")
            # Jangan fatal — lanjutkan ke file berikutnya
        else:
            print(f"  ✅ Retrofix OK: {fname}")


# ─────────────────────────────────────────────
# STEP 2: RECOMPUTE LABELS
# ─────────────────────────────────────────────

def run_recompute(chop_files: list[str], skip: bool = False) -> None:
    """Jalankan recompute_labels.py pada setiap file chop."""
    if skip:
        print("\n[STEP 2] SKIP recompute (--skip-recompute flag)")
        return

    print("\n" + "=" * 60)
    print("STEP 2: RECOMPUTE LABELS (all chop files)")
    print("=" * 60)

    recompute_script = BASE_DIR / "scripts" / "recompute_labels.py"

    for fname in chop_files:
        fpath = RAW_DIR / fname
        print(f"\n  → Recompute: {fname}")

        cmd = [
            sys.executable, str(recompute_script),
            "--input", str(fpath),
            "--output", str(fpath),        # overwrite in-place
            "--skip-vatic",                 # gunakan strike/resolution yang sudah ada di CSV
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"  ⚠️  Recompute WARNING (exit code {result.returncode}):")
            print(result.stderr[-500:] if result.stderr else "(no stderr)")
        else:
            print(f"  ✅ Recompute OK: {fname}")


# ─────────────────────────────────────────────
# STEP 3: SCHEMA ALIGNMENT & CONCATENATION
# ─────────────────────────────────────────────

def load_and_align() -> pd.DataFrame:
    """
    Load semua 5 file CSV, align schema, dan concatenate.

    Schema alignment strategy:
    - ttr_minutes (dry_run_155909 only) → convert ke ttr_seconds (× 60)
    - theoretical_exit_odds: fill NaN jika tidak ada
    - settlement_price + settlement_price_source: pertahankan, fill NaN
    """
    print("\n" + "=" * 60)
    print("STEP 3: SCHEMA ALIGNMENT & CONCATENATION")
    print("=" * 60)

    all_files = CHOP_FILES + DRY_RUN_FILES
    frames = []

    for fname in all_files:
        fpath = RAW_DIR / fname
        df = pd.read_csv(fpath)
        df["_source_file"] = fname
        print(f"  Loaded {fname}: {len(df)} rows × {len(df.columns)} cols")

        # ── Normalize ttr_minutes → ttr_seconds ──
        if "ttr_minutes" in df.columns and "ttr_seconds" not in df.columns:
            df["ttr_seconds"] = pd.to_numeric(
                df["ttr_minutes"], errors="coerce"
            ) * 60
            df = df.drop(columns=["ttr_minutes"])
            print(f"    → Converted ttr_minutes → ttr_seconds")

        frames.append(df)

    # ── Union all columns ──
    all_cols = set()
    for df in frames:
        all_cols |= set(df.columns)

    # ── Align: reindex semua frame ke superset kolom ──
    aligned_frames = []
    for df in frames:
        for col in all_cols:
            if col not in df.columns:
                df[col] = np.nan
        aligned_frames.append(df)

    # ── Concatenate ──
    master = pd.concat(aligned_frames, ignore_index=True)
    print(f"\n  Concatenated: {len(master)} rows × {len(master.columns)} cols")

    return master


# ─────────────────────────────────────────────
# STEP 4: SORT & DEDUP
# ─────────────────────────────────────────────

def sort_and_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Urutkan kronologis dan buang duplikat.

    Sort: berdasarkan timestamp (ascending) — mencegah lookahead bias.
    Dedup: berdasarkan timestamp + market_id (tick-level data).
    """
    print("\n" + "=" * 60)
    print("STEP 4: TIME-SERIES SORT & DEDUPLICATION")
    print("=" * 60)

    before = len(df)

    # Parse timestamp ke datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Sort chronological
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Sorted chronologically: {len(df)} rows")
    print(f"  Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # Drop duplicates pada timestamp + market_id
    df = df.drop_duplicates(subset=["timestamp", "market_id"], keep="last")
    after = len(df)
    dropped = before - after
    print(f"  Dropped {dropped} duplicate rows (by timestamp + market_id)")
    print(f"  Final: {after} rows")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# STEP 5: GENERATE LABEL & FINAL CLEANUP
# ─────────────────────────────────────────────

def generate_label_and_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buat kolom 'label' (binary target untuk XGBoost) dan cleanup final.

    label = 1 jika signal_correct == TRUE (WIN)
    label = 0 jika signal_correct == FALSE (LOSS)
    Rows tanpa label valid (N/A, PENDING) tetap dipertahankan tapi
    label = NaN — pipeline training akan filter sendiri.
    """
    print("\n" + "=" * 60)
    print("STEP 5: GENERATE LABEL & FINAL CLEANUP")
    print("=" * 60)

    # Generate label — handle mixed types: bool True/False, str TRUE/FALSE, NaN
    sc = df["signal_correct"].astype(str).str.upper()
    df["label"] = np.where(
        sc == "TRUE", 1,
        np.where(sc == "FALSE", 0, np.nan)
    )

    # Ensure obi_tfm_product & obi_tfm_alignment exist
    obi = pd.to_numeric(df.get("obi_value"), errors="coerce")
    tfm = pd.to_numeric(df.get("tfm_value"), errors="coerce")
    if "obi_tfm_product" not in df.columns:
        df["obi_tfm_product"] = obi * tfm
    if "obi_tfm_alignment" not in df.columns:
        df["obi_tfm_alignment"] = ((obi * tfm) > 0).astype(int)

    # Stats
    labeled = df["label"].notna().sum()
    wins = (df["label"] == 1).sum()
    losses = (df["label"] == 0).sum()
    unlabeled = df["label"].isna().sum()

    print(f"  Labeled rows   : {labeled}")
    print(f"    WIN  (label=1): {wins}")
    print(f"    LOSS (label=0): {losses}")
    print(f"  Unlabeled (NaN): {unlabeled}")
    if labeled > 0:
        print(f"  Win Rate       : {wins / labeled * 100:.2f}%")

    # Column ordering — put important cols first
    priority_cols = [
        "timestamp", "market_id", "slug", "signal_direction",
        "label", "signal_correct", "actual_outcome",
        "strike_price", "resolution_price",
        "entry_odds", "confidence_score",
    ]
    remaining = [c for c in df.columns if c not in priority_cols]
    col_order = [c for c in priority_cols if c in df.columns] + sorted(remaining)
    df = df[col_order]

    return df


# ─────────────────────────────────────────────
# STEP 6: SAVE & SUMMARY
# ─────────────────────────────────────────────

def save_master(df: pd.DataFrame) -> None:
    """Simpan alpha_v1_master.csv dan cetak summary."""
    print("\n" + "=" * 60)
    print("STEP 6: SAVE alpha_v1_master.csv")
    print("=" * 60)

    df.to_csv(OUTPUT_FILE, index=False)
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"\n  ✅  Saved: {OUTPUT_FILE}")
    print(f"  File size: {size_mb:.2f} MB")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Time span: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"\n  Unique markets: {df['market_id'].nunique()}")
    print(f"  Unique slugs: {df['slug'].nunique()}")

    # Source breakdown
    if "_source_file" in df.columns:
        print(f"\n  Source file breakdown:")
        for src, count in df["_source_file"].value_counts().items():
            print(f"    {src}: {count} rows")

    # Column list
    print(f"\n  Final columns ({len(df.columns)}):")
    for col in df.columns:
        null_pct = df[col].isna().mean() * 100
        print(f"    {col:35s} null={null_pct:5.1f}%")

    print("\n" + "=" * 60)
    print("CONSOLIDATION COMPLETE")
    print("=" * 60)
    print(f"\nNext step — train XGBoost Alpha V1:")
    print(f"  python -m model_training --data dataset/raw/alpha_v1_master.csv --output models/alpha_v1/ --version alpha_v1")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Consolidate Alpha V1 training dataset from 5 raw CSV files."
    )
    parser.add_argument(
        "--skip-retrofix", action="store_true",
        help="Skip retrofix_strikes.py (use if already run)"
    )
    parser.add_argument(
        "--skip-recompute", action="store_true",
        help="Skip recompute_labels.py (use if already run)"
    )
    args = parser.parse_args()

    # Step 1: Retrofix strikes
    run_retrofix(CHOP_FILES, skip=args.skip_retrofix)

    # Step 2: Recompute labels
    run_recompute(CHOP_FILES, skip=args.skip_recompute)

    # Step 3: Load & align schema
    master = load_and_align()

    # Step 4: Sort & dedup
    master = sort_and_dedup(master)

    # Step 5: Generate label & cleanup
    master = generate_label_and_cleanup(master)

    # Step 6: Save
    save_master(master)


if __name__ == "__main__":
    main()
