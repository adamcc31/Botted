"""
recompute_labels.py  (v2 — vectorized, strict comparisons, WIN/LOSS semantics)
===============================================================================
Pipeline ETL pasca retrofix_strikes.py:

  Layer 1 — Ground Truth Labels
    actual_outcome  : WIN  jika harga bergerak sesuai signal direction
                      LOSS jika tidak
                      N/A  untuk ABSTAIN/SKIP
    signal_correct  : TRUE / FALSE / N/A (cermin dari actual_outcome)

    Logika ketat (strict):
      BUY_UP   wins  iff  resolution_price  >  strike_price  (bukan >=)
      BUY_DOWN wins  iff  resolution_price  <  strike_price  (bukan <=)
    Tie (resolution == strike) → LOSS karena market tidak menguntungkan buyer.

  Layer 2 — Theoretical PnL
    WIN  → (1 / entry_odds) - 1
    LOSS → -1.0
    N/A  → 0.0

  Layer 3 — Predictive Feature: strike_distance_pct
    (chainlink_price - strike_price_baru) / strike_price_baru * 100
    Disesuaikan dengan tanda sesuai arah signal:
      BUY_UP  : positif = chainlink sudah di atas strike (bagus)
      BUY_DOWN: negatif = chainlink sudah di bawah strike (bagus)
    Kolom merefleksikan jarak spasial aktual, bukan label mana yang benar.

  Layer 4 — spread_filter_passed
    TIDAK dihitung ulang — spread adalah OBI historis, bukan fungsi strike_price.

Cara pakai:
    python recompute_labels.py \\
        --input  dataset_fixed.csv \\
        --output dataset_clean.csv \\
        --ml-output dataset_ml_ready.csv

    # Jika Vatic API tidak dibutuhkan (strike & resolution sudah ada di CSV):
    python recompute_labels.py --input dataset_fixed.csv --output dataset_clean.csv --skip-vatic
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Optional

import httpx
import numpy as np
import pandas as pd
from tqdm import tqdm

VATIC_BASE    = "https://api.vatic.trading"
BINANCE_BASE  = "https://api.binance.com"

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _make_client(proxy_url: Optional[str] = None) -> httpx.Client:
    """
    Buat httpx.Client dengan proxy opsional.
    Proxy format: http://user:pass@host:port  (QUOTAGUARD / VPN tunnel)
    """
    if proxy_url:
        return httpx.Client(proxy=proxy_url, timeout=30.0)
    return httpx.Client(timeout=30.0)


# ──────────────────────────────────────────────────────────────────────────────
# SUMBER 1 — VATIC  (Chainlink oracle, akurat untuk settlement)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_vatic_range(
    start_ts: int,
    end_ts: int,
    proxy_url: Optional[str] = None,
) -> list[dict]:
    """
    Fetch Chainlink 5-min price history dari Vatic.
    Endpoint publik — tidak memerlukan API key maupun IP whitelist.
    Catatan: Chainlink hanya menyimpan data ~14 hari terakhir.
    """
    fetch_end = end_ts + 600
    try:
        with _make_client(proxy_url) as client:
            r = client.get(
                f"{VATIC_BASE}/api/v1/history/chainlink/range",
                params={
                    "asset": "btc",
                    "type": "5min",
                    "start": (start_ts // 300) * 300,
                    "end":   fetch_end,
                },
            )
        r.raise_for_status()
        data     = r.json()
        # Format Vatic: {"now":..., "asset":..., "history": [...], "count": N}
        ticks_raw = data.get("history", [])
        if not isinstance(ticks_raw, list):
            ticks_raw = []
        normalized = []
        for tick in ticks_raw:
            ts = (
                tick.get("timestamp_start")
                or tick.get("timestamp")
                or tick.get("ts")
            )
            price = tick.get("price") or tick.get("value")
            if ts and price:
                normalized.append({"timestamp": int(ts), "price": float(price)})
        normalized.sort(key=lambda x: x["timestamp"])
        print(f"  [Vatic]   {len(normalized)} ticks OK")
        return normalized
    except Exception as e:
        print(f"  [Vatic]   ERROR: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# SUMBER 2 — BINANCE KLINES  (fallback publik, tanpa IP restriction)
# Catatan: Binance price ≈ Chainlink, tapi bukan identik.
# Akurasi cukup untuk recompute label historical; jangan pakai untuk live bot.
# ──────────────────────────────────────────────────────────────────────────────

def fetch_binance_range(start_ts: int, end_ts: int) -> list[dict]:
    """
    Fetch BTCUSDT 5-min klines dari Binance public REST API.
    Tidak memerlukan API key maupun IP khusus.
    Returns ticks dalam format yang sama dengan fetch_vatic_range.
    """
    fetch_end = end_ts + 600
    all_ticks: list[dict] = []
    cursor    = (start_ts // 300) * 300 * 1000          # Binance pakai milliseconds
    end_ms    = fetch_end * 1000
    MAX_LOOPS = 20                                        # safety guard

    try:
        with _make_client() as client:
            for _ in range(MAX_LOOPS):
                r = client.get(
                    f"{BINANCE_BASE}/api/v3/klines",
                    params={
                        "symbol":    "BTCUSDT",
                        "interval":  "5m",
                        "startTime": cursor,
                        "endTime":   end_ms,
                        "limit":     1000,
                    },
                )
                r.raise_for_status()
                candles = r.json()
                if not candles:
                    break
                for c in candles:
                    # Binance kline: [open_time, open, high, low, close, ...]
                    # Kita pakai open_time (ms) → open price sebagai "harga di awal candle"
                    # Ini setara dengan logika Vatic: harga di awal epoch window
                    ts_sec  = int(c[0]) // 1000
                    price   = float(c[1])           # open price
                    all_ticks.append({"timestamp": ts_sec, "price": price})
                last_ts = int(candles[-1][0])
                if last_ts >= end_ms or len(candles) < 1000:
                    break
                cursor = last_ts + 1
        all_ticks.sort(key=lambda x: x["timestamp"])
        print(f"  [Binance] {len(all_ticks)} ticks OK  (fallback — harga Binance, bukan Chainlink)")
        return all_ticks
    except Exception as e:
        print(f"  [Binance] ERROR: {e}")
        return []


def get_price_at_or_after(ticks: list[dict], target_ts: int) -> float | None:
    """Ambil harga pertama dengan timestamp >= target_ts."""
    for tick in ticks:
        if tick["timestamp"] >= target_ts:
            return tick["price"]
    return None


def extract_epoch(slug: str) -> int | None:
    """Parse unix epoch dari slug seperti 'btc-updown-5m-1777266000'."""
    try:
        e = int(str(slug).split("-")[-1])
        return e if e > 1_700_000_000 else None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 0 — VATIC ENRICHMENT  (opsional, bisa di-skip jika kolom sudah ada)
# ──────────────────────────────────────────────────────────────────────────────

def enrich_from_vatic(
    df: pd.DataFrame,
    proxy_url: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch strike_price dan resolution_price, lalu overwrite ke df.
    Urutan sumber:
      1. Vatic API  (Chainlink oracle, akurat) — memerlukan IP allowlist / proxy
      2. Binance klines  (fallback otomatis jika Vatic gagal)
    Hanya dipanggil jika --skip-vatic TIDAK diset.
    """
    df = df.copy()
    df["_epoch"] = df["slug"].apply(extract_epoch)
    valid_epochs = df["_epoch"].dropna().astype(int)

    if len(valid_epochs) == 0:
        raise ValueError("Tidak ada epoch valid di kolom slug.")

    epoch_min = int(valid_epochs.min())
    epoch_max = int(valid_epochs.max())
    print(f"\nEpoch range : {epoch_min} → {epoch_max + 300}")
    print(f"Unique epochs: {valid_epochs.nunique()}")

    # ── Coba Vatic dulu, fallback ke Binance ──
    print("\nFetching price history...")
    ticks = fetch_vatic_range(epoch_min, epoch_max + 300, proxy_url=proxy_url)
    source_label = "VATIC_RECOMPUTE"

    if not ticks:
        print("  Vatic gagal → mencoba Binance klines sebagai fallback...")
        ticks = fetch_binance_range(epoch_min, epoch_max + 300)
        source_label = "BINANCE_FALLBACK"

    if not ticks:
        raise RuntimeError(
            "Kedua sumber data gagal (Vatic + Binance).\n"
            "  • Vatic: endpoint /api/v1/history/chainlink/range — cek koneksi internet.\n"
            "           Catatan: Chainlink hanya menyimpan data ~14 hari terakhir.\n"
            "  • Binance: cek koneksi internet / firewall.\n"
            "  • Atau gunakan --skip-vatic jika kolom sudah ada di CSV."
        )

    # Build lookup epoch → price
    unique_epochs = sorted(valid_epochs.unique())
    epoch_to_strike: dict[int, float] = {}
    epoch_to_resolution: dict[int, float] = {}

    for epoch in unique_epochs:
        s = get_price_at_or_after(ticks, epoch)
        r = get_price_at_or_after(ticks, epoch + 300)
        if s:
            epoch_to_strike[epoch] = round(s, 5)
        if r:
            epoch_to_resolution[epoch] = round(r, 5)

    covered = sum(1 for e in unique_epochs if e in epoch_to_strike and e in epoch_to_resolution)
    print(f"Epochs lengkap (strike + resolution): {covered}/{len(unique_epochs)}")

    # Vectorized apply via map
    df["strike_price"]     = df["_epoch"].map(epoch_to_strike).combine_first(df["strike_price"])
    df["resolution_price"] = df["_epoch"].map(epoch_to_resolution).combine_first(df["resolution_price"])
    df["oracle_source"]    = np.where(
        df["_epoch"].isin(epoch_to_strike),
        source_label,
        df["oracle_source"],
    )

    df = df.drop(columns=["_epoch"])
    print(f"  strike_price updated    : {covered} epochs")
    print(f"  oracle_source           : {source_label}")
    if source_label == "BINANCE_FALLBACK":
        print(
            "  ⚠️  PERINGATAN: Data dari Binance, bukan Chainlink oracle.\n"
            "     Label WIN/LOSS mungkin berbeda ~0.01–0.05% vs resolusi Polymarket asli.\n"
            "     Untuk akurasi penuh, jalankan ulang dengan --proxy <QUOTAGUARD_URL>."
        )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 1 — GROUND TRUTH LABELS  (vectorized, strict comparisons)
# ──────────────────────────────────────────────────────────────────────────────

def recompute_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute actual_outcome dan signal_correct dengan logika ketat.

    Aturan WIN:
      BUY_UP   → resolution_price  >  strike_price   (strict greater-than)
      BUY_DOWN → resolution_price  <  strike_price   (strict less-than)
      Tie (==) dianggap LOSS karena buyer tidak mendapat profit.

    actual_outcome : WIN | LOSS | PENDING | N/A
    signal_correct : TRUE | FALSE | PENDING | N/A
    """
    df = df.copy()

    sp  = pd.to_numeric(df["strike_price"],    errors="coerce")
    rp  = pd.to_numeric(df["resolution_price"], errors="coerce")
    sig = df["signal_direction"].fillna("").str.upper()

    is_buy    = sig.isin(["BUY_UP", "BUY_DOWN"])
    is_buy_up = sig == "BUY_UP"
    is_buy_dn = sig == "BUY_DOWN"
    has_prices = sp.notna() & rp.notna()
    pending   = is_buy & (~has_prices)

    # Menang jika:  BUY_UP  → rp > sp  |  BUY_DOWN → rp < sp
    win_up   = is_buy_up & has_prices & (rp > sp)
    win_down = is_buy_dn & has_prices & (rp < sp)
    win      = win_up | win_down
    loss     = is_buy & has_prices & ~win

    # actual_outcome
    df["actual_outcome"] = np.where(
        ~is_buy,  "N/A",
        np.where(pending, "PENDING",
        np.where(win,     "WIN",
                          "LOSS"))
    )

    # signal_correct
    df["signal_correct"] = np.where(
        ~is_buy,  "N/A",
        np.where(pending, "PENDING",
        np.where(win,     "TRUE",
                          "FALSE"))
    )

    # ── Stats ──
    total_buy = is_buy.sum()
    resolved  = (win | loss).sum()
    wins      = win.sum()
    print(f"\n[Layer 1] Ground Truth Labels")
    print(f"  BUY rows         : {total_buy}")
    print(f"  Resolved         : {resolved}")
    print(f"  PENDING          : {pending.sum()}")
    print(f"  WIN              : {wins}")
    print(f"  LOSS             : {loss.sum()}")
    if resolved > 0:
        print(f"  Win Rate         : {wins/resolved*100:.2f}%")
    print(f"  actual_outcome   : {df['actual_outcome'].value_counts().to_dict()}")
    print(f"  signal_correct   : {df['signal_correct'].value_counts().to_dict()}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 2 — THEORETICAL PnL  (vectorized)
# ──────────────────────────────────────────────────────────────────────────────

def recompute_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute theoretical_pnl berdasarkan signal_correct yang sudah benar.

      WIN  → (1 / entry_odds) - 1   (misal odds=0.40 → PnL = +1.50 = +150%)
      LOSS → -1.0                   (kehilangan seluruh posisi)
      N/A / PENDING → 0.0
    """
    df = df.copy()
    sc      = df["signal_correct"].fillna("N/A").str.upper()
    odds    = pd.to_numeric(df["entry_odds"], errors="coerce")
    is_buy  = df["signal_direction"].fillna("").str.upper().isin(["BUY_UP", "BUY_DOWN"])

    # Win PnL: guard against odds=0 atau NaN
    win_pnl = np.where(
        (odds > 0) & odds.notna(),
        (1.0 / odds.where(odds > 0, np.nan)) - 1.0,
        np.nan,
    )

    df["theoretical_pnl"] = np.where(
        ~is_buy,           0.0,
        np.where(sc == "TRUE",  win_pnl,
        np.where(sc == "FALSE", -1.0,
                                 0.0))
    )

    resolved_pnl = df.loc[sc.isin(["TRUE", "FALSE"]), "theoretical_pnl"]
    print(f"\n[Layer 2] Theoretical PnL")
    print(f"  Rows updated     : {is_buy.sum()}")
    print(f"  Mean PnL (BUY)   : {resolved_pnl.mean():.4f}")
    print(f"  Total PnL (paper): {resolved_pnl.sum():.4f}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 3 — STRIKE DISTANCE PCT  (vectorized, tanda sesuai arah signal)
# ──────────────────────────────────────────────────────────────────────────────

def recompute_strike_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute strike_distance_pct setelah strike_price diupdate retrofix.

    Formula dasar (tidak tergantung arah):
        (chainlink_price - strike_price) / strike_price * 100

    Interpretasi per arah:
      BUY_UP  : nilai positif = chainlink sudah di atas strike → sinyal lebih kuat
      BUY_DOWN: nilai negatif = chainlink sudah di bawah strike → sinyal lebih kuat
    Kolom ini adalah ukuran spasial murni, bukan label kemenangan.

    Rows dengan chainlink_price atau strike_price null → NaN (biarkan XGBoost
    menanganinya lewat imputer atau tree split natural).
    """
    df = df.copy()
    cl = pd.to_numeric(df["chainlink_price"], errors="coerce")
    sp = pd.to_numeric(df["strike_price"],    errors="coerce")

    valid = cl.notna() & sp.notna() & (sp != 0)
    new_dist = np.where(
        valid,
        (cl - sp) / sp * 100,
        np.nan,
    )
    df["strike_distance_pct"] = np.round(new_dist, 6)

    updated = int(valid.sum())
    null_cl = int(cl.isna().sum())
    null_sp = int(sp.isna().sum())
    print(f"\n[Layer 3] strike_distance_pct")
    print(f"  Rows recomputed  : {updated}")
    print(f"  Skipped (null CL): {null_cl}")
    print(f"  Skipped (null SP): {null_sp}")
    if updated > 0:
        sample = pd.Series(new_dist[valid])
        print(f"  Range            : [{sample.min():.4f}%, {sample.max():.4f}%]")
        print(f"  Mean / Std       : {sample.mean():.4f}% / {sample.std():.4f}%")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# ML TRAINING FILTER
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "confidence_score",
    "obi_value",
    "tfm_value",
    "rv_value",
    "vol_percentile",
    "depth_ratio",
    "strike_distance_pct",
    "contest_urgency",
    "odds_delta_60s",
    "btc_return_1m",
    "spread_pct",
    "ttr_seconds",
]

CRITICAL_FEATURES = ["obi_value", "tfm_value", "confidence_score", "rv_value"]


def generate_ml_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Filter dan enriched dataset menjadi ML training-ready.

    Filter rules:
      - signal_direction IN (BUY_UP, BUY_DOWN)
      - signal_correct   IN (TRUE, FALSE)         ← sudah WIN/LOSS bersih
      - entry_odds_source == CLOB_LIVE
      - actual_outcome   IN (WIN, LOSS)
      - retrofix_status  TIDAK dimulai dengan ERROR (jika kolom ada)
    """
    print(f"\n{'='*60}")
    print("GENERATING ML TRAINING DATASET")
    print(f"{'='*60}")

    mask = (
        df["signal_direction"].isin(["BUY_UP", "BUY_DOWN"]) &
        df["signal_correct"].isin(["TRUE", "FALSE"]) &
        (df["entry_odds_source"] == "CLOB_LIVE") &
        df["actual_outcome"].isin(["WIN", "LOSS"])
    )

    if "retrofix_status" in df.columns:
        mask &= ~df["retrofix_status"].str.startswith("ERROR", na=False)

    df_ml = df[mask].copy()
    print(f"Rows setelah filter : {len(df_ml)}  (dari {len(df)} total)")

    # ── Derived features ──
    df_ml["obi_value"] = pd.to_numeric(df_ml["obi_value"], errors="coerce")
    df_ml["tfm_value"] = pd.to_numeric(df_ml["tfm_value"], errors="coerce")
    df_ml["obi_tfm_product"]   = df_ml["obi_value"] * df_ml["tfm_value"]
    df_ml["obi_tfm_alignment"] = (df_ml["obi_tfm_product"] > 0).astype(int)

    # ── Label biner untuk XGBoost ──
    df_ml["label"] = (df_ml["signal_correct"] == "TRUE").astype(int)

    # ── Feature audit ──
    ml_features = FEATURE_COLUMNS + ["obi_tfm_product", "obi_tfm_alignment"]
    available = [c for c in ml_features if c in df_ml.columns]
    missing   = [c for c in ml_features if c not in df_ml.columns]

    if missing:
        print(f"\n  ⚠️  Kolom fitur tidak ditemukan: {missing}")

    print(f"\nNull rate per fitur:")
    for col in available:
        null_pct = df_ml[col].isna().mean() * 100
        flag = "  ⚠️  >10% NULL" if null_pct > 10 else ""
        print(f"  {col:<28}: {null_pct:5.1f}%{flag}")

    # Drop rows dengan null di fitur kritis
    crit_avail = [c for c in CRITICAL_FEATURES if c in df_ml.columns]
    if crit_avail:
        before = len(df_ml)
        df_ml  = df_ml.dropna(subset=crit_avail)
        dropped = before - len(df_ml)
        if dropped:
            print(f"\n  Dropped {dropped} rows karena null di fitur kritis: {crit_avail}")

    # Class balance
    pos = df_ml["label"].mean()
    print(f"\nClass balance   : {pos*100:.1f}% WIN  /  {(1-pos)*100:.1f}% LOSS")
    print(f"Total ML rows   : {len(df_ml)}")
    print(f"Features tersedia: {len(available)}/{len(ml_features)}")
    print(f"Label distribution: {df_ml['label'].value_counts().to_dict()}")

    df_ml.to_csv(output_path, index=False)
    print(f"\nSaved ML dataset → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION REPORT  (post-processing sanity check)
# ──────────────────────────────────────────────────────────────────────────────

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Jalankan serangkaian sanity check dan cetak warning jika ada inkonsistensi.
    Ini TIDAK memodifikasi df, hanya melaporkan.
    """
    print(f"\n{'='*60}")
    print("VALIDATION REPORT")
    print(f"{'='*60}")

    errors = []

    # 1. WIN tapi resolution <= strike untuk BUY_UP
    buy_up_win = df[(df["signal_direction"] == "BUY_UP") & (df["actual_outcome"] == "WIN")]
    bad_up = buy_up_win[pd.to_numeric(buy_up_win["resolution_price"], errors="coerce") <=
                        pd.to_numeric(buy_up_win["strike_price"],     errors="coerce")]
    if len(bad_up):
        errors.append(f"  ❌  {len(bad_up)} BUY_UP WIN rows dengan resolution_price <= strike_price")

    # 2. WIN tapi resolution >= strike untuk BUY_DOWN
    buy_dn_win = df[(df["signal_direction"] == "BUY_DOWN") & (df["actual_outcome"] == "WIN")]
    bad_dn = buy_dn_win[pd.to_numeric(buy_dn_win["resolution_price"], errors="coerce") >=
                        pd.to_numeric(buy_dn_win["strike_price"],     errors="coerce")]
    if len(bad_dn):
        errors.append(f"  ❌  {len(bad_dn)} BUY_DOWN WIN rows dengan resolution_price >= strike_price")

    # 3. signal_correct != actual_outcome mapping
    buy_rows = df[df["signal_direction"].isin(["BUY_UP", "BUY_DOWN"])]
    mismatch = buy_rows[
        ((buy_rows["signal_correct"] == "TRUE")  & (buy_rows["actual_outcome"] != "WIN")) |
        ((buy_rows["signal_correct"] == "FALSE") & (buy_rows["actual_outcome"] != "LOSS"))
    ]
    if len(mismatch):
        errors.append(f"  ❌  {len(mismatch)} rows dengan mismatch signal_correct vs actual_outcome")

    # 4. theoretical_pnl negatif tapi signal_correct = TRUE
    bad_pnl = df[(df["signal_correct"] == "TRUE") & (df["theoretical_pnl"] < 0)]
    if len(bad_pnl):
        errors.append(f"  ❌  {len(bad_pnl)} WIN rows dengan theoretical_pnl negatif")

    # 5. Null resolution_price pada BUY rows non-PENDING
    buy_resolved = df[
        df["signal_direction"].isin(["BUY_UP", "BUY_DOWN"]) &
        ~df["actual_outcome"].isin(["PENDING", "N/A"])
    ]
    null_res = buy_resolved["resolution_price"].isna().sum()
    if null_res:
        errors.append(f"  ⚠️   {null_res} resolved BUY rows dengan resolution_price null")

    if errors:
        print("Issues ditemukan:")
        for e in errors:
            print(e)
    else:
        print("  ✅  Semua validasi PASSED. Dataset siap untuk XGBoost.")

    # Summary
    print(f"\nDataset summary:")
    print(f"  Total rows       : {len(df)}")
    print(f"  actual_outcome   : {df['actual_outcome'].value_counts().to_dict()}")
    print(f"  signal_correct   : {df['signal_correct'].value_counts().to_dict()}")
    pnl_buy = df[df["signal_direction"].isin(["BUY_UP", "BUY_DOWN"])]["theoretical_pnl"]
    print(f"  PnL mean (BUY)   : {pnl_buy.mean():.4f}")
    print(f"  PnL total (paper): {pnl_buy.sum():.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def recompute_labels(
    input_path: str,
    output_path: str,
    ml_output_path: Optional[str] = None,
    skip_vatic: bool = False,
    proxy_url: Optional[str] = None,
) -> None:
    print(f"\n{'='*60}")
    print("RECOMPUTE LABELS  v2  (vectorized · strict · WIN/LOSS)")
    print(f"{'='*60}")
    print(f"Input    : {input_path}")
    print(f"Output   : {output_path}")
    if ml_output_path:
        print(f"ML output: {ml_output_path}")
    vatic_status = "SKIP" if skip_vatic else (f"FETCH via proxy {proxy_url}" if proxy_url else "FETCH (direct)")
    print(f"Vatic    : {vatic_status}")

    # Load
    df = pd.read_csv(input_path)
    print(f"\nLoaded {len(df):,} rows  ×  {len(df.columns)} kolom")

    # ── Kolom wajib: buat jika tidak ada, jangan langsung crash ──
    REQUIRED_COLS = {
        "resolution_price" : np.nan,
        "strike_price"     : np.nan,
        "actual_outcome"   : "PENDING",
        "signal_correct"   : "N/A",
        "theoretical_pnl"  : 0.0,
        "strike_distance_pct": np.nan,
        "entry_odds"       : np.nan,
        "entry_odds_source": "UNKNOWN",
        "oracle_source"    : "UNKNOWN",
        "signal_direction" : "ABSTAIN",
        "chainlink_price"  : np.nan,
    }
    missing_cols = []
    for col, default in REQUIRED_COLS.items():
        if col not in df.columns:
            df[col] = default
            missing_cols.append(col)

    if missing_cols:
        print(f"\n⚠️  Kolom tidak ditemukan di CSV, dibuat dengan nilai default:")
        for col in missing_cols:
            print(f"   + {col} = {REQUIRED_COLS[col]!r}")
        if "resolution_price" in missing_cols:
            print(
                "\n   INFO: resolution_price tidak ada → semua BUY rows akan berstatus PENDING.\n"
                "   Gunakan tanpa --skip-vatic agar Vatic fetch resolution_price otomatis,\n"
                "   atau tambahkan kolom resolution_price ke CSV sebelum menjalankan script ini."
            )

    # Layer 0 — Vatic enrichment (opsional)
    if not skip_vatic:
        df = enrich_from_vatic(df, proxy_url=proxy_url)

    # Layer 1 — Ground truth
    df = recompute_ground_truth(df)

    # Layer 2 — PnL
    df = recompute_pnl(df)

    # Layer 3 — strike_distance_pct
    df = recompute_strike_distance(df)

    # Layer 4 — spread_filter_passed: TIDAK diubah (OBI historis)
    print(f"\n[Layer 4] spread_filter_passed → tidak diubah (OBI historis, bukan fungsi strike)")

    # Save clean dataset
    df.to_csv(output_path, index=False)
    print(f"\n✅  Saved clean dataset → {output_path}  ({len(df):,} rows)")

    # Validation
    validate_dataset(df)

    # ML dataset
    if ml_output_path:
        generate_ml_dataset(df, ml_output_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recompute labels, PnL, dan fitur prediktif pasca retrofix strike_price.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input",      required=True,  help="CSV input (output dari retrofix_strikes.py)")
    parser.add_argument("--output",     required=True,  help="CSV output bersih")
    parser.add_argument("--ml-output",  default=None,   help="Jika diset, generate CSV siap XGBoost")
    parser.add_argument("--skip-vatic", action="store_true",
                        help="Skip Vatic fetch; gunakan strike_price & resolution_price yang sudah ada di CSV")
    parser.add_argument("--proxy", default=None,
                        help=(
                            "URL proxy HTTP/SOCKS untuk bypass IP restriction Vatic.\n"
                            "Format  : http://user:pass@host:port\n"
                            "Contoh  : --proxy $QUOTAGUARD_URL\n"
                            "Gunakan ini jika menjalankan dari IP Indonesia (bukan Railway EU West)."
                        ))
    args = parser.parse_args()

    recompute_labels(
        input_path=args.input,
        output_path=args.output,
        ml_output_path=args.ml_output,
        skip_vatic=args.skip_vatic,
        proxy_url=args.proxy,
    )
    sys.exit(0)
