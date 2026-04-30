"""
ml/monitor.py
=============
Concept Drift Detection — Population Stability Index (PSI).

Pasar kripto sangat dinamis. Distribusi fitur bisa bergeser drastis
dalam hitungan jam (regime change, volatility spike, liquidity event).

Modul ini menghitung PSI antara distribusi training (baseline) dan
data inference terbaru (current) untuk mendeteksi apakah model
masih beroperasi dalam distribusi yang sama.

Interpretasi PSI:
  - PSI < 0.10  → Tidak ada perubahan signifikan
  - 0.10 ≤ PSI < 0.20  → Perubahan moderat, monitor
  - PSI ≥ 0.20  → Perubahan besar → CONCEPT DRIFT → retrain

Fokus utama: rv_value (realized volatility) dan vol_percentile
karena keduanya paling rentan terhadap regime shift.
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

import numpy as np
import pandas as pd

from .config import DRIFT_CFG, ALL_FEATURES

logger = logging.getLogger(__name__)

# Epsilon kecil untuk menghindari log(0) dan division-by-zero
_PSI_EPS: float = 1e-6

# Fitur prioritas yang dimonitor ketat
PRIORITY_FEATURES: List[str] = ["rv_value", "vol_percentile"]


# ---------------------------------------------------------------------------
# PSI Core Calculation
# ---------------------------------------------------------------------------

def _calculate_psi_single(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Hitung PSI untuk satu fitur.

    PSI = Σ (P_actual_i - P_expected_i) × ln(P_actual_i / P_expected_i)

    dimana P_i adalah proporsi observasi di bin ke-i.

    Args:
        expected: Distribusi training (baseline).
        actual:   Distribusi inference terbaru.
        n_bins:   Jumlah bin untuk diskretisasi.

    Returns:
        Nilai PSI (float).
    """
    # Hapus NaN
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        logger.warning("PSI: salah satu array kosong. Return 0.0.")
        return 0.0

    # Buat bin boundaries dari distribusi expected (training)
    # Menggunakan percentile agar bin tahan terhadap outlier
    breakpoints = np.percentile(
        expected,
        np.linspace(0, 100, n_bins + 1),
    )
    # Pastikan breakpoints unik (bisa terjadi jika banyak nilai identik)
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        # Semua nilai sama — tidak ada variasi
        return 0.0

    # Hitung proporsi per bin
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_pct = expected_counts / max(len(expected), 1)
    actual_pct = actual_counts / max(len(actual), 1)

    # Tambahkan epsilon untuk menghindari log(0) dan 0/0
    expected_pct = np.clip(expected_pct, _PSI_EPS, None)
    actual_pct = np.clip(actual_pct, _PSI_EPS, None)

    # PSI formula
    psi = float(np.sum(
        (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    ))

    return psi


def calculate_psi(
    df_baseline: pd.DataFrame,
    df_current: pd.DataFrame,
    features: Optional[List[str]] = None,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Hitung PSI untuk semua fitur yang diminta.

    Args:
        df_baseline: DataFrame training (baseline distribusi).
        df_current:  DataFrame inference terbaru.
        features:    List fitur yang dihitung. Default: ALL_FEATURES.
        n_bins:      Jumlah bin diskretisasi.

    Returns:
        Dict mapping feature_name → PSI value.
    """
    if features is None:
        features = [f for f in ALL_FEATURES if f in df_baseline.columns
                     and f in df_current.columns]

    psi_results: Dict[str, float] = {}

    for feat in features:
        if feat not in df_baseline.columns:
            logger.warning("PSI: fitur '%s' tidak ada di baseline. Skip.", feat)
            continue
        if feat not in df_current.columns:
            logger.warning("PSI: fitur '%s' tidak ada di current. Skip.", feat)
            continue

        psi_val = _calculate_psi_single(
            expected=df_baseline[feat].values.astype(np.float64),
            actual=df_current[feat].values.astype(np.float64),
            n_bins=n_bins,
        )
        psi_results[feat] = round(psi_val, 6)

    return psi_results


# ---------------------------------------------------------------------------
# Drift Alert System
# ---------------------------------------------------------------------------

def check_drift(
    psi_results: Dict[str, float],
    threshold: Optional[float] = None,
    priority_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluasi hasil PSI dan generate alert.

    Logika:
      - Jika PSI fitur PRIORITAS (rv_value, vol_percentile) > threshold
        → CRITICAL_WARNING
      - Jika PSI fitur lain > threshold → WARNING
      - Semua di bawah threshold → OK

    Args:
        psi_results:       Dict feature → PSI dari calculate_psi().
        threshold:         Batas PSI. Default dari DRIFT_CFG.psi_threshold.
        priority_features: Fitur yang memicu CRITICAL. Default: rv_value, vol_percentile.

    Returns:
        Dict berisi status, alerts, dan rekomendasi.
    """
    if threshold is None:
        threshold = DRIFT_CFG.psi_threshold

    if priority_features is None:
        priority_features = PRIORITY_FEATURES

    alerts: List[Dict[str, Any]] = []
    drifted_features: List[str] = []
    critical_drift: bool = False

    for feat, psi_val in psi_results.items():
        is_priority = feat in priority_features

        if psi_val >= threshold:
            drifted_features.append(feat)
            severity = "CRITICAL" if is_priority else "WARNING"

            if is_priority:
                critical_drift = True
                logger.critical(
                    "CRITICAL_WARNING: Concept Drift Detected — "
                    "'%s' PSI=%.4f (threshold=%.2f). "
                    "Model mungkin tidak valid untuk distribusi saat ini. "
                    "RETRAIN RECOMMENDED.",
                    feat, psi_val, threshold,
                )
            else:
                logger.warning(
                    "WARNING: Drift detected pada '%s' — PSI=%.4f (threshold=%.2f).",
                    feat, psi_val, threshold,
                )

            alerts.append({
                "feature": feat,
                "psi": psi_val,
                "threshold": threshold,
                "severity": severity,
                "is_priority": is_priority,
            })

        elif psi_val >= threshold * 0.5:
            # Moderate — belum kritis, tapi perlu diperhatikan
            logger.info(
                "MONITOR: '%s' PSI=%.4f (approaching threshold=%.2f).",
                feat, psi_val, threshold,
            )
        else:
            logger.debug("OK: '%s' PSI=%.4f — no drift.", feat, psi_val)

    # --- Tentukan status keseluruhan ---
    if critical_drift:
        overall_status = "CRITICAL_DRIFT"
        recommendation = (
            "Satu atau lebih fitur prioritas (rv_value/vol_percentile) "
            "mengalami drift signifikan. RETRAIN model segera atau "
            "downgrade ke deterministic rules."
        )
    elif drifted_features:
        overall_status = "MODERATE_DRIFT"
        recommendation = (
            f"Fitur berikut mengalami drift: {drifted_features}. "
            "Monitor lebih ketat dan pertimbangkan retrain dalam 24 jam."
        )
    else:
        overall_status = "STABLE"
        recommendation = "Semua fitur dalam distribusi normal. Model valid."

    result = {
        "status": overall_status,
        "critical_drift": critical_drift,
        "drifted_features": drifted_features,
        "alerts": alerts,
        "psi_values": psi_results,
        "threshold": threshold,
        "recommendation": recommendation,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    logger.info(
        "Drift check selesai: status=%s | %d/%d fitur drift",
        overall_status, len(drifted_features), len(psi_results),
    )

    return result


# ---------------------------------------------------------------------------
# PSI Baseline Management
# ---------------------------------------------------------------------------

class PSIBaselineManager:
    """
    Kelola baseline distribusi training untuk monitoring drift berkelanjutan.

    Baseline disimpan sebagai file .json berisi statistik per fitur
    (percentile boundaries + bin counts) dari training set.
    """

    def __init__(
        self,
        baseline_dir: str | Path,
        n_bins: int = 10,
    ) -> None:
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.n_bins = n_bins
        self._baseline_path = self.baseline_dir / "psi_baseline.json"

    def save_baseline(
        self,
        df_train: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Path:
        """
        Simpan distribusi training sebagai baseline.

        Menyimpan raw values per fitur (dicompact sebagai percentiles)
        agar bisa digunakan untuk menghitung PSI nantinya.
        """
        if features is None:
            features = [f for f in ALL_FEATURES if f in df_train.columns]

        baseline_data: Dict[str, Any] = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_samples": len(df_train),
            "n_bins": self.n_bins,
            "features": {},
        }

        for feat in features:
            vals = df_train[feat].dropna().values.astype(float)
            if len(vals) == 0:
                continue

            # Simpan breakpoints + expected proportions
            breakpoints = np.percentile(
                vals, np.linspace(0, 100, self.n_bins + 1)
            ).tolist()
            counts = np.histogram(vals, bins=np.unique(breakpoints))[0]
            proportions = (counts / max(len(vals), 1)).tolist()

            baseline_data["features"][feat] = {
                "breakpoints": breakpoints,
                "proportions": proportions,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "p25": float(np.percentile(vals, 25)),
                "p50": float(np.percentile(vals, 50)),
                "p75": float(np.percentile(vals, 75)),
            }

        with open(self._baseline_path, "w") as f:
            json.dump(baseline_data, f, indent=2)

        logger.info(
            "PSI baseline disimpan: %s (%d fitur, %d sampel)",
            self._baseline_path, len(baseline_data["features"]),
            baseline_data["n_samples"],
        )
        return self._baseline_path

    def load_baseline(self) -> Dict[str, Any]:
        """Load baseline dari disk."""
        if not self._baseline_path.exists():
            raise FileNotFoundError(
                f"PSI baseline tidak ditemukan: {self._baseline_path}. "
                "Jalankan save_baseline() terlebih dahulu."
            )

        with open(self._baseline_path, "r") as f:
            baseline = json.load(f)

        logger.info(
            "PSI baseline dimuat: %s (created=%s, %d fitur)",
            self._baseline_path,
            baseline.get("created_at", "unknown"),
            len(baseline.get("features", {})),
        )
        return baseline

    def check_current(
        self,
        df_current: pd.DataFrame,
        df_baseline: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Hitung PSI antara baseline dan data terbaru, lalu check drift.

        Convenience method yang memanggil calculate_psi() + check_drift().
        """
        psi_results = calculate_psi(
            df_baseline=df_baseline,
            df_current=df_current,
            features=features,
            n_bins=self.n_bins,
        )
        return check_drift(psi_results)


# ---------------------------------------------------------------------------
# Rolling Brier Monitoring
# ---------------------------------------------------------------------------

def rolling_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    window: Optional[int] = None,
) -> Tuple[np.ndarray, bool]:
    """
    Hitung rolling Brier score untuk mendeteksi degradasi model.

    Jika rolling Brier melewati degradation threshold, log warning.

    Args:
        y_true:  Label biner.
        y_prob:  Probabilitas terkalibrasi.
        window:  Ukuran window. Default dari DRIFT_CFG.

    Returns:
        Tuple (rolling_brier_array, is_degraded).
    """
    if window is None:
        window = DRIFT_CFG.brier_rolling_window

    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    n = len(y_true)
    if n < window:
        logger.warning(
            "Sampel (%d) < window (%d). Rolling Brier tidak bermakna.",
            n, window,
        )
        return np.array([]), False

    # Brier per sample: (y_true - y_prob)²
    brier_per_sample = (y_true - y_prob) ** 2

    # Rolling mean
    rolling = pd.Series(brier_per_sample).rolling(window=window).mean().values

    # Cek degradasi: apakah rolling Brier terbaru > threshold?
    latest_brier = rolling[-1]
    is_degraded = bool(
        np.isfinite(latest_brier)
        and latest_brier > DRIFT_CFG.brier_degradation_threshold
    )

    if is_degraded:
        logger.warning(
            "BRIER DEGRADATION: rolling Brier=%.4f > threshold=%.2f. "
            "Model mungkin sudah tidak akurat. Pertimbangkan downgrade "
            "ke deterministic rules.",
            latest_brier, DRIFT_CFG.brier_degradation_threshold,
        )
    else:
        logger.info(
            "Rolling Brier (window=%d): latest=%.4f (threshold=%.2f) — OK",
            window,
            latest_brier if np.isfinite(latest_brier) else float("nan"),
            DRIFT_CFG.brier_degradation_threshold,
        )

    return rolling, is_degraded
