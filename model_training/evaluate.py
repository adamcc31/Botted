"""
ml/evaluate.py
==============
Model Evaluation & EV Strategy Simulation.

Modul ini mengonsumsi artefak model terkalibrasi (base_model + Platt)
dan data uji untuk menghasilkan:
  1. Reliability Diagram (Calibration Curve) vs diagonal sempurna
  2. SHAP Summary Plot — validasi fitur top (edge_vs_crowd, obi_vol_interaction)
  3. Simulasi strategi EV pada underdog bets (odds < 0.30)

Semua plot disimpan ke disk untuk audit trail.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — tidak butuh display
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
    classification_report,
)

from .config import ALL_FEATURES, SELECTED_FEATURES, EV_CFG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Reliability Diagram (Calibration Curve)
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str | Path] = None,
    title: str = "Reliability Diagram — Calibrated Model",
) -> Dict[str, Any]:
    """
    Plot dan (opsional) simpan Reliability Diagram.

    Membandingkan fraksi positif aktual vs probabilitas prediksi rata-rata
    per bin terhadap diagonal sempurna (perfectly calibrated).

    Args:
        y_true:    Label biner ground truth.
        y_prob:    Probabilitas terkalibrasi P(WIN).
        n_bins:    Jumlah bin untuk calibration curve.
        save_path: Path file output (.png). None = tidak disimpan.
        title:     Judul plot.

    Returns:
        Dict berisi fraction_of_positives, mean_predicted, ece, brier, auc.
    """
    # --- Validasi input ---
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"Panjang y_true ({len(y_true)}) != y_prob ({len(y_prob)})"
        )

    # Handle edge case: semua label sama
    unique_labels = np.unique(y_true[~np.isnan(y_true)])
    if len(unique_labels) < 2:
        logger.warning(
            "y_true hanya mengandung satu kelas (%s). "
            "Reliability diagram tidak bermakna.",
            unique_labels,
        )

    # --- Hitung calibration curve ---
    fraction_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    # --- Metrik ringkasan ---
    ece = float(np.mean(np.abs(fraction_pos - mean_pred)))
    brier = float(brier_score_loss(y_true, y_prob))

    # AUC hanya valid jika ada kedua kelas
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    try:
        logloss = float(log_loss(y_true, y_prob))
    except ValueError:
        logloss = float("nan")

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Diagonal sempurna
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--", color="#6b7280", linewidth=1.5,
        label="Perfectly Calibrated",
    )

    # Calibration curve
    ax.plot(
        mean_pred, fraction_pos,
        marker="o", markersize=6, linewidth=2.0,
        color="#3b82f6", label=f"Model (ECE={ece:.4f})",
    )

    # Histogram distribusi probabilitas (background)
    ax2 = ax.twinx()
    ax2.hist(
        y_prob, bins=50, range=(0, 1),
        alpha=0.15, color="#8b5cf6", edgecolor="none",
    )
    ax2.set_ylabel("Count", fontsize=9, color="#8b5cf6")
    ax2.tick_params(axis="y", labelcolor="#8b5cf6", labelsize=8)

    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)

    # Annotasi metrik
    textstr = (
        f"AUC:    {auc:.4f}\n"
        f"Brier:  {brier:.4f}\n"
        f"LogLoss:{logloss:.4f}\n"
        f"ECE:    {ece:.4f}\n"
        f"N:      {len(y_true)}"
    )
    props = dict(boxstyle="round,pad=0.4", facecolor="#f0f9ff", alpha=0.9)
    ax.text(
        0.98, 0.02, textstr,
        transform=ax.transAxes,
        fontsize=8, verticalalignment="bottom", horizontalalignment="right",
        bbox=props, fontfamily="monospace",
    )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Reliability diagram disimpan: %s", save_path)

    plt.close(fig)

    result = {
        "fraction_of_positives": fraction_pos.tolist(),
        "mean_predicted": mean_pred.tolist(),
        "ece": ece,
        "brier": brier,
        "auc": auc,
        "logloss": logloss,
        "n_samples": len(y_true),
    }
    logger.info(
        "Reliability metrics: AUC=%.4f | Brier=%.4f | ECE=%.4f | LogLoss=%.4f",
        auc, brier, ece, logloss,
    )
    return result


# ---------------------------------------------------------------------------
# 2. SHAP Analysis
# ---------------------------------------------------------------------------

def plot_shap_summary(
    model,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str | Path] = None,
    max_display: int = 20,
    title: str = "SHAP Feature Importance",
) -> Dict[str, float]:
    """
    SHAP Summary Plot untuk memvalidasi fitur top.

    Menggunakan TreeExplainer (optimal untuk XGBoost) untuk menghitung
    SHAP values dan menghasilkan summary beeswarm plot.

    Args:
        model:         XGBClassifier (base model, bukan CalibratedClassifierCV).
        X:             Feature matrix (np.ndarray).
        feature_names: Nama fitur. Default ALL_FEATURES.
        save_path:     Path output .png.
        max_display:   Jumlah fitur yang ditampilkan.
        title:         Judul plot.

    Returns:
        Dict mapping feature_name → mean(|SHAP|), sorted descending.
    """
    import shap

    if feature_names is None:
        feature_names = ALL_FEATURES

    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"Jumlah feature_names ({len(feature_names)}) "
            f"!= jumlah kolom X ({X.shape[1]})"
        )

    logger.info("Menghitung SHAP values untuk %d sampel...", X.shape[0])

    # TreeExplainer — optimal dan exact untuk tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # --- Mean |SHAP| per fitur ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_dict: Dict[str, float] = {}
    for fname, val in sorted(
        zip(feature_names, mean_abs_shap), key=lambda x: -x[1]
    ):
        importance_dict[fname] = round(float(val), 6)

    # Log top-5
    top5 = list(importance_dict.items())[:5]
    logger.info("SHAP Top-5 Features:")
    for rank, (fname, val) in enumerate(top5, 1):
        logger.info("  #%d: %-25s  mean|SHAP|=%.6f", rank, fname, val)

    # --- Validasi fitur kunci ---
    _validate_key_features(importance_dict)

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP summary plot disimpan: %s", save_path)

    plt.close("all")

    return importance_dict


def _validate_key_features(
    importance_dict: Dict[str, float],
    expected_top: Optional[List[str]] = None,
    top_n: int = 10,
) -> None:
    """
    Validasi bahwa fitur kunci berada di top-N SHAP.

    Jika `edge_vs_crowd` atau `obi_vol_interaction` TIDAK di top-N,
    log WARNING — ini mengindikasikan feature engineering mungkin
    perlu di-review.
    """
    if expected_top is None:
        expected_top = ["edge_vs_crowd", "obi_vol_interaction"]

    ranked = list(importance_dict.keys())[:top_n]

    for feat in expected_top:
        if feat not in ranked:
            logger.warning(
                "⚠ SHAP WARNING: '%s' TIDAK di top-%d fitur. "
                "Review feature engineering — fitur ini seharusnya dominan. "
                "Ranked: %s",
                feat, top_n, ranked,
            )
        else:
            position = ranked.index(feat) + 1
            logger.info(
                "✓ '%s' berada di posisi #%d (top-%d). OK.",
                feat, position, top_n,
            )


# ---------------------------------------------------------------------------
# 3. EV Strategy Simulation (Underdog Focus)
# ---------------------------------------------------------------------------

def calculate_ev(
    p_win: float,
    entry_odds: float,
) -> float:
    """
    Hitung Expected Value untuk satu taruhan.

    Formula:
        EV = P(WIN) × ((1 / entry_odds) - 1) - (1 - P(WIN))

    Interpretasi:
        - EV > 0 → taruhan menguntungkan secara statistik
        - EV > ev_threshold → layak dieksekusi

    Edge cases:
        - entry_odds == 0  → return -inf (odds invalid)
        - NaN input        → return NaN
    """
    # Handle NaN
    if np.isnan(p_win) or np.isnan(entry_odds):
        return float("nan")

    # Handle division by zero
    if entry_odds <= 0.0:
        return float("-inf")

    # Clip p_win ke [0, 1]
    p_win = float(np.clip(p_win, 0.0, 1.0))

    payout_multiplier = (1.0 / entry_odds) - 1.0
    ev = p_win * payout_multiplier - (1.0 - p_win)
    return float(ev)


def simulate_ev_strategy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    entry_odds: np.ndarray,
    underdog_cutoff: float = 0.30,
    ev_threshold: float = 0.0,
    bet_size: float = 1.0,
) -> Dict[str, Any]:
    """
    Simulasi strategi EV pada underdog bets.

    Alur:
        1. Filter hanya taruhan dengan entry_odds < underdog_cutoff
        2. Hitung EV untuk setiap taruhan
        3. Filter hanya taruhan dengan EV > ev_threshold
        4. Hitung statistik simulasi: Win Rate, Avg EV, Total PnL

    Args:
        y_true:           Label biner ground truth (1=win, 0=lose).
        y_prob:           Probabilitas terkalibrasi P(WIN).
        entry_odds:       Odds masuk dari CLOB.
        underdog_cutoff:  Batas atas odds untuk underdog (default 0.30).
        ev_threshold:     EV minimum untuk eksekusi (default 0.0).
        bet_size:         Ukuran taruhan flat per sinyal (unit).

    Returns:
        Dict berisi semua statistik simulasi.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    entry_odds = np.asarray(entry_odds, dtype=np.float64)

    n_total = len(y_true)
    if not (len(y_prob) == len(entry_odds) == n_total):
        raise ValueError(
            "Panjang y_true, y_prob, dan entry_odds harus sama. "
            f"Got {n_total}, {len(y_prob)}, {len(entry_odds)}."
        )

    logger.info(
        "═══════════════════════════════════════════════════════════"
    )
    logger.info("EV STRATEGY SIMULATION — Underdog Focus")
    logger.info(
        "═══════════════════════════════════════════════════════════"
    )
    logger.info("Total samples: %d", n_total)

    # --- Step 1: Filter underdog ---
    underdog_mask = entry_odds < underdog_cutoff
    n_underdog = int(underdog_mask.sum())
    logger.info(
        "Underdog filter (odds < %.2f): %d / %d (%.1f%%)",
        underdog_cutoff, n_underdog, n_total,
        100.0 * n_underdog / max(n_total, 1),
    )

    if n_underdog == 0:
        logger.warning("Tidak ada taruhan underdog. Simulasi berhenti.")
        return _empty_sim_result(n_total, underdog_cutoff, ev_threshold)

    # --- Step 2: Hitung EV per taruhan underdog ---
    ev_values = np.array([
        calculate_ev(p, o)
        for p, o in zip(y_prob[underdog_mask], entry_odds[underdog_mask])
    ])

    # --- Step 3: Filter EV > threshold ---
    valid_ev_mask = np.isfinite(ev_values)
    positive_ev_mask = valid_ev_mask & (ev_values > ev_threshold)
    n_positive_ev = int(positive_ev_mask.sum())

    logger.info(
        "EV > %.2f filter: %d / %d underdog bets (%.1f%%)",
        ev_threshold, n_positive_ev, n_underdog,
        100.0 * n_positive_ev / max(n_underdog, 1),
    )

    if n_positive_ev == 0:
        logger.warning("Tidak ada taruhan dengan EV positif. Simulasi berhenti.")
        return _empty_sim_result(n_total, underdog_cutoff, ev_threshold)

    # --- Step 4: Statistik taruhan yang dieksekusi ---
    exec_y_true = y_true[underdog_mask][positive_ev_mask]
    exec_y_prob = y_prob[underdog_mask][positive_ev_mask]
    exec_odds = entry_odds[underdog_mask][positive_ev_mask]
    exec_ev = ev_values[positive_ev_mask]

    win_rate = float(exec_y_true.mean())
    avg_ev = float(exec_ev.mean())
    median_ev = float(np.median(exec_ev))
    avg_odds = float(exec_odds.mean())
    avg_p_win = float(exec_y_prob.mean())

    # --- PnL Calculation ---
    # Win: profit = bet_size × ((1/odds) - 1)
    # Lose: loss = -bet_size
    pnl_per_bet = np.where(
        exec_y_true == 1,
        bet_size * ((1.0 / np.clip(exec_odds, 1e-6, None)) - 1.0),  # profit on win
        -bet_size,  # loss on lose
    )
    total_pnl = float(np.sum(pnl_per_bet))
    avg_pnl_per_bet = float(np.mean(pnl_per_bet))
    max_drawdown = float(np.min(np.cumsum(pnl_per_bet)))
    peak_pnl = float(np.max(np.cumsum(pnl_per_bet)))

    # Sharpe-like ratio (if enough trades)
    if len(pnl_per_bet) > 1 and np.std(pnl_per_bet) > 0:
        sharpe_ratio = float(np.mean(pnl_per_bet) / np.std(pnl_per_bet))
    else:
        sharpe_ratio = float("nan")

    # --- Print results ---
    logger.info("───────────────────────────────────────────────────────────")
    logger.info("SIMULATION RESULTS:")
    logger.info("───────────────────────────────────────────────────────────")
    logger.info("  Bets Executed:     %d / %d total", n_positive_ev, n_total)
    logger.info("  Win Rate:          %.2f%% (%d / %d)",
                win_rate * 100, int(exec_y_true.sum()), n_positive_ev)
    logger.info("  Average EV:        %.4f", avg_ev)
    logger.info("  Median EV:         %.4f", median_ev)
    logger.info("  Average Odds:      %.4f", avg_odds)
    logger.info("  Average P(WIN):    %.4f", avg_p_win)
    logger.info("  Total PnL:         %.4f units", total_pnl)
    logger.info("  Avg PnL/Bet:       %.4f units", avg_pnl_per_bet)
    logger.info("  Max Drawdown:      %.4f units", max_drawdown)
    logger.info("  Peak PnL:          %.4f units", peak_pnl)
    logger.info("  Sharpe Ratio:      %.4f", sharpe_ratio)
    logger.info("───────────────────────────────────────────────────────────")

    result = {
        "n_total": n_total,
        "n_underdog": n_underdog,
        "n_executed": n_positive_ev,
        "underdog_cutoff": underdog_cutoff,
        "ev_threshold": ev_threshold,
        "win_rate": round(win_rate, 4),
        "avg_ev": round(avg_ev, 4),
        "median_ev": round(median_ev, 4),
        "avg_odds": round(avg_odds, 4),
        "avg_p_win": round(avg_p_win, 4),
        "total_pnl": round(total_pnl, 4),
        "avg_pnl_per_bet": round(avg_pnl_per_bet, 4),
        "max_drawdown": round(max_drawdown, 4),
        "peak_pnl": round(peak_pnl, 4),
        "sharpe_ratio": round(sharpe_ratio, 4) if np.isfinite(sharpe_ratio) else None,
        "pnl_curve": np.cumsum(pnl_per_bet).tolist(),
    }

    return result


def _empty_sim_result(
    n_total: int,
    underdog_cutoff: float,
    ev_threshold: float,
) -> Dict[str, Any]:
    """Return empty simulation result ketika tidak ada taruhan yang memenuhi syarat."""
    return {
        "n_total": n_total,
        "n_underdog": 0,
        "n_executed": 0,
        "underdog_cutoff": underdog_cutoff,
        "ev_threshold": ev_threshold,
        "win_rate": 0.0,
        "avg_ev": 0.0,
        "median_ev": 0.0,
        "avg_odds": 0.0,
        "avg_p_win": 0.0,
        "total_pnl": 0.0,
        "avg_pnl_per_bet": 0.0,
        "max_drawdown": 0.0,
        "peak_pnl": 0.0,
        "sharpe_ratio": None,
        "pnl_curve": [],
    }


# ---------------------------------------------------------------------------
# Convenience: Full Evaluation Suite
# ---------------------------------------------------------------------------

def run_full_evaluation(
    base_model,
    platt_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    entry_odds: np.ndarray,
    feature_names: Optional[List[str]] = None,
    output_dir: Optional[str | Path] = None,
    underdog_cutoff: float = 0.30,
) -> Dict[str, Any]:
    """
    Jalankan semua evaluasi sekaligus:
      1. Reliability Diagram
      2. SHAP Analysis
      3. EV Strategy Simulation

    Args:
        base_model:      XGBClassifier (model dasar).
        platt_model:     LogisticRegression (Platt calibrator).
        X_test:          Feature matrix test set.
        y_test:          Label test set.
        entry_odds:      Odds masuk per sample.
        feature_names:   Nama fitur.
        output_dir:      Direktori output untuk artefak.
        underdog_cutoff: Batas odds underdog.

    Returns:
        Dict berisi semua hasil evaluasi.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if feature_names is None:
        feature_names = SELECTED_FEATURES

    logger.info("=" * 60)
    logger.info("FULL MODEL EVALUATION SUITE")
    logger.info("=" * 60)

    # --- Generate calibrated probabilities ---
    raw_proba = base_model.predict_proba(X_test)[:, 1]
    # Support both IsotonicRegression (.predict) and LogisticRegression (.predict_proba)
    if hasattr(platt_model, 'predict_proba'):
        y_prob = platt_model.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
    else:
        y_prob = platt_model.predict(raw_proba)
        y_prob = np.clip(y_prob, 0.001, 0.999)

    results: Dict[str, Any] = {}

    # --- 1. Reliability Diagram ---
    logger.info("\n[1/3] Reliability Diagram")
    calib_save = (output_dir / "reliability_diagram.png") if output_dir else None
    results["calibration"] = plot_reliability_diagram(
        y_true=y_test,
        y_prob=y_prob,
        save_path=calib_save,
    )

    # --- 2. SHAP Analysis ---
    logger.info("\n[2/3] SHAP Analysis")
    shap_save = (output_dir / "shap_summary.png") if output_dir else None
    results["shap_importance"] = plot_shap_summary(
        model=base_model,
        X=X_test,
        feature_names=feature_names,
        save_path=shap_save,
    )

    # --- 3. EV Strategy Simulation ---
    logger.info("\n[3/3] EV Strategy Simulation")
    results["ev_simulation"] = simulate_ev_strategy(
        y_true=y_test,
        y_prob=y_prob,
        entry_odds=entry_odds,
        underdog_cutoff=underdog_cutoff,
        ev_threshold=EV_CFG.ev_threshold,
    )

    # --- Classification Report (bonus) ---
    threshold = 0.5
    y_pred_binary = (y_prob >= threshold).astype(int)
    report = classification_report(
        y_test, y_pred_binary, output_dict=True, zero_division=0
    )
    results["classification_report"] = report
    logger.info(
        "\nClassification Report (threshold=%.2f):\n%s",
        threshold,
        classification_report(y_test, y_pred_binary, zero_division=0),
    )

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)

    return results
