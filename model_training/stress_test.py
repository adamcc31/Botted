"""
ml/stress_test.py
=================
Dual Execution Stress Test — menjalankan pipeline pada dua dataset
dan menghasilkan laporan perbandingan metrik side-by-side.

Usage:
    python -m model_training.stress_test
"""

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_file: Optional[str] = None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )

logger = logging.getLogger("stress_test")


# ---------------------------------------------------------------------------
# Preprocessing: Prepare dataset_clean.csv for pipeline consumption
# ---------------------------------------------------------------------------

def preprocess_clean_dataset(path: Path) -> pd.DataFrame:
    """
    dataset_clean.csv lacks 'label', 'obi_tfm_product', 'obi_tfm_alignment'
    and uses string-based signal_correct. Fix all issues here.
    """
    logger.info("Preprocessing dataset_clean.csv ...")
    df = pd.read_csv(path)
    n_before = len(df)

    # 1. Create 'label' from signal_correct (string -> binary)
    if "label" not in df.columns:
        if "signal_correct" in df.columns:
            # Drop PENDING rows
            pending_mask = df["signal_correct"].astype(str).str.upper() == "PENDING"
            n_pending = pending_mask.sum()
            df = df[~pending_mask].copy()
            logger.info("  Dropped %d PENDING rows", n_pending)

            # Map TRUE->1, FALSE->0
            df["label"] = df["signal_correct"].astype(str).str.upper().map(
                {"TRUE": 1, "FALSE": 0}
            )
            # Drop rows that didn't map
            unmapped = df["label"].isna().sum()
            if unmapped > 0:
                logger.warning("  Dropping %d rows with unmappable signal_correct", unmapped)
                df = df.dropna(subset=["label"]).copy()
            df["label"] = df["label"].astype(int)
            logger.info("  Created 'label' column: %s", df["label"].value_counts().to_dict())
        else:
            raise ValueError("Cannot create 'label': no signal_correct column")

    # 2. Create missing RAW_FEATURES
    if "obi_tfm_product" not in df.columns:
        if "obi_value" in df.columns and "tfm_value" in df.columns:
            df["obi_tfm_product"] = df["obi_value"] * df["tfm_value"]
            logger.info("  Created obi_tfm_product from obi_value × tfm_value")
        else:
            df["obi_tfm_product"] = 0.0
            logger.warning("  obi_tfm_product set to 0.0 (source cols missing)")

    if "obi_tfm_alignment" not in df.columns:
        if "obi_value" in df.columns and "tfm_value" in df.columns:
            df["obi_tfm_alignment"] = (
                np.sign(df["obi_value"]) == np.sign(df["tfm_value"])
            ).astype(float)
            logger.info("  Created obi_tfm_alignment from sign(obi) == sign(tfm)")
        else:
            df["obi_tfm_alignment"] = 0.0
            logger.warning("  obi_tfm_alignment set to 0.0 (source cols missing)")

    logger.info(
        "  Preprocessing done: %d → %d rows (%d dropped)",
        n_before, len(df), n_before - len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Run a single pipeline execution
# ---------------------------------------------------------------------------

def run_pipeline_single(
    data_path: Path,
    output_dir: Path,
    run_name: str,
    preprocess: bool = False,
) -> Dict[str, Any]:
    """
    Execute one pipeline run, capturing all results and any crashes.
    """
    from .pipeline import TrainingPipeline
    from .features import build_features
    from .config import ALL_FEATURES, TARGET_COL

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  RUN: %-51s║", run_name)
    logger.info("║  Data: %-50s║", str(data_path.name))
    logger.info("╚" + "═" * 58 + "╝")

    result: Dict[str, Any] = {
        "run_name": run_name,
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "success": False,
        "crash": None,
        "warnings": [],
    }

    t0 = time.time()

    try:
        # --- Load & optionally preprocess ---
        if preprocess:
            df = preprocess_clean_dataset(data_path)
            # Save preprocessed to temp CSV for pipeline consumption
            temp_csv = output_dir / "_preprocessed.csv"
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(temp_csv, index=False)
            effective_path = temp_csv
            result["n_rows_raw"] = len(pd.read_csv(data_path))
            result["n_rows_after_preprocess"] = len(df)
        else:
            effective_path = data_path
            df_raw = pd.read_csv(data_path)
            result["n_rows_raw"] = len(df_raw)
            result["n_rows_after_preprocess"] = len(df_raw)

        # --- Execute pipeline ---
        pipeline = TrainingPipeline(
            data_path=effective_path,
            output_dir=output_dir,
            data_format="csv",
            test_ratio=0.15,
            model_version=f"stress_test_{run_name}",
        )

        pipeline_result = pipeline.run_full_pipeline()
        result["success"] = True
        result["pipeline_result"] = pipeline_result

        # --- Extract detailed metrics ---
        result["n_rows_after_dedup"] = len(pipeline.df_clean) if pipeline.df_clean is not None else None
        result["n_train"] = len(pipeline.df_train) if pipeline.df_train is not None else None
        result["n_calib"] = len(pipeline.df_calib) if pipeline.df_calib is not None else None
        result["n_test"] = len(pipeline.df_test) if pipeline.df_test is not None else None
        result["n_markets_train"] = (
            pipeline.df_train["market_id"].nunique()
            if pipeline.df_train is not None and "market_id" in pipeline.df_train.columns
            else None
        )

        # CV metrics
        if pipeline.cv_result:
            result["cv_mean_auc"] = pipeline.cv_result.get("mean_auc")
            result["cv_std_auc"] = pipeline.cv_result.get("std_auc")
            result["cv_mean_brier"] = pipeline.cv_result.get("mean_brier")
            result["cv_oof_auc"] = pipeline.cv_result.get("oof_auc")
            result["cv_oof_ece_raw"] = pipeline.cv_result.get("oof_ece_raw")
            result["cv_optimal_trees"] = pipeline.cv_result.get("n_estimators_optimal")

        # Training/calibration metrics
        if pipeline.train_result:
            result["calib_auc"] = pipeline.train_result["metrics"].get("calib_auc")
            result["calib_brier"] = pipeline.train_result["metrics"].get("calib_brier")
            result["calib_ece"] = pipeline.train_result["metrics"].get("calib_ece")
            result["best_iteration"] = pipeline.train_result.get("best_iteration")

        # Quality gates
        if pipeline.gate_result:
            result["gates_all_pass"] = pipeline.gate_result.get("all_pass")
            result["gate_details"] = {
                name: {"value": c["value"], "threshold": c["threshold"], "pass": c["pass"]}
                for name, c in pipeline.gate_result.get("checks", {}).items()
            }

        # Evaluation metrics
        if pipeline.eval_result:
            cal = pipeline.eval_result.get("calibration", {})
            result["eval_auc"] = cal.get("auc")
            result["eval_brier"] = cal.get("brier")
            result["eval_ece"] = cal.get("ece")
            result["eval_logloss"] = cal.get("logloss")

            shap_imp = pipeline.eval_result.get("shap_importance", {})
            result["shap_top10"] = list(shap_imp.items())[:10] if shap_imp else []
            result["shap_edge_vs_crowd_rank"] = _get_rank(shap_imp, "edge_vs_crowd")
            result["shap_obi_vol_rank"] = _get_rank(shap_imp, "obi_vol_interaction")

            ev_sim = pipeline.eval_result.get("ev_simulation", {})
            result["ev_n_total"] = ev_sim.get("n_total")
            result["ev_n_underdog"] = ev_sim.get("n_underdog")
            result["ev_n_executed"] = ev_sim.get("n_executed")
            result["ev_win_rate"] = ev_sim.get("win_rate")
            result["ev_avg_ev"] = ev_sim.get("avg_ev")
            result["ev_total_pnl"] = ev_sim.get("total_pnl")
            result["ev_sharpe"] = ev_sim.get("sharpe_ratio")
            result["ev_max_drawdown"] = ev_sim.get("max_drawdown")

    except Exception as e:
        result["crash"] = {
            "exception": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }
        logger.error("RUN '%s' CRASHED: %s", run_name, e)
        logger.error(traceback.format_exc())

    result["elapsed_seconds"] = round(time.time() - t0, 1)

    return result


def _get_rank(importance_dict: Dict[str, float], feature: str) -> Optional[int]:
    """Get 1-indexed rank of a feature in the SHAP importance dict."""
    if not importance_dict:
        return None
    ranked = list(importance_dict.keys())
    if feature in ranked:
        return ranked.index(feature) + 1
    return None


# ---------------------------------------------------------------------------
# Comparison Report
# ---------------------------------------------------------------------------

def generate_comparison_report(
    result_a: Dict[str, Any],
    result_b: Dict[str, Any],
) -> str:
    """Generate a formatted side-by-side comparison report."""

    lines = []
    lines.append("=" * 72)
    lines.append("  STRESS TEST — DUAL EXECUTION COMPARISON REPORT")
    lines.append("=" * 72)
    lines.append("")

    # --- Crash Check ---
    lines.append("─" * 72)
    lines.append("1. CRASH CHECK (Pipeline Stability)")
    lines.append("─" * 72)
    for r in [result_a, result_b]:
        status = "✅ PASS (No Crash)" if r["success"] else "❌ CRASHED"
        lines.append(f"  {r['run_name']:20s}: {status}  [{r['elapsed_seconds']:.1f}s]")
        if r["crash"]:
            lines.append(f"    Exception: {r['crash']['type']}: {r['crash']['exception']}")
            # First 5 lines of traceback
            tb_lines = r['crash']['traceback'].strip().split('\n')
            for tl in tb_lines[-5:]:
                lines.append(f"    {tl}")
    lines.append("")

    if not result_a["success"] or not result_b["success"]:
        lines.append("⚠ One or both runs crashed. Partial report below.")
        lines.append("")

    # --- Data Volume ---
    lines.append("─" * 72)
    lines.append("2. DATA VOLUME")
    lines.append("─" * 72)
    header = f"  {'Metric':<30s} {'Run A (ml_ready)':>18s} {'Run B (clean)':>18s}"
    lines.append(header)
    lines.append("  " + "─" * 68)

    vol_metrics = [
        ("Raw rows (input)", "n_rows_raw"),
        ("After preprocess", "n_rows_after_preprocess"),
        ("After dedup", "n_rows_after_dedup"),
        ("Train split", "n_train"),
        ("Calibration split", "n_calib"),
        ("Test split", "n_test"),
        ("Unique markets (train)", "n_markets_train"),
    ]
    for label, key in vol_metrics:
        va = result_a.get(key, "—")
        vb = result_b.get(key, "—")
        lines.append(f"  {label:<30s} {str(va):>18s} {str(vb):>18s}")
    lines.append("")

    # --- CV Metrics ---
    lines.append("─" * 72)
    lines.append("3. CROSS-VALIDATION METRICS (GroupKFold)")
    lines.append("─" * 72)
    cv_metrics = [
        ("Mean AUC", "cv_mean_auc"),
        ("Std AUC", "cv_std_auc"),
        ("OOF AUC", "cv_oof_auc"),
        ("Mean Brier", "cv_mean_brier"),
        ("OOF ECE (raw)", "cv_oof_ece_raw"),
        ("Optimal trees", "cv_optimal_trees"),
    ]
    lines.append(f"  {'Metric':<30s} {'Run A':>18s} {'Run B':>18s}")
    lines.append("  " + "─" * 68)
    for label, key in cv_metrics:
        va = _fmt(result_a.get(key))
        vb = _fmt(result_b.get(key))
        lines.append(f"  {label:<30s} {va:>18s} {vb:>18s}")
    lines.append("")

    # --- Calibration Metrics ---
    lines.append("─" * 72)
    lines.append("4. CALIBRATION METRICS (Platt-Calibrated)")
    lines.append("─" * 72)
    cal_metrics = [
        ("Calib AUC", "calib_auc"),
        ("Calib Brier", "calib_brier"),
        ("Calib ECE", "calib_ece"),
        ("Test AUC", "eval_auc"),
        ("Test Brier", "eval_brier"),
        ("Test ECE", "eval_ece"),
        ("Test LogLoss", "eval_logloss"),
        ("Best Iteration", "best_iteration"),
    ]
    lines.append(f"  {'Metric':<30s} {'Run A':>18s} {'Run B':>18s}")
    lines.append("  " + "─" * 68)
    for label, key in cal_metrics:
        va = _fmt(result_a.get(key))
        vb = _fmt(result_b.get(key))
        lines.append(f"  {label:<30s} {va:>18s} {vb:>18s}")
    lines.append("")

    # --- Quality Gates ---
    lines.append("─" * 72)
    lines.append("5. QUALITY GATES")
    lines.append("─" * 72)
    for r in [result_a, result_b]:
        passed = r.get("gates_all_pass", "—")
        status = "✅ ALL PASS" if passed is True else ("❌ FAILED" if passed is False else "—")
        lines.append(f"  {r['run_name']:20s}: {status}")
        if r.get("gate_details"):
            for gname, ginfo in r["gate_details"].items():
                gs = "✓" if ginfo["pass"] else "✗"
                lines.append(
                    f"    {gs} {gname:<18s}: {ginfo['value']:.4f} "
                    f"(threshold: {ginfo['threshold']:.4f})"
                )
    lines.append("")

    # --- SHAP Validation ---
    lines.append("─" * 72)
    lines.append("6. SHAP FEATURE IMPORTANCE VALIDATION")
    lines.append("─" * 72)
    for r in [result_a, result_b]:
        lines.append(f"  {r['run_name']}:")
        evc_rank = r.get("shap_edge_vs_crowd_rank", "—")
        obi_rank = r.get("shap_obi_vol_rank", "—")

        evc_status = "✅" if isinstance(evc_rank, int) and evc_rank <= 5 else "⚠"
        obi_status = "✅" if isinstance(obi_rank, int) and obi_rank <= 5 else "⚠"
        lines.append(f"    {evc_status} edge_vs_crowd      → Rank #{evc_rank}")
        lines.append(f"    {obi_status} obi_vol_interaction → Rank #{obi_rank}")

        if r.get("shap_top10"):
            lines.append("    Top-10 SHAP:")
            for i, (fname, val) in enumerate(r["shap_top10"], 1):
                lines.append(f"      #{i:2d}: {fname:<28s} {val:.6f}")
        lines.append("")

    # --- EV Simulation ---
    lines.append("─" * 72)
    lines.append("7. EV STRATEGY SIMULATION (Underdog odds < 0.30)")
    lines.append("─" * 72)
    ev_metrics = [
        ("Total test samples", "ev_n_total"),
        ("Underdog bets (filtered)", "ev_n_underdog"),
        ("Bets executed (EV > thr)", "ev_n_executed"),
        ("Win Rate", "ev_win_rate"),
        ("Average EV", "ev_avg_ev"),
        ("Total PnL (units)", "ev_total_pnl"),
        ("Sharpe Ratio", "ev_sharpe"),
        ("Max Drawdown", "ev_max_drawdown"),
    ]
    lines.append(f"  {'Metric':<30s} {'Run A':>18s} {'Run B':>18s}")
    lines.append("  " + "─" * 68)
    for label, key in ev_metrics:
        va = _fmt(result_a.get(key))
        vb = _fmt(result_b.get(key))
        lines.append(f"  {label:<30s} {va:>18s} {vb:>18s}")
    lines.append("")

    # --- Summary ---
    lines.append("═" * 72)
    lines.append("  SUMMARY")
    lines.append("═" * 72)
    if result_a["success"] and result_b["success"]:
        lines.append("  Both pipelines executed successfully. No crashes detected.")
    else:
        lines.append("  ⚠ One or more pipelines crashed. See crash details above.")
    lines.append(f"  Run A elapsed: {result_a['elapsed_seconds']:.1f}s")
    lines.append(f"  Run B elapsed: {result_b['elapsed_seconds']:.1f}s")
    lines.append("═" * 72)

    return "\n".join(lines)


def _fmt(val: Any) -> str:
    """Format a value for display."""
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_logging()

    base = Path(__file__).resolve().parent.parent
    dataset_dir = base / "dataset"
    models_dir = base / "models"

    data_a = dataset_dir / "dataset_ml_ready.csv"
    data_b = dataset_dir / "dataset_clean.csv"
    output_a = models_dir / "test_ml_ready"
    output_b = models_dir / "test_clean"

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  STRESS TEST — DUAL EXECUTION                           ║")
    logger.info("╚" + "═" * 58 + "╝")

    # --- Run A: dataset_ml_ready.csv ---
    result_a = run_pipeline_single(
        data_path=data_a,
        output_dir=output_a,
        run_name="Run_A_ml_ready",
        preprocess=False,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("")

    # --- Run B: dataset_clean.csv ---
    result_b = run_pipeline_single(
        data_path=data_b,
        output_dir=output_b,
        run_name="Run_B_clean",
        preprocess=True,  # needs label creation + missing cols
    )

    # --- Generate Report ---
    report = generate_comparison_report(result_a, result_b)
    print("\n\n")
    print(report)

    # Save report
    report_path = models_dir / "stress_test_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Report saved to: %s", report_path)

    # Save raw results as JSON
    for name, res in [("result_a", result_a), ("result_b", result_b)]:
        res_path = models_dir / f"stress_test_{name}.json"
        # filter non-serializable
        serializable = {k: v for k, v in res.items()
                        if k != "pipeline_result"}
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)


if __name__ == "__main__":
    main()
