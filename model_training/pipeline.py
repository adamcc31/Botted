"""
ml/pipeline.py
==============
Training Orchestrator — Entry point utama untuk pipeline ML.

Menyatukan seluruh komponen menjadi alur berurutan:
  1. Load data (dataset.py)
  2. Validasi & deduplikasi
  3. Rekayasa fitur (features.py)
  4. Split chronological → train / calib / test
  5. Impute missing values
  6. GroupKFold cross-validation
  7. Final training + Platt calibration (trainer.py)
  8. Quality gates check
  9. Evaluasi & simulasi EV (evaluate.py)
  10. Simpan model .pkl, metadata, dan PSI baseline

Penggunaan:
    python -m model_training.pipeline --data data/signals.csv --output models/v1/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Pipeline modules
from .config import (
    ALL_FEATURES, SELECTED_FEATURES, TARGET_COL, META_COLS,
    SPLIT_CFG, XGB_CFG, QUALITY_GATES, EV_CFG, DRIFT_CFG,
)
from .dataset import (
    load_from_csv,
    load_from_sqlite,
    validate_dataset,
    deduplicate_per_market,
    chronological_split,
    get_market_groups,
    impute_features,
    apply_imputer,
)
from .features import build_features, get_feature_matrix, validate_no_leakage
from .trainer import (
    cross_validate,
    train_and_calibrate,
    run_quality_gates,
    save_model,
)
from .evaluate import run_full_evaluation, plot_reliability_diagram
from .monitor import PSIBaselineManager


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def _setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Konfigurasi logging stdout + optional file."""
    handlers = []

    # Console handler — rapi dan readable
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    handlers.append(console_handler)

    # File handler (opsional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

class TrainingPipeline:
    """
    Orchestrator utama untuk training pipeline XGBoost meta-model.

    Setiap method merepresentasikan satu langkah pipeline yang
    bisa dijalankan secara independen (testing) atau berurutan
    via run_full_pipeline().
    """

    def __init__(
        self,
        data_path: str | Path,
        output_dir: str | Path,
        data_format: str = "csv",
        test_ratio: float = 0.15,
        model_version: Optional[str] = None,
    ) -> None:
        """
        Args:
            data_path:     Path ke dataset (CSV atau SQLite).
            output_dir:    Direktori output untuk model dan artefak.
            data_format:   "csv" atau "sqlite".
            test_ratio:    Proporsi data untuk final test set.
            model_version: ID versi model. Auto-generate jika None.
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.data_format = data_format
        self.test_ratio = test_ratio
        self.model_version = model_version or time.strftime("%Y%m%d_%H%M%S")

        # State yang diisi selama pipeline
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        self.df_train: Optional[pd.DataFrame] = None
        self.df_calib: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
        self.imputer_vals: Optional[pd.Series] = None
        self.cv_result: Optional[dict] = None
        self.train_result: Optional[dict] = None
        self.gate_result: Optional[dict] = None
        self.eval_result: Optional[dict] = None
        self.model_path: Optional[Path] = None

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load Data
    # ------------------------------------------------------------------

    def step_load_data(self) -> pd.DataFrame:
        """Load dataset dari CSV atau SQLite."""
        logger.info("=" * 60)
        logger.info("STEP 1: LOAD DATA")
        logger.info("=" * 60)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset tidak ditemukan: {self.data_path}"
            )

        if self.data_format == "sqlite":
            self.df_raw = load_from_sqlite(self.data_path)
        else:
            self.df_raw = load_from_csv(self.data_path)

        logger.info(
            "Data loaded: %d rows × %d columns",
            self.df_raw.shape[0], self.df_raw.shape[1],
        )
        return self.df_raw

    # ------------------------------------------------------------------
    # Step 2: Validate & Clean
    # ------------------------------------------------------------------

    def step_validate_and_clean(self) -> pd.DataFrame:
        """Validasi kualitas data dan deduplikasi per market."""
        logger.info("=" * 60)
        logger.info("STEP 2: VALIDATE & CLEAN")
        logger.info("=" * 60)

        if self.df_raw is None:
            raise RuntimeError("Panggil step_load_data() terlebih dahulu.")

        # Validasi
        validation_results = validate_dataset(self.df_raw)
        logger.info("Validation results: %s", json.dumps(validation_results, indent=2))

        # Check gates
        if validation_results.get("sample_gate") == "FAIL":
            logger.warning(
                "⚠ SAMPLE GATE FAIL — dataset mungkin terlalu kecil. "
                "Pipeline tetap berjalan untuk eksperimen, tapi model "
                "TIDAK boleh di-deploy."
            )

        # Deduplikasi
        self.df_clean = deduplicate_per_market(self.df_raw)

        # Leakage check
        validate_no_leakage(self.df_clean)
        logger.info("✓ No leakage detected.")

        logger.info(
            "Clean data: %d rows × %d columns",
            self.df_clean.shape[0], self.df_clean.shape[1],
        )
        return self.df_clean

    # ------------------------------------------------------------------
    # Step 3: Split Data
    # ------------------------------------------------------------------

    def step_split_data(self) -> None:
        """Split chronological → train / calib / test."""
        logger.info("=" * 60)
        logger.info("STEP 3: CHRONOLOGICAL SPLIT")
        logger.info("=" * 60)

        if self.df_clean is None:
            raise RuntimeError("Panggil step_validate_and_clean() terlebih dahulu.")

        if self.test_ratio > 0:
            self.df_train, self.df_calib, self.df_test = chronological_split(
                self.df_clean,
                calib_ratio=SPLIT_CFG.calib_holdout_ratio,
                test_ratio=self.test_ratio,
            )
            logger.info(
                "Split: train=%d | calib=%d | test=%d",
                len(self.df_train), len(self.df_calib), len(self.df_test),
            )
        else:
            self.df_train, self.df_calib = chronological_split(
                self.df_clean,
                calib_ratio=SPLIT_CFG.calib_holdout_ratio,
                test_ratio=0.0,
            )
            self.df_test = self.df_calib.copy()  # Fallback: use calib as test
            logger.info(
                "Split: train=%d | calib=%d | test=calib (no separate test set)",
                len(self.df_train), len(self.df_calib),
            )

    # ------------------------------------------------------------------
    # Step 4: Feature Engineering & Imputation
    # ------------------------------------------------------------------

    def step_engineer_features(self) -> None:
        """Build engineered features dan impute NaN."""
        logger.info("=" * 60)
        logger.info("STEP 4: FEATURE ENGINEERING & IMPUTATION")
        logger.info("=" * 60)

        if self.df_train is None:
            raise RuntimeError("Panggil step_split_data() terlebih dahulu.")

        # Impute — fit pada training, apply ke calib dan test
        self.df_train, self.imputer_vals = impute_features(
            self.df_train, strategy="median"
        )
        self.df_calib = apply_imputer(self.df_calib, self.imputer_vals)
        self.df_test = apply_imputer(self.df_test, self.imputer_vals)

        logger.info("Imputer fitted on %d training samples.", len(self.df_train))
        logger.info("Imputer values:\n%s", self.imputer_vals.to_string())

    # ------------------------------------------------------------------
    # Step 5: Cross-Validation
    # ------------------------------------------------------------------

    def step_cross_validate(self) -> dict:
        """GroupKFold cross-validation pada training set."""
        logger.info("=" * 60)
        logger.info("STEP 5: CROSS-VALIDATION (GroupKFold)")
        logger.info("=" * 60)

        if self.df_train is None:
            raise RuntimeError("Panggil step_engineer_features() terlebih dahulu.")

        # Build features untuk CV
        df_train_feat = build_features(self.df_train)
        X_train = df_train_feat[SELECTED_FEATURES].values.astype(np.float32)
        y_train = df_train_feat[TARGET_COL].values.astype(np.int32)
        groups = get_market_groups(df_train_feat)

        self.cv_result = cross_validate(X_train, y_train, groups)

        logger.info(
            "CV Result: mean_AUC=%.4f ±%.4f | OOF_AUC=%.4f | Brier=%.4f",
            self.cv_result["mean_auc"],
            self.cv_result["std_auc"],
            self.cv_result["oof_auc"],
            self.cv_result["mean_brier"],
        )
        return self.cv_result

    # ------------------------------------------------------------------
    # Step 6: Train & Calibrate
    # ------------------------------------------------------------------

    def step_train_and_calibrate(self) -> dict:
        """Final training XGBoost + Platt calibration."""
        logger.info("=" * 60)
        logger.info("STEP 6: FINAL TRAINING + PLATT CALIBRATION")
        logger.info("=" * 60)

        if self.df_train is None or self.df_calib is None:
            raise RuntimeError("Panggil step_split_data() terlebih dahulu.")

        self.train_result = train_and_calibrate(
            self.df_train,
            self.df_calib,
            imputer_vals=self.imputer_vals,
        )

        logger.info(
            "Training complete: AUC=%.4f | Brier=%.4f | ECE=%.4f | best_iter=%d",
            self.train_result["metrics"]["calib_auc"],
            self.train_result["metrics"]["calib_brier"],
            self.train_result["metrics"]["calib_ece"],
            self.train_result["best_iteration"],
        )
        return self.train_result

    # ------------------------------------------------------------------
    # Step 7: Quality Gates
    # ------------------------------------------------------------------

    def step_quality_gates(self) -> dict:
        """Evaluasi quality gates — model harus PASS semua."""
        logger.info("=" * 60)
        logger.info("STEP 8: QUALITY GATES CHECK")
        logger.info("=" * 60)

        if self.cv_result is None or self.train_result is None:
            raise RuntimeError(
                "Panggil step_cross_validate() dan step_train_and_calibrate() dahulu."
            )

        # Pass eval_result so test ECE can be checked
        self.gate_result = run_quality_gates(
            self.cv_result, self.train_result,
            eval_result=self.eval_result,
        )

        if self.gate_result["all_pass"]:
            logger.info("ALL QUALITY GATES PASSED -- model layak disimpan.")
        else:
            failed = [
                name for name, c in self.gate_result["checks"].items()
                if not c["pass"]
            ]
            logger.warning(
                "QUALITY GATES FAILED: %s -- model TIDAK akan disimpan.", failed
            )

        # Log detail setiap gate
        for name, check in self.gate_result["checks"].items():
            status = "PASS" if check["pass"] else "FAIL"
            logger.info(
                "  %s: %s (value=%.4f, threshold=%.4f)",
                name, status, check["value"], check["threshold"],
            )

        return self.gate_result

    # ------------------------------------------------------------------
    # Step 8: Evaluate & Simulate
    # ------------------------------------------------------------------

    def step_evaluate(self) -> dict:
        """Full evaluation: reliability diagram, SHAP, EV simulation."""
        logger.info("=" * 60)
        logger.info("STEP 7: EVALUATION & SIMULATION")
        logger.info("=" * 60)

        if self.train_result is None or self.df_test is None:
            raise RuntimeError(
                "Panggil step_train_and_calibrate() terlebih dahulu."
            )

        # Prepare test data
        df_test_feat = build_features(self.df_test)
        X_test = df_test_feat[SELECTED_FEATURES].values.astype(np.float32)
        y_test = df_test_feat[TARGET_COL].values.astype(np.int32)

        # Entry odds untuk simulasi EV
        if "entry_odds" in df_test_feat.columns:
            entry_odds = df_test_feat["entry_odds"].values.astype(np.float64)
        else:
            logger.warning(
                "entry_odds tidak tersedia di test set. "
                "Menggunakan synthetic odds (uniform 0.05-0.50)."
            )
            rng = np.random.default_rng(42)
            entry_odds = rng.uniform(0.05, 0.50, size=len(y_test))

        eval_dir = self.output_dir / "evaluation"
        self.eval_result = run_full_evaluation(
            base_model=self.train_result["base_model"],
            platt_model=self.train_result["platt"],
            X_test=X_test,
            y_test=y_test,
            entry_odds=entry_odds,
            feature_names=SELECTED_FEATURES,
            output_dir=eval_dir,
        )

        return self.eval_result

    # ------------------------------------------------------------------
    # Step 9: Save Model & Artifacts
    # ------------------------------------------------------------------

    def step_save(self) -> Path:
        """Simpan model .pkl, metadata, dan PSI baseline."""
        logger.info("=" * 60)
        logger.info("STEP 9: SAVE MODEL & ARTIFACTS")
        logger.info("=" * 60)

        if not self.gate_result or not self.gate_result["all_pass"]:
            logger.warning(
                "Quality gates belum PASS. Model TIDAK disimpan. "
                "Override: set gate_result['all_pass'] = True untuk force save."
            )
            # Tidak raise error — biarkan user memutuskan
            if self.gate_result and not self.gate_result["all_pass"]:
                logger.error(
                    "ABORT: model gagal quality gates. Perbaiki model atau "
                    "longgarkan threshold di config.py sebelum retry."
                )
                return self.output_dir

        # Save model
        self.model_path = save_model(
            train_result=self.train_result,
            cv_result=self.cv_result,
            gate_result=self.gate_result,
            output_dir=self.output_dir,
            model_version=self.model_version,
        )
        logger.info("Model disimpan: %s", self.model_path)

        # Save PSI baseline dari training data
        if self.df_train is not None:
            psi_manager = PSIBaselineManager(
                baseline_dir=self.output_dir,
            )
            df_train_feat = build_features(self.df_train)
            psi_baseline_path = psi_manager.save_baseline(df_train_feat)
            logger.info("PSI baseline disimpan: %s", psi_baseline_path)

        # Save evaluation results
        if self.eval_result is not None:
            eval_path = self.output_dir / "evaluation_results.json"
            # Filter non-serializable
            serializable = _make_serializable(self.eval_result)
            with open(eval_path, "w") as f:
                json.dump(serializable, f, indent=2)
            logger.info("Evaluation results disimpan: %s", eval_path)

        # Save imputer
        if self.imputer_vals is not None:
            imputer_path = self.output_dir / "imputer_vals.json"
            with open(imputer_path, "w") as f:
                json.dump(self.imputer_vals.to_dict(), f, indent=2)
            logger.info("Imputer values disimpan: %s", imputer_path)

        return self.model_path

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(self) -> dict:
        """
        Jalankan seluruh pipeline end-to-end.

        Returns:
            Dict berisi semua hasil dari setiap langkah.
        """
        pipeline_start = time.time()

        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║  XGBoost META-MODEL TRAINING PIPELINE                    ║")
        logger.info("║  Version: %-47s║", self.model_version)
        logger.info("╚" + "═" * 58 + "╝")
        logger.info("")

        try:
            # 1. Load
            self.step_load_data()

            # 2. Validate & Clean
            self.step_validate_and_clean()

            # 3. Split
            self.step_split_data()

            # 4. Feature Engineering & Imputation
            self.step_engineer_features()

            # 5. Cross-Validation
            self.step_cross_validate()

            # 6. Train & Calibrate
            self.step_train_and_calibrate()

            # 7. Evaluate (BEFORE gates — test ECE needed for gate check)
            self.step_evaluate()

            # 8. Quality Gates (now includes test ECE from step 7)
            self.step_quality_gates()

            # 9. Save
            self.step_save()

        except Exception as e:
            logger.exception("Pipeline GAGAL pada langkah terakhir: %s", e)
            raise

        elapsed = time.time() - pipeline_start

        logger.info("")
        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║  PIPELINE COMPLETE                                       ║")
        logger.info("╚" + "═" * 58 + "╝")
        logger.info("Total time: %.1f seconds", elapsed)
        logger.info("Output dir: %s", self.output_dir)

        return {
            "model_version": self.model_version,
            "model_path": str(self.model_path) if self.model_path else None,
            "output_dir": str(self.output_dir),
            "cv_result": self.cv_result,
            "train_metrics": self.train_result["metrics"] if self.train_result else None,
            "gate_result": self.gate_result,
            "elapsed_seconds": round(elapsed, 1),
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types untuk JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    else:
        return obj


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "XGBoost Meta-Model Training Pipeline — "
            "Polymarket 5-minute underdog strategy"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m model_training.pipeline --data data/signals.csv --output models/v1/\n"
            "  python -m model_training.pipeline --data data/signals.db --format sqlite\n"
        ),
    )

    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path ke dataset (CSV atau SQLite).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models/latest/",
        help="Direktori output untuk model dan artefak. Default: models/latest/",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["csv", "sqlite"],
        default="csv",
        help="Format data input. Default: csv.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proporsi data untuk final test set. Default: 0.15.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="ID versi model. Auto-generated jika tidak diisi.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Level logging. Default: INFO.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path file log (opsional).",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point utama."""
    args = parse_args()

    # Setup logging
    _setup_logging(log_level=args.log_level, log_file=args.log_file)

    logger.info("Starting pipeline with arguments:")
    logger.info("  data:       %s", args.data)
    logger.info("  output:     %s", args.output)
    logger.info("  format:     %s", args.format)
    logger.info("  test_ratio: %.2f", args.test_ratio)
    logger.info("  version:    %s", args.version or "(auto)")

    # Run pipeline
    pipeline = TrainingPipeline(
        data_path=args.data,
        output_dir=args.output,
        data_format=args.format,
        test_ratio=args.test_ratio,
        model_version=args.version,
    )

    result = pipeline.run_full_pipeline()

    # Summary
    logger.info("")
    logger.info("Pipeline summary:")
    logger.info("  Model version:  %s", result["model_version"])
    logger.info("  Model path:     %s", result["model_path"])
    logger.info("  CV mean AUC:    %.4f", result["cv_result"]["mean_auc"])
    logger.info("  Gates passed:   %s", result["gate_result"]["all_pass"])
    logger.info("  Elapsed:        %.1f seconds", result["elapsed_seconds"])


if __name__ == "__main__":
    main()
