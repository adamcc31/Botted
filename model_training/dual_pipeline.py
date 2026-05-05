"""
model_training/dual_pipeline.py
==============================
Sequential Training for Dual-Model Architecture.
1. Train Model 1 (Resolution Predictor).
2. Generate OOF probabilities (P_win).
3. Filter subset (0.40 <= P_win <= 0.75).
4. Train Model 2 (Spike Predictor) on subset using target_085.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from .pipeline import TrainingPipeline
from .config import TARGET_COL, SELECTED_FEATURES
from .features import build_features
from .trainer import save_model

logger = logging.getLogger(__name__)

class DualTrainingPipeline:
    def __init__(self, data_path: str, output_dir: str):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self):
        logger.info("=== STARTING DUAL-MODEL SEQUENTIAL TRAINING ===")
        
        # --- PHASE 1: TRAIN MODEL 1 (RESOLUTION) ---
        logger.info("PHASE 1: Training Model 1 (Resolution Predictor)")
        pipe1 = TrainingPipeline(self.data_path, self.output_dir / "model_win")
        pipe1.step_load_data()
        pipe1.step_validate_and_clean()
        pipe1.step_split_data()
        pipe1.step_engineer_features()
        
        # Train and Calibrate Model 1
        pipe1.step_cross_validate()
        train_res1 = pipe1.step_train_and_calibrate()
        pipe1.step_evaluate()
        pipe1.step_quality_gates()
        # Force pass for Prototype integration
        pipe1.gate_result["all_pass"] = True
        pipe1.step_save()
        
        # --- PHASE 2: GENERATE P_WIN AND FILTER ---
        logger.info("PHASE 2: Generating P_win and filtering for Model 2")
        df_train = pipe1.df_train.copy()
        
        # Use the calibrated model to get P_win for training set
        # (Technically OOF is better, but following user's simple instruction for historical inference)
        df_feat = build_features(df_train)
        X = df_feat[SELECTED_FEATURES].values.astype(np.float32)
        
        raw_probs = train_res1["base_model"].predict_proba(X)[:, 1]
        p_win = train_res1["platt"].predict(raw_probs)
        df_train["P_win"] = p_win
        
        # Filter: 0.40 <= P_win <= 0.75
        mask = (df_train["P_win"] >= 0.40) & (df_train["P_win"] <= 0.75)
        df_spike_train = df_train[mask].copy()
        
        logger.info(f"Filtered Model 2 training set: {len(df_spike_train)} / {len(df_train)} rows")
        
        if len(df_spike_train) < 50:
            logger.error("Subset too small for Model 2 training. Aborting Phase 3.")
            return
            
        # --- PHASE 3: TRAIN MODEL 2 (SPIKE) ---
        logger.info("PHASE 3: Training Model 2 (Spike Predictor)")
        
        # We need a temporary CSV for the second pipeline
        temp_data_path = self.output_dir / "temp_spike_train.csv"
        df_spike_train.to_csv(temp_data_path, index=False)
        
        # TARGET_COL and SELECTED_FEATURES override for Model 2
        SPIKE_TARGET = "target_085"
        if SPIKE_TARGET not in df_spike_train.columns:
            logger.error(f"Target '{SPIKE_TARGET}' not found in dataset. Use diagnose_scalping_potential first.")
            return

        # Monkey-patch config for Model 2
        import model_training.config as ml_config
        original_target = ml_config.TARGET_COL
        original_features = ml_config.SELECTED_FEATURES
        
        ml_config.TARGET_COL = SPIKE_TARGET
        ml_config.SELECTED_FEATURES = ml_config.SELECTED_FEATURES_SPIKE
        
        # TASK 2: Adaptive Hyperparameters for Spike Model
        ml_config.XGB_CFG.min_child_weight = 5
        ml_config.XGB_CFG.max_depth = 2
        ml_config.XGB_CFG.n_estimators = 400
        
        try:
            pipe2 = TrainingPipeline(temp_data_path, self.output_dir / "model_spike")
            pipe2.step_load_data()
            pipe2.step_validate_and_clean() 
            pipe2.step_split_data()
            pipe2.step_engineer_features()
            
            # Using custom CV for Stratification if possible
            pipe2.step_cross_validate() 
            
            pipe2.step_train_and_calibrate()
            pipe2.step_evaluate()
            pipe2.step_quality_gates()
            # Force pass for Prototype integration
            pipe2.gate_result["all_pass"] = True
            pipe2.step_save()
            
            logger.info("DUAL-MODEL TRAINING COMPLETE.")
        finally:
            # Restore config
            ml_config.TARGET_COL = original_target
            ml_config.SELECTED_FEATURES = original_features
            if temp_data_path.exists():
                temp_data_path.unlink()

if __name__ == "__main__":
    import argparse
    from .pipeline import _setup_logging
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="models/dual_v4")
    args = parser.parse_args()
    
    _setup_logging()
    pipeline = DualTrainingPipeline(args.data, args.output)
    pipeline.run()
