"""
model_training/dual_inference.py
===============================
Dual-Model Inference Engine.
Loads model_win and model_spike and implements the Decision Matrix.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from .inference import XGBoostGate

logger = logging.getLogger(__name__)

class DualXGBoostGate:
    def __init__(self):
        self.gate_win = XGBoostGate()
        self.gate_spike = XGBoostGate()
        self._is_loaded = False

    def load_models(self, dual_model_dir: str | Path):
        dual_model_dir = Path(dual_model_dir)
        self.gate_win.load_model(dual_model_dir / "model_win")
        
        spike_path = dual_model_dir / "model_spike"
        if spike_path.exists():
            self.gate_spike.load_model(spike_path)
            self._is_loaded = True
        else:
            logger.warning(f"Spike model not found at {spike_path}. Running in Single-Model mode.")
            self._is_loaded = True # Still loaded, but only win gate

    def evaluate_dual_signal(
        self,
        raw_features: Dict[str, Any],
        entry_odds: float,
        ev_threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Decision Matrix implementation.
        """
        # 1. Infer Model 1 (Win)
        res_win = self.gate_win.evaluate_signal(raw_features, entry_odds, ev_threshold)
        
        p_win = res_win["p_win"]
        
        # Decision Matrix
        # Default strategy
        strategy = "HOLD_TO_MATURITY"
        
        # Condition A: Conviction Hold
        if p_win > 0.80:
            return {
                **res_win,
                "strategy": "HOLD_TO_MATURITY",
                "tp_target": 1.0,
                "p_spike": None
            }
            
        # Condition B: Scalping Mode (Marginal Zone)
        if 0.40 <= p_win <= 0.75 and self.gate_spike.is_loaded:
            res_spike = self.gate_spike.predict_quality(raw_features)
            p_spike = res_spike["p_win"] # This is p_spike because it's model_spike
            
            if p_spike > 0.80:
                # Override decision if spike is high but win was marginal
                return {
                    "decision": "PASS",
                    "reason": f"SCALP: P_win={p_win:.4f} (marginal), P_spike={p_spike:.4f} (high)",
                    "p_win": p_win,
                    "p_spike": p_spike,
                    "ev": res_win["ev"], # Keep original EV
                    "strategy": "SCALPING",
                    "tp_target": 0.85,
                    "entry_odds": entry_odds,
                    "confidence": "SCALP_HIGH",
                    "kelly_fraction": res_win["kelly_fraction"]
                }
            else:
                # P_win is marginal and no spike predicted
                # Keep original resolution from win gate
                return {
                    **res_win,
                    "strategy": "HOLD_TO_MATURITY",
                    "p_spike": p_spike
                }
                
        # Fallback to single model result
        return {
            **res_win,
            "strategy": "HOLD_TO_MATURITY",
            "p_spike": None
        }

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
