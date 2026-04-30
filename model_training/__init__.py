"""
model_training
==============
XGBoost Meta-Model Pipeline untuk Polymarket 5-minute underdog strategy.

Modul:
  - config.py    : Konstanta, hyperparameter, dan quality gates
  - features.py  : Feature engineering (edge_vs_crowd, interactions)
  - dataset.py   : Data loading, validasi, deduplikasi, splitting
  - trainer.py   : GroupKFold CV, XGBoost training, Platt calibration
  - evaluate.py  : Reliability diagram, SHAP analysis, EV simulation
  - monitor.py   : Concept drift detection (PSI), rolling Brier
  - inference.py : Live production gate (XGBoostGate class)
  - pipeline.py  : Training orchestrator (entry point)
"""

from .config import (
    ALL_FEATURES,
    RAW_FEATURES,
    ENGINEERED_FEATURES,
    TARGET_COL,
    SPLIT_CFG,
    XGB_CFG,
    QUALITY_GATES,
    EV_CFG,
    DRIFT_CFG,
)
from .inference import XGBoostGate

__all__ = [
    "ALL_FEATURES",
    "RAW_FEATURES",
    "ENGINEERED_FEATURES",
    "TARGET_COL",
    "SPLIT_CFG",
    "XGB_CFG",
    "QUALITY_GATES",
    "EV_CFG",
    "DRIFT_CFG",
    "XGBoostGate",
]
