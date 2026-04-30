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

# Lazy import for XGBoostGate — avoid pulling in joblib/xgboost/sklearn
# at module load time if they haven't been installed yet.
def __getattr__(name):
    if name == "XGBoostGate":
        from .inference import XGBoostGate
        return XGBoostGate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
