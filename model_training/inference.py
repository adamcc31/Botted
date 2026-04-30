"""
ml/inference.py
===============
Live Production Gate — XGBoostGate.

Class ini dirancang untuk diimpor oleh sistem bot live yang berjalan
secara asinkron. Bertanggung jawab untuk:
  1. Memuat artefak CalibratedClassifierCV (XGBoost + Platt)
  2. Transformasi fitur mentah dari bot → feature vector
  3. Prediksi P(WIN) terkalibrasi
  4. Kalkulasi Expected Value (EV)
  5. Keputusan PASS/REJECT berdasarkan quality gates

Thread-safe: semua state immutable setelah load_model().
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import joblib
import numpy as np
import pandas as pd

from .config import (
    RAW_FEATURES, ENGINEERED_FEATURES, SELECTED_FEATURES,
    EV_CFG, DRIFT_CFG,
)
from .features import build_features

logger = logging.getLogger(__name__)


class XGBoostGate:
    """
    Layer 3 Quality Gate — ML meta-model untuk memvalidasi sinyal bot.

    Alur penggunaan:
        gate = XGBoostGate()
        gate.load_model("models/latest/")
        result = gate.predict_quality(raw_features_dict)
        ev = gate.calculate_ev(result["p_win"], entry_odds=0.15)
    """

    def __init__(self) -> None:
        self._base_model = None
        self._platt_model = None
        self._imputer_vals: Optional[pd.Series] = None
        self._feature_names: List[str] = SELECTED_FEATURES
        self._model_version: Optional[str] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Apakah model sudah dimuat dan siap inference."""
        return self._is_loaded

    @property
    def model_version(self) -> Optional[str]:
        """Versi model yang aktif."""
        return self._model_version

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        """Metadata lengkap model (metrics, gates, dll)."""
        return self._metadata

    # ------------------------------------------------------------------
    # Load Model
    # ------------------------------------------------------------------

    def load_model(
        self,
        model_dir: str | Path,
    ) -> None:
        """
        Muat artefak model dari direktori.

        Direktori harus mengandung:
          - model.pkl : bundle {base_model, platt, imputer_vals, feature_names}
          - metadata.json : info versi dan metrics

        Args:
            model_dir: Path ke direktori artefak model.

        Raises:
            FileNotFoundError: Jika model.pkl tidak ditemukan.
            RuntimeError: Jika bundle tidak valid.
        """
        model_dir = Path(model_dir)
        model_path = model_dir / "model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact tidak ditemukan: {model_path}. "
                "Jalankan pipeline.py terlebih dahulu."
            )

        logger.info("Memuat model dari %s ...", model_dir)
        t0 = time.time()

        bundle = joblib.load(model_path)

        # Validasi bundle
        required_keys = {"base_model", "platt", "feature_names"}
        missing = required_keys - set(bundle.keys())
        if missing:
            raise RuntimeError(
                f"Model bundle tidak valid — missing keys: {missing}. "
                f"Bundle keys: {list(bundle.keys())}"
            )

        self._base_model = bundle["base_model"]
        self._platt_model = bundle["platt"]
        self._imputer_vals = bundle.get("imputer_vals")
        self._feature_names = bundle["feature_names"]
        self._model_version = bundle.get("version", "unknown")

        # Load metadata jika tersedia
        meta_path = model_dir / "metadata.json"
        if meta_path.exists():
            import json
            with open(meta_path, "r") as f:
                self._metadata = json.load(f)

        self._is_loaded = True

        logger.info(
            "Model dimuat dalam %.2fs — version=%s | n_features=%d",
            time.time() - t0,
            self._model_version,
            len(self._feature_names),
        )

    # ------------------------------------------------------------------
    # Predict Quality
    # ------------------------------------------------------------------

    def predict_quality(
        self,
        raw_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Prediksi kualitas sinyal dari fitur mentah.

        Alur:
          1. Konversi dict → DataFrame satu baris
          2. Jalankan FeatureEngineer (build_features) untuk
             membuat ENGINEERED_FEATURES
          3. Handle NaN via imputer
          4. Prediksi P(WIN) terkalibrasi via XGBoost + Platt

        Args:
            raw_features: Dict dari fitur mentah (output signal engine bot).
                          Harus mengandung semua kolom di RAW_FEATURES.

        Returns:
            Dict berisi:
              - p_win:           float — probabilitas terkalibrasi
              - p_win_raw:       float — probabilitas sebelum Platt
              - confidence:      str   — "HIGH" / "MEDIUM" / "LOW"
              - feature_vector:  list  — vector fitur untuk debugging
              - missing_features: list — fitur yang tidak ada di input

        Raises:
            RuntimeError: Jika model belum dimuat.
        """
        if not self._is_loaded:
            raise RuntimeError(
                "Model belum dimuat. Panggil load_model() terlebih dahulu."
            )

        # --- Step 1: Dict → DataFrame ---
        df = pd.DataFrame([raw_features])

        # Cek fitur yang hilang
        missing_raw = [f for f in RAW_FEATURES if f not in df.columns]
        if missing_raw:
            logger.warning(
                "Missing raw features: %s. Akan diisi NaN.",
                missing_raw,
            )
            for feat in missing_raw:
                df[feat] = np.nan

        # --- Step 2: Feature Engineering ---
        try:
            df = build_features(df)
        except (ValueError, KeyError) as e:
            logger.error(
                "Feature engineering gagal: %s. "
                "Mengembalikan prediksi fallback.",
                e,
            )
            return self._fallback_prediction(raw_features, str(e))

        # --- Step 3: Handle NaN via imputer ---
        if self._imputer_vals is not None:
            for col, val in self._imputer_vals.items():
                if col in df.columns and df[col].isna().any():
                    df[col] = df[col].fillna(val)

        # Fill remaining NaN dengan 0 (last resort)
        feature_cols = [
            f for f in self._feature_names if f in df.columns
        ]
        remaining_missing = [
            f for f in self._feature_names if f not in df.columns
        ]
        for feat in remaining_missing:
            df[feat] = 0.0
            logger.warning("Fitur '%s' tidak ada — set ke 0.0.", feat)

        df[self._feature_names] = df[self._feature_names].fillna(0.0)

        # --- Step 4: Prediksi ---
        X = df[self._feature_names].values.astype(np.float32)

        # Raw probability dari XGBoost
        p_raw = float(self._base_model.predict_proba(X)[:, 1][0])

        # Calibrated probability via Isotonic (atau Platt fallback)
        if hasattr(self._platt_model, "predict_proba"):
            p_cal = float(
                self._platt_model.predict_proba(
                    np.array([[p_raw]])
                )[:, 1][0]
            )
        else:
            p_cal = float(
                self._platt_model.predict(
                    np.array([p_raw])
                )[0]
            )

        # Clip ke [0.30, 0.72] berdasarkan reality-accurate backtest
        p_cal = float(np.clip(p_cal, 0.30, 0.72))
        p_raw = float(np.clip(p_raw, 0.0, 1.0))

        # --- Confidence tier ---
        if p_cal >= 0.65:
            confidence = "HIGH"
        elif p_cal >= 0.45:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        result = {
            "p_win": p_cal,
            "p_win_raw": p_raw,
            "confidence": confidence,
            "feature_vector": X[0].tolist(),
            "missing_features": missing_raw + remaining_missing,
            "model_version": self._model_version,
        }

        logger.debug(
            "Prediction: P(WIN)=%.4f (raw=%.4f) | confidence=%s",
            p_cal, p_raw, confidence,
        )

        return result

    # ------------------------------------------------------------------
    # Calculate EV
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_ev(
        p_win: float,
        entry_odds: float,
    ) -> float:
        """
        Hitung Expected Value untuk satu taruhan.

        Formula:
            EV = P(WIN) × ((1 / entry_odds) - 1) - (1 - P(WIN))

        Dimana:
            - (1 / entry_odds) - 1 = payout multiplier jika menang
            - (1 - P(WIN)) = probabilitas kalah (= kehilangan stake)

        Edge cases:
            - entry_odds ≤ 0   → return -inf
            - NaN input         → return NaN
            - p_win di luar [0,1] → di-clip

        Args:
            p_win:       Probabilitas menang terkalibrasi.
            entry_odds:  Odds masuk dari CLOB (0.01 - 0.99).

        Returns:
            Expected Value (float).
        """
        # Handle NaN
        if np.isnan(p_win) or np.isnan(entry_odds):
            return float("nan")

        # Handle invalid odds
        if entry_odds <= 0.0:
            return float("-inf")

        # Clip p_win
        p_win = float(np.clip(p_win, 0.30, 0.72))

        # Fee/slippage accounting
        POLYMARKET_FEE = 0.02
        SLIPPAGE_ESTIMATE = 0.005

        # EV = P(WIN) × net_payout_if_win - P(LOSE) × stake
        payout_multiplier = ((1.0 / entry_odds) - 1.0) * (1.0 - POLYMARKET_FEE - SLIPPAGE_ESTIMATE)
        ev = p_win * payout_multiplier - (1.0 - p_win)

        return float(ev)

    # ------------------------------------------------------------------
    # Full Gate Decision
    # ------------------------------------------------------------------

    def evaluate_signal(
        self,
        raw_features: Dict[str, Any],
        entry_odds: float,
        ev_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end signal evaluation: predict + EV + decision.

        Mengembalikan keputusan PASS/REJECT beserta semua metrik
        pendukung untuk logging dan audit.

        Args:
            raw_features: Fitur mentah dari signal engine.
            entry_odds:   Odds masuk target.
            ev_threshold: EV minimum. Default dari EV_CFG.

        Returns:
            Dict berisi decision, p_win, ev, dan detail.
        """
        if ev_threshold is None:
            ev_threshold = EV_CFG.ev_threshold

        # ── HARD GATE: Reject immediately if critical features are missing ──
        # These are the RAW inputs that feed the 5 SELECTED_FEATURES.
        # obi_vol_interaction is ENGINEERED (obi_value × vol_percentile),
        # so we check its parents instead.
        # If ANY is missing or NaN, the model output is meaningless.
        # p_win=0.0 guarantees EV=-1.0 → never executes.
        REQUIRED_FEATURES = [
            "entry_odds", "contest_urgency", "depth_ratio",
            "tfm_value", "obi_value", "vol_percentile",
        ]
        missing_critical = []
        for feat in REQUIRED_FEATURES:
            val = raw_features.get(feat)
            if val is None:
                missing_critical.append(feat)
            else:
                try:
                    if np.isnan(float(val)):
                        missing_critical.append(feat)
                except (TypeError, ValueError):
                    missing_critical.append(feat)

        if missing_critical:
            reason = f"HARD REJECT: missing critical features: {missing_critical}"
            logger.warning("missing_features_hard_reject: %s", missing_critical)
            return {
                "decision": "REJECT",
                "reason": reason,
                "p_win": 0.0,
                "p_win_raw": 0.0,
                "ev": -1.0,
                "entry_odds": entry_odds,
                "ev_threshold": ev_threshold,
                "confidence": "REJECTED",
                "kelly_fraction": 0.0,
                "model_version": self._model_version,
                "missing_features": missing_critical,
            }

        # Predict
        prediction = self.predict_quality(raw_features)
        p_win = prediction["p_win"]

        # EV
        ev = self.calculate_ev(p_win, entry_odds)

        # Decision — FALLBACK circuit-breaker FIRST (defense-in-depth)
        # Even if P=0.0 already guarantees EV=-1.0, we hard-reject
        # FALLBACK predictions as an additional safety layer.
        if prediction["confidence"] == "FALLBACK":
            decision = "REJECT"
            reason = (
                f"HARD REJECT: feature engineering failed — "
                f"prediction is FALLBACK (error: {prediction.get('error', 'unknown')}). "
                f"Refusing to trade with degraded data."
            )
            logger.warning("FALLBACK CIRCUIT-BREAKER: %s", reason)
        elif np.isnan(ev) or not np.isfinite(ev):
            decision = "REJECT"
            reason = "EV is NaN or infinite — input data invalid."
        elif ev >= ev_threshold:
            decision = "PASS"
            reason = f"EV={ev:.4f} >= threshold={ev_threshold:.4f}"
        else:
            decision = "REJECT"
            reason = f"EV={ev:.4f} < threshold={ev_threshold:.4f}"

        # Kelly sizing (optional info)
        if decision == "PASS" and entry_odds > 0:
            kelly_fraction = self._kelly_bet_size(p_win, entry_odds)
        else:
            kelly_fraction = 0.0

        result = {
            "decision": decision,
            "reason": reason,
            "p_win": p_win,
            "p_win_raw": prediction["p_win_raw"],
            "ev": ev,
            "entry_odds": entry_odds,
            "ev_threshold": ev_threshold,
            "confidence": prediction["confidence"],
            "kelly_fraction": kelly_fraction,
            "model_version": self._model_version,
            "missing_features": prediction["missing_features"],
        }

        log_level = logging.INFO if decision == "PASS" else logging.DEBUG
        logger.log(
            log_level,
            "Gate %s: P(WIN)=%.4f | EV=%.4f | odds=%.4f | kelly=%.4f | %s",
            decision, p_win, ev, entry_odds, kelly_fraction, reason,
        )

        return result

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kelly_bet_size(
        p_win: float,
        entry_odds: float,
    ) -> float:
        """
        Hitung fractional Kelly bet size.

        Kelly% = (p × b - q) / b
        dimana b = (1/odds) - 1 (payout ratio), q = 1 - p

        Dibatasi oleh kelly_fraction dan max_kelly_pct dari config.
        """
        if entry_odds <= 0 or np.isnan(p_win) or np.isnan(entry_odds):
            return 0.0

        p = float(np.clip(p_win, 0.30, 0.72))
        q = 1.0 - p
        
        # Fee/slippage accounting untuk net odds
        POLYMARKET_FEE = 0.02
        SLIPPAGE_ESTIMATE = 0.005
        b = ((1.0 / entry_odds) - 1.0) * (1.0 - POLYMARKET_FEE - SLIPPAGE_ESTIMATE)

        if b <= 0:
            return 0.0

        full_kelly = (p * b - q) / b
        if full_kelly <= 0:
            return 0.0

        # Apply Quarter-Kelly + 3% hard cap
        quarter_kelly = full_kelly * 0.25
        capped = min(quarter_kelly, 0.03)

        return round(float(capped), 6)

    @staticmethod
    def _fallback_prediction(
        raw_features: Dict[str, Any],
        error_msg: str,
    ) -> Dict[str, Any]:
        """
        Kembalikan prediksi fallback saat feature engineering gagal.

        P(WIN) = 0.0 → menjamin EV = -1.0 untuk semua odds.

        KRITIS: Jangan pernah gunakan P=0.5 di sini!
        Pada underdog odds (misal 0.20), P=0.5 menghasilkan:
          EV = 0.5 × ((1/0.20) - 1) - 0.5 = 0.5 × 4 - 0.5 = +1.5
        Bot akan MENGEKSEKUSI trade saat sistem buta — FATAL.

        P=0.0 menjamin:
          EV = 0.0 × ((1/odds) - 1) - 1.0 = -1.0  (untuk semua odds > 0)
        → Selalu ditolak oleh EV gate.
        → Ditambah hard-reject circuit-breaker di evaluate_signal().
        """
        logger.error(
            "FALLBACK PREDICTION (P=0.0) — feature engineering GAGAL: %s. "
            "Trade akan di-REJECT secara otomatis.",
            error_msg,
        )
        return {
            "p_win": 0.0,
            "p_win_raw": 0.0,
            "confidence": "FALLBACK",
            "feature_vector": [],
            "missing_features": list(raw_features.keys()),
            "model_version": "FALLBACK",
            "error": error_msg,
        }
