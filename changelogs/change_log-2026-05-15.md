# Change Log - 2026-05-15
## Sprint: V5 Architecture Restoration & Risk Equilibrium

### Summary
This session focused on the critical forensic audit, data restoration, and architectural hardening of the **Slingger Hunter V5** trading engine. The engine was transitioned from a failing production state to a verified, data-backed architecture with rigorous risk controls.

### Key Changes
#### 1. Volatility Paradox Fix (Feature Engineering)
- **Problem:** Mismatch between training volatility (Polymarket internal) and live inference (Binance BTC RV).
- **Solution:** Re-implemented `btc_realized_vol_prior_30m` in `main.py` using internal Polymarket order book standard deviation.
- **Result:** Feature importance rank jumped to #3 in the model, significantly improving prediction accuracy.

#### 2. Full Data Restoration & Merging
- **Extraction:** Pulled 140k+ rows of missing log data (May 9-15) from Railway production volume.
- **Merge:** Updated `dataset/clob_log/CLOB_MASTER.csv` to 302,440 rows.
- **Cleanup:** Sanitized dataset from "phantom columns" (`btc_vs_strike_pct`, `obi_value`, `rv_value`).

#### 3. Model Retraining (V5.0.1)
- **Performance:** Retrained XGBoost model with 5-Fold GroupKFold.
- **Metrics:** Achieved **0.7278 OOF AUC** (Target was >0.65).
- **EV Projection:** Confirmed **25.8% Expected Value** at the 0.65 probability threshold.

#### 4. Risk Equilibrium & Hardening
- **Half-Kelly:** Applied a 0.5x multiplier to Kelly sizing in `src/utils.py` to dampen drawdown volatility.
- **Hard Ceiling:** Implemented a **$20.0 absolute max stake** per trade to prevent slippage on thin 5m markets.
- **Threshold Lock:** Enforced a minimum **0.65 probability threshold** in `metadata.json`.

#### 5. Stability & Memory Fixes (OOM & Capital Reset)
- **OOM Prevention:** Implemented surgical cleanup of state dictionaries (`_shadow_scalps` and `_completed_markets`) to prevent unbounded memory growth in long sessions on Railway.
- **Capital Sync:** Resolved the fatal divergence between V5 and V1 capital. V5 now dynamically pulls from `self._dry_run.capital` (canonical source) to ensure all dry-run statistics are unified and consistent.

#### 6. Production Safety (Dry-Run Mandate)
- **Isolation:** Explicitly tagged V5 logic with `[MANDAT-DRYRUN]` in `main.py`.
- **Forward-Test:** V5 is now strictly locked in shadow-entry mode; no real capital will be deployed until the end-of-month validation.

### Technical Artifacts
- **Model:** `models/slingger_hunter_v5/model.json`
- **Metadata:** `models/slingger_hunter_v5/metadata.json`
- **Core Logic:** `main.py` -> `_run_slingger_v5`, `src/utils.py` -> `compute_position_size`.

### Status: LOCKED & READY FOR DEPLOYMENT
