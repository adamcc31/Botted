# Change Log - 2026-06-02
## Sprint: 5.5 — Dry-Run Gate Removal, V5 Data Pipeline Audit & Shadow Prediction

### Summary
Sprint 5.5 melakukan tiga blok perubahan: perpanjangan batas waktu dry-run dari 48 jam ke 720 jam (30 hari), audit mendalam terhadap V5 feature pipeline yang mengungkap ketidaksesuaian kritis antara fitur yang dicatat di CSV dan fitur aktual yang dipakai model, serta implementasi shadow prediction logging untuk sinyal yang diblokir oleh spread filter.

---

### BLOCK 1 — Dry-Run Gate & Tag Cleanup

#### 1. Timeout 48h → 720h (30 hari)
- **File:** `config/config.json` (baris 66), `main.py` (baris 533)
- **Perubahan:** `max_duration_hours: 48` → `720` di config; default hardcoded di `_dry_run_time_guard()` disamakan ke 720.
- **Alasan:** Batas 48 jam terlalu pendek untuk mengumpulkan data statistik yang bermakna sebelum gate live menyala.

#### 2. Rebranding Tag Telegram `[ALPHA V1]` → `[SYSTEM]`
- **File:** `src/telegram_notifier.py`
- **Method yang diubah:** `system_health()`, `dry_run_limit()`, `session_finished()`
- **Alasan:** Ketiga notifikasi ini adalah lifecycle system-level, bukan output dari pipeline Alpha V1. Method V1 murni (`order_execution`, `order_result`, dll.) tidak disentuh.

---

### BLOCK 2 — V5 Feature Relevance Audit (Investigasi)

#### Temuan Kritis
Model V5 aktual (`models/slingger_hunter_v5/metadata.json`) menggunakan **12 fitur CLOB-native**:
```
yes_price_t0, no_price_t0, clob_spread_t0, yes_depth_t0, no_depth_t0,
depth_imbalance_t0, price_velocity_30s, depth_trend_30s,
btc_realized_vol_prior_30m, ttr_at_signal, market_hour_utc, day_of_week
```
Sementara `dry_run_*.csv` selama ini merekam fitur-fitur V1 (`obi_value`, `tfm_value`, `odds_delta_60s`, dll.) yang **tidak ada** dalam pipeline inferensi V5. Tanpa fix ini, retrain Phase 3 akan menggunakan feature set yang salah.

#### Klasifikasi Kolom
| Kategori | Kolom |
|---|---|
| **V5_CORE** | `ttr_seconds`, `odds_yes`/`no`, `entry_odds`, `yes_price_t0`→`btc_realized_vol_prior_30m` (baru) |
| **V5_META** | `timestamp`, `market_id`, `slug`, `spread_pct`, `actual_outcome`, `confidence_score`, dll. |
| **V1_REMNANT** (dihapus) | `theoretical_exit_odds`, `theoretical_pnl`, `signal_correct`, `vol_percentile`, `odds_yes_60s_ago`, `odds_delta_60s` |
| **AMBIGUOUS** | `obi_value`, `tfm_value`, `depth_ratio`, `contest_urgency` (dipertahankan untuk backward compat) |

---

### BLOCK 3 — V5 Data Pipeline Implementation

#### A) Hapus 6 Kolom V1_REMNANT dari dry_run CSV
- **File:** `src/exporter.py` → `export_signals()` SQL query
- **Kolom yang dihapus dari SELECT:** `theoretical_exit_odds` (hardcoded 1.0), `theoretical_pnl` (LEAKAGE), `signal_correct`, `vol_percentile`, `odds_yes_60s_ago`, `odds_delta_60s`
- **Catatan:** Kolom-kolom ini tetap ada di tabel SQLite (tidak di-DROP) untuk backward compatibility dengan historical data.

#### B) Tambah 9 Kolom V5 Aktual ke dry_run CSV & Database
- **File:** `src/database.py`, `src/dry_run.py`, `src/exporter.py`
- **Kolom baru di `SignalRecord`:** `yes_price_t0`, `no_price_t0`, `clob_spread_t0`, `yes_depth_t0`, `no_depth_t0`, `depth_imbalance_t0`, `price_velocity_30s`, `depth_trend_30s`, `btc_realized_vol_prior_30m`
- **Perubahan `record_signal()`:** Signature ditambah parameter `v5_features: dict = None`; INSERT statement diperbarui menghapus 5 field V1 dan menambah 9 field V5.
- **Railway migration:** Dilakukan manual via `ALTER TABLE signals ADD COLUMN` untuk 9 kolom baru. Semua `[OK]`.

#### C) Shadow Prediction Logging
- **File baru:** `data/exports/{session_id}/dry_run_shadow_{session_id}.csv` (per sesi)
- **Trigger:** Hanya aktif di `dry-run` mode, saat spread filter mengembalikan `SKIP` atau `WAIT`
- **Method baru di `main.py`:**
  - `_build_v5_features_from_clob(market, clob_state)` — ekstrak 9 fitur V5 dari CLOBState yang tersedia tanpa memerlukan oracle price
  - `_run_shadow_prediction(market, clob_state, oracle_price, spread_result, v5_feats)` — jalankan model V5 untuk YES dan NO side, tulis ke shadow CSV
- **Method baru di `src/exporter.py`:** `record_shadow(record: dict)` — append ke CSV dengan schema lengkap 27 kolom
- **Jaminan isolasi:** Shadow prediction tidak mengubah jalur eksekusi trade apapun. Diproteksi oleh `try/except` sehingga error tidak propagate ke trading loop.
- **Shadow CSV schema (27 kolom):**
  ```
  timestamp, market_id, slug, ttr_seconds, spread_pct, spread_blocked_reason,
  yes_price_t0, no_price_t0, clob_spread_t0, yes_depth_t0, no_depth_t0,
  depth_imbalance_t0, price_velocity_30s, depth_trend_30s, btc_realized_vol_prior_30m,
  ttr_at_signal, market_hour_utc, day_of_week,
  shadow_signal_yes, shadow_prob_yes, shadow_kelly_yes, shadow_tier_yes,
  shadow_signal_no, shadow_prob_no, shadow_kelly_no, shadow_tier_no,
  actual_outcome
  ```

---

### Files Changed
| File | Perubahan |
|---|---|
| `config/config.json` | `max_duration_hours` 48 → 720 |
| `main.py` | `_dry_run_time_guard` default 720, spread filter blocks + v5_features, `_build_v5_features_from_clob()`, `_run_shadow_prediction()` |
| `src/database.py` | +9 kolom V5 ke `SignalRecord` |
| `src/dry_run.py` | `record_signal()` signature + INSERT V5, hapus V1 fields |
| `src/exporter.py` | `export_signals()` SQL clean, `record_shadow()` method baru |
| `src/telegram_notifier.py` | 3 tag `[ALPHA V1]` → `[SYSTEM]` |

### Railway Migration
```
[OK] Added column: yes_price_t0
[OK] Added column: no_price_t0
[OK] Added column: clob_spread_t0
[OK] Added column: yes_depth_t0
[OK] Added column: no_depth_t0
[OK] Added column: depth_imbalance_t0
[OK] Added column: price_velocity_30s
[OK] Added column: depth_trend_30s
[OK] Added column: btc_realized_vol_prior_30m
Migration complete.
```
Database: `/app/data/trading.db` (SQLite on Railway volume)

### Status: DEPLOYED — Tidak memerlukan restart. Bot mengisi kolom V5 mulai sinyal berikutnya.
