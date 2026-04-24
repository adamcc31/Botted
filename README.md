# Polymarket BTC Trading Bot: Latency Arbitrage & Slow Skew Engine

![Version](https://img.shields.io/badge/version-2.0.0--serverless-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-green)
![Architecture](https://img.shields.io/badge/architecture-No--DB%20%7C%20Flat%20Files-orange)
![Status](https://img.shields.io/badge/status-production--ready-success)

Dokumentasi teknis ini merangkum arsitektur, filosofi strategi, dan panduan operasional dari bot trading otomatis Polymarket. Bot ini telah dirancang ulang sepenuhnya untuk menjadi sistem **Serverless & Stateless**, yang mengeksploitasi inefisiensi kecepatan (*Latency Arbitrage*) pada market biner Bitcoin.

---

## 1. Filosofi Strategi: Deteksi "Slow Skew"

Bot ini **BUKAN** bot peramal arah harga (Directional Trading) yang menebak apakah harga Bitcoin akan naik atau turun dalam 4 jam ke depan. Sebaliknya, bot ini beroperasi di bawah payung **Arbitrase Kecepatan (Latency/Micro-Mispricing Arbitrage)**.

### Mekanisme Keuntungan (The Alpha):
1. **The Anchor (Binance):** Harga Spot Binance bereaksi dalam hitungan milidetik (*ultra-low latency*).
2. **The Lag (Polymarket CLOB):** Market Maker (algoritma penyedia likuiditas) di Polymarket sering kali membutuhkan waktu 30-60 detik untuk memperbarui harga order book (CLOB) setelah terjadi pergerakan ekstrem di Binance.
3. **The Math (Fair Probability):** Setiap detik, bot mengkonversi harga Binance Spot menjadi *Probabilitas Objektif* menggunakan model **Black-Scholes Digital Option**.
4. **The Execution:** Jika Binance melonjak drastis dan peluang *Fair* sebuah token YES adalah 48% ($0.48), tetapi Polymarket masih tertinggal dan menjual token tersebut seharga 30% ($0.30), bot akan secara instan memborong token tersebut.

Keuntungan didapatkan dari "kesenjangan respons pasar" (Slow Skew), bukan dari prediksi masa depan murni.

---

## 2. Arsitektur Sistem (Serverless & Lightweight)

Pemutakhiran arsitektur besar-besaran telah dilakukan agar bot ideal untuk *deployment Cloud (Railway/VPS)* dengan intervensi nol.

- **Non-Database (No-DB):** Proyek ini **100% bebas dari Database** (Tidak ada SQLite, tidak ada PostgreSQL). Semua data persisten diubah menjadi sistem *Flat-File CSV* memori rendah.
- **Dynamic Market Discovery:** Bot tidak bergantung pada satu URL/Slug market mati. Modul `market_discovery.py` akan otomatis menyapu seluruh API Polymarket, menemukan market BTC biner yang likuid, mengunci durasi (1-12 jam), mengeksekusi, dan otomatis berputar (*rotate*) ke market baru ketika market lama kedaluwarsa.
- **Asynchronous Execution:** Menggunakan `asyncio` untuk `BinanceFeed` (WebSocket) dan `CLOBFeed` (API Polling HTTPX) berjalan paralel dalam skala milidetik.

---

## 3. Alur Kerja (Lifecycle)

1. **Market Scanning:** Bot secara otomatis mendeteksi market "Bitcoin Price Action" di Polymarket dengan likuiditas minimum dan *spread* wajar ($>10 USD depth, <7% vig).
2. **Real-time Synchronization:** Menarik data WebSocket Binance dan *Order Book* Polymarket secara terus-menerus.
3. **Mispricing Calculation:** `FairProbabilityEngine` membandingkan probabilitas matematis dari Spot saat ini terhadap harga riil `ask/bid` CLOB.
4. **Signal & Risk Gate:** Saat deteksi *mispricing* melebihi ambang batas *safety margin* (>5%), `signal_generator.py` menerbitkan sinyal `BUY_YES` atau `BUY_NO`.
5. **Execution (Dry-Run / Live):** Bot mendaftarkan trade secara simulasi (Dry-Run) atau mengeksekusinya di relayer *on-chain* Polymarket secara tanpa gas (*gasless via Polygon*).

---

## 4. Observabilitas & Notifikasi Telegram (6-Hour Loop)

Karena bot beroperasi 24/7 di *background* VPS (Serverless / Docker), sistem pemantauan dialihkan secara penuh ke ekspor CSV dan notifikasi Telegram.

1. **Exporter CSV (`src/exporter.py`):**
   - Setiap keputusan, sesi, dan rekaman CLOB (`clob_log.csv`) serta Trades (`trades.csv`) disimpan ke `data/exports/<session_id>/`.
   
2. **Telegram 6-Hour Automated Reporter:**
   - Bot memiliki skema pengulangan otomatis (Tiap 6 Jam / `21600 detik`).
   - Merangkum metrik (`Win Rate`, `PnL`, Porsi *Signals*, Total Trade).
   - **Sinkronisasi File:** Setiap 6 jam, bot tidak hanya mengirim teks pesan, tetapi **secara fisik mengunggah dan mengirim file `trades.csv`** terbaru langsung ke Telegram Anda. Ini memastikan *review* performa bisa dilakukan dari ponsel tanpa mengakses server.

---

## 5. Deployment Guide (Railway / Docker VPS)

Sistem telah di-*containerize* standar lewat `.env` dan `Dockerfile` minimal.

### Environment Variables Wajib (`.env`)
```ini
POLYMARKET_PRIVATE_KEY=0x... (Wallet Polygon)
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

### Build & Run
Bot dikonfigurasi melalui `railway.toml` untuk menjamin otomatisasi.

**Local / Manual Docker:**
```bash
# Instalasi Lightweight
pip install -r requirements.txt

# Eksekusi Opserver Dry-Run Mode
python main.py --mode dry-run
```

**Aturan `max_market_vig`:** Batasan toleransi ketebalan (*spread width*) standar diset ke **0.07 (7%)**. Angka empiris market riil Polymarket adalah ~0.1%. Batasan 7% hanya digunakan sebagai *Circuit Breaker* seandainya API market membanderol harga error. 

---

## 6. Pengembangan Masa Depan (Riset HF Ticks)

Bukti pendukung *Latency Arbitrage* kini disematkan di folder dataset HF (`dataset/btc_5m_hf_ticks.parquet` berisi ~13,3 Juta baris).

Langkah riset optimasi berikutnya adalah merancang sistem uji balikan (*backtest*) murni yang dapat mensimulasikan order dari *tick* historis guna mendeteksi jarak keterlambatan spesifik (milidetik) optimal yang menghubungkan lonjakan WebSocket Binance dan resync API Polymarket.

---
*Arsitektur & Konsep diperbarui: April 2026. Dioptimalkan untuk "High-Frequency Dry Run" Server Deployment.*
