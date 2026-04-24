# Polymarket BTC Trading Bot

## Apa Itu Bot Ini?

Bot ini dirancang secara khusus untuk mengeksekusi trading pada pasar (markets) *binary options* Polymarket, berfokus utama pada *market* **Bitcoin (BTC) UP/DOWN** berdurasi pendek, seperti interval 5-menit dan 15-menit. Secara esensi, fungsi bot adalah memindai pasar untuk mencari *mispricing* (kesalahan penetapan harga) antara *fair probability* yang diamati secara real-time dari data mikrostruktur dan harga riil *order book* yang ditawarkan di bursa Polymarket.

Strategi inti bot ini bekerja dengan mendeteksi adanya **sentiment-microstructure divergence**. Fenomena ini terjadi ketika pasar/kerumunan (crowd) Polymarket mengalami *Fear Of Missing Out* (FOMO) ekstrem ke salah satu probabilitas arah harga (misalnya, rasio *odds* bergerak secara agresif menjadi 70:30 atau lebih), sedangkan bot melakukan verifikasi sentimen tersebut melalui struktur *order book* riil (book imbalance, trade flow) serta kondisi volatilitas dari bursa utama, yakni Binance. Jika analisis mikrostruktur Binance tidak menunjukkan bukti dinding transaksi jual/beli yang menjustifikasi probabilitas ekstrem tersebut, bot mendeteksi *mispricing* dan secara cerdas mengambil peluang pada arah sebaliknya yang lebih masuk akal.

Ini bukan dirancang sebagai strategi *high win-rate* konvensional, melainkan berfokus penuh pada **asymmetric payout strategy**. Karena *trading* dikunci pada harga *odds* yang sangat rendah pada posisi underdog (misal, 0.20 - 0.35), pembayaran atas tebakan benar yang dihasilkan sangat besar. Bot tidak membutuhkan rasio menang yang absolut. Memenangkan 1 dari 3 transaksi (win rate ~33%) dalam kondisi rentang *odds* 0.20 akan tetap menjaga portofolio berada dalam *Expected Value* (EV) dan *Return on Investment* (ROI) yang positif. Bot mengeksekusi *pipeline* perputaran yang beroperasi penuh (*fully automated*): mulai dari *market discovery*, ekstraksi fitur-sinyal, validasi masuk (entry validation), paper/live trading, hingga proses penyelesaian hasil tutup pasar.

## Bagaimana Prediksi Dibuat?

Sinyal prediksi bot dibangun menggunakan jalur pipa pemrosesan dari ujung-ke-ujung (*end-to-end*) sebagai berikut:

**2.1 Market Discovery**
- Bot secara periodik memantau **Gamma API** Polymarket untuk menemukan kontrak opsi (*active markets*) berjangka pendek.
- Penyaringan ketat berjalan dengan memantau URL slug secara dinamis, menggunakan variabel `POLYMARKET_SLUG_PREFIX` (dengan default standar: `btc-updown-5m`).
- Ekstraksi informasi target *Strike Price* tidak mengandalkan harga buka Binance, namun diambil mutlak secara parsing dari teks pertanyaan (market question text) maupun target API *metadata* sehingga nilainya selalu konstan.
- *Time-To-Resolution* (TTR) dihitung secara mutlak dan menentukan apakah sebuah *market* memasuki zona jendela *entry* (*entry window*) yang valid.

**2.2 Dual Feed Price Alignment**
- Menggunakan arsitektur keamanan dua sumber data harga (*Dual Feed*): Pertama berasal dari **Binance direct WS** (`wss://stream.binance.com`), dan kedua berasal dari **Chainlink via Polymarket RTDS** (`wss://ws-live-data.polymarket.com`).
- Mengapa diperlukan dua sumber? *Market flow* dan fitur likuiditas paling akurat dibaca dari bursa besar seperti Binance. Namun, *settlement* harga akhir dari market BTC UP/DOWN dilakukan mutlak berdasarkan agregasi **Chainlink**. Bot memerlukan RTDS Chainlink sebagai penyeimbang nilai *strike relative* dan *settlement-aligned price*.
- Bot menggunakan **Spread Filter** (*pre-entry gate*) yang dengan tegas menolak *entry* masuk jika selisih / *divergence* persentase harga Binance dan Chainlink melampaui toleransi, guna menekan kerugian dari *basis risk*.

**2.3 Feature Engine**
Sebanyak 24 kelompok fitur dihitung seketika (real-time):
- **Order flow features**: Mencakup Order Book Imbalance (*OBI*), Trade Flow Momentum (*TFM*), serta metrik rasio kedalaman *depth_ratio* ditarik dan dirangkum dari rekaman transaksi Binance.
- **Volatility features**: Mencakup Realized Volatility (*RV*) serta indikator posisi volatilitas persentil (*vol_percentile*), memanfaatkan kline data riwayat historis 1-menit dari Binance.
- **Strike-relative features**: Mencakup *strike_distance_pct* dan *contest_urgency*, angka ini dikalkulasi dengan **hanya bersumber** pada patokan harga spot *Chainlink oracle*.
- **CLOB features**: Likuiditas di Polymarket dikalkulasi dari *Central Limit Order Book* (CLOB) *WebSocket*. Meliputi *odds_yes_mid*, struktur bukaan harga (*spread*), hingga penghitungan margin keamanan (*market_vig*).

**2.4 Fair Probability (q_fair)**
- Prediksi mendasarkan formulanya pada pendekatan **Black-Scholes digital option** khusus instrumen *binary market*.
- Perhitungan `q_fair = Φ(d2)` mendefinisikan *fair probability*. Nilai dasar harga spot terkini (*spot S*) menggunakan *oracle Chainlink price* agar terkalibrasi lurus dengan penutupan kontrak. Tidak akan menggunakan Binance sebagai substitusi!
- Sebagai pasangannya, level turbulensi alias volatilitas (σ) akan terus bersandar dari observasi statistik fluktuasi 1-menit yang tertangkap dari sistem Binance.

**2.5 Signal Decision**
- Bot menyulut peringatan untuk aksi eksekusi `BUY_UP`, `BUY_DOWN`, atau menghindar (`ABSTAIN`).
- Ambang batas keyakinan (threshold confidence) digunakan: perpaduan sinyal model (`P_model`) lalu diurangi *uncertainty buffer* harus mampu menyisakan celah marjin profit (margin of safety) atas kondisi *odds* yang ada pada *orderbook*.
- Bot mengevaluasi kedua arah (`BUY_UP` dan `BUY_DOWN`) secara independen dalam satu siklus evaluasi. `edge_up` dan `edge_down` dihitung dari `q_fair` dan `(1 - q_fair)` masing-masing dibandingkan dengan harga aktual CLOB. Hanya arah dengan *edge* tertinggi yang dieksekusi, dan hanya satu posisi yang diizinkan per `market_id` dalam satu waktu (*One-Bet Rule*).
- Ada aturan proteksi **Rotation lock** & **Dwell time**, bot tidak akan loncat agresif (*rotate*) apabila jeda pendinginan *cooldown* antarmarket dalam 15 menit berjalan tidak wajar.

## Kapan Bot Memasang Bid?

Bot ini ditata sangat diskriminatif terhadap order. Bid (Order masuk) **hanya akan terjadi apabila KESELURUHAN TUJUH KONDISI berikut terpenuhi** di waktu bersamaan:

1. **TTR dalam window optimal**: Sisa masa aktif (*Time-To-Resolution*) berada sangat presisi pada cakupan limit batas *entry window* yang telah dirancang.
2. **Margin of Safety & Odds Imbalance**: Margin sisa di salah satu sisi setelah pengkalkulasian sinyal terbukti jauh melebihi limit batas minimal tingkat keamanan (`margin_of_safety`).
3. **Spread filter PASS**: Selisih diferensiasi perbedaan harga `|Binance - Chainlink| / Chainlink` harus patuh berada lebih kecil (atau sama dengan) batas normal penyimpangan yang diizinkan (`SPREAD_THRESHOLD_NORMAL_PCT`).
4. **Oracle price valid**: Feed Chainlink dari Polymarket secara absolut terbukti tersedia, bukan dari data kusam (*stale*) dan respons API berada dalam standar latensi aktual (`< stale_threshold_seconds`).
5. **Signal confidence mencukupi**: Rekomendasi probabilitas mesin melebihi toleransi risiko *deadband* harga terengah opsi yang tersedia.
6. **Bebas Market Lock**: Mekanisme pembekuan rotasi tidak sedang bekerja; waktu adaptasi di pasar terpenuhi (`dwell_minutes`).
7. **One-Bet Rule**: Tidak ada posisi aktif atau sinyal pending untuk `market_id` yang sama. Jika dua sinyal berlawanan muncul secara simultan (situasi reversal extreme), hanya sinyal dengan `P_model` lebih tinggi yang dieksekusi.

Sebaliknya, bot akan menghindar dari bid dan berstatus penonton jika:
- **ABSTAIN**: Tidak ada *edge* / kekuatan prediksi, pasar terlalu lesu (*Liquidity Block*), atau posisi teoretis harga bergerak terlalu berdempetan dengan pergerakan normal (*No Trade Zone*).
- **SKIP (Spread)**: Terdapat ketidakselarasan ekstrem pada indikator harga *feed*. Selisih antara harga pusat bursa Binance vs Oracle Chainlink dinilai membahayakan presisi.
- **SKIP (Oracle)**: *Chainlink price* tidak dapat diproses atau mendadak mati. (Bot tidak akan mengambil risiko meniru data harga pengganti secara paksa!).
- **Rotation lock**: Kondisi peralihan antarmarket terjadi sebelum rentang adaptasi (dwell time) terpenuhi.

## Sumber Data

| Sumber | Data | Digunakan Untuk | Endpoint |
|---|---|---|---|
| **Binance WebSocket** | L2 order book, aggTrades, OHLCV (1m & 15m) | *Order flow features*, pergerakan *depth* buku asimetris, dan kalkulasi volatilitas (*volatility*). | `wss://stream.binance.com` |
| **Polymarket RTDS** | Chainlink BTC/USD *real-time* | Patokan dasar pergerakan harga (*settlement-aligned spot price S*) bagi *Fair Probability*. | `wss://ws-live-data.polymarket.com` |
| **Polymarket CLOB WS** | *Order book* instrumen aset YES/NO | Menilai *odds* langsung (harga tawaran entry yang aktual), margin perantara, & volume kedalaman. | `wss://ws-subscriptions-clob.polymarket.com/ws/market` |
| **Gamma API** | *Active markets*, info relasi *strike price*, & usia aset | Mengelola antrean status penemuan pasar (Market Discovery). | `https://gamma-api.polymarket.com` |
| **Chainlink Oracle** | BTC/USD agregat yang menentukan hasil | Menjalin patokan penyelesaian taruhan. | *(Via RTDS)* |

> [!IMPORTANT] 
> Seluruh instrumen *binary options* Polymarket yang ditargetkan (terutama BTC) ditarik resolusi *settlement* akhirnya merujuk pada agregat **Chainlink oracle**, bukan indikator Binance semata. Oleh karena itu, seluruh porsi perhitungan jarak dari sasaran (seperti: *strike-relative calculations* dan `q_fair`) di program ini **telah dipaksa menggunakan standar rujukan *Chainlink price***, bukan Binance.

## Metrik Kunci

Performa sistem dilaporkan secara mendetail, dengan rekam agregat logistik laporan operasional via Telegram dan penyimpanan terstruktur.

**Per Session (Diarsip dalam tabel `signals` pada SQLite):**
- **Total signals evaluated**: Kuantitas siklus proses evaluasi keputusan (*decision loop*).
- **Breakdown**: Klasifikasi komposisi sinyal terhitung secara rinci (Aktivasi ke posisi `BUY_UP` / `BUY_DOWN` / `ABSTAIN` / `SKIP`).
- **SKIP breakdown**: Statistik alasan kegagalan kondisi teknis eksekusi, dibelah antara penolakan batas deviasi (*Spread filter*) dengan penolakan akses *API oracle* mandek (*Oracle unavailable*).
- **Win rate**: Tingkat rasio kemenangan tebakan riil bot diuji secara mandiri (terhitung mutlak untuk record parameter `signal_correct IN ('TRUE', 'FALSE')` pada akhir jatuh tempo).
- **Average spread_pct**: Kondisi metrik penyimpangan selisih rujukan harga (Binance vs Chainlink) selama fase penarikan aset.
- **Binance fallback count**: Pencatatan darurat ke penarikan alternatif (Kondisi absolut sistem yang sehat di *environment* Polymarket mengindikasikan status ini memegang rekor nilai **0**).

**Per Trade (Dikomputasi ke dalam output rekam jejak berkas `trades.csv`):**
- Posisi harga tebakan taruhan awal (*Entry odds*), pencapaian akhir di jatuh tempo (*exit odds*), & rekapitulasi performa probabilitas simulasi hasil PnL teoretis (*Theoretical PnL*).
- Indikator asal nilai penentu *settlement_price_source* (`CHAINLINK` vs `BINANCE_FALLBACK`).

**Penjelasan Expected Value (EV):**
Mesin pengambil keputusan menerapkan konsep *asymmetric payout* secara murni. Jika probabilitas *odds* suatu instrumen berada di rasio harga **0.20** (20 sen pembagian rasio probabilitas), maka kemenangan tebakan itu berpotensi meledakkan hadiah murni senilai pengkalian keuntungan bersih 4x sampai 5x (*payout*).
Meskipun bot meleset dua siklus awal taruhan (`-2 poin unit kerugian`), keberhasilan satu *trade* siklus kemenangan membalikkan portofolio ke akumulasi laba bersih (+4 poin unit). Strateginya tidak diukur dalam parameter mengejar probabilitas tembakan memenangkan tebakan, namun berpusat absolut pada penemuan celah deviasi **mispricing detection** dari dinamika kerumunan di polymarket.

## Tech Stack

| Komponen | Teknologi | Alasan |
|---|---|---|
| **Core bot** | Python (3.x) | Bahasa standar yang mendukung kerangka pemrosesan dan ekstraksi komputasi numerik saintifik (*polars*, *numpy*, *pandas*). |
| **Async runtime** | `asyncio` | Skalabilitas performa untuk memelihara lusinan soket koneksi antarmuka ganda API secara serentak *Concurrent WebSocket connections*. |
| **Database** | SQLite via SQLAlchemy (`aiosqlite`) | Konsep basis data terselip ringkas di penyimpanan yang bersifat tanpa penanganan terpusat rumit, namun sangat kaya kueri analitik. |
| **Data validation** | Pydantic (v2) | Pengetatan paksaan terhadap aturan kontrak periksa skema antar-alur (*Schema enforcement*). |
| **Logging** | `structlog` | Pengiriman informasi pemrosesan mesin komprehensif ke alur JSON logging pada eksekusi produksi (Railway). |
| **Exchange feed** | `websockets` + `py-clob-client` | Menjembatani *push feed* seketika order harga langsung dari inti infrastruktur Binance + Polymarket WS. |
| **Notifications** | Telegram Bot API | Monitoring intervensi pengawasan pelaporan langsung ke perangkat pengguna (Remote monitoring). |
| **Deployment** | Railway (EU West) | Pemilihan instansi dekat pangkalan letak geografi *datacenter server* infrastruktur pusat Binance Europe tanpa halangan pemblokiran IP koneksi jaringan ISP. |

## Arsitektur Sistem

Berikut visualisasi aliran pergerakan lalu lintas data secara menyeluruh melalui alur sistem operasional `main.py`:

```text
[Binance WS] ──────────────────────────────┐
  depth20, aggTrade, kline_1m, kline_15m   │
                                           ▼
[Polymarket RTDS WS] ──────────► [DualFeed] ──► [SpreadFilter]
  crypto_prices_chainlink                  │         │
                                           │    PASS / SKIP
[Polymarket CLOB WS] ──────────► [CLOBFeed]│         │
  book (YES/NO odds)                       │         ▼
                                           └──► [FeatureEngine]
[Gamma API] ────────────────────► [MarketDiscovery]   │
  active markets, strike, TTR                         │
                                                      ▼
                                               [FairProbability]
                                               q_fair = Φ(d2)
                                                      │
                                                      ▼
                                               [SignalGenerator]
                                               BUY_UP / BUY_DOWN / ABSTAIN
                                                      │
                                                      ▼
                                               [DryRunEngine / LiveEngine]
                                               Paper trade / Live order
                                                      │
                                                      ▼
                                               [SQLite] ──► [Exporter]
                                               signals.db    CSV + Telegram
```

## Konfigurasi

Semua konfigurasi statis ditaruh pada file `config/config.json`.
Sedangkan hal yang berkaitan kredensial ditempatkan di `.env`.

Tabel referensi dasar rujukan kunci (Mengacu ke parameter model kerangka `.env.example` aktual):

| Variable | Default | Deskripsi |
|---|---|---|
| `BINANCE_API_KEY` | - | Kunci API Binance. (Baca izin terbatas: Baca saja / READ-ONLY sufficient) |
| `BINANCE_API_SECRET` | - | Rahasia kunci otentikasi API Binance. |
| `POLYMARKET_PRIVATE_KEY` | - | Ekstrak *Private Key Wallet Polygon* terdaftar untuk verifikasi tanda tangan otomatis (*CreateOrDeriveApiKey*) untuk otorisasi *CLOB API*. |
| `POLY_BUILDER_API_KEY` | - | Kredensial *Builder L2 (Settings Polymarket)* relayer mutlak tanpa bea gas (*gasless*). |
| `POLY_BUILDER_SECRET` | - | Rahasia builder Polymarket. |
| `POLY_BUILDER_PASSPHRASE` | - | Penyelarasan kerahasiaan ekstensi pengaman akses builder. |
| `DATABASE_URL` | - | *URL connection* untuk menyambung basis data. (Jika kosong, terpasang default SQLite di `./data/trading.db`). |
| `LIVE_MODE` | `false` | Kondisi operasi keamanan yang menahan bot supaya beraktivitas dengan aset kosong simulasi (Kunci aktivasi). |
| `LOG_LEVEL` | `INFO` | Rujukan indikasi filter tingkat detail *output log*. |
| `ENVIRONMENT` | `development` | Kondisi sistem *cloud*. Jika di atur ke `production`, mode *Structured Log (JSON)* menyala. |
| `QUOTAGUARD_URL` | - | *Proxy IP Statik Railway*. Tidak diperlukan jika deploy di Railway EU West. Hanya dibutuhkan jika menjalankan bot dari IP Indonesia langsung. |
| `POLYMARKET_SLUG_PREFIX` | `btc-updown-5m` | Pilihan filter *tag/slug market* dari Gamma API yang difokuskan pada pengintaian durasi. |
| `SPREAD_THRESHOLD_NORMAL_PCT`| `0.05` | Batas wajar selisih penyimpangan rujukan harga antara Binance dan Chainlink. |
| `SPREAD_THRESHOLD_ELEVATED_PCT` | `0.12` | Ambang toleransi akhir (elevasi deviasi). Apabila lewat batas maksimum, bot menunda evaluasi sinyal secara instan (Skip). |

## Cara Menjalankan

Berikut eksekusi dasar rutinitas terminal yang dilakukan di lingkup environment pengembangan lokal maupun implementasi Railway Cloud.

```bash
# 1. Install seluruh pustaka *requirements*
pip install -r requirements.txt

# 2. Persiapkan duplikat kerangka perantara environment
cp .env.example .env
# Tambahkan identitas Private Key, Secret dan Proxy (Sesuai kebutuhan platform) pada .env

# 3. Menjalankan skema uji coba algoritma / Paper Trading (Aman - Tidak menanggung beban finansial)
python main.py --mode dry-run

# 4. Meloloskan bot dalam transaksi aset Live Trading riil (Lolos gerbang ganda)
python main.py --mode live --confirm-live
```

**Informasi Operasional Khusus:**
- Konfigurasi penyambungan terhadap bursa besar seperti *Binance API* terindikasi terkena restriksi akses blokir langsung oleh IP Komersial dari penyedia wilayah operasional negara Anda (Indonesia). Eksekusi simulasi mode *dry-run* sangat diwajibkan menggunakan jalur penunjang kelancaran IP semacam *VPN/Proxy* transparan.
- Lingkungan pengujian penyebaran program server di *cloud hosting* disarankan mengeksekusi langsung penyebaran ke *node region* **Railway EU West** — lokasi latensi ping yang luar biasa stabil kepada perbatasan simpul datacenter server terdekat Binance Eropa dan bersih dari penolakan ISP.
- Penyimpanan lokal riwayat sesi (Tabel `signals` log) dapat ditemukan menetap utuh dalam sub-direktori *folder* `data/trading.db`.

## Status & Roadmap (Fase Development)

Penanda arah pengembangan arsitektur proyek integrasi pemrosesan sistem Bot. 

**Tahap Selesai (Selesai Implementasi Penuh):**
- [x] Data infrastructure (Binance WS, Polymarket CLOB WS, RTDS)
- [x] Dual feed monitoring (Binance direct + Chainlink via RTDS)
- [x] Spread filter pre-entry gate
- [x] Oracle-aligned pricing (BUG fix koreksi patokan dasar harga pada modul `fair_probability` dan `feature_engine`)
- [x] Signal logging ke SQLite database dengan *outcome settler* otomatis
- [x] Telegram reporting pelaporan rutin interval dan keluaran ekspor kompilasi *shutdown CSV*
- [x] Configurable market filter penargetan URL rentang masa waktu instrumen spesifik dari pemindai `POLYMARKET_SLUG_PREFIX`
- [x] CLOB WebSocket *push real-time* berlanjut (sepenuhnya memensiunkan beban proses interogasi HTTP REST *polling* yang mengantongi penolakan 429 *Too Many Requests*)\r
- [x] One-Bet-Per-Market Rule: hanya satu posisi aktif per `market_id`, sinyal duplikat diblokir otomatis

**Arah Eksplorasi Pengembangan Lanjutan (Roadmap):**
- [ ] Menerapkan rancang bangun konstruksi pelatihan kerangka *ML model training* secara berkala mengambil *accumulated signal data* historis log mandiri
- [ ] Penugasan uji lapangan transmisi sinyal eksekusi *Live trading* terstruktur mengaitkan pada jalur *Polymarket CLOB API*
- [ ] Rekalibrasi *Expand* dan dukungan parameter korelasi instrumen kripto unggulan yang lain, mencakup spektrum pasar instrumen ETH dan aset SOL
- [ ] Penyeteman parameter ketepatan *Kalibrasi threshold spread filter* ditarik mendasar dari persentase data performa kondisi aktual pasar (Evaluasi *post 48h paper trading* periodik)
