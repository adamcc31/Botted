import asyncio
import os
import time
import sys
import logging
import ssl
import traceback
import requests
import aiohttp
import json
from datetime import datetime
from collections import deque
import numpy as np
from dotenv import load_dotenv

# GLOBAL SSL BYPASS FOR CLOUD STABILITY
ssl._create_default_https_context = ssl._create_unverified_context

# Load credentials
load_dotenv()

# Dependencies
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds

# Setup Logging
LOG_DIR = "/app/logs" if os.environ.get("RAILWAY_ENVIRONMENT") else "logs"
os.makedirs(LOG_DIR, exist_ok=True)

class InfoFilter(logging.Filter):
    def filter(self, record): return record.levelno < logging.ERROR

log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(log_formatter)
stdout_handler.addFilter(InfoFilter())
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.ERROR)
stderr_handler.setFormatter(log_formatter)
file_handler = logging.FileHandler(f"{LOG_DIR}/execution_v4.log", encoding='utf-8')
file_handler.setFormatter(log_formatter)

logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, stderr_handler, file_handler])
logger = logging.getLogger(__name__)

# CONFIG: VERIFIED BTC/USD Price Feed ID (2026 Active)
BTC_FEED_ID = "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43"
PYTH_HERMES_BASE = "https://hermes.pyth.network/v2/updates/price/latest"
BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

class SlowSkewBotV4:
    def __init__(self):
        # Configuration
        self.tfm_window_mins = 15
        self.baseline_window_mins = 240
        self.z_threshold = 1.0 
        self.notif_threshold = 5.0 
        
        # State
        self.yes_token_id = None
        self.target_market_name = "Target Discovery Active..."
        self.price_buffer = deque(maxlen=self.baseline_window_mins)
        self.last_minute_price = None
        self.last_minute_ts = None
        self.last_heartbeat = 0
        self.last_tg_heartbeat = 0
        self.last_csv_sent = 0
        self.is_redundant_mode = False
        self.pyth_fail_count = 0
        
        # Clients
        self.clob_client = None
        self.telemetry_file = f"{LOG_DIR}/execution_v4_telemetry.csv"
        self._init_telemetry()

    def _init_telemetry(self):
        if not os.path.exists(self.telemetry_file):
            with open(self.telemetry_file, "w") as f:
                f.write("trigger_ts,z_score,poly_up_ask,poly_up_bid,ask_size,bid_size,latency_ms,status,source\n")

    def init_clob_client(self):
        private_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
        if not private_key:
            return False
        try:
            temp_client = ClobClient(host="https://clob.polymarket.com", chain=137, key=private_key)
            creds = temp_client.derive_api_key()
            self.clob_client = ClobClient(host="https://clob.polymarket.com", key=private_key, chain=137, creds=creds)
            
            # V2_MIGRATION: API key startup check
            try:
                self.clob_client.get_order_book("test")
            except Exception as e:
                if "401" in str(e) or "Unauthorized" in str(e):
                    logger.critical("v2_auth_failed_regenerate_api_keys", extra={"error": str(e), "action": "Login to Polymarket UI after 12:00 UTC April 28, regenerate L2 API keys"})
                    sys.exit(1)
                    
            logger.info("Polymarket CLOB Client Initialized.")
            return True
        except Exception:
            logger.error(f"CLOB Init Failed: {traceback.format_exc()}")
            return False

    def bootstrap_discovery(self, max_attempts=5, delay_s=15):
        manual_id = os.environ.get("TARGET_TOKEN_ID")
        if manual_id:
            self.yes_token_id = manual_id
            logger.info(f"Using manual TARGET_TOKEN_ID: {manual_id}")
            return True

        min_volume = float(os.environ.get("MIN_VOLUME_24HR", 1000.0))

        for attempt in range(1, max_attempts + 1):
            logger.info(f"Searching for Active Bitcoin target (Attempt {attempt}/{max_attempts})...")
            try:
                # Query /markets with high limit to find buried daily price-action
                r = requests.get(
                    'https://gamma-api.polymarket.com/markets', 
                    params={
                        'active': 'true',
                        'closed': 'false',
                        'order': 'volume24hr',
                        'ascending': 'false',
                        'limit': 200
                    },
                    timeout=10
                )
                r.raise_for_status()
                markets = r.json()
                
                candidates = []
                skipped_reasons = []

                for m in markets:
                    q = m.get('question', '')
                    
                    # 1. Technical filters
                    if not m.get('active') or m.get('closed') or not m.get('enableOrderBook'):
                        continue
                    
                    # 2. Strict Price-Action Pattern (Above/Up/Down/Below)
                    q_low = q.lower()
                    is_btc = "bitcoin" in q_low or "btc" in q_low
                    is_pa = any(p in q_low for p in ['above', 'up', 'down', 'below'])
                    
                    if not (is_btc and is_pa):
                        if len(skipped_reasons) < 5 and is_btc:
                            skipped_reasons.append(f"Non-PA: '{q[:40]}...'")
                        continue
                            
                    # 3. Volume Check
                    vol = float(m.get('volume24hr') or m.get('volume') or 0.0)
                    if vol < min_volume:
                        if len(skipped_reasons) < 5:
                            skipped_reasons.append(f"Low-Vol: '${vol:,.0f}' for '{q[:40]}...'")
                        continue
                        
                    candidates.append({'market': m, 'volume': vol})
                
                if not candidates and skipped_reasons:
                    logger.warning(f"No candidates passed filters. Top rejections: {skipped_reasons}")
                
                # Sort by volume descending
                candidates.sort(key=lambda x: x['volume'], reverse=True)
                
                for cand in candidates:
                    m = cand['market']
                    ids = m.get('clobTokenIds')
                    if isinstance(ids, str):
                        try: ids = json.loads(ids)
                        except: ids = [ids.replace('[','').replace(']','').replace('"','')]
                    
                    if not ids: continue
                    tid = ids[0]
                    
                    try:
                        # Validate token with a quick orderbook pull
                        self.clob_client.get_order_book(tid)
                        self.yes_token_id = tid
                        self.target_market_name = m.get('question', 'Unknown Bitcoin Market')
                        logger.info(f"🎯 LIQUID TARGET FOUND: {self.target_market_name} (Vol: ${cand['volume']:,.2f})")
                        return True
                    except Exception as e:
                        logger.debug(f"Token {tid} invalid: {e}")
                        continue
                
                if attempt < max_attempts:
                    logger.warning(f"No valid targets found on attempt {attempt}. Retrying in {delay_s}s...")
                    time.sleep(delay_s)
                else:
                    logger.error("All discovery attempts exhausted.")
                    
            except Exception as e:
                logger.error(f"Discovery attempt {attempt} error: {e}")
                if attempt < max_attempts:
                    time.sleep(delay_s)
        
        return False

    async def send_alert(self, session, title, message):
        tg_token = os.environ.get("TELEGRAM_TOKEN")
        tg_chat = os.environ.get("TELEGRAM_CHAT_ID")
        if tg_token and tg_chat:
            try:
                url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
                payload = {"chat_id": tg_chat, "text": f"🔥 {title}\n\n{message}"}
                async with session.post(url, json=payload, timeout=5) as resp:
                    if resp.status == 200: return
            except: pass

        webhook_url = os.environ.get("NOTIF_WEBHOOK_URL")
        if not webhook_url: return
        try:
            payload = {"content": f"🔥 **{title}**\n{message}"}
            async with session.post(webhook_url, json=payload, timeout=5) as resp: pass
        except: pass

    async def send_document(self, session, file_path, caption=""):
        tg_token = os.environ.get("TELEGRAM_TOKEN")
        tg_chat = os.environ.get("TELEGRAM_CHAT_ID")
        if tg_token and tg_chat and os.path.exists(file_path):
            try:
                url = f"https://api.telegram.org/bot{tg_token}/sendDocument"
                with open(file_path, 'rb') as f:
                    data = aiohttp.FormData()
                    data.add_field('chat_id', str(tg_chat))
                    data.add_field('caption', caption)
                    data.add_field('document', f, filename=os.path.basename(file_path))
                    
                    async with session.post(url, data=data, timeout=15) as resp:
                        if resp.status == 200:
                            logger.info(f"Telegram Doc Sent: {file_path}")
                        else:
                            logger.warning(f"Failed to send Doc. Status: {resp.status}")
            except Exception as e:
                logger.error(f"Failed to send Telegram Doc: {e}")

    async def get_binance_price(self, session):
        try:
            async with session.get(BINANCE_API_URL, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data.get('price', 0))
                return None
        except Exception: return None

    async def get_pyth_price(self, session):
        """Primary Oracle: V2 Hermes with Verified Feed ID."""
        # Querying with encoded array param correctly
        params = {"ids[]": [BTC_FEED_ID]}
        try:
            async with session.get(PYTH_HERMES_BASE, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.pyth_fail_count = 0
                    self.is_redundant_mode = False
                    
                    # Hermes V2 response parsing
                    parsed = data.get('parsed', []) if isinstance(data, dict) else []
                    if parsed:
                        p_info = parsed[0].get('price', {})
                        raw_px = p_info.get('price')
                        expo = p_info.get('expo')
                        if raw_px is not None and expo is not None:
                            return float(raw_px) * (10 ** int(expo))
                else:
                    logger.warning(f"Pyth Primary failed ({resp.status}). Activating Binance Fallback...")
        except Exception as e:
            logger.debug(f"Pyth Error: {e}")
        
        self.pyth_fail_count += 1
        self.is_redundant_mode = True
        return await self.get_binance_price(session)

    async def main_loop(self):
        async with aiohttp.ClientSession() as session:
            logger.info("🟢 CLOUD BOT ACTIVE (v28: Verified Ingest).")
            await self.send_alert(session, "[SYSTEM ONLINE]", 
                                 f"Verified Feed ID: {BTC_FEED_ID[:10]}...\nTarget: `{self.target_market_name}`")
            
            # Boot-up file send verification
            if os.path.exists(self.telemetry_file):
                await self.send_document(session, self.telemetry_file, caption="🚀 Boot-up Telemetry Audit")
            self.last_csv_sent = time.time()
            
            while True:
                try:
                    price = await self.get_pyth_price(session)
                    if not price: 
                        await asyncio.sleep(5)
                        continue
                        
                    now = datetime.now()
                    current_minute = now.replace(second=0, microsecond=0)
                    
                    if self.last_heartbeat == 0 or (time.time() - self.last_heartbeat > 300):
                        source = "Binance (Backup)" if self.is_redundant_mode else "Pyth (Primary)"
                        logger.info(f"💓 HEARTBEAT | Source: {source} | BTC: ${price:,.2f}")
                        self.last_heartbeat = time.time()
                    
                    if time.time() - self.last_tg_heartbeat > 14400:
                        await self.send_alert(session, "[HEARTBEAT]", f"BTC Stable: ${price:,.2f}")
                        self.last_tg_heartbeat = time.time()

                    # 5 Hours = 18000 seconds
                    if time.time() - self.last_csv_sent > 18000:
                        if os.path.exists(self.telemetry_file):
                            await self.send_document(session, self.telemetry_file, caption="📊 5-Hour Telemetry Audit")
                        self.last_csv_sent = time.time()

                    if self.last_minute_ts and current_minute > self.last_minute_ts:
                        momentum = price - self.last_minute_price
                        self.price_buffer.append(momentum)
                        await self.evaluate_signal(session, price)
                        self.last_minute_price = price
                    
                    if not self.last_minute_ts:
                        self.last_minute_price = price
                        logger.info(f"Baseline set: ${price:,.2f}")
                    
                    self.last_minute_ts = current_minute
                    
                    # Intelligent Polling: 15s base + backoff for failed Pyth
                    backoff_delay = min(self.pyth_fail_count * 15, 60) if self.is_redundant_mode else 0
                    await asyncio.sleep(15 + backoff_delay)
                except Exception as e:
                    logger.error(f"Fatal Loop: {e}")
                    await asyncio.sleep(15)

    async def evaluate_signal(self, session, current_price):
        if len(self.price_buffer) < 2: return
        history = list(self.price_buffer)
        margin_mult = 1.001 if self.is_redundant_mode else 1.0
        
        mu, sigma = np.mean(history), np.std(history)
        if sigma < 1e-9: return
        
        z = (history[-1] - mu) / sigma
        eff_threshold = self.z_threshold * margin_mult
        
        if abs(z) > eff_threshold:
            mode = "REDUNDANT" if self.is_redundant_mode else "PRIMARY"
            logger.info(f"🚨 SKEW ALERT [{mode}]: Z={z:.2f}")
            await self.execute_dry_run(z)
            if abs(z) >= self.notif_threshold:
                await self.send_alert(session, f"HIGH MOMENTUM ({mode})", 
                                     f"Z-Score: `{z:.2f}`\nBTC: `${current_price:,.2f}`")

    async def execute_dry_run(self, z):
        start_ms = time.time() * 1000
        try:
            ask, bid, ask_sz, bid_sz = 0.0, 0.0, 0.0, 0.0
            if self.clob_client and self.yes_token_id:
                book = self.clob_client.get_order_book(self.yes_token_id)
                if book.asks:
                    ask, ask_sz = float(book.asks[-1].price), float(book.asks[-1].size)
                if book.bids:
                    bid, bid_sz = float(book.bids[-1].price), float(book.bids[-1].size)
            
            latency = time.time() * 1000 - start_ms
            with open(self.telemetry_file, "a") as f:
                f.write(f"{datetime.now().isoformat()},{z:.2f},{ask},{bid},{ask_sz},{bid_sz},{latency:.2f},SUCCESS,{'REDUNDANT' if self.is_redundant_mode else 'PRIMARY'}\n")
            logger.info(f"✔ Telemetry OK: {latency:.1f}ms")
        except Exception: pass

async def main():
    bot = SlowSkewBotV4()
    if not bot.init_clob_client(): return
    if not bot.bootstrap_discovery():
        logger.error("COULD NOT FIND TARGET.")
        return
    await bot.main_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        logger.error(f"Terminal Fatal: {traceback.format_exc()}")
