import asyncio
import sys
import os
sys.path.append(os.getcwd())
from datetime import datetime, timezone, timedelta
from collections import deque
import pandas as pd
import numpy as np

# Mocking parts of the system for testing
from src.config_manager import ConfigManager
from src.schemas import ActiveMarket, CLOBState
from src.clob_feed import CLOBFeed
from src.binance_feed import BinanceFeed
from src.feature_engine import FeatureEngine

async def test_velocity_extraction():
    print("--- STARTING VELOCITY EXTRACTION SANITY CHECK ---")
    config = ConfigManager()
    clob_feed = CLOBFeed(config)
    binance_feed = BinanceFeed(config)
    engine = FeatureEngine(config)
    
    token_id = "test_token_yes"
    active_market = ActiveMarket(
        market_id="test_market",
        slug="test-slug",
        question="Is it testing?",
        strike_price=70000.0,
        T_open=datetime.now(timezone.utc) - timedelta(minutes=10),
        T_resolution=datetime.now(timezone.utc) + timedelta(minutes=5),
        TTR_minutes=5.0,
        clob_token_ids={"YES": token_id, "NO": "test_token_no"}
    )
    
    # 1. Populate CLOB History (Simulate 20 seconds ago)
    hist_time = datetime.now(timezone.utc) - timedelta(seconds=20)
    hist_book = {
        "asset_id": token_id,
        "asks": [{"price": "0.55", "size": "100"}],
        "bids": [{"price": "0.45", "size": "100"}]
    }
    
    for tid in [token_id, "test_token_no"]:
        clob_feed._clob_history[tid] = deque(maxlen=100)
        clob_feed._clob_history[tid].append({
            "timestamp": hist_time,
            "book": {
                "asset_id": tid,
                "asks": [{"price": "0.55", "size": "100"}],
                "bids": [{"price": "0.45", "size": "100"}]
            }
        })
        
        # Add current book to history too so we have len >= 2
        clob_feed._clob_history[tid].append({
            "timestamp": datetime.now(timezone.utc),
            "book": {
                "asset_id": tid,
                "asks": [{"price": "0.52", "size": "100"}],
                "bids": [{"price": "0.48", "size": "100"}]
            }
        })
    
    # 2. Current State
    clob_state = CLOBState(
        market_id="test_market",
        timestamp=datetime.now(timezone.utc),
        yes_ask=0.52,
        yes_bid=0.48,
        no_ask=0.52,
        no_bid=0.48,
        yes_depth_usd=200.0,
        no_depth_usd=100.0,
        market_vig=0.04,
        is_liquid=True
    )
    
    # Mock Binance Feed data for minimal FeatureEngine requirements
    binance_feed._latest_price = 70100.0
    for i in range(25):
        binance_feed._ohlcv_buffer.append({
            "open": 70000.0, "high": 70100.0, "low": 69900.0, "close": 70050.0, 
            "volume": 10.0, "close_time": int(time.time()*1000)
        })
    
    # 3. Compute Features
    print("Computing features with history...")
    fv = engine.compute(
        binance_feed=binance_feed,
        clob_feed=clob_feed,
        active_market=active_market,
        clob_state=clob_state,
        oracle_price=70100.0
    )
    
    if fv:
        # Extract the last two features (velocity)
        features_dict = dict(zip(fv.feature_names, fv.values))
        spread_vel = features_dict.get("clob_spread_vel")
        depth_delta = features_dict.get("clob_depth_delta")
        
        print(f"Feature Vector Length: {len(fv.values)}")
        print(f"clob_spread_vel: {spread_vel}")
        print(f"clob_depth_delta: {depth_delta}")
        
        # Current spread: 0.52 - 0.48 = 0.04
        # Historical spread: 0.55 - 0.45 = 0.10
        # Vel: (0.04 - 0.10) / 15.0 = -0.004
        expected_vel = (0.04 - 0.10) / 15.0
        
        print(f"Expected Spread Velocity: ~{expected_vel:.6f}")
        
        if abs(spread_vel - expected_vel) < 1e-6:
            print("SUCCESS: Spread velocity matches expected value.")
        else:
            print("FAILURE: Spread velocity mismatch.")
            
        if abs(depth_delta) > 0:
            print("SUCCESS: Depth delta is non-zero as expected.")
        else:
            print("WARNING: Depth delta is zero (check depth calculation).")
    else:
        print("FAILURE: Feature computation returned None.")

if __name__ == "__main__":
    import time
    asyncio.run(test_velocity_extraction())
