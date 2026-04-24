"""
test_ultrashort.py — Validation tests for ultra-short market alignment (Fix 2-5).

Tests:
  1. TTR window correctness for 5-minute markets
  2. TTR window correctness for 10-minute markets
  3. Near-expiry bias disabled for ultra-short
  4. Market pattern matching for ultra-short keywords
  5. Signal generation on ultra-short market with large edge
  6. Signal ABSTAIN when edge insufficient on ultra-short
  7. Backtest simulation: synthetic 5-minute market cycle
"""

import unittest
import math
from datetime import datetime, timedelta, timezone

from src.signal_generator import SignalGenerator
from src.fair_probability import FairProbabilityEngine
from src.market_discovery import MarketDiscovery
from src.dry_run import DryRunEngine
from src.schemas import (
    ActiveMarket,
    CLOBState,
    FeatureMetadata,
    FeatureVector,
    SignalResult,
    ApprovedBet,
)


class DummyConfig:
    def __init__(self, overrides=None):
        self._overrides = overrides or {}

    def get(self, key, default=None):
        return self._overrides.get(key, default)


class FakeBinanceFeed:
    def __init__(self, latest_price: float, ohlcv_1m_buffer):
        self.latest_price = latest_price
        self.ohlcv_1m_buffer = ohlcv_1m_buffer


def _make_closes(start, step_log_return, n):
    closes = [start]
    for _ in range(n):
        closes.append(closes[-1] * math.exp(step_log_return))
    return [{"close": c} for c in closes]


FEATURE_NAMES = [
    "OBI", "TFM_normalized", "VAM", "RV", "vol_percentile",
    "depth_ratio", "price_vs_ema20", "binance_spread_bps",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "TTR_normalized", "TTR_sin", "TTR_cos", "strike_distance_pct",
    "contest_urgency", "ttr_x_obi", "ttr_x_tfm", "ttr_x_strike",
    "clob_yes_mid", "clob_yes_spread", "clob_no_spread", "market_vig",
]


def make_feature_vector(now, market_id, current_price, ttr_minutes):
    md = FeatureMetadata(
        timestamp=now,
        bar_close_time=now,
        market_id=market_id,
        strike_price=current_price,
        current_btc_price=current_price,
        TTR_minutes=ttr_minutes,
        TTR_phase="ENTRY_WINDOW",
        compute_lag_ms=0.0,
    )
    values_by_name = {
        "vol_percentile": 0.5,
        "binance_spread_bps": 2.0,
        "depth_ratio": 1.2,
        "strike_distance_pct": 0.0,
    }
    values = [values_by_name.get(n, 0.0) for n in FEATURE_NAMES]
    return FeatureVector(values=values, feature_names=FEATURE_NAMES, metadata=md)


def make_clob(market_id, yes_ask, yes_bid, no_ask, no_bid):
    vig = yes_ask + no_ask - 1.0
    return CLOBState(
        market_id=market_id,
        timestamp=datetime.now(timezone.utc),
        yes_ask=yes_ask, yes_bid=yes_bid,
        no_ask=no_ask, no_bid=no_bid,
        yes_depth_usd=5000.0, no_depth_usd=5000.0,
        market_vig=vig, is_liquid=True, is_stale=False,
    )


def make_ultrashort_market(now, lifespan_minutes=5.0, ttr_minutes=3.0,
                            strike_price=83000.0, market_id="ultra1"):
    T_open = now - timedelta(minutes=(lifespan_minutes - ttr_minutes))
    T_resolution = now + timedelta(minutes=ttr_minutes)
    return ActiveMarket(
        market_id=market_id,
        slug=market_id,
        question=f"BTC 5 minute up or down from ${strike_price:,.0f}?",
        strike_price=strike_price,
        T_open=T_open,
        T_resolution=T_resolution,
        TTR_minutes=ttr_minutes,
        clob_token_ids={"YES": "0xYES", "NO": "0xNO"},
        settlement_exchange="BINANCE",
        settlement_instrument="BTCUSDT",
        settlement_granularity="1m",
        settlement_price_type="close",
        resolution_source="Binance",
    )


# ══════════════════════════════════════════════════════════════
# Test 1: TTR Window for 5-minute market
# ══════════════════════════════════════════════════════════════
class TestTTRWindowUltraShort(unittest.TestCase):
    def test_5min_market_ttr_window(self):
        """5-minute market → entry window 0.5 to 4.0 minutes."""
        cfg = DummyConfig()
        sg = SignalGenerator(cfg)
        now = datetime.now(timezone.utc)
        market = make_ultrashort_market(now, lifespan_minutes=5.0, ttr_minutes=3.0)
        ttr_min, ttr_max = sg._resolve_ttr_window(market)
        self.assertAlmostEqual(ttr_min, 0.5, places=2)  # 5.0 * 0.10
        self.assertAlmostEqual(ttr_max, 4.0, places=2)  # 5.0 * 0.80

    def test_10min_market_ttr_window(self):
        """10-minute market → entry window 1.0 to 8.0 minutes."""
        cfg = DummyConfig()
        sg = SignalGenerator(cfg)
        now = datetime.now(timezone.utc)
        market = make_ultrashort_market(now, lifespan_minutes=10.0, ttr_minutes=5.0)
        ttr_min, ttr_max = sg._resolve_ttr_window(market)
        self.assertAlmostEqual(ttr_min, 1.0, places=2)  # 10.0 * 0.10
        self.assertAlmostEqual(ttr_max, 8.0, places=2)  # 10.0 * 0.80

    def test_15min_market_falls_through_to_short(self):
        """15-minute market (> 10 min) → uses short window (5-45 min)."""
        cfg = DummyConfig()
        sg = SignalGenerator(cfg)
        now = datetime.now(timezone.utc)
        market = make_ultrashort_market(now, lifespan_minutes=15.0, ttr_minutes=7.0)
        ttr_min, ttr_max = sg._resolve_ttr_window(market)
        self.assertEqual(ttr_min, 5.0)
        self.assertEqual(ttr_max, 45.0)

    def test_discovery_mirror_matches_signal(self):
        """MarketDiscovery._resolve_signal_ttr_window must match SignalGenerator."""
        cfg = DummyConfig()
        sg = SignalGenerator(cfg)
        md = MarketDiscovery(cfg)
        now = datetime.now(timezone.utc)
        market = make_ultrashort_market(now, lifespan_minutes=5.0, ttr_minutes=3.0)

        sig_window = sg._resolve_ttr_window(market)
        disc_window = md._resolve_signal_ttr_window(market)
        self.assertEqual(sig_window, disc_window)


# ══════════════════════════════════════════════════════════════
# Test 2: Near-Expiry Bias Disabled
# ══════════════════════════════════════════════════════════════
class TestNearExpiryBias(unittest.TestCase):
    def test_ultrashort_no_expiry_inflation(self):
        """Ultra-short market: near_w should be 0 → no vol floor inflation."""
        cfg = DummyConfig()
        engine = FairProbabilityEngine(cfg)
        now = datetime.now(timezone.utc)

        feed = FakeBinanceFeed(
            latest_price=83000.0,
            ohlcv_1m_buffer=_make_closes(83000.0, 0.0005, 200),
        )

        # Ultra-short: 5 min lifespan, 2 min TTR
        market = make_ultrashort_market(now, lifespan_minutes=5.0, ttr_minutes=2.0,
                                         strike_price=83000.0)
        result = engine.compute(feed, market, clob_state=None)

        # near_w should be 0 for ultra-short, meaning diagnostics near_w == 0
        self.assertEqual(result.diagnostics.get("near_w"), 0.0)

    def test_normal_market_has_expiry_inflation(self):
        """Normal 1-hour market at 2 min TTR: near_w should be > 0."""
        cfg = DummyConfig()
        engine = FairProbabilityEngine(cfg)
        now = datetime.now(timezone.utc)

        feed = FakeBinanceFeed(
            latest_price=83000.0,
            ohlcv_1m_buffer=_make_closes(83000.0, 0.0005, 200),
        )

        # Normal: 60 min lifespan, 2 min TTR
        market = ActiveMarket(
            market_id="normal1",
            slug="normal1",
            question="Bitcoin Up or Down from $83,000?",
            strike_price=83000.0,
            T_open=now - timedelta(minutes=58),
            T_resolution=now + timedelta(minutes=2),
            TTR_minutes=2.0,
            clob_token_ids={"YES": "y", "NO": "n"},
            settlement_exchange="BINANCE",
            settlement_instrument="BTCUSDT",
            settlement_granularity="1m",
            settlement_price_type="close",
            resolution_source="Binance",
        )
        result = engine.compute(feed, market, clob_state=None)
        self.assertGreater(result.diagnostics.get("near_w", 0), 0.0)


# ══════════════════════════════════════════════════════════════
# Test 3: Market Pattern Matching
# ══════════════════════════════════════════════════════════════
class TestMarketPatterns(unittest.TestCase):
    def test_5min_pattern_matches(self):
        """Ultra-short market patterns should match 5-minute questions."""
        cfg = DummyConfig()
        md = MarketDiscovery(cfg)

        test_cases = [
            {"question": "BTC 5 minute up or down from $83,000?", "expect": True},
            {"question": "Bitcoin 5 min candle prediction", "expect": True},
            {"question": "BTC 5min market", "expect": True},
            {"question": "Bitcoin up or down", "expect": True},
            {"question": "Will ETH reach $5000?", "expect": False},
        ]
        for tc in test_cases:
            with self.subTest(question=tc["question"]):
                result = md._is_btc_up_down_market({"question": tc["question"], "description": ""})
                self.assertEqual(result, tc["expect"], f"Failed for: {tc['question']}")

    def test_old_pattern_removed(self):
        """'what price will bitcoin hit' should NOT match anymore."""
        cfg = DummyConfig()
        md = MarketDiscovery(cfg)
        result = md._is_btc_up_down_market({
            "question": "what price will bitcoin hit next month?",
            "description": ""
        })
        self.assertFalse(result)


# ══════════════════════════════════════════════════════════════
# Test 4: Signal on Ultra-Short Market
# ══════════════════════════════════════════════════════════════
class TestUltraShortSignal(unittest.TestCase):
    def test_buy_yes_with_large_edge(self):
        """BUY_UP when fair prob >> CLOB ask on ultra-short market."""
        cfg = DummyConfig()
        sg = SignalGenerator(cfg)
        now = datetime.now(timezone.utc)
        market = make_ultrashort_market(now, lifespan_minutes=5.0, ttr_minutes=3.0,
                                         strike_price=82000.0)

        # CLOB lagging: YES ask still at 0.30 but fair prob is ~0.70
        clob = make_clob("ultra1", yes_ask=0.30, yes_bid=0.28,
                          no_ask=0.71, no_bid=0.70)
        fv = make_feature_vector(now, "ultra1", current_price=83000.0, ttr_minutes=3.0)

        # P_model = 0.70 (from fair_prob), u = 0.03 (ultra-short default)
        res = sg.evaluate(P_model=0.70, uncertainty_u=0.03,
                           clob_state=clob, active_market=market, feature_vector=fv)
        self.assertEqual(res.signal, "BUY_UP")
        # Edge should be: 0.70 - 0.30 - 0.03 = 0.37 (huge)
        self.assertGreater(res.edge_yes, 0.30)

    def test_abstain_when_no_edge(self):
        """ABSTAIN when fair prob ≈ CLOB ask on ultra-short."""
        cfg = DummyConfig()
        sg = SignalGenerator(cfg)
        now = datetime.now(timezone.utc)
        market = make_ultrashort_market(now, lifespan_minutes=5.0, ttr_minutes=3.0,
                                         strike_price=83000.0)

        # CLOB efficient: no mispricing
        clob = make_clob("ultra1", yes_ask=0.52, yes_bid=0.50,
                          no_ask=0.50, no_bid=0.48)
        fv = make_feature_vector(now, "ultra1", current_price=83000.0, ttr_minutes=3.0)

        res = sg.evaluate(P_model=0.51, uncertainty_u=0.03,
                           clob_state=clob, active_market=market, feature_vector=fv)
        self.assertEqual(res.signal, "ABSTAIN")

    def test_ttr_gate_blocks_expired_ultrashort(self):
        """TTR gate blocks signal if market is past entry window."""
        cfg = DummyConfig()
        sg = SignalGenerator(cfg)
        now = datetime.now(timezone.utc)
        # Market with only 0.3 min left (below 0.5 min minimum for 5-min market)
        market = make_ultrashort_market(now, lifespan_minutes=5.0, ttr_minutes=0.3,
                                         strike_price=83000.0)
        clob = make_clob("ultra1", yes_ask=0.30, yes_bid=0.28,
                          no_ask=0.71, no_bid=0.70)
        fv = make_feature_vector(now, "ultra1", current_price=83000.0, ttr_minutes=0.3)

        res = sg.evaluate(P_model=0.70, uncertainty_u=0.03,
                           clob_state=clob, active_market=market, feature_vector=fv)
        self.assertEqual(res.signal, "ABSTAIN")
        self.assertEqual(res.abstain_reason, "TTR_PHASE")


# ══════════════════════════════════════════════════════════════
# Test 5: Backtest Simulation — 5-Minute Market Cycle
# ══════════════════════════════════════════════════════════════
class TestBacktestSimulation(unittest.TestCase):
    """
    Simulates a series of 5-minute markets with known BTC price movements.
    Verifies the DryRunEngine can correctly resolve trades and compute metrics.
    """

    def test_backtest_5min_cycle(self):
        """Simulate 10 ultra-short market cycles with realistic edge."""
        cfg = DummyConfig({
            "dry_run.pass_win_rate": 0.53,
            "dry_run.pass_profit_factor": 1.1,
            "dry_run.pass_dry_run_score": 0.7,
            "dry_run.pass_max_drawdown": -0.15,
            "dry_run.min_trades_per_session": 5,
            "dry_run.abort_consecutive_losses": 10,
            "dry_run.abstain_summary_every_signals": 0,
        })
        sg = SignalGenerator(cfg)
        dry_run = DryRunEngine(cfg, initial_capital=100.0)

        import asyncio

        now = datetime.now(timezone.utc)
        results = []

        # Simulate 10 market cycles
        scenarios = [
            # (strike, btc_at_entry, btc_at_resolution, clob_yes_ask)
            (83000, 83200, 83500, 0.30),   # BTC far above strike → YES wins, CLOB lagging
            (83000, 82800, 82500, 0.70),   # BTC below strike → NO wins, CLOB lagging
            (84000, 84100, 84300, 0.35),   # BTC above strike → YES wins
            (84000, 83900, 83700, 0.65),   # BTC below strike → NO wins
            (85000, 85200, 85400, 0.25),   # BTC far above → YES wins, huge lag
            (85000, 84800, 84600, 0.75),   # BTC below → NO wins
            (82000, 82300, 82500, 0.28),   # YES wins
            (82000, 81700, 81500, 0.72),   # NO wins
            (83500, 83600, 83800, 0.40),   # Moderate edge YES
            (83500, 83400, 83200, 0.60),   # Moderate edge NO
        ]

        trades_submitted = 0
        for i, (strike, btc_entry, btc_resolve, clob_yes_ask) in enumerate(scenarios):
            market = ActiveMarket(
                market_id=f"sim_{i}",
                slug=f"sim_{i}",
                question=f"BTC 5 minute up or down from ${strike:,}?",
                strike_price=float(strike),
                T_open=now + timedelta(minutes=i*6),
                T_resolution=now + timedelta(minutes=i*6 + 5),
                TTR_minutes=3.0,
                clob_token_ids={"YES": "0xY", "NO": "0xN"},
                settlement_exchange="BINANCE",
                settlement_instrument="BTCUSDT",
                settlement_granularity="1m",
                settlement_price_type="close",
                resolution_source="Binance",
            )

            clob = make_clob(f"sim_{i}",
                              yes_ask=clob_yes_ask,
                              yes_bid=max(0.01, clob_yes_ask - 0.02),
                              no_ask=1.0 - clob_yes_ask + 0.01,
                              no_bid=max(0.01, 1.0 - clob_yes_ask - 0.01))
            fv = make_feature_vector(now, f"sim_{i}",
                                      current_price=float(btc_entry),
                                      ttr_minutes=3.0)

            # Compute fair probability using BS
            engine = FairProbabilityEngine(cfg)
            feed = FakeBinanceFeed(
                latest_price=float(btc_entry),
                ohlcv_1m_buffer=_make_closes(float(btc_entry), 0.0003, 200),
            )
            fair_result = engine.compute(feed, market, clob_state=clob,
                                          as_of_time=now + timedelta(minutes=i*6 + 2))

            signal = sg.evaluate(
                P_model=fair_result.q_fair,
                uncertainty_u=fair_result.uncertainty_u,
                clob_state=clob,
                active_market=market,
                feature_vector=fv,
            )
            dry_run.record_signal(signal)

            if signal.signal != "ABSTAIN":
                approved = ApprovedBet(
                    signal=signal,
                    bet_size=min(5.0, dry_run.capital * 0.05),
                    kelly_fraction=0.05,
                    kelly_multiplier=0.5,
                )
                trade = dry_run.simulate_trade(signal, approved, market)
                resolved = asyncio.get_event_loop().run_until_complete(
                    dry_run.resolve_trade(trade, float(btc_resolve))
                )
                trades_submitted += 1
                results.append({
                    "cycle": i,
                    "signal": signal.signal,
                    "outcome": resolved.outcome,
                    "pnl": resolved.pnl_usd,
                    "edge_yes": round(signal.edge_yes, 4),
                    "edge_no": round(signal.edge_no, 4),
                    "P_model": round(signal.P_model, 4),
                    "clob_yes_ask": clob_yes_ask,
                })

        # Compute final metrics
        metrics = dry_run.compute_session_metrics(model_version="test_v1")

        # Print results for visibility
        print("\n" + "=" * 70)
        print("BACKTEST SIMULATION — 5-Minute Ultra-Short Market")
        print("=" * 70)
        for r in results:
            status = "WIN" if r["outcome"] == "WIN" else "LOSS"
            print(f"  Cycle {r['cycle']:2d} | {r['signal']:7s} | {status:4s} | "
                  f"PnL: ${r['pnl']:+.2f} | Fair: {r['P_model']:.4f} | "
                  f"CLOB_ask: {r['clob_yes_ask']:.2f} | "
                  f"edge_Y: {r['edge_yes']:+.4f} edge_N: {r['edge_no']:+.4f}")

        print("-" * 70)
        print(f"  Trades Submitted:  {trades_submitted}")
        print(f"  Trades Executed:   {metrics.trades_executed}")
        print(f"  Win Rate:          {metrics.win_rate*100:.1f}%")
        print(f"  Total PnL:         ${metrics.total_pnl_usd:+.2f}")
        print(f"  Profit Factor:     {metrics.profit_factor:.2f}")
        print(f"  Max Drawdown:      {metrics.max_drawdown:.4f}")
        print(f"  Capital Start:     ${metrics.capital_start:.2f}")
        print(f"  Capital End:       ${metrics.capital_end:.2f}")
        print(f"  Signals Evaluated: {metrics.total_signals_evaluated}")
        print(f"  Signals Abstained: {metrics.signals_abstained}")
        print(f"  Dry Run Score:     {metrics.dry_run_score:.4f}")
        print(f"  Pass/Fail:         {metrics.pass_fail}")
        print("=" * 70)

        # Assertions: bot should trade and metrics should be computable
        self.assertGreater(trades_submitted, 0, "Bot should have executed at least 1 trade")
        self.assertIsNotNone(metrics.win_rate)
        self.assertIsNotNone(metrics.total_pnl_usd)
        self.assertGreater(metrics.total_signals_evaluated, 0)


if __name__ == "__main__":
    unittest.main()
