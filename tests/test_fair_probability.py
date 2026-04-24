import unittest
from datetime import datetime, timedelta, timezone

from src.fair_probability import FairProbabilityEngine
from src.schemas import ActiveMarket, CLOBState


class DummyConfig:
    def __init__(self, overrides=None):
        self._overrides = overrides or {}

    def get(self, key, default=None):
        return self._overrides.get(key, default)


class FakeBinanceFeed:
    def __init__(self, latest_price: float, ohlcv_1m_buffer):
        self.latest_price = latest_price
        self.ohlcv_1m_buffer = ohlcv_1m_buffer


def _make_closes(start: float, step_log_return: float, n: int):
    closes = [start]
    import math

    for _ in range(n):
        closes.append(closes[-1] * math.exp(step_log_return))
    # Engine expects list of dicts with "close"
    return [{"close": c} for c in closes]


class FairProbabilityTests(unittest.TestCase):
    def test_probability_monotone_in_strike(self):
        cfg = DummyConfig()
        engine = FairProbabilityEngine(cfg)

        now = datetime.now(timezone.utc)
        res = now + timedelta(minutes=10)

        clob = CLOBState(
            market_id="m",
            timestamp=now,
            yes_ask=0.6,
            yes_bid=0.59,
            no_ask=0.41,
            no_bid=0.4,
            yes_depth_usd=1000,
            no_depth_usd=1000,
            market_vig=0.01,
            is_liquid=True,
            is_stale=False,
        )

        feed = FakeBinanceFeed(
            latest_price=100.0,
            ohlcv_1m_buffer=_make_closes(100.0, 0.001, 200),
        )

        m1 = ActiveMarket(
            market_id="m",
            question="q",
            strike_price=95.0,
            T_open=now - timedelta(hours=1),
            T_resolution=res,
            TTR_minutes=10.0,
            clob_token_ids={"YES": "y", "NO": "n"},
            settlement_exchange="BINANCE",
            settlement_instrument="BTCUSDT",
            settlement_granularity="1m",
            settlement_price_type="close",
            resolution_source="Binance",
        )
        m2 = m1.model_copy(update={"strike_price": 105.0})

        q1 = engine.compute(feed, m1, clob_state=clob).q_fair
        q2 = engine.compute(feed, m2, clob_state=clob).q_fair
        self.assertGreater(q1, q2)

    def test_uncertainty_increases_near_expiry(self):
        cfg = DummyConfig()
        engine = FairProbabilityEngine(cfg)

        now = datetime.now(timezone.utc)
        feed = FakeBinanceFeed(
            latest_price=100.0,
            ohlcv_1m_buffer=_make_closes(100.0, 0.0005, 200),
        )
        common = dict(
            market_id="m",
            question="q",
            strike_price=100.0,
            T_open=now - timedelta(hours=1),
            clob_token_ids={"YES": "y", "NO": "n"},
            settlement_exchange="BINANCE",
            settlement_instrument="BTCUSDT",
            settlement_granularity="1m",
            settlement_price_type="close",
            resolution_source="Binance",
        )
        m_far = ActiveMarket(
            **common,
            T_resolution=now + timedelta(minutes=20),
            TTR_minutes=20.0,
        )
        m_near = ActiveMarket(
            **common,
            T_resolution=now + timedelta(minutes=2),
            TTR_minutes=2.0,
        )

        res_far = engine.compute(feed, m_far, clob_state=None)
        res_near = engine.compute(feed, m_near, clob_state=None)
        self.assertGreater(res_near.uncertainty_u, res_far.uncertainty_u)

    def test_sigma_estimator_behavior(self):
        closes = _make_closes(100.0, 0.0, 50)  # constant price => 0 variance
        close_vals = [x["close"] for x in closes]
        sigma = FairProbabilityEngine._realized_sigma_ann_from_closes(
            close_vals, window_n=30
        )
        self.assertGreaterEqual(sigma, 0.0)


if __name__ == "__main__":
    unittest.main()

