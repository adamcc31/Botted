import unittest
from datetime import datetime, timedelta, timezone

from src.signal_generator import SignalGenerator
from src.schemas import (
    ActiveMarket,
    CLOBState,
    FeatureMetadata,
    FeatureVector,
)


class DummyConfig:
    def __init__(self, overrides=None):
        self._overrides = overrides or {}

    def get(self, key, default=None):
        return self._overrides.get(key, default)


FEATURE_NAMES = [
    "OBI",
    "TFM_normalized",
    "VAM",
    "RV",
    "vol_percentile",
    "depth_ratio",
    "price_vs_ema20",
    "binance_spread_bps",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "TTR_normalized",
    "TTR_sin",
    "TTR_cos",
    "strike_distance_pct",
    "contest_urgency",
    "ttr_x_obi",
    "ttr_x_tfm",
    "ttr_x_strike",
    "clob_yes_mid",
    "clob_yes_spread",
    "clob_no_spread",
    "market_vig",
]


def make_feature_vector(now, market_id, current_price, ttr_minutes):
    md = FeatureMetadata(
        timestamp=now,
        bar_close_time=now,
        market_id=market_id,
        strike_price=100.0,
        current_btc_price=current_price,
        TTR_minutes=ttr_minutes,
        TTR_phase="ENTRY_WINDOW",
        compute_lag_ms=0.0,
    )

    # Values mostly irrelevant for signal_v2; set the ones it reads.
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
        yes_ask=yes_ask,
        yes_bid=yes_bid,
        no_ask=no_ask,
        no_bid=no_bid,
        yes_depth_usd=1000.0,
        no_depth_usd=1000.0,
        market_vig=vig,
        is_liquid=True,
        is_stale=False,
    )


class SignalV2Tests(unittest.TestCase):
    def test_no_trade_zone_abstains_near_mid(self):
        cfg = DummyConfig()
        sg = SignalGenerator(cfg)

        now = datetime.now(timezone.utc)
        market_id = "m1"
        clob = make_clob(market_id, yes_ask=0.70, yes_bid=0.68, no_ask=0.31, no_bid=0.30)

        active = ActiveMarket(
            market_id=market_id,
            slug=market_id,
            question="q",
            strike_price=100.0,
            T_open=now - timedelta(hours=1),
            T_resolution=now + timedelta(minutes=7),
            TTR_minutes=7.0,
            clob_token_ids={"YES": "y", "NO": "n"},
            settlement_exchange="BINANCE",
            settlement_instrument="BTCUSDT",
            settlement_granularity="1m",
            settlement_price_type="close",
            resolution_source="Binance",
        )
        fv = make_feature_vector(now, market_id, current_price=100.0, ttr_minutes=7.0)

        mid_yes = (clob.yes_bid + clob.yes_ask) / 2.0
        res = sg.evaluate(P_model=mid_yes, uncertainty_u=0.02, clob_state=clob, active_market=active, feature_vector=fv)
        self.assertEqual(res.signal, "ABSTAIN")
        self.assertEqual(res.abstain_reason, "NO_TRADE_ZONE")

    def test_entry_buy_yes_when_edge_large(self):
        cfg = DummyConfig()
        sg = SignalGenerator(cfg)

        now = datetime.now(timezone.utc)
        market_id = "m2"
        clob = make_clob(market_id, yes_ask=0.60, yes_bid=0.59, no_ask=0.41, no_bid=0.40)

        active = ActiveMarket(
            market_id=market_id,
            slug=market_id,
            question="q",
            strike_price=100.0,
            T_open=now - timedelta(hours=1),
            T_resolution=now + timedelta(minutes=7),
            TTR_minutes=7.0,
            clob_token_ids={"YES": "y", "NO": "n"},
            settlement_exchange="BINANCE",
            settlement_instrument="BTCUSDT",
            settlement_granularity="1m",
            settlement_price_type="close",
            resolution_source="Binance",
        )
        fv = make_feature_vector(now, market_id, current_price=100.0, ttr_minutes=7.0)

        res = sg.evaluate(P_model=0.70, uncertainty_u=0.02, clob_state=clob, active_market=active, feature_vector=fv)
        self.assertEqual(res.signal, "BUY_YES")

    def test_basis_risk_halt_when_non_binance_near_resolution(self):
        cfg = DummyConfig()
        sg = SignalGenerator(cfg)

        now = datetime.now(timezone.utc)
        market_id = "m3"
        clob = make_clob(market_id, yes_ask=0.60, yes_bid=0.59, no_ask=0.41, no_bid=0.40)

        active = ActiveMarket(
            market_id=market_id,
            slug=market_id,
            question="q",
            strike_price=100.0,
            T_open=now - timedelta(hours=1),
            T_resolution=now + timedelta(minutes=5),
            TTR_minutes=5.0,
            clob_token_ids={"YES": "y", "NO": "n"},
            settlement_exchange="PYTH",
            settlement_instrument="BTCUSDT",
            settlement_granularity="unknown",
            settlement_price_type="unknown",
            resolution_source="Pyth",
        )
        fv = make_feature_vector(now, market_id, current_price=100.0, ttr_minutes=5.0)

        res = sg.evaluate(P_model=0.70, uncertainty_u=0.02, clob_state=clob, active_market=active, feature_vector=fv)
        self.assertEqual(res.signal, "ABSTAIN")
        self.assertEqual(res.abstain_reason, "BASIS_RISK_BLOCK")


if __name__ == "__main__":
    unittest.main()

