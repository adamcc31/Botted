"""
zone_matrix.py — Empirical Zone Classifier for Predator Architecture.

Classifies every signal into Alpha / Death / Neutral zones based on
three physical variables extracted from forensic_full_spectrum.csv:
  - distance_to_strike (absolute USD)
  - ttr_minutes (time to resolution)
  - entry_odds (CLOB ask price)

This module is PASSIVE — it does not modify any production code.
It is consumed by simulate_predator.py for quarantine testing
and will be wired into risk_manager.py only after stress validation.

Zone data source: scripts/output/forensic_full_spectrum.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List


@dataclass(frozen=True)
class ZoneResult:
    zone_id: str           # e.g. "A1", "D3", "NEUTRAL"
    zone_type: str         # "ALPHA", "DEATH", "NEUTRAL"
    tier: int              # 1=highest priority, 3=lowest, 0=neutral/death
    empirical_wr: float    # historical win rate from forensic data
    empirical_ev: float    # historical NetEV per $1
    kelly_fraction: float  # Full Kelly f* from empirical WR + median odds
    sample_n: int          # number of historical samples


# ── Alpha Zones ──────────────────────────────────────────────────
# Empirical WR and EV extracted directly from forensic_full_spectrum.csv
ALPHA_ZONES: List[dict] = [
    {
        "zone_id": "A3", "tier": 2,
        "ttr_min": 3.0,  "ttr_max": 5.0,
        "dist_min": 30,  "dist_max": 60,
        "odds_min": 0.25, "odds_max": 0.40,
        "wr": 0.400, "ev": 0.266, "kelly": 0.026, "n": 25, # Quarter-Kelly (10.4% / 4)
    },
    {
        "zone_id": "A4", "tier": 1,
        "ttr_min": 1.5,  "ttr_max": 3.0,
        "dist_min": 15,  "dist_max": 30,
        "odds_min": 0.55, "odds_max": 0.70,
        "wr": 0.750, "ev": 0.170, "kelly": 0.150, "n": 44, # Capped at 15% (was 33.3%)
    },
    {
        "zone_id": "A5", "tier": 2,
        "ttr_min": 1.5,  "ttr_max": 3.0,
        "dist_min": 0,   "dist_max": 15,
        "odds_min": 0.55, "odds_max": 0.70,
        "wr": 0.722, "ev": 0.146, "kelly": 0.150, "n": 18, # Capped at 15% (was 25.9%)
    },
    {
        "zone_id": "A6", "tier": 3,
        "ttr_min": 3.0,  "ttr_max": 5.0,
        "dist_min": 100, "dist_max": 99999,
        "odds_min": 0.70, "odds_max": 1.00,
        "wr": 1.000, "ev": 0.129, "kelly": 0.150, "n": 30,  # Capped at 15%
    },
    {
        "zone_id": "A7", "tier": 2,
        "ttr_min": 1.5,  "ttr_max": 3.0,
        "dist_min": 0,   "dist_max": 15,
        "odds_min": 0.25, "odds_max": 0.40,
        "wr": 0.387, "ev": 0.096, "kelly": 0.085, "n": 75, # Uncapped (under 15%)
    },
    {
        "zone_id": "A8", "tier": 2,
        "ttr_min": 3.0,  "ttr_max": 5.0,
        "dist_min": 30,  "dist_max": 60,
        "odds_min": 0.55, "odds_max": 0.70,
        "wr": 0.708, "ev": 0.090, "kelly": 0.150, "n": 65, # Capped at 15% (was 22.1%)
    },
]

# ── Death Zones & Pruned Zones ───────────────────────────────────
DEATH_ZONES: List[dict] = [
    # PRUNED ALPHAS -> DEATH ZONE (for strict abstention)
    {
        "zone_id": "A1_PRUNED",
        "ttr_min": 3.0,  "ttr_max": 5.0,
        "dist_min": 30,  "dist_max": 60,
        "odds_min": 0.15, "odds_max": 0.25,
        "wr": 0.276, "ev": -0.116, "n": 29,
    },
    {
        "zone_id": "A2_PRUNED",
        "ttr_min": 1.5,  "ttr_max": 3.0,
        "dist_min": 15,  "dist_max": 30,
        "odds_min": 0.15, "odds_max": 0.25,
        "wr": 0.297, "ev": -0.197, "n": 37,
    },
    # ORIGINAL DEATH ZONES
    {
        "zone_id": "D1",
        "ttr_min": 1.5,  "ttr_max": 3.0,
        "dist_min": 0,   "dist_max": 15,
        "odds_min": 0.15, "odds_max": 0.25,
        "wr": 0.074, "ev": -0.666, "n": 27,
    },
    {
        "zone_id": "D2",
        "ttr_min": 1.5,  "ttr_max": 3.0,
        "dist_min": 30,  "dist_max": 60,
        "odds_min": 0.0,  "odds_max": 0.15,
        "wr": 0.053, "ev": -0.553, "n": 38,
    },
    {
        "zone_id": "D3",
        "ttr_min": 1.5,  "ttr_max": 3.0,
        "dist_min": 30,  "dist_max": 60,
        "odds_min": 0.15, "odds_max": 0.25,
        "wr": 0.119, "ev": -0.399, "n": 42,
    },
    {
        "zone_id": "D4",
        "ttr_min": 3.0,  "ttr_max": 5.0,
        "dist_min": 0,   "dist_max": 15,
        "odds_min": 0.55, "odds_max": 0.70,
        "wr": 0.429, "ev": -0.317, "n": 21,
    },
    {
        "zone_id": "D5",
        "ttr_min": 3.0,  "ttr_max": 5.0,
        "dist_min": 15,  "dist_max": 30,
        "odds_min": 0.25, "odds_max": 0.40,
        "wr": 0.250, "ev": -0.276, "n": 52,
    },
    {
        "zone_id": "D6",
        "ttr_min": 3.0,  "ttr_max": 5.0,
        "dist_min": 15,  "dist_max": 30,
        "odds_min": 0.55, "odds_max": 0.70,
        "wr": 0.476, "ev": -0.244, "n": 42,
    },
]

# Neutral zone sentinel
_NEUTRAL = ZoneResult(
    zone_id="NEUTRAL", zone_type="NEUTRAL", tier=0,
    empirical_wr=0.0, empirical_ev=0.0, kelly_fraction=0.0, sample_n=0,
)


def _match_zone(ttr: float, dist: float, odds: float, zone: dict) -> bool:
    """Check if coordinates fall within a zone's bounds."""
    return (
        zone["ttr_min"] <= ttr < zone["ttr_max"]
        and zone["dist_min"] <= dist < zone["dist_max"]
        and zone["odds_min"] <= odds < zone["odds_max"]
    )


def classify_zone(
    ttr_minutes: float,
    distance_to_strike_usd: float,
    entry_odds: float,
) -> ZoneResult:
    """
    Classify a signal into Alpha, Death, or Neutral zone.

    Args:
        ttr_minutes: Time to resolution in minutes.
        distance_to_strike_usd: abs(btc_price - strike_price) in USD.
        entry_odds: CLOB ask price (probability) for the side being traded.

    Returns:
        ZoneResult with zone metadata.

    Priority: Death zones checked FIRST (hardblock takes precedence).
    """
    # Death zones first — hardblock overrides everything
    for dz in DEATH_ZONES:
        if _match_zone(ttr_minutes, distance_to_strike_usd, entry_odds, dz):
            return ZoneResult(
                zone_id=dz["zone_id"],
                zone_type="DEATH",
                tier=0,
                empirical_wr=dz["wr"],
                empirical_ev=dz["ev"],
                kelly_fraction=0.0,
                sample_n=dz["n"],
            )

    # Alpha zones
    for az in ALPHA_ZONES:
        if _match_zone(ttr_minutes, distance_to_strike_usd, entry_odds, az):
            return ZoneResult(
                zone_id=az["zone_id"],
                zone_type="ALPHA",
                tier=az["tier"],
                empirical_wr=az["wr"],
                empirical_ev=az["ev"],
                kelly_fraction=az["kelly"],
                sample_n=az["n"],
            )

    return _NEUTRAL
