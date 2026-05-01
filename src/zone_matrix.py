"""
zone_matrix.py — Empirical Zone Classifier for Predator Architecture V4.

Classifies every signal into Alpha / Death / Neutral zones based on
three physical variables:
  - distance_to_strike (absolute USD)
  - ttr_minutes (time to resolution)
  - entry_odds (CLOB ask price)

V4 Redesign Note:
This version uses the combined V1 + V3 dataset (~34,000 signals).
V3 is_win is defined as signal_correct (direction accuracy).
V4-A3 includes pre-filter signals; executable trades may be fewer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List


@dataclass(frozen=True)
class ZoneResult:
    zone_id: str           # e.g. "V4-A1", "V4-D1", "NEUTRAL"
    zone_type: str         # "ALPHA", "DEATH", "NEUTRAL"
    tier: int              # 1=highest priority, 3=lowest, 0=neutral/death
    empirical_wr: float    # historical win rate
    empirical_ev: float    # historical NetEV per $1
    kelly_fraction: float  # multiplier for Kelly (1.0=Full, 0.5=Half, 0.25=Quarter)
    kelly_cap: float       # max bet as fraction of capital (e.g. 0.20 for 20%)
    sample_n: int          # number of historical samples


# ── Alpha Zones V4 ───────────────────────────────────────────────
ALPHA_ZONES_V4: List[dict] = [
    {
        "zone_id": "V4-A1", "tier": 1,
        "ttr_min": 1.5, "ttr_max": 3.0,
        "dist_min": 15, "dist_max": 30,
        "odds_min": 0.55, "odds_max": 0.70,
        "wr": 0.775, "ev": 0.158, "kelly_fraction": 1.0, "kelly_cap": 0.20, "n": 49
    },
    {
        "zone_id": "V4-A2", "tier": 2,
        "ttr_min": 3.0, "ttr_max": 5.0,
        "dist_min": 30, "dist_max": 60,
        "odds_min": 0.55, "odds_max": 0.70,
        "wr": 0.721, "ev": 0.089, "kelly_fraction": 0.5, "kelly_cap": 0.15, "n": 61
    },
    {
        "zone_id": "V4-A3", "tier": 2,
        "ttr_min": 1.5, "ttr_max": 5.0,
        "dist_min": 30, "dist_max": 60,
        "odds_min": 0.70, "odds_max": 1.0,
        "wr": 0.834, "ev": 0.067, "kelly_fraction": 0.5, "kelly_cap": 0.15, "n": 113
    },
    {
        "zone_id": "V4-A4", "tier": 3,
        "ttr_min": 3.0, "ttr_max": 5.0,
        "dist_min": 0, "dist_max": 15,
        "odds_min": 0.40, "odds_max": 0.55,
        "wr": 0.489, "ev": 0.022, "kelly_fraction": 0.25, "kelly_cap": 0.05, "n": 90
    },
]

# ── Death Zones V4 ───────────────────────────────────────────────
DEATH_ZONES_V4: List[dict] = [
    {
        "zone_id": "V4-D1",
        "ttr_min": 3.0, "ttr_max": 5.0,
        "dist_min": 60, "dist_max": 100,
        "odds_min": 0.70, "odds_max": 1.0,
        "wr": 0.666, "ev": -0.133, "n": 39
    },
    {
        "zone_id": "V4-D2",
        "ttr_min": 3.0, "ttr_max": 5.0,
        "dist_min": 15, "dist_max": 30,
        "odds_min": 0.55, "odds_max": 0.70,
        "wr": 0.488, "ev": -0.121, "n": 43
    },
    {
        "zone_id": "V4-D3",
        "ttr_min": 1.5, "ttr_max": 3.0,
        "dist_min": 0, "dist_max": 15,
        "odds_min": 0.15, "odds_max": 0.25,
        "wr": 0.098, "ev": -0.113, "n": 51
    },
    {
        "zone_id": "V4-D4",
        "ttr_min": 1.5, "ttr_max": 5.0,
        "dist_min": 15, "dist_max": 30,
        "odds_min": 0.25, "odds_max": 0.40,
        "wr": 0.253, "ev": -0.067, "n": 155
    },
]

# Neutral zone sentinel
_NEUTRAL = ZoneResult(
    zone_id="NEUTRAL", zone_type="NEUTRAL", tier=0,
    empirical_wr=0.0, empirical_ev=0.0, kelly_fraction=0.0, kelly_cap=0.0, sample_n=0,
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
    Classify a signal into Alpha, Death, or Neutral zone using V4 logic.

    Priority: Death zones checked FIRST (hardblock takes precedence).
    """
    # ── Hard Constraints for NEUTRAL ──
    if ttr_minutes < 1.5 or ttr_minutes > 5.0 or distance_to_strike_usd > 100:
        return _NEUTRAL

    # ── Death Zones V4 ──
    for dz in DEATH_ZONES_V4:
        if _match_zone(ttr_minutes, distance_to_strike_usd, entry_odds, dz):
            return ZoneResult(
                zone_id=dz["zone_id"],
                zone_type="DEATH",
                tier=0,
                empirical_wr=dz["wr"],
                empirical_ev=dz["ev"],
                kelly_fraction=0.0,
                kelly_cap=0.0,
                sample_n=dz["n"],
            )

    # ── Alpha Zones V4 ──
    for az in ALPHA_ZONES_V4:
        if _match_zone(ttr_minutes, distance_to_strike_usd, entry_odds, az):
            return ZoneResult(
                zone_id=az["zone_id"],
                zone_type="ALPHA",
                tier=az["tier"],
                empirical_wr=az["wr"],
                empirical_ev=az["ev"],
                kelly_fraction=az["kelly_fraction"],
                kelly_cap=az["kelly_cap"],
                sample_n=az["n"],
            )

    return _NEUTRAL


# ── Legacy V3 Implementation ──────────────────────────────────────

ALPHA_ZONES_V3: List[dict] = [
    {"zone_id": "A3", "ttr_min": 3.0, "ttr_max": 5.0, "dist_min": 30, "dist_max": 60, "odds_min": 0.25, "odds_max": 0.40, "wr": 0.400, "ev": 0.266, "kelly": 0.026, "n": 25},
    {"zone_id": "A4", "ttr_min": 1.5, "ttr_max": 3.0, "dist_min": 15, "dist_max": 30, "odds_min": 0.55, "odds_max": 0.70, "wr": 0.750, "ev": 0.170, "kelly": 0.150, "n": 44},
    {"zone_id": "A5", "ttr_min": 1.5, "ttr_max": 3.0, "dist_min": 0, "dist_max": 15, "odds_min": 0.55, "odds_max": 0.70, "wr": 0.722, "ev": 0.146, "kelly": 0.150, "n": 18},
    {"zone_id": "A6", "ttr_min": 3.0, "ttr_max": 5.0, "dist_min": 100, "dist_max": 99999, "odds_min": 0.70, "odds_max": 1.00, "wr": 1.000, "ev": 0.129, "kelly": 0.150, "n": 30},
    {"zone_id": "A7", "ttr_min": 1.5, "ttr_max": 3.0, "dist_min": 0, "dist_max": 15, "odds_min": 0.25, "odds_max": 0.40, "wr": 0.387, "ev": 0.096, "kelly": 0.085, "n": 75},
    {"zone_id": "A8", "ttr_min": 3.0, "ttr_max": 5.0, "dist_min": 30, "dist_max": 60, "odds_min": 0.55, "odds_max": 0.70, "wr": 0.708, "ev": 0.090, "kelly": 0.150, "n": 65},
]

DEATH_ZONES_V3: List[dict] = [
    {"zone_id": "A1_PRUNED", "ttr_min": 3.0, "ttr_max": 5.0, "dist_min": 30, "dist_max": 60, "odds_min": 0.15, "odds_max": 0.25, "wr": 0.276, "ev": -0.116, "n": 29},
    {"zone_id": "A2_PRUNED", "ttr_min": 1.5, "ttr_max": 3.0, "dist_min": 15, "dist_max": 30, "odds_min": 0.15, "odds_max": 0.25, "wr": 0.297, "ev": -0.197, "n": 37},
    {"zone_id": "D1", "ttr_min": 1.5, "ttr_max": 3.0, "dist_min": 0, "dist_max": 15, "odds_min": 0.15, "odds_max": 0.25, "wr": 0.074, "ev": -0.666, "n": 27},
    {"zone_id": "D2", "ttr_min": 1.5, "ttr_max": 3.0, "dist_min": 30, "dist_max": 60, "odds_min": 0.0, "odds_max": 0.15, "wr": 0.053, "ev": -0.553, "n": 38},
    {"zone_id": "D3", "ttr_min": 1.5, "ttr_max": 3.0, "dist_min": 30, "dist_max": 60, "odds_min": 0.15, "odds_max": 0.25, "wr": 0.119, "ev": -0.399, "n": 42},
    {"zone_id": "D4", "ttr_min": 3.0, "ttr_max": 5.0, "dist_min": 0, "dist_max": 15, "odds_min": 0.55, "odds_max": 0.70, "wr": 0.429, "ev": -0.317, "n": 21},
    {"zone_id": "D5", "ttr_min": 3.0, "ttr_max": 5.0, "dist_min": 15, "dist_max": 30, "odds_min": 0.25, "odds_max": 0.40, "wr": 0.250, "ev": -0.276, "n": 52},
    {"zone_id": "D6", "ttr_min": 3.0, "ttr_max": 5.0, "dist_min": 15, "dist_max": 30, "odds_min": 0.55, "odds_max": 0.70, "wr": 0.476, "ev": -0.244, "n": 42},
]


def classify_zone_v3_legacy(
    ttr_minutes: float,
    distance_to_strike_usd: float,
    entry_odds: float,
) -> ZoneResult:
    """Legacy V3 classifier for reference and rollback."""
    for dz in DEATH_ZONES_V3:
        if _match_zone(ttr_minutes, distance_to_strike_usd, entry_odds, dz):
            return ZoneResult(dz["zone_id"], "DEATH", 0, dz["wr"], dz["ev"], 0.0, 0.0, dz["n"])
    for az in ALPHA_ZONES_V3:
        if _match_zone(ttr_minutes, distance_to_strike_usd, entry_odds, az):
            return ZoneResult(az["zone_id"], "ALPHA", 1, az["wr"], az["ev"], az["kelly"], 0.15, az["n"])
    return _NEUTRAL
