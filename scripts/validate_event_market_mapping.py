"""
Validate Polymarket event-vs-market mapping for BTC strike markets.

Goal:
1) Prove/disprove hypothesis that strike is often in child market metadata
   (e.g. groupItemTitle), not always in parent-like question text.
2) Show which rows are actually tradable for the bot (token IDs, orderbook, TTR).

Usage examples:
  python scripts/validate_event_market_mapping.py --slug what-price-will-bitcoin-hit-on-april-8
  python scripts/validate_event_market_mapping.py --event-search "bitcoin up or down" --limit-events 3
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import httpx

GAMMA_API_BASE = "https://gamma-api.polymarket.com"

STRIKE_PATTERNS = [
    r"\$([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",
    r"above\s+\$?([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",
    r"below\s+\$?([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",
    r"up\s+or\s+down\s+from\s+\$?([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",
]


@dataclass
class MarketCheck:
    market_id: str
    question: str
    group_item_title: str
    event_slug: str
    strike_question: Optional[float]
    strike_group: Optional[float]
    has_tokens: bool
    is_binary_yes_no: bool
    enable_orderbook: bool
    active: bool
    closed: bool
    ttr_minutes: Optional[float]
    settlement_binance_1m: bool

    @property
    def inferred_strike(self) -> Optional[float]:
        return self.strike_question if self.strike_question is not None else self.strike_group

    @property
    def parent_like_missing_strike(self) -> bool:
        q = self.question.lower()
        return ("up or down" in q) and (self.strike_question is None)

    @property
    def tradable_fit(self) -> bool:
        if not self.inferred_strike:
            return False
        if not self.has_tokens or not self.enable_orderbook:
            return False
        if not self.active or self.closed:
            return False
        if self.ttr_minutes is not None and self.ttr_minutes <= 0:
            return False
        return self.settlement_binance_1m


def _safe_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _extract_strike(text: str) -> Optional[float]:
    if not text:
        return None
    for pattern in STRIKE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        try:
            val = float(match.group(1).replace(",", ""))
            if 1_000 <= val <= 1_000_000:
                return val
        except Exception:
            continue
    return None


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _ascii_text(value: str) -> str:
    return value.encode("ascii", errors="replace").decode("ascii")


def _infer_binance_1m(market: dict[str, Any]) -> bool:
    rules = " ".join(
        [
            str(market.get("description", "")),
            str(market.get("resolutionSource", "")),
            str(market.get("uma_resolution_rules", "")),
        ]
    ).lower()
    return ("binance" in rules) and ("1m" in rules or "1 minute" in rules or "1-minute" in rules)


def _extract_has_tokens(market: dict[str, Any]) -> bool:
    clob_ids = _safe_json_list(market.get("clobTokenIds"))
    if len(clob_ids) >= 2 and all(str(x).strip() for x in clob_ids[:2]):
        return True

    tokens = market.get("tokens", [])
    if isinstance(tokens, list):
        yes = False
        no = False
        for t in tokens:
            outcome = str(t.get("outcome", "")).strip().upper()
            token_id = t.get("token_id") or t.get("tokenId") or t.get("clobTokenId") or ""
            if outcome == "YES" and str(token_id).strip():
                yes = True
            if outcome == "NO" and str(token_id).strip():
                no = True
        return yes and no
    return False


def _is_binary_yes_no(market: dict[str, Any]) -> bool:
    outcomes = [str(x).strip().lower() for x in _safe_json_list(market.get("outcomes"))]
    return set(outcomes) == {"yes", "no"}


def _iter_target_events(
    client: httpx.Client,
    slug: Optional[str],
    event_search: Optional[str],
    limit_events: int,
) -> Iterable[dict[str, Any]]:
    if slug:
        r = client.get(f"{GAMMA_API_BASE}/events", params={"slug": slug})
        r.raise_for_status()
        events = r.json()
        if isinstance(events, list):
            yield from events[:limit_events]
        return

    params: dict[str, Any] = {
        "active": "true",
        "closed": "false",
        "limit": max(20, limit_events * 10),
        "order": "volume",
        "ascending": "false",
    }
    if event_search:
        params["search"] = event_search
    r = client.get(f"{GAMMA_API_BASE}/events", params=params)
    r.raise_for_status()
    events = r.json()
    if not isinstance(events, list):
        return
    for e in events[:limit_events]:
        yield e


def _event_markets_from_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    markets = event.get("markets", [])
    if isinstance(markets, list) and markets:
        return markets
    return []


def run_validation(slug: Optional[str], event_search: Optional[str], limit_events: int) -> int:
    now = datetime.now(timezone.utc)
    checks: list[MarketCheck] = []

    with httpx.Client(timeout=30.0) as client:
        events = list(_iter_target_events(client, slug, event_search, limit_events))
        if not events:
            print("No events found.")
            return 1

        for event in events:
            event_slug = str(event.get("slug", ""))
            markets = _event_markets_from_event(event)
            if not markets:
                r = client.get(
                    f"{GAMMA_API_BASE}/markets",
                    params={"active": "true", "closed": "false", "eventId": event.get("id"), "limit": 500},
                )
                r.raise_for_status()
                payload = r.json()
                markets = payload if isinstance(payload, list) else []

            for m in markets:
                q = str(m.get("question", ""))
                g = str(m.get("groupItemTitle", ""))
                end_str = m.get("endDate") or m.get("endDateIso")
                end_dt = _parse_ts(end_str if isinstance(end_str, str) else None)
                ttr = None
                if end_dt is not None:
                    ttr = (end_dt - now).total_seconds() / 60.0

                checks.append(
                    MarketCheck(
                        market_id=str(m.get("conditionId") or m.get("condition_id") or m.get("id") or ""),
                        question=q,
                        group_item_title=g,
                        event_slug=event_slug,
                        strike_question=_extract_strike(q),
                        strike_group=_extract_strike(g),
                        has_tokens=_extract_has_tokens(m),
                        is_binary_yes_no=_is_binary_yes_no(m),
                        enable_orderbook=bool(m.get("enableOrderBook")),
                        active=bool(m.get("active", False)),
                        closed=bool(m.get("closed", False)),
                        ttr_minutes=ttr,
                        settlement_binance_1m=_infer_binance_1m(m),
                    )
                )

    parent_like = [c for c in checks if c.parent_like_missing_strike]
    strike_from_group = [c for c in checks if c.strike_question is None and c.strike_group is not None]
    tradable_fit = [c for c in checks if c.tradable_fit]

    print("=== VALIDATION SUMMARY ===")
    print(f"markets_scanned={len(checks)}")
    print(f"parent_like_missing_strike={len(parent_like)}")
    print(f"strike_recovered_from_group_item_title={len(strike_from_group)}")
    print(f"tradable_fit_candidates={len(tradable_fit)}")

    print("\n=== SAMPLE: PARENT-LIKE MISSING STRIKE ===")
    for c in parent_like[:10]:
        q = _ascii_text(c.question[:90])
        g = _ascii_text(c.group_item_title[:60])
        ttr = "NA" if c.ttr_minutes is None else f"{c.ttr_minutes:.1f}"
        print(
            f"- market_id={c.market_id} ttr_min={ttr} "
            f"tokens={c.has_tokens} orderbook={c.enable_orderbook} "
            f"question='{q}' group='{g}'"
        )

    print("\n=== SAMPLE: STRIKE RECOVERED FROM GROUP ITEM ===")
    for c in strike_from_group[:10]:
        q = _ascii_text(c.question[:70])
        g = _ascii_text(c.group_item_title[:50])
        print(
            f"- market_id={c.market_id} strike={c.inferred_strike} "
            f"question='{q}' group='{g}'"
        )

    print("\n=== TOP TRADABLE FIT CANDIDATES ===")
    # Sort by shortest positive TTR first (more immediately actionable)
    tradable_fit_sorted = sorted(
        tradable_fit,
        key=lambda x: (x.ttr_minutes if x.ttr_minutes is not None else 1e18),
    )
    for c in tradable_fit_sorted[:15]:
        ttr = "NA" if c.ttr_minutes is None else f"{c.ttr_minutes:.1f}"
        q = _ascii_text(c.question[:70])
        g = _ascii_text(c.group_item_title[:50])
        print(
            f"- market_id={c.market_id} strike={c.inferred_strike} ttr_min={ttr} "
            f"tokens={c.has_tokens} binance1m={c.settlement_binance_1m} "
            f"question='{q}' group='{g}'"
        )

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Polymarket event-market mapping for BTC discovery.")
    parser.add_argument("--slug", type=str, default=None, help="Exact event slug (recommended for targeted validation).")
    parser.add_argument(
        "--event-search",
        type=str,
        default="bitcoin",
        help="Event search term when slug is not provided.",
    )
    parser.add_argument("--limit-events", type=int, default=3, help="How many events to inspect.")
    args = parser.parse_args()

    code = run_validation(
        slug=args.slug,
        event_search=args.event_search,
        limit_events=max(1, args.limit_events),
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
