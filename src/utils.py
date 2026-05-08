import math

def compute_position_size(capital: float,
                           swing_prob: float,
                           entry_odds: float,
                           exit_odds: float,
                           kelly_fraction: float = 0.15,
                           max_pct: float = 0.05,
                           fee: float = 0.02,
                           spread: float = 0.005) -> dict:
    """
    Kelly-based sizing untuk binary outcome market.
    Uses RETURN RATIO, not absolute price delta.
    """

    if entry_odds <= 0:
        return {
            'stake_usd': 0.0,
            'shares': 0.0,
            'stake_pct': 0.0,
            'max_win_usd': 0.0,
            'max_loss_usd': 0.0,
            'full_kelly': -1.0,
        }

    # RETURN RATIO PER $1 RISKED
    gross_return_ratio = (
        (exit_odds - entry_odds) / entry_odds
    )

    # AFTER FEES
    b_adj = gross_return_ratio * (1 - fee)

    q = 1 - swing_prob

    if b_adj <= 0:
        full_kelly = -1.0
    else:
        full_kelly = (
            (b_adj * swing_prob) - q
        ) / b_adj

    # Clamp
    stake_pct = min(
        max(kelly_fraction * full_kelly, 0.0),
        max_pct
    )

    stake_usd = capital * stake_pct

    shares = (
        stake_usd / entry_odds
        if entry_odds > 0
        else 0.0
    )

    max_win_usd = (
        (exit_odds - entry_odds)
        * (1 - fee)
        * shares
        - spread
    )

    max_loss_usd = -stake_usd - spread

    return {
        'stake_usd': round(stake_usd, 2),
        'shares': round(shares, 2),
        'stake_pct': stake_pct,
        'max_win_usd': round(max_win_usd, 2),
        'max_loss_usd': round(max_loss_usd, 2),
        'full_kelly': round(full_kelly, 4),
    }

def fmt_money(v: float) -> str:
    sign = "+" if v >= 0 else "-"
    return f"{sign}${abs(v):.2f}"
