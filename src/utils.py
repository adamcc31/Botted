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
    
    [FIX-EV-1] Returns zero stake if Kelly criterion is negative (negative edge).
    This prevents any capital deployment into provably negative EV situations.
    V1 gold standard: RiskManager._compute_bet_size returns (0,0,0) when zone.kelly_fraction <= 0.
    """

    zero_result = {
        'stake_usd': 0.0,
        'shares': 0.0,
        'stake_pct': 0.0,
        'max_win_usd': 0.0,
        'max_loss_usd': 0.0,
        'full_kelly': -1.0,
        'ev': 0.0,
    }

    if entry_odds <= 0 or entry_odds >= 1.0:
        return zero_result

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

    # [FIX-EV-1] Early exit on negative Kelly (negative EV)
    # Threshold -0.05 allows tiny negative rounds but blocks clearly bad bets.
    # Production data shows UNDERDOG_<35% has full_kelly ~ -0.75 (EV=-29%).
    if full_kelly < -0.05:
        return {**zero_result, 'full_kelly': round(full_kelly, 4)}

    # [FIX-V5-RISK] Implement Half-Kelly (0.5 multiplier) and Hard Ceiling ($20.0)
    # 1. Apply 0.5 multiplier to fractional Kelly to dampen drawdown volatility
    risk_adjusted_kelly = kelly_fraction * full_kelly * 0.5
    
    # 2. Clamp percentage
    stake_pct = min(
        max(risk_adjusted_kelly, 0.0),
        max_pct
    )

    # 3. Calculate nominal stake and apply $20.0 HARD CEILING for slippage protection
    # Even if compounding equity grows, we do not exceed $20 per trade on thin 5m markets.
    stake_usd = min(capital * stake_pct, 20.0)
    
    # Recalculate stake_pct after nominal ceiling
    stake_pct = stake_usd / capital if capital > 0 else 0.0

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
    
    # Expected Value (for diagnostic logging)
    ev = swing_prob * (gross_return_ratio * (1 - fee)) - q

    return {
        'stake_usd': round(stake_usd, 2),
        'shares': round(shares, 2),
        'stake_pct': stake_pct,
        'max_win_usd': round(max_win_usd, 2),
        'max_loss_usd': round(max_loss_usd, 2),
        'full_kelly': round(full_kelly, 4),
        'ev': round(ev, 4),
    }

def fmt_money(v: float) -> str:
    sign = "+" if v >= 0 else "-"
    return f"{sign}${abs(v):.2f}"
