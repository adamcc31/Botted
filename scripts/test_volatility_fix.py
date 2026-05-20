import numpy as np
import pandas as pd
import math

def compute_btc_realized_vol_live(binance_ohlcv_30m: pd.DataFrame) -> float:
    """
    Compute annualized realized volatility from 30min Binance OHLCV.
    """
    closes = binance_ohlcv_30m['close']
    log_returns = np.log(closes / closes.shift(1)).dropna()
    sigma = log_returns.std()
    annualization_factor = np.sqrt(105120)
    volatility_annualized = sigma * annualization_factor
    return float(volatility_annualized)

def run_test():
    print("Running Volatility Feature Fix Unit Tests...")
    
    # 1. Create mock 5-minute bars with 50% annualized volatility
    # Trading year = 105,120 bars of 5-minutes.
    # Annualized vol = 0.50 -> 5-min std = 0.50 / sqrt(105120)
    std_5m = 0.50 / np.sqrt(105120)
    print(f"Target 5m log-return std: {std_5m:.6f}")
    
    np.random.seed(42)
    log_returns = np.random.normal(0, std_5m, 5)  # 6 closes -> 5 returns
    closes = [65000.0]
    for r in log_returns:
        closes.append(closes[-1] * np.exp(r))
        
    df_mock = pd.DataFrame({'close': closes})
    vol = compute_btc_realized_vol_live(df_mock)
    print(f"Calculated Volatility: {vol:.4f}")
    
    # Check if within typical range
    assert 0.30 <= vol <= 0.80, f"Expected volatility between 0.30 and 0.80, got {vol:.4f}"
    print("Test 1 (Realistic Volatility Check) PASSED!")

    # 2. Check low volatility
    std_5m_low = 0.05 / np.sqrt(105120)
    log_returns_low = np.random.normal(0, std_5m_low, 5)
    closes_low = [65000.0]
    for r in log_returns_low:
        closes_low.append(closes_low[-1] * np.exp(r))
    df_mock_low = pd.DataFrame({'close': closes_low})
    vol_low = compute_btc_realized_vol_live(df_mock_low)
    print(f"Low Volatility: {vol_low:.4f}")
    assert vol_low < 0.15, f"Expected low volatility, got {vol_low:.4f}"
    print("Test 2 (Low Volatility Check) PASSED!")

    # 3. Check high volatility
    std_5m_high = 1.50 / np.sqrt(105120)
    log_returns_high = np.random.normal(0, std_5m_high, 5)
    closes_high = [65000.0]
    for r in log_returns_high:
        closes_high.append(closes_high[-1] * np.exp(r))
    df_mock_high = pd.DataFrame({'close': closes_high})
    vol_high = compute_btc_realized_vol_live(df_mock_high)
    print(f"High Volatility: {vol_high:.4f}")
    assert vol_high > 1.0, f"Expected high volatility, got {vol_high:.4f}"
    print("Test 3 (High Volatility Check) PASSED!")
    
    print("All volatility tests passed successfully!")

if __name__ == "__main__":
    run_test()
