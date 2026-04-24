import pandas as pd
import numpy as np
import structlog
from pathlib import Path

logger = structlog.get_logger(__name__)

def analyze_lead_lag():
    logger.info("loading_hf_dataset", path="dataset/btc_5m_hf_ticks.parquet")
    df = pd.read_parquet('dataset/btc_5m_hf_ticks.parquet')
    
    # Sort by datetime to ensure time series integrity
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 1. Basic Correlation (Static)
    corr = df[['binance_price', 'up_mid']].corr().iloc[0, 1]
    logger.info("static_correlation", raw_pearson=round(corr, 4))
    
    # 2. Returns Correlation (1-minute delta)
    # Sampling to 1-minute to reduce noise
    df_1m = df.set_index('datetime').resample('1min').last().dropna()
    
    df_1m['binance_ret'] = df_1m['binance_price'].pct_change()
    df_1m['poly_ret'] = df_1m['up_mid'].pct_change()
    
    ret_corr = df_1m[['binance_ret', 'poly_ret']].corr().iloc[0, 1]
    logger.info("returns_correlation", pearson_1m=round(ret_corr, 4))
    
    # 3. Lead-Lag Analysis (Shifting Binance)
    # Does Binance moving NOW predict Polymarket moving LATER?
    lags = [1, 2, 5, 10] # minutes
    for lag in lags:
        df_1m[f'binance_ret_lag_{lag}'] = df_1m['binance_ret'].shift(lag)
        lag_corr = df_1m[[f'binance_ret_lag_{lag}', 'poly_ret']].corr().iloc[0, 1]
        logger.info("lagged_correlation", shift_minutes=lag, pearson=round(lag_corr, 4))

    # 4. Target Definition (Delta 5m)
    # We want to predict Polymarket move in next 5 mins
    df_1m['target_5m'] = df_1m['poly_ret'].shift(-5)
    final_corr = df_1m[['binance_ret', 'target_5m']].corr().iloc[0, 1]
    logger.info("predictive_power", binance_now_vs_poly_5m_later=round(final_corr, 4))

if __name__ == "__main__":
    analyze_lead_lag()
