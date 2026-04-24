import pandas as pd
import matplotlib.pyplot as plt

def visualize_momentum():
    df = pd.read_parquet('dataset/btc_5m_hf_ticks.parquet')
    
    # Selection: One specific event_slug that has many ticks
    event = df['event_slug'].iloc[1000] # Pick a sample market
    subset = df[df['event_slug'] == event].sort_values('datetime')
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Binance Price', color='tab:blue')
    ax1.plot(subset['datetime'], subset['binance_price'], color='tab:blue', label='Binance')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Polymarket Mid (Up)', color='tab:red')
    ax2.plot(subset['datetime'], subset['up_mid'], color='tab:red', label='Poly Up Mid')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.title(f"Momentum Check: {event}")
    fig.tight_layout()
    plt.savefig('tmp/momentum_viz.png')
    print(f"Visualization saved to tmp/momentum_viz.png for event: {event}")

if __name__ == "__main__":
    visualize_momentum()
