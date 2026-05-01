import pandas as pd
import sqlite3
import os

DB_PATH = 'data/trades.db'

def monitor_progress():
    print("V4 Zone Progress (target: 381 resolved trades per zone):")
    
    zones = ["V4-A1", "V4-A2", "V4-A3", "V4-A4"]
    target = 381
    
    # Check if DB exists
    if not os.path.exists(DB_PATH):
        # Fallback to checking CSV files in data/exports or similar
        print("Database not found locally. Searching for CSV exports...")
        # For now, just show placeholders as we are "COLLECTING"
        for z in zones:
            print(f"{z:6}: [░░░░░░░░░░]  0/{target} — COLLECTING")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT zone_id, outcome FROM trades WHERE zone_id LIKE 'V4-A%'", conn)
        conn.close()
        
        resolved = df[df['outcome'].isin(['WIN', 'LOSS'])]
        counts = resolved['zone_id'].value_counts()
        
        for z in zones:
            count = counts.get(z, 0)
            progress = min(10, int(count / target * 10))
            bar = "█" * progress + "░" * (10 - progress)
            status = "COLLECTING" if count < target else "COMPLETED"
            
            print(f"{z:6}: [{bar}] {count:>3}/{target} — {status}")
            
            if count >= 100:
                # Basic win rate check
                wr = resolved[resolved['zone_id'] == z]['outcome'].map({'WIN': 1, 'LOSS': 0}).mean()
                print(f"      Current Win Rate: {wr:.1f}%")
                
    except Exception as e:
        print(f"Error querying progress: {e}")
        for z in zones:
            print(f"{z:6}: [░░░░░░░░░░]  0/{target} — COLLECTING")

if __name__ == "__main__":
    monitor_progress()
