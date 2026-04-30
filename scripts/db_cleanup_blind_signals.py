"""
db_cleanup_blind_signals.py — Mark contaminated signals from blind P=0.5 execution.

Run on production Railway volume:
  python scripts/db_cleanup_blind_signals.py --db /path/to/bot.db

This script:
  1. Counts signals with NULL entry_odds, tfm_value, or missing ML features
  2. Sets is_complete = 0 on those rows to exclude from analysis
  3. Reports the number of affected rows
"""

import argparse
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="bot_data.db",
                        help="Path to SQLite database")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count only, don't update")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Check if is_complete column exists
    cursor.execute("PRAGMA table_info(signals)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if "is_complete" not in columns:
        print("Adding 'is_complete' column (default=1)...")
        cursor.execute("ALTER TABLE signals ADD COLUMN is_complete INTEGER DEFAULT 1")
        conn.commit()

    # Count contaminated rows
    query = """
        SELECT COUNT(*) FROM signals
        WHERE entry_odds IS NULL
           OR tfm_norm IS NULL
           OR obi_value IS NULL
           OR vol_percentile IS NULL
           OR depth_ratio IS NULL
           OR contest_urgency IS NULL
    """
    cursor.execute(query)
    count = cursor.fetchone()[0]
    print(f"Contaminated rows found: {count}")

    if count == 0:
        print("No cleanup needed.")
        conn.close()
        return

    if args.dry_run:
        print("Dry-run mode — no changes made.")
        conn.close()
        return

    # Update
    update_query = """
        UPDATE signals SET is_complete = 0
        WHERE entry_odds IS NULL
           OR tfm_norm IS NULL
           OR obi_value IS NULL
           OR vol_percentile IS NULL
           OR depth_ratio IS NULL
           OR contest_urgency IS NULL
    """
    cursor.execute(update_query)
    affected = cursor.rowcount
    conn.commit()
    print(f"Marked {affected} rows as is_complete=0")

    conn.close()


if __name__ == "__main__":
    main()
