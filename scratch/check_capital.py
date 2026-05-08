import sqlite3
import os

db_path = '/app/data/trading.db'
if not os.path.exists(db_path):
    print(f"Error: {db_path} not found")
    exit(1)

conn = sqlite3.connect(db_path)
cur = conn.cursor()
try:
    cur.execute('SELECT * FROM capital ORDER BY rowid DESC LIMIT 1')
    row = cur.fetchone()
    print(f"Capital: {row}")
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()
